import logging
import math
from typing import List, Optional

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Sampler
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import RandomSampler, SequentialSampler
from tqdm.auto import tqdm
from transformers import Trainer
from transformers.trainer import SequentialDistributedSampler
from transformers.trainer_utils import EvalLoopOutput, EvalPrediction

logger = logging.getLogger(__name__)


class GroupRandomSampler(Sampler):

    def __init__(self, data_source, group):
        self.data_source = data_source
        self.group = group
        assert len(data_source) % group == 0

    @property
    def num_samples(self):
        return len(self.data_source)

    def __iter__(self):
        n = len(self.data_source) // self.group
        perm = torch.randperm(n)
        perm = (perm[:, None] * self.group + torch.arange(self.group)).view(-1)
        return iter(perm.tolist())

    def __len__(self):
        return self.num_samples


class GroupDistributredSampler(DistributedSampler):
    def __init__(self, dataset, group_size, num_replicas=None, rank=None, shuffle=True, seed=0):
        self.group_size = group_size
        assert len(
            dataset) % group_size == 0, f'length of dataset should be a multiple of group size, but get length {len(dataset)} and group size {group_size}'
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError('Requires distributed package to be available')
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError('Requires distributed package to be available')
            rank = dist.get_rank()
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.num_groups = math.ceil(len(self.dataset) / self.group_size / self.num_replicas)
        self.num_samples = self.num_groups * self.group_size
        self.total_size = self.num_samples * self.num_replicas
        self.shuffle = shuffle
        self.seed = seed

    def __iter__(self):
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            n = len(self.dataset) // self.group_size
            perm = torch.randperm(n, generator=g)
            perm = (perm[:, None] * self.group_size + torch.arange(self.group_size)).view(-1)
            indices = perm.tolist()
        else:
            indices = list(range(len(self.dataset)))
        indices += indices[:self.total_size - len(indices)]
        assert len(indices) == self.total_size
        sub_indices = []
        for group_id in range(self.num_groups):
            sub_indices += indices[(group_id * self.num_replicas + self.rank) * self.group_size
                                   :(group_id * self.num_replicas + self.rank + 1) * self.group_size]
        assert len(sub_indices) == self.num_samples
        return iter(sub_indices)


class KGCTrainer(Trainer):

    def use_group_shuffle(self, num_neg):
        self.group_shuffle = True
        self.num_neg = num_neg

    def stop_group_shuffle(self):
        self.group_shuffle = False

    def get_train_dataloader(self) -> DataLoader:
        if self.train_dataset is None:
            raise ValueError('Trainer: training requires a train_dataset.')
        if hasattr(self, 'group_shuffle') and self.group_shuffle:
            train_sampler = self._get_group_sampler()
        else:
            train_sampler = self._get_train_sampler()
        return DataLoader(
            self.train_dataset,
            batch_size=self.args.train_batch_size,
            sampler=train_sampler,
            collate_fn=self.data_collator,
            drop_last=self.args.dataloader_drop_last
        )

    def get_eval_dataloader(self, eval_dataset=None) -> DataLoader:
        if eval_dataset is None and self.eval_dataset is None:
            raise ValueError('Trainer: evaluation requires an eval_dataset.')
        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
        if hasattr(self, 'group_shuffle') and self.group_shuffle:
            sampler = SequentialSampler(eval_dataset)
        sampler = SequentialSampler(eval_dataset)
        data_loader = DataLoader(
            eval_dataset,
            sampler=sampler,
            batch_size=self.args.eval_batch_size,
            collate_fn=self.data_collator,
            drop_last=self.args.dataloader_drop_last
        )
        return data_loader

    def get_test_dataloader(self, test_dataset) -> DataLoader:
        sampler = SequentialSampler(test_dataset)
        data_loader = DataLoader(
            test_dataset,
            sampler=sampler,
            batch_size=self.args.eval_batch_size,
            collate_fn=self.data_collator,
            drop_last=self.args.dataloader_drop_last
        )
        return data_loader

    def _get_group_sampler(self) -> Optional[torch.utils.data.sampler.Sampler]:
        if isinstance(self.train_dataset, torch.utils.data.IterableDataset):
            return None
        else:
            return (
                GroupRandomSampler(self.train_dataset, self.num_neg * 3 + 1)
            )

    def _get_train_sampler(self) -> Optional[torch.utils.data.sampler.Sampler]:
        if isinstance(self.train_dataset, torch.utils.data.IterableDataset):
            return None
        else:
            return (
                RandomSampler(self.train_dataset)
            )

    def prediction_loop(
        self,
        dataloader: DataLoader,
        description: str,
        prediction_loss_only: Optional[bool] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> EvalLoopOutput:
        prediction_loss_only = prediction_loss_only if prediction_loss_only is not None else self.args.prediction_loss_only
        model = self.model
        if self.args.n_gpu > 1:
            model = torch.nn.DataParallel(model)
        else:
            model = self.model
        batch_size = dataloader.batch_size
        logger.info('***** Running %s *****', description)
        logger.info('  Num examples = %d', self.num_examples(dataloader))
        logger.info('  Batch size = %d', batch_size)
        eval_losses: List[float] = []
        preds: torch.Tensor = None
        label_ids: torch.Tensor = None
        model.eval()
        if self.args.past_index >= 0:
            past = None
        for inputs in tqdm(dataloader, desc=description, disable=False):
            has_labels = any((inputs.get(k) is not None for k in ['labels', 'lm_labels', 'masked_lm_labels']))
            for (k, v) in inputs.items():
                if isinstance(v, torch.Tensor):
                    inputs[k] = v.to(self.args.device)
            if self.args.past_index >= 0:
                inputs['mems'] = past
            with torch.no_grad():
                outputs = model(**inputs)
                if has_labels:
                    (step_eval_loss, logits) = outputs[:2]
                    eval_losses += [step_eval_loss.mean().item()]
                else:
                    logits = outputs[0]
                if self.args.past_index >= 0:
                    past = outputs[self.args.past_index if has_labels else self.args.past_index - 1]
            if not prediction_loss_only:
                if preds is None:
                    preds = logits.detach()
                else:
                    preds = torch.cat((preds, logits.detach()), dim=0)
                if inputs.get('labels') is not None:
                    if label_ids is None:
                        label_ids = inputs['labels'].detach()
                    else:
                        label_ids = torch.cat((label_ids, inputs['labels'].detach()), dim=0)
        if preds is not None:
            preds = preds.cpu().numpy()
        if label_ids is not None:
            label_ids = label_ids.cpu().numpy()

        num_examples = self.num_examples(dataloader)

        if self.compute_metrics is not None and preds is not None and (label_ids is not None):
            metrics = self.compute_metrics(EvalPrediction(predictions=preds, label_ids=label_ids))
                
        else:
            metrics = {}
        if len(eval_losses) > 0:
            metrics[f"{metric_key_prefix}_loss"] = np.mean(eval_losses)
        for key in list(metrics.keys()):
            if not key.startswith('eval_'):
                metrics[f'eval_{key}'] = metrics.pop(key)
        return EvalLoopOutput(predictions=preds, label_ids=label_ids, metrics=metrics,num_samples=num_examples)
