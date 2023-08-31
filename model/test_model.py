import torch
import torch.nn as nn
from transformers.models.bert.modeling_bert import BertModel, BertOnlyMLMHead, BertPreTrainedModel
from transformers import BertModel
from torch_geometric.nn import GAT, LSTM

class BertKGE(BertModel):

    def __init__(self, config):
        super().__init__(config)

        # 初始化GAT
        self.gat = GAT(config.hidden_size, config.num_attention_heads)

        # 初始化LSTM
        self.lstm = LSTM(config.hidden_size, config.num_attention_heads)

    def forward(self, input_ids, attention_mask, token_type_ids,
                head_mask, inputs_embeds, adj, edge_type, batch_inputs):

        # 获取BERT的输出
        output = self.bert(input_ids, attention_mask, token_type_ids,
                          head_mask, inputs_embeds)

        # 获取头实体和尾实体的嵌入
        head_embeddings = output[0][batch_inputs[:, 0], :]
        tail_embeddings = output[0][batch_inputs[:, 2], :]

        # 使用GAT融合邻居信息
        head_embeddings, edge_embeddings = self.gat(
            head_embeddings, adj, edge_type)

        # 使用LSTM融合从头实体出发的多跳路径信息
        head_embeddings, _ = self.lstm(head_embeddings)

        # 返回最终的嵌入
        return head_embeddings, tail_embeddings
    


class PoolingTripletDecoder(nn.Module):

    def __init__(self, margin):
        super().__init__()
        self.margin = margin

    def forward(self, encoding_triplet):
        assert encoding_triplet.size(1) == 3
        z = (
            self.margin
            - 0.5 * (
                ((encoding_triplet[:, 0, :] + encoding_triplet[:, 1, :] - encoding_triplet[:, 2, :]) ** 2).sum(dim=-1)
                / encoding_triplet.size(-1) ** 0.5
            )
        )
        return z


class BertPoolingForTripletPrediction(BertPreTrainedModel):

    def __init__(self, config, margin, text_loss_weight, pos_weight):
        super().__init__(config)
        self.bert = BertModel(config)
        self.trip_decoder = PoolingTripletDecoder(margin)
        embed_size = config.vocab_size
        config.vocab_size = config.real_vocab_size
        self.cls = BertOnlyMLMHead(config)
        self.predict_mode = False
        self.pos_weight = pos_weight
        self.text_loss_weight = text_loss_weight
        config.vocab_size = embed_size
        self.init_weights()
        # self.lstm_fusion = LSTMFusionModel(lstm_input_size, lstm_hidden_size)
        # self.gat_fusion = GATFusionModel(gat_input_size, gat_num_nodes, gat_hidden_size)

    def set_train_mode(self):
        self.predict_mode = False

    def set_predict_mode(self):
        self.predict_mode = True

    def pooling_encoder(self, sequence_output, pooling_mask):
        pooling_mask = pooling_mask.type_as(sequence_output)
        mask_len = torch.sum(pooling_mask, dim=1, keepdim=True)
        mask_sum = torch.matmul(pooling_mask.unsqueeze(1), sequence_output).squeeze()
        mask_mean = torch.div(mask_sum, mask_len)
        return mask_mean

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        pooling_head_mask=None,
        pooling_rel_mask=None,
        pooling_tail_mask=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        mlm_labels=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        **kwargs
    ):
        assert kwargs == {}, f'Unexpected keyword arguments: {list(kwargs.keys())}.'
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states
        )
        sequence_output = outputs[0]
        prediction_scores = self.cls(sequence_output)
        # lstm_fused = self.lstm_fusion(lstm_inputs)
        # gat_fused = self.gat_fusion(gat_inputs)
        head_encoding = self.pooling_encoder(sequence_output, pooling_head_mask)
        rel_encoding = self.pooling_encoder(sequence_output, pooling_rel_mask)
        tail_encoding = self.pooling_encoder(sequence_output, pooling_tail_mask)
        trip_encoding = torch.stack([head_encoding, rel_encoding, tail_encoding], dim=1)
        triplet_scores = self.trip_decoder(trip_encoding)
        if self.predict_mode:
            return (triplet_scores,)
        
        # 训练模型，计算损失函数
        loss = None
        
        trip_loss_fct = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight)
        lm_loss_fct = nn.CrossEntropyLoss()
        if labels is not None:
            loss = trip_loss_fct(triplet_scores, labels.float())
        if mlm_labels is not None and prediction_scores is not None:
            masked_lm_loss = (
                self.text_loss_weight
                * lm_loss_fct(prediction_scores.view(-1, self.config.real_vocab_size), mlm_labels.view(-1))
            )
            loss = masked_lm_loss if loss is None else loss + masked_lm_loss
        output = (triplet_scores,)
        return (loss,) + output if loss is not None else output
