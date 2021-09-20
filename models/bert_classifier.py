import torch
import torch.nn as nn
from BERT.bert_text_dataset import BERT_PRETRAINED_MODEL
from constants import MAX_KB_ARR_LENGTH
from transformers.tokenization_bert import BertTokenizer
from pytorch_lightning import LightningModule
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from typing import Callable, List
from transformers import BertModel, BertConfig
from utils_bert import load_new_tokens
import os


class LightningHyperparameters:
    def __init__(self, *initial_data, **kwargs):
        for dictionary in initial_data:
            for key in dictionary:
                setattr(self, key, dictionary[key])
        for key in kwargs:
            setattr(self, key, kwargs[key])


class Linear_Layer(nn.Module):
    def __init__(self, input_size: int, output_size: int, dropout: float = None,
                 batch_norm: bool = False, layer_norm: bool = False, activation: Callable = F.relu):
        super().__init__()
        self.linear = nn.Linear(input_size, output_size)
        if type(dropout) is float and dropout > 0.0:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None
        if batch_norm:
            self.batch_norm = nn.BatchNorm1d(output_size)
        else:
            self.batch_norm = None
        if layer_norm:
            self.layer_norm = nn.LayerNorm(output_size)
        else:
            self.layer_norm = None
        self.activation = activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        linear_out = self.linear(x)
        if self.dropout:
            linear_out = self.dropout(linear_out)
        if self.batch_norm:
            linear_out = self.batch_norm(linear_out)
        if self.layer_norm:
            linear_out = self.layer_norm(linear_out)
        if self.activation:
            linear_out = self.activation(linear_out)
        return linear_out


class HAN_Attention_Pooler_Layer(nn.Module):
    def __init__(self, h_dim: int):
        super().__init__()
        self.linear_in = Linear_Layer(h_dim, h_dim, activation=torch.tanh)
        self.softmax = nn.Softmax(dim=-1)
        self.decoder_h = nn.Parameter(torch.randn(h_dim), requires_grad=True)

    def forward(self, encoder_h_seq: torch.Tensor, mask: torch.Tensor = None) -> (torch.Tensor, torch.Tensor):
        """
        Args:
            encoder_h_seq (:class:`torch.FloatTensor` [batch size, sequence length, dimensions]): Data
                over which to apply the attention mechanism.
            mask (:class:`torch.BoolTensor` [batch size, sequence length]): Mask
                for padded sequences of variable length.

        Returns:
            :class:`tuple` with `output` and `weights`:
            * **output** (:class:`torch.LongTensor` [batch size, output length, dimensions]):
              Tensor containing the attended features.
            * **weights** (:class:`torch.FloatTensor` [batch size, output length, query length]):
              Tensor containing attention weights.
        """
        batch_size, seq_len, h_dim = encoder_h_seq.size()

        encoder_h_seq = self.linear_in(encoder_h_seq.contiguous().view(-1, h_dim))
        encoder_h_seq = encoder_h_seq.view(batch_size, seq_len, h_dim)

        # (batch_size, 1, dimensions) * (batch_size, seq_len, dimensions) -> (batch_size, seq_len)
        attention_scores = torch.bmm(self.decoder_h.expand((batch_size, h_dim)).unsqueeze(1), encoder_h_seq.transpose(1, 2).contiguous())

        # Compute weights across every context sequence
        attention_scores = attention_scores.view(batch_size, -1)
        if mask is not None:
            if mask.dtype is not torch.bool:
                mask = mask.bool()
            attention_scores[~mask] = float("-inf")
        attention_weights = self.softmax(attention_scores)

        # (batch_size, 1, query_len) * (batch_size, query_len, dimensions) -> (batch_size, dimensions)
        output = torch.bmm(attention_weights.unsqueeze(1), encoder_h_seq).squeeze()
        return output, attention_weights

    @staticmethod
    def create_mask(valid_lengths: torch.Tensor, max_len: int = None) -> torch.Tensor:
        if not max_len:
            max_len = valid_lengths.max()
        return torch.arange(max_len, dtype=valid_lengths.dtype, device=valid_lengths.device).expand(len(valid_lengths), max_len) < valid_lengths.unsqueeze(1)


class BertPretrainedClassifier(nn.Module):
    def __init__(self, batch_size: int = 8, dropout: float = 0.1, label_size: int = 2,
                 loss_func: Callable = nn.BCELoss, bert_pretrained_model: str = BERT_PRETRAINED_MODEL,
                 bert_state_dict: str = None, name: str = "OOB", device: torch.device = None):
        super(BertPretrainedClassifier, self).__init__()
        self.name = f"{self.__class__.__name__}-{name}"
        self.batch_size = batch_size
        self.label_size = label_size
        self.dropout = dropout
        self.loss_func = loss_func()
        self.device = device
        self.bert_pretrained_model = bert_pretrained_model
        self.bert_state_dict = bert_state_dict
        self.bert = BertPretrainedClassifier.load_frozen_bert(bert_pretrained_model, bert_state_dict)
        self.tokenizer = BertTokenizer.from_pretrained(bert_pretrained_model,
                                                       do_lower_case=bool(BERT_PRETRAINED_MODEL.endswith("uncased")))
        new_tokens = load_new_tokens()
        self.tokenizer.add_tokens(new_tokens)
        self.bert.resize_token_embeddings(len(self.tokenizer))
        self.embeddings = self.bert.embeddings.word_embeddings
        self.hidden_size = self.bert.config.hidden_size
        self.pooler = HAN_Attention_Pooler_Layer(self.hidden_size)
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=-1)
        self.loss_func_ce = CrossEntropyLoss()

    @staticmethod
    def load_frozen_bert(bert_pretrained_model, bert_state_dict):
        if bert_state_dict:
            print(f"Loading pretrained BERT model from: %s" % bert_state_dict)
            config = BertConfig.from_pretrained(os.path.dirname(bert_state_dict))
            fine_tuned_state_dict = torch.load(bert_state_dict, map_location=torch.device('cpu'))
            bert = BertModel.from_pretrained(bert_pretrained_model, config=config, state_dict=fine_tuned_state_dict)
        else:
            print(f"Loading pretrained BERT model from scratch")
            bert = BertModel.from_pretrained(bert_pretrained_model)
        for p in bert.parameters():
            p.requires_grad = False
        return bert

    def forward(self, input_ids, input_mask, labels, kb_arr):  # kb_arr需要添加到日志解析中去！同时labels是否需要padding以对齐？
        last_hidden_states_seq, _ = self.bert(input_ids, attention_mask=input_mask)
        pooled_seq_vector, attention_weights = self.pooler(last_hidden_states_seq, input_mask)
        kb_emb = self.embeddings(kb_arr)

        u_temp = pooled_seq_vector.unsqueeze(1).expand_as(kb_emb)
        prob_logit = torch.sum(kb_emb*u_temp, dim=2)

        # # Masking pad kb tokens
        # comparison_tensor = torch.ones_like(kb_arr, dtype=torch.int64) * EntityPredictionDataset.PAD_TOKEN_IDX  # Matrix to compare
        # mask = torch.eq(kb_arr, comparison_tensor)  # The mask
        # dummy_scores = torch.ones_like(prob_logit) * -99999.0
        # masked_prob_logit = torch.where(mask, dummy_scores, prob_logit)

        # scores = self.sigmoid(masked_prob_logit)
        scores = self.softmax(prob_logit)
        loss = self.loss_func_ce(prob_logit.view(-1, MAX_KB_ARR_LENGTH), labels.view(-1))
        return loss, prob_logit

    def get_trainable_params(self, recurse: bool = True) -> (List[nn.Parameter], int):
        parameters = list(filter(lambda p: p.requires_grad, self.parameters(recurse)))
        num_trainable_parameters = sum([p.flatten().size(0) for p in parameters])
        return parameters, num_trainable_parameters


class LightningBertPretrainedClassifier(LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        self.bert_classifier = BertPretrainedClassifier(**hparams.bert_params)

    def parameters(self, recurse: bool = ...):
        return self.bert_classifier.parameters(recurse)

    def configure_optimizers(self):
        parameters_list = self.bert_classifier.get_trainable_params()[0]
        if parameters_list:
            return torch.optim.Adam(parameters_list)
        else:
            return [] # PyTorch Lightning hack for test mode with frozen model

    def forward(self, *args):
        return self.bert_classifier.forward(*args)

    def get_trainable_params(self, recurse: bool = True) -> (List[nn.Parameter], int):
        parameters = list(filter(lambda p: p.requires_grad, self.parameters(recurse)))
        num_trainable_parameters = sum([p.flatten().size(0) for p in parameters])
        return parameters, num_trainable_parameters

    def training_step(self, batch, batch_idx):
        input_ids, input_mask, labels, unique_ids, kb_arr = batch
        loss, scores, pooler_attention_weights = self.forward(input_ids, input_mask, labels, kb_arr)
        return {"loss": loss, "log": {"batch_num": batch_idx, "train_loss": loss}}


class DebiasedBertPretrainedClassifier(LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        self.classifier_bias = LightningBertPretrainedClassifier.load_from_checkpoint(self.hparams.ckpt_path)
        self.classifier_debias = LightningBertPretrainedClassifier(self.hparams)
        self.froze_classifier_parameters(self.classifier_bias)
        self.loss_func_ce = nn.CrossEntropyLoss()

    def parameters(self, recurse: bool = ...):
        parameter_list_b = [p for p in self.classifier_bias.parameters(recurse)]
        parameter_list_db = [p for p in self.classifier_debias.parameters(recurse)]
        return parameter_list_b + parameter_list_db

    def configure_optimizers(self):
        parameters_list_b = self.classifier_bias.get_trainable_params()[0]
        parameters_list_db = self.classifier_debias.get_trainable_params()[0]
        parameters_list = parameters_list_b + parameters_list_db
        if parameters_list:
            return torch.optim.Adam(parameters_list)
        else:
            return [] # PyTorch Lightning hack for test mode with frozen model

    def froze_classifier_parameters(self, cls):
        for p in cls.parameters():
            p.requires_grad = False
        return

    def forward(self, input_ids, input_mask, labels, kb_arr):
        loss_bias, logits_bias, _ = self.classifier_bias(input_ids, input_mask, labels, kb_arr)
        loss_debias, logits_debias, _ = self.classifier_debias(input_ids, input_mask, labels, kb_arr)
        # logits_final = logits_debias - logits_bias
        logits_final = logits_debias
        loss = self.loss_func_ce(logits_final.view(-1, MAX_KB_ARR_LENGTH), labels.view(-1))
        return loss, logits_final

    def training_step(self, batch, batch_idx):
        input_ids, input_mask, labels, unique_ids, kb_arr = batch
        loss, scores = self.forward(input_ids, input_mask, labels, kb_arr)
        return {"loss": loss, "log": {"batch_num": batch_idx, "train_loss": loss}}

