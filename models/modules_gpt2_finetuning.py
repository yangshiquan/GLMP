import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.config import *
from utils.utils_general import _cuda
from .grad_reverse_layer import GradReverseLayerFunction
import random
# from transformers.modeling_bert import BertModel
from transformers import GPT2Config, GPT2LMHeadModel, GPT2Model


class AutoregressiveLM(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, dropout, tokenizer, n_layers=1):
        super(AutoregressiveLM, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.dropout = dropout
        self.config = GPT2Config.from_pretrained("gpt2")
        self.gpt2 = GPT2LMHeadModel.from_pretrained("gpt2", config=self.config)
        self.gpt2.resize_token_embeddings(len(tokenizer))

    def forward(self, input_seqs, input_lengths, hidden=None):
        # Note: we run this all at once (over multiple batches of multiple sequences)
        # print("input_seqs in size: ", input_seqs.size())
        batch_size, max_len = len(input_lengths), max(input_lengths)
        length_tensor = torch.tensor(input_lengths, dtype=torch.int64).unsqueeze(1).expand(batch_size, max_len)
        comparison_tensor = torch.arange(0, max_len).expand_as(length_tensor)
        mask = torch.lt(comparison_tensor, length_tensor).long()
        # mask = torch.lt(comparison_tensor, length_tensor).long().cuda()
        output = self.gpt2(input_ids=input_seqs, labels=input_seqs, attention_mask=mask)
        loss, logits = output.loss, output.logits
        return loss, logits

