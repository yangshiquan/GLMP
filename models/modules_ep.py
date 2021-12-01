import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.config import *
from utils.utils_general import _cuda
from .grad_reverse_layer import GradReverseLayerFunction
import random
# from transformers.modeling_bert import BertModel
from transformers import BertModel


class ContextRNNEP(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, dropout, tokenizer, n_layers=1):
        super(ContextRNNEP, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.dropout = dropout
        self.dropout_layer = nn.Dropout(dropout)
        self.embedding = nn.Embedding(input_size, hidden_size, padding_idx=PAD_token)
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers, dropout=dropout, bidirectional=True)
        # self.W = nn.Linear(2 * hidden_size, hidden_size)
        self.W = nn.Linear(768, hidden_size)
        self.W2 = nn.Linear(hidden_size, hidden_size)
        self.W3 = nn.Linear(hidden_size, output_size)
        self.bert = BertModel.from_pretrained(BERT_PRETRAINED_MODEL)
        self.bert.resize_token_embeddings(len(tokenizer))

    def get_state(self, bsz):
        """Get cell states and hidden states."""
        return _cuda(torch.zeros(2, bsz, self.hidden_size))

    def forward(self, input_seqs, input_lengths, hidden=None):
        # Note: we run this all at once (over multiple batches of multiple sequences)
        # print("input_seqs in size: ", input_seqs.size())
        # embedded = self.embedding(input_seqs.contiguous().view(input_seqs.size(0), -1).long())
        # embedded = embedded.view(input_seqs.size()+(embedded.size(-1),))
        # embedded = torch.sum(embedded, 2).squeeze(2)
        # embedded = self.dropout_layer(embedded)
        # hidden = self.get_state(input_seqs.size(1))
        # # print("input_seqs out size: ", input_seqs.size())
        # # print("embedded size: ", embedded.size())
        # if input_lengths:
        #     embedded = nn.utils.rnn.pack_padded_sequence(embedded, input_lengths, batch_first=False)
        # outputs, hidden = self.gru(embedded, hidden)
        # if input_lengths:
        #    outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=False)
        # prob_logits = self.W3(self.W2(self.W(torch.cat((hidden[0], hidden[1]), dim=1))))
        batch_size, max_len = len(input_lengths), max(input_lengths)
        length_tensor = torch.tensor(input_lengths, dtype=torch.int64).unsqueeze(1).expand(batch_size, max_len)
        comparison_tensor = torch.arange(0, max_len).expand_as(length_tensor)
        # mask = torch.lt(comparison_tensor, length_tensor).long()
        mask = torch.lt(comparison_tensor, length_tensor).long().cuda()
        output = self.bert(input_seqs, mask)
        prob_logits = self.W3(self.W2(self.W(output[1])))
        return prob_logits


class AttrProxy(object):
    """
    Translates index lookups into attribute lookups.
    To implement some trick which able to use list of nn.Module in a nn.Module
    see https://discuss.pytorch.org/t/list-of-nn-module-in-a-nn-module/219/2
    """

    def __init__(self, module, prefix):
        self.module = module
        self.prefix = prefix

    def __getitem__(self, i):
        return getattr(self.module, self.prefix + str(i))
