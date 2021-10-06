import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.config import *
from utils.utils_general import _cuda
from .grad_reverse_layer import GradReverseLayerFunction
import random

MAX_KB_LEN = 100
MAX_INPUT_LEN = 500


class ContextRNN(nn.Module):
    def __init__(self, input_size, hidden_size, dropout, n_layers=1):
        super(ContextRNN, self).__init__()      
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers     
        self.dropout = dropout
        self.dropout_layer = nn.Dropout(dropout)
        self.embedding = nn.Embedding(input_size, hidden_size, padding_idx=PAD_token)
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers, dropout=dropout, bidirectional=True)
        self.W = nn.Linear(2*hidden_size, hidden_size)

    def get_state(self, bsz):
        """Get cell states and hidden states."""
        return _cuda(torch.zeros(2, bsz, self.hidden_size))

    def forward(self, input_seqs, input_lengths, hidden=None):
        # Note: we run this all at once (over multiple batches of multiple sequences)
        # print("input_seqs in size: ", input_seqs.size())
        embedded = self.embedding(input_seqs.contiguous().view(input_seqs.size(0), -1).long()) 
        embedded = embedded.view(input_seqs.size()+(embedded.size(-1),))
        embedded = torch.sum(embedded, 2).squeeze(2) 
        embedded = self.dropout_layer(embedded)
        hidden = self.get_state(input_seqs.size(1))
        # print("input_seqs out size: ", input_seqs.size())
        # print("embedded size: ", embedded.size())
        if input_lengths:
            embedded = nn.utils.rnn.pack_padded_sequence(embedded, input_lengths, batch_first=False)
        outputs, hidden = self.gru(embedded, hidden)
        if input_lengths:
           outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=False)   
        # pdb.set_trace()
        hidden = self.W(torch.cat((hidden[0], hidden[1]), dim=1)).unsqueeze(0)
        outputs = self.W(outputs)
        return outputs.transpose(0,1), hidden


class EntityPredictionRNN(nn.Module):
    def __init__(self, input_size, hidden_size, dropout, shared_emb, num_labels, n_layers=1):
        super(EntityPredictionRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.dropout = dropout
        self.dropout_layer = nn.Dropout(dropout)
        self.embedding = shared_emb
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers, dropout=dropout, bidirectional=False)
        self.W = nn.Linear(2*hidden_size, hidden_size)
        self.intent_prediction = UserIntentPredictionHead(hidden_size, num_labels)
        self.entity_prediction = EntityPredictionHead(hidden_size, shared_emb)

    def get_state(self, bsz):
        """Get cell states and hidden states."""
        return _cuda(torch.zeros(1, bsz, self.hidden_size))

    def forward(self, input_seqs, input_lengths, kb_arr, global_pointer, hidden=None):
        # Note: we run this all at once (over multiple batches of multiple sequences)
        # print("input_seqs in size: ", input_seqs.size())
        # embedded = self.embedding(input_seqs.contiguous().view(input_seqs.size(0), -1).long())
        # embedded = embedded.view(input_seqs.size()+(embedded.size(-1),))
        # embedded = torch.sum(embedded, 2).squeeze(2)
        # embedded = self.dropout_layer(embedded)

        input_seqs = input_seqs.transpose(0, 1).cuda()
        # input_seqs = input_seqs.transpose(0, 1)
        embedded = self.embedding(input_seqs.contiguous().view(input_seqs.size(0), -1).long())
        embedded = embedded.view(input_seqs.size()+(embedded.size(-1),))
        # embedded = torch.sum(embedded, 2).squeeze(2)
        embedded = self.dropout_layer(embedded)
        hidden = self.get_state(input_seqs.size(1))
        # print("input_seqs out size: ", input_seqs.size())
        # print("embedded size: ", embedded.size())
        if input_lengths:
            embedded = nn.utils.rnn.pack_padded_sequence(embedded, input_lengths, batch_first=False)
        outputs, hidden = self.gru(embedded, hidden)
        if input_lengths:
           outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=False)
        # pdb.set_trace()
        intent_logits = self.intent_prediction(hidden.squeeze(0))
        # entity_logits = self.entity_prediction(hidden.squeeze(0), kb_arr, global_pointer)
        entity_logits = self.entity_prediction(hidden.squeeze(0), kb_arr.cuda(), global_pointer)
        # hidden = self.W(torch.cat((hidden[0], hidden[1]), dim=1)).unsqueeze(0)
        # outputs = self.W(outputs)
        return entity_logits, intent_logits


class EntityPredictionHead(nn.Module):
    def __init__(self, hidden_size, shared_emb):
        super(EntityPredictionHead, self).__init__()
        self.classifier = nn.Linear(hidden_size, hidden_size)
        self.embeddings = shared_emb

    def forward(self, hidden_state, kb_arr, global_pointer):
        kb_emb = self.embeddings(kb_arr)
        # kb_emb = kb_emb * global_pointer.unsqueeze(2).expand_as(kb_emb)
        u_temp = hidden_state.unsqueeze(1).expand_as(kb_emb)
        prob_logits = torch.sum(kb_emb * u_temp, dim=2)
        return prob_logits


class UserIntentPredictionHead(nn.Module):
    def __init__(self, hidden_size, num_labels):
        super(UserIntentPredictionHead, self).__init__()
        self.classifier = nn.Linear(hidden_size, num_labels)
        self.sigmoid = nn.Sigmoid()
        self.alpha = 0.1

    def forward(self, hidden_state):
        reversed_hidden_state = GradReverseLayerFunction.apply(hidden_state, self.alpha)
        output = self.classifier(reversed_hidden_state)
        # output = self.classifier(hidden_state)
        return output


class ExternalKnowledge(nn.Module):
    def __init__(self, vocab, embedding_dim, hop, dropout):
        super(ExternalKnowledge, self).__init__()
        self.max_hops = hop
        self.embedding_dim = embedding_dim
        self.dropout = dropout
        self.dropout_layer = nn.Dropout(dropout) 
        for hop in range(self.max_hops+1):
            C = nn.Embedding(vocab, embedding_dim, padding_idx=PAD_token)
            # C.weight.data.normal_(0, 0.1)
            t = torch.randn(vocab, embedding_dim) * 0.1
            t[PAD_token, :] = torch.zeros(1, embedding_dim)
            C.weight.data = t
            self.add_module("C_{}".format(hop), C)
        self.C = AttrProxy(self, "C_")
        self.softmax = nn.Softmax(dim=1)
        self.sigmoid = nn.Sigmoid()
        self.conv_layer = nn.Conv1d(embedding_dim, embedding_dim, 5, padding=2)
        self.projector = nn.Linear(768, embedding_dim)
        self.projector2 = nn.Linear(768, embedding_dim)

    def add_lm_embedding(self, full_memory, kb_len, conv_len, hiddens):
        for bi in range(full_memory.size(0)):
            start, end = kb_len[bi], kb_len[bi]+conv_len[bi]
            full_memory[bi, start:end, :] = full_memory[bi, start:end, :] + hiddens[bi, :conv_len[bi], :]
        return full_memory

    def load_memory(self, story, hidden):
        # Forward multiple hop mechanism
        u = [hidden.squeeze(0)]
        story_size = story.size()
        self.m_story = []
        for hop in range(self.max_hops):
            embed_A = self.C[hop](story.contiguous().view(story_size[0], -1))#.long()) # b * (m * s) * e
            embed_A = embed_A.view(story_size+(embed_A.size(-1),)) # b * m * s * e
            # embed_A = torch.sum(embed_A, 2).squeeze(2) # b * m * e
            # if not args["ablationH"]:
            #     embed_A = self.add_lm_embedding(embed_A, kb_len, conv_len, dh_outputs)
            embed_A = self.dropout_layer(embed_A)
            
            if(len(list(u[-1].size()))==1): 
                u[-1] = u[-1].unsqueeze(0) ## used for bsz = 1.
            u_temp = u[-1].unsqueeze(1).expand_as(embed_A)
            prob_logit = torch.sum(embed_A*u_temp, 2)
            prob_   = self.softmax(prob_logit)
            
            embed_C = self.C[hop+1](story.contiguous().view(story_size[0], -1).long())
            embed_C = embed_C.view(story_size+(embed_C.size(-1),)) 
            # embed_C = torch.sum(embed_C, 2).squeeze(2)
            # if not args["ablationH"]:
            #     embed_C = self.add_lm_embedding(embed_C, kb_len, conv_len, dh_outputs)

            prob = prob_.unsqueeze(2).expand_as(embed_C)
            o_k  = torch.sum(embed_C*prob, 1)
            u_k = u[-1] + o_k
            u.append(u_k)
            self.m_story.append(embed_A)
        self.m_story.append(embed_C)
        return self.sigmoid(prob_logit), u[-1]

    def forward(self, query_vector, global_pointer):
        u = [query_vector]
        for hop in range(self.max_hops):
            m_A = self.m_story[hop] 
            if not args["ablationG"]:
                m_A = m_A * global_pointer.unsqueeze(2).expand_as(m_A) 
            if(len(list(u[-1].size()))==1): 
                u[-1] = u[-1].unsqueeze(0) ## used for bsz = 1.
            u_temp = u[-1].unsqueeze(1).expand_as(m_A)
            prob_logits = torch.sum(m_A*u_temp, 2)
            prob_soft   = self.softmax(prob_logits)
            m_C = self.m_story[hop+1] 
            if not args["ablationG"]:
                m_C = m_C * global_pointer.unsqueeze(2).expand_as(m_C)
            prob = prob_soft.unsqueeze(2).expand_as(m_C)
            o_k  = torch.sum(m_C*prob, 1)
            u_k = u[-1] + o_k
            u.append(u_k)
        return prob_soft, prob_logits


class LocalMemoryDecoder(nn.Module):
    def __init__(self, shared_emb, lang, embedding_dim, hop, dropout):
        super(LocalMemoryDecoder, self).__init__()
        self.num_vocab = lang.n_words
        self.lang = lang
        self.max_hops = hop
        self.embedding_dim = embedding_dim
        self.dropout = dropout
        self.dropout_layer = nn.Dropout(dropout) 
        self.C = shared_emb 
        self.softmax = nn.Softmax(dim=1)
        self.sketch_rnn = nn.GRU(embedding_dim, embedding_dim, dropout=dropout)
        self.relu = nn.ReLU()
        self.projector = nn.Linear(2*embedding_dim, embedding_dim)
        self.conv_layer = nn.Conv1d(embedding_dim, embedding_dim, 5, padding=2)
        self.softmax = nn.Softmax(dim = 1)
        self.mlp = nn.Linear(embedding_dim, embedding_dim)

    def forward(self, extKnow, story_size, story_lengths, copy_list, encode_hidden,
                target_batches, max_target_length, batch_size, use_teacher_forcing,
                get_decoded_words, global_pointer, conv_arr_plain, kb_arr_plain_new, kb_arr_plain, ent_labels, input_ids, input_lens, cl_ent_pred, conv_arr):
        # Initialize variables for vocab and pointer
        all_decoder_outputs_vocab = _cuda(torch.zeros(max_target_length, batch_size, self.num_vocab))
        # all_decoder_outputs_ptr = _cuda(torch.zeros(max_target_length, batch_size, story_size[1]))
        all_decoder_outputs_ptr = _cuda(torch.zeros(max_target_length, batch_size, MAX_KB_LEN))
        # all_decoder_outputs_ptr = _cuda(torch.zeros(max_target_length, batch_size, kb_arr_plain.shape[1]))
        all_decoder_outputs_topv = _cuda(torch.zeros(max_target_length, batch_size, 1))
        all_decoder_outputs_intents = _cuda(torch.zeros(max_target_length, batch_size, self.lang.n_annotators))
        all_decoder_outputs_ptr_biased = _cuda(torch.zeros(max_target_length, batch_size, MAX_KB_LEN))
        decoder_input = _cuda(torch.LongTensor([SOS_token] * batch_size))
        # memory_mask_for_step = _cuda(torch.ones(story_size[0], story_size[1]))
        kb_lens = [len(ele) for ele in kb_arr_plain_new]
        memory_mask_for_step = _cuda(torch.ones(batch_size, MAX_KB_LEN))
        # memory_mask_for_step = _cuda(torch.ones(batch_size, kb_arr_plain.shape[1]))
        decoded_fine, decoded_coarse = [], []
        
        hidden = self.relu(self.projector(encode_hidden)).unsqueeze(0)

        # Start to generate word-by-word
        for t in range(max_target_length):
            embed_q = self.dropout_layer(self.C(decoder_input)) # b * e
            if len(embed_q.size()) == 1: embed_q = embed_q.unsqueeze(0)
            _, hidden = self.sketch_rnn(embed_q.unsqueeze(0), hidden)
            p_vocab = self.attend_vocab(self.C.weight, hidden.squeeze(0))
            all_decoder_outputs_vocab[t] = p_vocab
            _, topvi = p_vocab.data.topk(1)
            all_decoder_outputs_topv[t] = topvi

            # compute bert input for kb entity prediction
            input_ids, input_lens = self.compute_entity_prediction_input(conv_arr_plain, target_batches, t, batch_size, kb_arr_plain)
            entity_logits, intent_logits = extKnow(input_ids, input_lens, kb_arr_plain, global_pointer)
            bias_feas = cl_ent_pred(input_ids, input_lens)
            bias_feas_mlp = self.mlp(bias_feas)
            bias_preds = extKnow.entity_prediction(bias_feas_mlp, kb_arr_plain.cuda(), global_pointer)
            # bias_preds = extKnow.entity_prediction(bias_feas_mlp, kb_arr_plain, global_pointer)
            all_decoder_outputs_ptr[t] = entity_logits
            all_decoder_outputs_intents[t] = intent_logits
            all_decoder_outputs_ptr_biased[t] = bias_preds
            # all_decoder_outputs_ptr_biased[t] = entity_logits
            prob_soft = entity_logits - bias_preds
            # prob_soft = entity_logits

            if use_teacher_forcing:
                decoder_input = target_batches[:,t]
            else:
                decoder_input = topvi.squeeze()

            if get_decoded_words:

                search_len = min(5, min(story_lengths))
                prob_soft = prob_soft * memory_mask_for_step
                _, toppi = prob_soft.data.topk(search_len)
                temp_f, temp_c = [], []

                for bi in range(batch_size):
                    token = topvi[bi].item() #topvi[:,0][bi].item()
                    temp_c.append(self.lang.index2word[token])

                    if '@' in self.lang.index2word[token]:
                        cw = 'UNK'
                        for i in range(search_len):
                            # if toppi[:,i][bi] < story_lengths[bi]-1:
                            if toppi[:, i][bi] < kb_lens[bi] - 1:
                                cw = copy_list[bi][toppi[:,i][bi].item()]
                                break
                        temp_f.append(cw)

                        if args['record']:
                            memory_mask_for_step[bi, toppi[:,i][bi].item()] = 0
                    else:
                        temp_f.append(self.lang.index2word[token])

                decoded_fine.append(temp_f)
                decoded_coarse.append(temp_c)

        return all_decoder_outputs_vocab, all_decoder_outputs_ptr, decoded_fine, decoded_coarse, all_decoder_outputs_ptr_biased

    def compute_entity_prediction_input(self,
                                        conv_arr_plain,
                                        target_batches,
                                        current_step,
                                        batch_size,
                                        kb_arr_plain):
        bert_input_arr = []
        for bt in range(batch_size):
            conv_arr_plain_t = conv_arr_plain[bt]
            output_plain_t = target_batches[bt, :(current_step + 1)]
            for t in range(current_step + 1):
                conv_arr_plain_t.append(self.lang.index2word[output_plain_t[t].item()])
            bert_input = " ".join(conv_arr_plain_t)
            bert_input_arr.append(bert_input)

        # convert to id and padding
        lens = [len(ele.split(" ")) for ele in bert_input_arr]
        max_len = max(lens)
        padded_seqs = torch.zeros(batch_size, max_len).long()
        # lengths = [len(seq) for seq in kb_arr_plain]
        # max_kb_len = 1 if max(lengths) == 0 else max(lengths)
        # kb_arr_padded = torch.zeros(batch_size, MAX_KB_LEN).long()
        for i, seq in enumerate(bert_input_arr):
            end = lens[i]
            word_ids = []
            word_list = seq.split(" ")
            for word in word_list:
                word_id = self.lang.word2index[word] if word in self.lang.word2index else UNK_token
                word_ids.append(word_id)
            padded_seqs[i, :end] = torch.Tensor(word_ids[:end])
        # for i, ele in enumerate(kb_arr_plain):
        #     kb_arr_ids = []
        #     for ent in ele:
        #         kb_arr_id = self.lang.word2index[ent] if ent in self.lang.word2index else UNK_token
        #         kb_arr_ids.append(kb_arr_id)
        #     kb_arr_padded[i, :len(kb_arr_ids)] = torch.Tensor(kb_arr_ids)
        return padded_seqs, lens

    def attend_vocab(self, seq, cond):
        scores_ = cond.matmul(seq.transpose(1,0))
        # scores = F.softmax(scores_, dim=1)
        return scores_


class Similarity(nn.Module):
    """
    Dot product or cosine similarity
    """

    def __init__(self, temp):
        super().__init__()
        self.temp = temp
        self.cos = nn.CosineSimilarity(dim=-1)

    def forward(self, x, y):
        return self.cos(x, y) / self.temp


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
