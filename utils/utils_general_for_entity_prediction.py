import torch
import torch.utils.data as data
import torch.nn as nn
from utils.config import *


def _cuda(x):
    if USE_CUDA:
        return x.cuda()
    else:
        return x


class Lang:
    def __init__(self):
        self.word2index = {}
        self.index2word = {PAD_token: "PAD", SOS_token: "SOS", EOS_token: "EOS", UNK_token: 'UNK'}
        self.n_words = len(self.index2word)  # Count default tokens
        self.word2index = dict([(v, k) for k, v in self.index2word.items()])
        self.ent2index = {}
        self.index2ent = {}
        self.n_ents = 0
        self.intent2index = {}
        self.state2index = {}
        self.annotator2index = {}
        self.index2intent = {}
        self.index2state = {}
        self.index2annotator = {}
        self.n_intents = 0
        self.n_state_values = {}
        self.n_annotators = 0

    def index_words(self, story, trg=False):
        if trg:
            for word in story.split(' '):
                self.index_word(word)
        else:
            for word_triple in story:
                for word in word_triple:
                    self.index_word(word)

    def index_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.index2word[self.n_words] = word
            self.n_words += 1

    def index_entity(self, ent):
        if ent not in self.ent2index:
            self.ent2index[ent] = self.n_ents
            self.index2ent[self.n_ents] = ent
            self.n_ents += 1

    def index_intent(self, intent):
        if intent not in self.intent2index:
            self.intent2index[intent] = self.n_intents
            self.index2intent[self.n_intents] = intent
            self.n_intents += 1

    def index_state_values(self, state, value):
        if state not in self.state2index:
            self.state2index[state] = {}
            self.index2state[state] = {}
            self.n_state_values[state] = 0
        if value not in self.state2index[state]:
            self.state2index[state][value] = self.n_state_values[state]
            self.index2state[state][self.n_state_values[state]] = value
            self.n_state_values[state] += 1

    def index_annotator(self, annotator_id):
        if annotator_id not in self.annotator2index:
            self.annotator2index[annotator_id] = self.n_annotators
            self.index2annotator[self.n_annotators] = annotator_id
            self.n_annotators += 1

    def re_initialize(self):
        self.intent2index = {}
        self.state2index = {}
        self.annotator2index = {}
        self.index2intent = {}
        self.index2state = {}
        self.index2annotator = {}
        self.n_intents = 0
        self.n_state_values = {}
        self.n_annotators = 0


class Dataset(data.Dataset):
    """Custom data.Dataset compatible with data.DataLoader."""

    def __init__(self, data_info, src_word2id, trg_word2id, ent2id):
        """Reads source and target sequences from txt files."""
        self.data_info = {}
        for k in data_info.keys():
            self.data_info[k] = data_info[k]

        self.num_total_seqs = len(data_info['input'])
        self.src_word2id = src_word2id
        self.trg_word2id = trg_word2id
        self.ent2id = ent2id

    def __getitem__(self, index):
        """Returns one data pair (source and target)."""
        input = self.data_info['input'][index]
        input = self.preprocess(input, self.src_word2id, trg=False)
        target = self.data_info['target'][index]
        target = self.preprocess(target, self.ent2id)

        # processed information
        data_info = {}
        for k in self.data_info.keys():
            try:
                data_info[k] = locals()[k]
            except:
                data_info[k] = self.data_info[k][index]

        return data_info

    def __len__(self):
        return self.num_total_seqs

    def preprocess(self, sequence, word2id, trg=True):
        """Converts words to ids."""
        if trg:
            story = [word2id[word] if word in word2id else UNK_token for word in sequence.split(' ')]
            story = torch.Tensor(story).to(dtype=torch.long)
        else:
            story = []
            for i, word_triple in enumerate(sequence):
                if isinstance(word_triple, list):
                    story.append([])
                    for ii, word in enumerate(word_triple):
                        temp = word2id[word] if word in word2id else UNK_token
                        story[i].append(temp)
                else:
                    temp = word2id[word_triple] if word_triple in word2id else UNK_token
                    story.append(temp)
            story = torch.Tensor(story)
        return story

    def collate_fn(self, data):
        def merge(sequences, story_dim):
            lengths = [len(seq) for seq in sequences]
            max_len = 1 if max(lengths) == 0 else max(lengths)
            if (story_dim):
                padded_seqs = torch.ones(len(sequences), max_len, MEM_TOKEN_SIZE).long()
                for i, seq in enumerate(sequences):
                    end = lengths[i]
                    if len(seq) != 0:
                        padded_seqs[i, :end, :] = seq[:end]
            else:
                padded_seqs = torch.ones(len(sequences), max_len).long()
                for i, seq in enumerate(sequences):
                    end = lengths[i]
                    padded_seqs[i, :end] = seq[:end]
            return padded_seqs, lengths

        def merge_kb(sequences):
            padded_seqs = torch.zeros(len(sequences), 100).long()
            for i, seq in enumerate(sequences):
                end = len(seq)
                padded_seqs[i, :end] = seq[:end]
            return padded_seqs

        def merge_index(sequences):
            lengths = [len(seq) for seq in sequences]
            padded_seqs = torch.zeros(len(sequences), max(lengths)).float()
            for i, seq in enumerate(sequences):
                end = lengths[i]
                padded_seqs[i, :end] = seq[:end]
            return padded_seqs, lengths

        # sort a list by sequence length (descending order) to use pack_padded_sequence
        data.sort(key=lambda x: len(x['input']), reverse=True)
        item_info = {}
        for key in data[0].keys():
            item_info[key] = [d[key] for d in data]

        # merge sequences
        input, input_arr_lengths = merge(item_info['input'], True)

        # convert to contiguous and cuda
        input = _cuda(input.transpose(0,1).contiguous())

        # processed information
        data_info = {}
        for k in item_info.keys():
            try:
                data_info[k] = locals()[k]
            except:
                data_info[k] = item_info[k]

        # add additional information
        data_info['input_arr_lengths'] = input_arr_lengths

        return data_info


def get_seq(pairs, lang, batch_size, type):
    data_info = {}
    for k in pairs[0].keys():
        data_info[k] = []

    for pair in pairs:
        for k in pair.keys():
            data_info[k].append(pair[k])
        if (type):
            lang.index_words(pair['input'])
            lang.index_words(pair['target'], trg=True)
            lang.index_entity(pair['target'])

    dataset = Dataset(data_info, lang.word2index, lang.word2index, lang.ent2index)
    data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                              batch_size=batch_size,
                                              # shuffle = type,
                                              shuffle=False,
                                              collate_fn=dataset.collate_fn)
    return data_loader


def compute_dataset_length(data_length, batch_size):
    return int(data_length / batch_size)