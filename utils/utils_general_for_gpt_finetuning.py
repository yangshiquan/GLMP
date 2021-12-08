import torch
import torch.utils.data as data
from utils.config import *
# from transformers.tokenization_bert import BertTokenizer
from transformers import GPT2Tokenizer
import random


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

    def index_special_tokens(self, special_tokens):
        for token in special_tokens:
            self.index_entity(token)

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

        self.num_total_seqs = len(data_info['gpt_input'])
        self.src_word2id = src_word2id
        self.trg_word2id = trg_word2id
        self.ent2id = ent2id
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        self.special_tokens_dict = {'additional_special_tokens': list(ent2id.keys())}
        self.tokenizer.add_special_tokens(self.special_tokens_dict)

    def __getitem__(self, index):
        """Returns one data pair (source and target)."""
        gpt_input = self.data_info['gpt_input'][index]
        gpt_input = self.preprocess(gpt_input, self.src_word2id, trg=False)

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

    def trunc_seq(self, tokens, max_num_tokens=1024):
        """Truncates a pair of sequences to a maximum sequence length. Lifted from Google's BERT repo."""
        l = 0
        r = len(tokens.split(" "))
        trunc_tokens = list(tokens.split(" "))
        while r - l > max_num_tokens:
            # We want to sometimes truncate from the front and sometimes from the
            # back to add more randomness and avoid biases.
            if random.random() < 0.5:
                l += 1
            else:
                r -= 1
        return trunc_tokens[l:r]

    def preprocess(self, sequence, word2id, trg=True):
        """Converts words to ids."""
        if trg:
            story = [word2id[word] if word in word2id else UNK_token for word in sequence.split(' ')]
            story = torch.Tensor(story).to(dtype=torch.long)
        else:
            truncated_sequence = self.trunc_seq(sequence, max_num_tokens=self.tokenizer.model_max_length)
            story = self.tokenizer.convert_tokens_to_ids(truncated_sequence)
            # story = torch.Tensor(story)
        return story

    def collate_fn(self, data):
        def merge(sequences):
            lengths = [len(seq) for seq in sequences]
            max_len = 1 if max(lengths) == 0 else max(lengths)
            padded_seqs = torch.ones(len(sequences), max_len).long()
            for i, seq in enumerate(sequences):
                end = lengths[i]
                padded_seqs[i, :end] = torch.Tensor(seq[:end]).long()
            return padded_seqs, lengths

        # sort a list by sequence length (descending order) to use pack_padded_sequence
        data.sort(key=lambda x: len(x['gpt_input']), reverse=True)
        item_info = {}
        for key in data[0].keys():
            item_info[key] = [d[key] for d in data]

        # merge sequences
        gpt_input, input_arr_lengths = merge(item_info['gpt_input'])

        # convert to contiguous and cuda
        gpt_input = _cuda(gpt_input.contiguous())

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
            lang.index_words(pair['gpt_input'])
            lang.index_special_tokens(pair['gpt_input'].split(" "))
            lang.index_special_tokens(pair['kb_arr_plain'])

    dataset = Dataset(data_info, lang.word2index, lang.word2index, lang.ent2index)
    data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                              batch_size=batch_size,
                                              # shuffle = type,
                                              shuffle=False,
                                              collate_fn=dataset.collate_fn)
    return data_loader


def compute_dataset_length(data_length, batch_size):
    return int(data_length / batch_size)