from torch.optim import lr_scheduler
from torch import optim
from models.modules_gpt2_finetuning import *
from utils.measures import wer, moses_multi_bleu
import numpy as np


class GPT2(nn.Module):
    def __init__(self, hidden_size, lang, max_resp_len, path, task, tokenizer, lr, n_layers, dropout):
        super(GPT2, self).__init__()
        self.name = "GPT2"
        self.task = task
        self.input_size = lang.n_words
        self.output_size = lang.n_words
        self.hidden_size = hidden_size
        self.lang = lang
        self.lr = lr
        self.n_layers = n_layers
        self.dropout = dropout
        self.max_resp_len = max_resp_len
        self.decoder_hop = n_layers
        self.tokenizer = tokenizer
        self.softmax = nn.Softmax(dim=0)

        if path:
            if USE_CUDA:
                print("MODEL {} LOADED".format(str(path)))
                self.encoder = torch.load(str(path) + '/gpt2.th')
            else:
                print("MODEL {} LOADED".format(str(path)))
                self.encoder = torch.load(str(path) + '/gpt2.th', lambda storage, loc: storage)
        else:
            self.encoder = AutoregressiveLM(lang.n_words, lang.n_ents, hidden_size, dropout, tokenizer)

        # Initialize optimizers and criterion
        self.encoder_optimizer = optim.Adam(self.encoder.parameters(), lr=lr)
        self.scheduler = lr_scheduler.ReduceLROnPlateau(self.encoder_optimizer, mode='max', factor=0.5, patience=1,
                                                        min_lr=0.0001, verbose=True)
        self.criterion_ce = nn.CrossEntropyLoss()
        self.reset()

        if USE_CUDA:
            self.encoder.cuda()

    def print_loss(self):
        print_loss_avg = self.loss / self.print_every
        self.print_every += 1
        return 'L:{:.2f}'.format(print_loss_avg)

    def save_model(self, dec_type):
        name_data = "DelexRespGen/" if self.task == '' else "BABI/"
        layer_info = str(self.n_layers)
        directory = 'save/' + args["addName"] + name_data + str(self.task) + 'HDD' + str(
            self.hidden_size) + 'BSZ' + str(args['batch']) + 'DR' + str(self.dropout) + 'L' + layer_info + 'lr' + str(
            self.lr) + str(dec_type)
        if not os.path.exists(directory):
            os.makedirs(directory)
        torch.save(self.encoder, directory + '/gpt2.th')

    def reset(self):
        self.loss, self.print_every = 0, 1

    def _cuda(self, x):
        if USE_CUDA:
            return torch.Tensor(x).cuda()
        else:
            return torch.Tensor(x)

    def train_batch(self, data, clip, epoch, reset=0):
        if reset: self.reset()
        # Zero gradients of both optimizers
        self.encoder_optimizer.zero_grad()

        # Encode
        loss, _ = self.encoder(data['gpt_input'], data['input_arr_lengths'])

        loss.backward()

        # Clip gradient norms
        ec = torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), clip)

        # Update parameters with optimizers
        self.encoder_optimizer.step()
        self.loss += loss.item()

        return loss

    def evaluate(self, dev, matric_best, early_stop=None):
        print("STARTING EVALUATION")
        # Set to not-training mode to disable dropout
        self.encoder.train(False)

        # Add evaluation and metrics calculation logic here.
        total, null_total, all = 0, 0, 0
        total_correct, null_total_correct, all_correct = 0, 0, 0
        pbar = tqdm(enumerate(dev), total=len(dev))

        hyp, ref = [], []
        for j, data_dev in pbar:
            _, prob_logits = self.encoder(data_dev['gpt_input'], data_dev['input_arr_lengths'])
            labels = data_dev['response']
            predictions = torch.argmax(prob_logits, dim=-1)
            decoded_sentences = [" ".join(self.tokenizer.convert_ids_to_tokens(elm)) for elm in predictions.tolist()]
            # golden_sentences = [" ".join(self.tokenizer.convert_ids_to_tokens(elm)) for elm in labels.tolist()]
            for sent in decoded_sentences:
                sent_new = ' '.join([sent.split('<|endofresponse|>')[0], "<|endofresponse|>"])
                if '<|response|>' in sent_new:
                    tmp = sent_new.split('<|response|>')[-1]
                    tmp = tmp.strip(' ,.')
                    tmp = tmp.replace('<|endofresponse>|', '')
                    tmp = tmp.replace('<|endoftext|>', '')
                    # tokens = self.tokenizer.encode(tmp)
                    # new_tokens = []
                    # for tok in tokens:
                    #     if tok in self.tokenizer.encode(self.tokenizer._eos_token):
                    #         continue
                    #     new_tokens.append(tok)
                    # response = self.tokenizer.decode(new_tokens).strip(' ,.')
                    response = tmp
                else:
                    response = ''
                hyp.append(response)
            for sent in labels:
                tmp = sent.strip()
                ref.append(tmp)

        # Set back to training mode
        self.encoder.train(True)

        bleu_score = moses_multi_bleu(np.array(hyp), np.array(ref), lowercase=True)
        print("BLEU SCORE:\t" + str(bleu_score))
        if (bleu_score >= matric_best):
            self.save_model('BLEU-' + str(bleu_score))
            print("MODEL SAVED")
        return bleu_score
