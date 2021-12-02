from torch.optim import lr_scheduler
from torch import optim
from models.modules_cl import *


class GLMP(nn.Module):
    def __init__(self, hidden_size, lang, max_resp_len, path, task, lr, n_layers, dropout):
        super(GLMP, self).__init__()
        self.name = "GLMP"
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
        self.softmax = nn.Softmax(dim=0)

        if path:
            if USE_CUDA:
                print("MODEL {} LOADED".format(str(path)))
                self.encoder = torch.load(str(path) + '/cl_enc.th')
            else:
                print("MODEL {} LOADED".format(str(path)))
                self.encoder = torch.load(str(path) + '/cl_enc.th', lambda storage, loc: storage)
        else:
            self.encoder = ContextRNNCL(lang.n_words, lang.n_ents, hidden_size, dropout)

        # if not args['pretrain']:
        #     for param in self.encoder.embedding.parameters():
        #         param.requires_grad = False
        #     for param in self.encoder.gru.parameters():
        #         param.requires_grad = False

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
        if args['pretrain']:
            name_data = "CL-PRETRAIN/" if self.task == '' else "BABI/"
        else:
            name_data = "CL-FINETUNE/" if self.task == '' else "BABI/"
        layer_info = str(self.n_layers)
        directory = 'save/GLMP-' + args["addName"] + name_data + str(self.task) + 'HDD' + str(
            self.hidden_size) + 'BSZ' + str(args['batch']) + 'DR' + str(self.dropout) + 'L' + layer_info + 'lr' + str(
            self.lr) + str(dec_type)
        if not os.path.exists(directory):
            os.makedirs(directory)
        torch.save(self.encoder, directory + '/cl_enc.th')

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

        if args['pretrain']:
            # Encode
            paired_example = data['paired_example']
            cos_sim, labels = self.encoder(paired_example, data['paired_example_lengths'], args['pretrain'])

            # loss = self.criterion_ce(cos_sim, labels)
            loss = self.criterion_ce(cos_sim, labels.cuda())

            loss.backward()

            # Clip gradient norms
            ec = torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), clip)

            # Update parameters with optimizers
            self.encoder_optimizer.step()
            self.loss += loss.item()

            return loss
        else:
            # Encode
            prob_logits = self.encoder(data['input'], data['input_arr_lengths'], args['pretrain'])

            # loss = self.criterion_ce(prob_logits, torch.tensor(data['target'], dtype=int))
            loss = self.criterion_ce(prob_logits, torch.tensor(data['target'], dtype=int).cuda())

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
        for j, data_dev in pbar:
            prob_logits = self.encoder(data_dev['input'], data_dev['input_arr_lengths'], args['pretrain'])
            labels = torch.tensor(data_dev['target'], dtype=int)
            predictions = torch.argmax(prob_logits, dim=-1)
            decoded_ents = [self.lang.index2ent[elm] for elm in predictions.tolist()]
            golden_ents = [self.lang.index2ent[elm] for elm in labels.tolist()]
            for idx, elm in enumerate(golden_ents):
                if elm != "NULL":
                    pred_ent = decoded_ents[idx]
                    total += 1
                    all += 1
                    if elm == pred_ent:
                        total_correct += 1
                        all_correct += 1
                else:
                    null_total += 1
                    all += 1
                    pred_ent = decoded_ents[idx]
                    if elm == pred_ent:
                        null_total_correct += 1
                        all_correct += 1

            # correct = predictions.eq(labels.view_as(predictions).cuda()).double()
            # total_correct += correct.sum()
            # total += predictions.size(0)

        # Set back to training mode
        self.encoder.train(True)

        acc_score = total_correct / float(total)
        acc_score_null = null_total_correct / float(null_total)
        acc_score_all = all_correct / float(all)
        print("ACC SCORE (Tail): " + str(acc_score))
        print("ACC SCORE (Head): " + str(acc_score_null))
        print("ACC SCORE (All): " + str(acc_score_all))
        if (acc_score >= matric_best):
            self.save_model('ACC-{:.4f}'.format(acc_score))
            print("MODEL SAVED")

        return acc_score

