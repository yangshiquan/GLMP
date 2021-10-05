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
                self.encoder = torch.load(str(path) + '/enc.th')
            else:
                print("MODEL {} LOADED".format(str(path)))
                self.encoder = torch.load(str(path) + '/enc.th', lambda storage, loc: storage)
        else:
            self.encoder = ContextRNNCL(lang.n_words, hidden_size, dropout)

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
        name_data = "CL-PRETRAIN/" if self.task == '' else "BABI/"
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

        # Encode
        paired_example = data['paired_example']
        cos_sim, labels = self.encoder(paired_example, data['paired_example_lengths'])

        loss = self.criterion_ce(cos_sim, labels)

        loss.backward()

        # Clip gradient norms
        ec = torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), clip)

        # Update parameters with optimizers
        self.encoder_optimizer.step()
        self.loss += loss.item()

        return loss

