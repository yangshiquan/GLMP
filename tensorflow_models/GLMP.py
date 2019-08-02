import tensorflow as tf
from utils.config import *


class GLMP(tf.keras.Model):
    def __init__(self, hidden_size, lang, max_resp_len, path, task, lr, n_layers, dropout):
        super(GLMP, self).__init__()
        self.name = 'GLMP'
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
        self.softmax = tf.keras.layers.Softmax(0)

        if USE_CUDA:
