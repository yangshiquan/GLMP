import tensorflow as tf


class ContextRNN(tf.keras.Model):
    def __init__(self, input_size, hidden_size, dropout, n_layers=1):
        super(ContextRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.dropout = dropout
        self.dropout_layer = tf.keras.layers.Dropout(dropout)
        self.embedding = tf.keras.layers.Embedding(input_size, hidden_size)  # different: pad token embedding mask.
        self.gru = tf.keras.layers.Bidirectional(
            tf.keras.layers.GRU(
                hidden_size,
                dropout=dropout,
                return_sequences=True,
                return_state=True))  # different: initializer, input shape.
        # self.gru2 = tf.keras.layers.GRU(hidden_size, dropout=dropout, return_state=True, return_sequences=True)
        self.W = tf.keras.layers.Dense(hidden_size)  # different: bias should be explicitly assigned.

    def initialize_hidden_state(self, batch_size):
        return tf.zeros((batch_size, self.hidden_size))

    def call(self, input_seqs, input_lengths, hidden=None, training=True):
        embedded = self.embedding(tf.reshape(input_seqs, [input_seqs.get_shape()[0], -1]))  # different: pad token embedding not masked. input_seqs: batch_size * input_length * MEM_TOKEN_SIZE.
        embedded = tf.reshape(
            embedded, [input_seqs.get_shape()[0], input_seqs.get_shape()[1], input_seqs.get_shape()[2], embedded.get_shape()[-1]])  # embedded: batch_size * input_length * MEM_TOKEN_SIZE * embedding_dim.
        embedded = tf.math.reduce_sum(embedded, 2)  # embedded: batch_size * input_length * embedding_dim.
        embedded = self.dropout_layer(embedded)
        hidden = self.initialize_hidden_state(input_seqs.get_shape()[0])
        outputs, hidden = self.gru(embedded,
                                  initial_state=hidden,
                                  training=training)  # different: padded token not mask in forward calculation, need a flag to indicate train or test if using dropout. No pack_padded_sequence and pad_packed_sequence.
        hidden_hat = tf.concat([hidden[0], hidden[1]], 1)
        hidden = self.W(hidden_hat)  # different: no unsqueeze(0).
        outputs = self.W(outputs)  # different: no need to transpose(0, 1) because the first dimension is already batch_size.
        return outputs, hidden




