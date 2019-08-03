import tensorflow as tf
from utils.config import *


class ExternalKnowledge(tf.keras.Model):
    def __init__(self, vocab, embedding_dim, hop, dropout):
        super(ExternalKnowledge, self).__init__()
        self.max_hops = hop
        self.embedding_dim = embedding_dim
        self.vocab = vocab
        self.dropout = dropout
        self.dropout_layer = tf.keras.layers.Dropout(self.dropout)
        self.C_1 = tf.keras.layers.Embedding(self.vocab,
                                          self.embedding_dim,
                                          embeddings_initializer=tf.random_normal_initializer(0.0, 0.1))  # different: no masking for pad token, pad token embedding does not equal zero, only support one hop.
        self.C_2 = tf.keras.layers.Embedding(self.vocab,
                                          self.embedding_dim,
                                          embeddings_initializer=tf.random_normal_initializer(0.0, 0.1))  # different: no masking for pad token, pad token embedding does not equal zero, only support one hop.
        self.softmax = tf.keras.layers.Softmax(1)
        self.sigmoid = tf.keras.activations.sigmoid()

    def add_lm_embedding(self, full_memory, kb_len, conv_len, hiddens):
        for bi in range(full_memory.get_shape(0)):
            start, end = kb_len[bi], kb_len[bi] + conv_len[bi]
            full_memory[bi, start:end, :] = full_memory[bi, start:end, :] + hiddens[bi, :conv_len[bi], :]
        return full_memory

    def load_memory(self, story, kb_len, conv_len, hidden, dh_outputs, training=True):
        u = [hidden]  # different: hidden without squeeze(0), hidden: batch_size * embedding_size.
        story_size = story.get_shape()
        self.m_story = []
        embedding_A = self.C_1(tf.reshape(story, [story_size[0], -1]))  # story: batch_size * seq_len * MEM_TOKEN_SIZE, embedding_A: batch_size * memory_size * MEM_TOKEN_SIZE * embedding_dim.
        embedding_A = tf.reshape(embedding_A, [story_size[0], story_size[1], story_size[2], embedding_A.get_shape[-1]])  # embedding_A: batch_size * memory_size * MEM_TOKEN_SIZE * embedding_dim.
        embedding_A = tf.math.reduce_sum(embedding_A, 2)  # embedding_A: batch_size * memory_size * embedding_dim.
        if not args['ablationH']:
            embedding_A = self.add_lm_embedding(embedding_A, kb_len, conv_len, dh_outputs)
        if training:
            embedding_A = self.dropout_layer(embedding_A, training=training)

        u_temp = tf.tile(tf.expand_dims(u[-1], 1), embedding_A.get_shape())  # u_temp: batch_size * memory_size * embedding_dim.
        prob_logits = tf.math.reduce_sum((embedding_A * u_temp), 2)  # prob_logits: batch_size * memory_size
        prob_soft = self.softmax(prob_logits)  # prob_soft: batch_size * memory_size

        embedding_C = self.C_2(tf.reshape(story, [story_size[0], -1]))
        embedding_C = tf.reshape(embedding_C, [story_size[0], story_size[1], story_size[2], embedding_C.get_shape[-1]])
        embedding_C = tf.math.reduce_sum(embedding_C, 2)  # embedding_C: batch_size * memory_size * embedding_dim.
        if not args['ablationH']:
            embedding_C = self.add_lm_embedding(embedding_C, kb_len, conv_len, dh_outputs)

        prob_soft_temp = tf.tile(tf.expand_dims(prob_soft, 2), embedding_C.get_shape())  # prob_soft_temp: batch_size * memory_size * embedding_dim.
        u_k = u[-1] + tf.math.reduce_sum((embedding_C * prob_soft_temp), 1)
        u.append(u_k)
        self.m_story.append(embedding_A)
        self.m_story.append(embedding_C)
        return self.sigmoid(prob_logits), u[-1]

    def call(self, query_vector, global_pointer, training=True):
        u = [query_vector]  # query_vector: batch_size * embedding_dim.
        embed_A = self.m_story[0]  # embed_A: batch_size * memory_size * embedding_dim.
        if not args['ablationG']:
            embed_A = embed_A * tf.tile(tf.expand_dims(global_pointer, 2), embed_A.get_shape())

        u_temp = tf.tile(tf.expand_dims(u[-1], 1), embed_A.get_shape())  # u_temp: batch_size * memory_size * embedding_dim.
        prob_logits = tf.math.reduce_sum((embed_A * u_temp), 2)  # prob_logits: batch_size * memory_size.
        prob_soft = self.softmax(prob_logits)  # prob_soft: batch_size * memory_size.

        embed_C = self.m_story[1]  # embed_C: batch_size * memory_size * embedding_dim.
        if not args['ablationG']:
            embed_C = embed_C * tf.tile(tf.expand_dims(global_pointer, 2), embed_C.get_shape())

        prob_soft_temp = tf.tile(tf.expand_dims(prob_soft, 2), embed_C.get_shape())  # prob_soft_temp: batch_size * memory_size * embedding_dim.
        u_k = u[-1] + tf.math.reduct_sum((embed_C * prob_soft_temp), 1)  # u_k: batch_size * embedding_dim.
        u.append(u_k)
        return prob_soft, prob_logits