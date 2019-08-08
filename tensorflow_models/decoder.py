import tensorflow as tf
from utils.config import *
import numpy as np
import pdb


class LocalMemoryDecoder(tf.keras.Model):
    def __init__(self, shared_emb, lang, embedding_dim, hop, dropout):
        super(LocalMemoryDecoder, self).__init__()
        self.num_vocab = lang.n_words
        self.lang = lang
        self.max_hops = hop
        self.embedding_dim = embedding_dim
        self.dropout = dropout
        self.dropout_layer = tf.keras.layers.Dropout(dropout)
        self.C = shared_emb
        self.softmax = tf.keras.layers.Softmax(1)
        self.sketch_rnn = tf.keras.layers.GRU(embedding_dim,
                                              dropout=dropout,
                                              return_sequences=True,
                                              return_state=True)  # different: need to set training flag if using dropout.
        self.relu = tf.keras.layers.ReLU()
        self.projector = tf.keras.layers.Dense(embedding_dim)
        self.softmax = tf.keras.layers.Softmax(1)

    def attend_vocab(self, seq, cond):
        scores_ = tf.matmul(cond, tf.transpose(seq))  # different: no softmax layer, need to check loss function.
        return scores_

    def call(self, extKnow, story_size, story_lengths, copy_list, encode_hidden,
             target_batches, max_target_length, batch_size, use_teacher_forcing,
             get_decoded_words, global_pointer, training=True):
        # all_decoder_outputs_vocab = tf.zeros([max_target_length.numpy()[0], batch_size, self.num_vocab])  # max_target_length * batch_size * num_vocab.
        # all_decoder_outputs_ptr = tf.zeros([max_target_length.numpy()[0], batch_size, story_size[1]])  # max_target_length * batch_size * memory_size.
        # memory_mask_for_step = tf.ones([story_size[0], story_size[1]])  # batch_size * memory_size.
        memory_mask_for_step = np.ones((story_size[0], story_size[1]))  # batch_size * memory_size.
        decoded_fine, decoded_coarse = [], []

        decoder_input = tf.constant([SOS_token] * batch_size)  # batch_size.
        hidden = self.relu(self.projector(encode_hidden))  # batch_size * embedding_dim.

        all_decoder_outputs_vocab = []
        all_decoder_outputs_ptr = []
        for t in range(max_target_length):
            embed_q = self.C(decoder_input)
            if training:
                embed_q = self.dropout_layer(embed_q, training=training)  # batch_size * embedding_dim.
            if len(embed_q.get_shape()) == 1:
                embed_q = tf.expand_dims(embed_q, 0)
            _, hidden = self.sketch_rnn(tf.expand_dims(embed_q, 1),
                                        initial_state=hidden,
                                        training=training)  # 1 * batch_size * embedding_dim.
            query_vector = hidden  # need to check meaning of hidden[0], query_vector: batch_size * embedding_dim.

            p_vocab = self.attend_vocab(self.C.embeddings.numpy(), hidden)  # self.C.read_value: num_vocab * embedding_dim, p_vocab: batch_size * num_vocab.
            # all_decoder_outputs_vocab[t] = p_vocab
            all_decoder_outputs_vocab.append(p_vocab)
            _, topvi = tf.math.top_k(p_vocab)  # topvi: batch_size * 1.

            prob_soft, prob_logits = extKnow(query_vector, global_pointer, training=training)  # query_vector: batch_size * embedding_dim, global_pointer: batch_size * memory_size.
            # all_decoder_outputs_ptr[t] = prob_logits  # need to check whether use softmax or not, prob_logits: batch_size * memory_size.
            all_decoder_outputs_ptr.append(prob_logits)

            if use_teacher_forcing:
                decoder_input = target_batches[:, t]  # decoder_input: batch_size, target_batches[:, t].
            else:
                decoder_input = tf.squeeze(topvi)  # decoder_input: batch_size.

            if get_decoded_words:
                search_len = min(5, min(story_lengths))
                prob_soft = prob_soft * memory_mask_for_step
                _, toppi = tf.math.top_k(prob_soft, k=search_len)  # toppi: batch_size * search_len.
                temp_f, temp_c = [], []

                for bi in range(batch_size):
                    token = topvi[bi, 0].numpy().astype(int)
                    temp_c.append(self.lang.index2word[token])

                    if '@' in self.lang.index2word[token]:
                        cw = 'UNK'
                        for i in range(search_len):
                            if toppi[bi, i].numpy().astype(int) < story_lengths[bi] - 1:
                                cw = copy_list[bi][toppi[bi, i].numpy().astype(int)]
                                break
                        temp_f.append(cw)
                        if args['record']:
                            memory_mask_for_step[bi, toppi[bi, i].numpy().astype(int)] = 0
                    else:
                        temp_f.append(self.lang.index2word[token])
                decoded_coarse.append(temp_c)
                decoded_fine.append(temp_f)
        all_decoder_outputs_vocab_out = tf.stack(all_decoder_outputs_vocab, axis=0)
        all_decoder_outputs_ptr_out = tf.stack(all_decoder_outputs_ptr, axis=0)
        return all_decoder_outputs_vocab_out, all_decoder_outputs_ptr_out, decoded_fine, decoded_coarse