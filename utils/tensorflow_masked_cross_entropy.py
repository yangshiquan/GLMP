import tensorflow as tf


def sequence_mask(sequence_length, max_len=None):
    '''
    Generate mask matrix according to sequence length information.
    :param sequence_length:
    :param max_len:
    :return:
    '''
    if max_len is None:
        max_len = sequence_length.numpy().max()
    batch_size = sequence_length.shape[0]
    seq_range = tf.range(0, max_len)  # 1 * max_len.
    seq_range_expand = tf.tile(tf.expand_dims(seq_range, 0), [batch_size, 1])  # seq_range_expand: batch_size * max_len.
    seq_length_expand = tf.tile(tf.expand_dims(sequence_length, 1), [1, seq_range_expand.shape[1]])
    return tf.cast((seq_range_expand < seq_length_expand), dtype=tf.int32)


def generate_indices(target):
    '''
    Generate tensor slice index matrix when using tf.gather_nd.
    :param target:
    :return:
    '''
    max_len = target.shape[0]
    indices = [[i, target[i, 0]] for i in range(max_len)]
    return indices


def masked_cross_entropy(logits, target, length):
    '''
    Masked cross entropy loss at timestep granularity.
    The sigmoid_cross_entropy_with_logits in tensorflow
    only support at sample granularity.

    Args:
        logits: A Variable containing a FloatTensor of size
            (batch, max_len, num_classes) which contains the unnormalized probability for each class.
        target: A Variable containing a LongTensor of size
            (batch, max_len) which contains the index of the true class for each corresponding step.
        length: A Variable containing a LongTensor of size
            (batch,) which contains the length of each data in a batch.
    Returns:
        loss: An average loss value for single timestep masked by the length.
    '''
    logits_flat = tf.reshape(logits, [-1, logits.get_shape()[-1]])  # logits: batch_size * max_len * num_classes, logits_flat: (batch_size * max_len) * num_classes.
    log_probs_flat = tf.nn.log_softmax(logits_flat)  # log_probs_flat: (batch_size * max_len) * num_classes.
    target_flat = tf.reshape(target, [-1, 1])  # target_flat: (batch_size * max_len) * 1.
    losses_flat = -tf.gather_nd(log_probs_flat, generate_indices(target_flat))  # loss_flat: (batch_size * max_len) * 1.
    losses = tf.reshape(losses_flat, target.shape)  # losses: batch_size * max_len.
    mask = sequence_mask(sequence_length=length, max_len=target.shape[1])
    # print(losses)
    # print(mask)
    losses = losses * tf.cast(mask, tf.float32)
    loss = tf.reduce_sum(losses) / tf.reduce_sum(tf.cast(length, tf.float32))
    return loss


# ==============================
# Unit Test
# ==============================
if __name__ == '__main__':
    ret = masked_cross_entropy(tf.Variable([[[0.1, 0.2, 0.3], [0.2, 0.3, 0.4], [0.4, 0.5, 0.6]]]), tf.Variable([[0, 1, 2]]), tf.Variable([2]))
    print(ret)
