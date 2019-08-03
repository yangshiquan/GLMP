import tensorflow as tf


def masked_cross_entropy(logits, target, length):
    '''
    Masked cross entropy loss at timestep granularity. The sigmoid_cross_entropy_with_logits in tensorflow only support at sample granularity.
    :param logits:
    :param target:
    :param length:
    :return:
    '''





    return