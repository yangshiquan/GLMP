import torch.nn as nn
import numpy as np
import torch
import tensorflow as tf

sigmoid = nn.Sigmoid()
out = sigmoid(torch.Tensor([[0.0, 0.0, 0.0], [0.1, 0.1, 0.1]]))

loss = nn.BCELoss()
torch_bce_loss = loss(torch.Tensor(sigmoid(torch.Tensor([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]))), torch.Tensor([[1, 0, 1], [1, 1, 1]]))
print("torch_bce_loss:", torch_bce_loss)

tensorflow_bce_loss = tf.compat.v1.losses.sigmoid_cross_entropy(tf.cast(tf.convert_to_tensor([[1, 0, 1], [1, 1, 1]]), dtype=tf.double), tf.cast(tf.convert_to_tensor([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]), dtype=tf.double))
tensorflow_bce_loss2 = tf.nn.sigmoid_cross_entropy_with_logits(tf.cast(tf.convert_to_tensor([[1, 0, 1], [1, 1, 1]]), dtype=tf.double), tf.cast(tf.convert_to_tensor([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]), dtype=tf.double))
loss2 = tf.cast(tf.reduce_sum(tensorflow_bce_loss2)/(tensorflow_bce_loss2.shape[0]*tensorflow_bce_loss2.shape[1]), dtype=tf.float32)
print("tensorflow_bce_loss:", tensorflow_bce_loss)
print("tensorflow_bce_loss2:", loss2)
