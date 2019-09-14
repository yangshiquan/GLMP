import tensorflow as tf
from tensorflow.python.ops import array_ops
from tensorflow.python.keras import backend as K


class GraphGRUCell(tf.keras.Model):
    '''
    Cell class for GraphGRU layer.
    '''
    def __init__(self,
                 units,
                 input_dim,
                 recurrent_size=4,
                 activation='tanh',
                 recurrent_activation='sigmoid',
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 recurrent_initializer='orthogonal',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 recurrent_regularizer=None,
                 bias_regularizer=None,
                 kernel_constraint=None,
                 recurrent_constraint=None,
                 bias_constraint=None,
                 dropout=0.,
                 recurrent_dropout=0.,
                 **kwargs):
        super(GraphGRUCell, self).__init__(**kwargs)
        self.units = units
        self.input_dim = input_dim
        self.recurrent_size = recurrent_size

        self.activation = activation
        self.recurrent_activation = recurrent_activation
        self.use_bias = use_bias

        self.kernel_initializer = kernel_initializer
        self.recurrent_initializer = recurrent_initializer
        self.bias_initializer = bias_initializer

        self.kernel_regularizer = kernel_regularizer
        self.recurrent_regularizer = recurrent_regularizer
        self.bias_regularizer = bias_regularizer

        self.kernel_constraint = kernel_constraint
        self.recurrent_constraint = recurrent_constraint
        self.bias_constraint = bias_constraint

        self.dropout = min(1., max(0., dropout))
        self.recurrent_dropout = min(1., max(0., recurrent_dropout))

        self.kernel = self.add_weight(  # self.kernel: input_dim*(3*embedding_dim)
            name='kernel',
            shape=(input_dim, 3 * units),
            initializer=kernel_initializer,
            regularizer=kernel_regularizer,
            constraint=kernel_constraint
        )

        self.recurrent_kernel = self.add_weight(  # self.recurrent_kernel: recurrent_size*embedding_dim*(3*embedding_dim)
            name='recurrent_kernel',
            shape=(recurrent_size, units, 3 * units),
            initializer=recurrent_initializer,
            regularizer=recurrent_regularizer,
            constraint=recurrent_constraint
        )

        if use_bias:
            self.bias = self.add_weight(  # self.bias: (recurrent_size+1)*(3*embedding_dim)
                name='bias',
                shape=((recurrent_size + 1), 3 * units),
                initializer=bias_initializer,
                regularizer=bias_regularizer,
                constraint=bias_constraint
            )
        else:
            self.bias = None

    def call(self, inputs, states, training=True):  # inputs: batch_size*embedding_dim, states:4*batch_size*embedding_dim
        batch_size = inputs.shape[0]
        if self.use_bias:
            unstacked_biases = array_ops.unstack(self.bias)  # unstacked_biases: (recurrent_size+1)*embedding_dim
            input_bias, recurrent_bias = unstacked_biases[0], unstacked_biases[1:]  # input_bias: (3*embedding_dim), recurrent_bias: recurrent_size*(3*embedding_dim)

        matrix_x = K.dot(inputs, self.kernel)  # matrix_x: batch_size*(3*embedding_dim)
        if self.use_bias:
            # biases: bias_z_i, bias_r_i, bias_h_i
            matrix_x = K.bias_add(matrix_x, input_bias)

        x_z = matrix_x[:, :self.units]  # x_z: batch_size*embedding_dim
        x_r = matrix_x[:, self.units: 2 * self.units]  # x_r: batch_size*embedding_dim
        x_h = matrix_x[:, 2 * self.units:]  # x_h: batch_size*embedding_dim

        accumulate_h = array_ops.zeros([batch_size, self.units])  # accumulate_h: batch_size*embedding_dim
        accumulate_z_h = array_ops.zeros([batch_size, self.units])  # accumulate_z_h: batch_size*embedding_dim
        accumulate_z = array_ops.zeros([batch_size, self.units])  # accumulate_z: batch_size*embedding_dim
        for k in range(self.recurrent_size):
            matrix_inner = K.dot(states[k], self.recurrent_kernel[k])  # matrix_inner: batch_size*(3*embedding_dim), states[k]: batch_size*embedding_dim
            if self.use_bias:
                matrix_inner = K.bias_add(matrix_inner, recurrent_bias[k])
            recurrent_z = matrix_inner[:, :self.units]  # recurrent_z: batch_size*embedding_dim
            recurrent_r = matrix_inner[:, self.units: 2 * self.units]  # recurrent_r: batch_size*embedding_dim

            z = self.recurrent_activation(x_z + recurrent_z)  # z: batch_size*embedding_dim
            r = self.recurrent_activation(x_r + recurrent_r)  # r: batch_size*embedding_dim

            recurrent_h = r * matrix_inner[:, 2 * self.units:]  # recurrent_h: batch_size*embedding_dim
            accumulate_h = accumulate_h + recurrent_h  # accumulate_h: batch_size*embedding_dim

            accumulate_z_h = accumulate_z_h + z * states[k]  # accumulate_z_h: batch_size*embedding_dim
            accumulate_z = accumulate_z + z  # accumulate_z: batch_size*embedding_dim

        hh = self.activation(x_h + accumulate_h / self.recurrent_size)  # hh: batch_size*embedding_dim
        h = (1 - accumulate_z / self.recurrent_size) * hh + accumulate_z_h / self.recurrent_size  # h: batch_size*embedding_dim
        return h, [h]