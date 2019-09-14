import tensorflow as tf
from tensorflow.python.util import nest
from tensorflow.python.keras.utils import generic_utils
from tensorflow.python.ops import array_ops
from tensorflow.python.framework import dtypes as dtypes_module
from tensorflow.python.ops import math_ops
from tensorflow.python.framework import ops
from tensorflow_models.Libraries.GraphGRUCell import GraphGRUCell


class RNN(tf.keras.Model):
    '''
    Base class for RNN layer.
    '''
    def __init__(self,
                 units,
                 input_dim,
                 recurrent_size=4,
                 return_sequences=False,
                 return_state=False,
                 go_backwards=False,
                 unroll=True,
                 time_major=False,
                 zero_output_for_mask=True,
                 **kwargs):
        super(RNN, self).__init__(**kwargs)
        self.units = units
        self.input_dim = input_dim
        self.recurrent_size = recurrent_size
        self.return_sequences = return_sequences
        self.return_state = return_state
        self.go_backwards = go_backwards
        self.unroll = unroll
        self.time_major = time_major
        self.zero_output_for_mask = zero_output_for_mask
        self.supports_masking = True
        self.cell = GraphGRUCell(units, input_dim, recurrent_size)

    def call(self,
             inputs,  # inputs: batch_size*max_len*embedding_dim
             dependencies,  # dependencies: batch_size*max_len*recurrent_size
             mask=None,  # mask: batch_size*max_len
             initial_states=None,  # initial_states: 4*batch_size*embedding_dim
             training=True):
        if mask is not None:
            mask = nest.flatten(mask)[0]

        timesteps = inputs.shape[0] if self.time_major else inputs.shape[1]
        if self.unroll and timesteps is None:
            raise ValueError('Cannot unroll a RNN if the time dimension is undefined.')

        def step(inputs, states, training):
            output, new_states = self.cell(inputs, states, training)  # inputs: batch_size*embedding_dim, states: 4*batch_size*embedding_dim
            if not nest.is_sequence(new_states):
                new_states = [new_states]
            return output, new_states

        def swap_batch_timestep(input_t):
            axes = list(range(len(input_t.shape)))
            axes[0], axes[1] = 1, 0
            return array_ops.transpose(input_t, axes)

        if not self.time_major:
            inputs = nest.map_structure(swap_batch_timestep, inputs)  # inputs: max_len*batch_size*embedding_dim
            dependencies = swap_batch_timestep(dependencies)  # dependencies: max_len*batch_size*recurrent_size

        flatted_inputs = nest.flatten(inputs)  # inputs: max_len*batch_size*embedding_dim
        time_steps = flatted_inputs[0].shape[0]
        batch = flatted_inputs[0].shape[1]

        for input_ in flatted_inputs:
            input_.shape.with_rank_at_least(3)

        if mask is not None:  # mask: batch_size*max_len
            if mask.dtype != dtypes_module.bool:
                mask = math_ops.cast(mask, dtypes_module.bool)
            if len(mask.shape) == 2:
                mask = array_ops.expand_dims(mask, axis=-1)  # mask: batch_size*max_len*1
            if not self.time_major:
                mask = swap_batch_timestep(mask)  # mask: max_len*batch_size*1

        def _expand_mask(mask_t, input_t, fixed_dim=1):  # mask_t: batch_size*1, input_t: batch_size*embedding_dim
            assert not nest.is_sequence(mask_t)
            assert not nest.is_sequence(input_t)
            rank_diff = len(input_t.shape) - len(mask_t.shape)  # rand_diff: 0
            for _ in range(rank_diff):
                mask_t = array_ops.expand_dims(mask_t, -1)
            multiples = [1] * fixed_dim + input_t.shape.as_list()[fixed_dim:]  # multiples: [1, embedding_dim]
            return array_ops.tile(mask_t, multiples)

        if self.unroll:
            if not time_steps:
                raise ValueError('Unrolling requires a fixed number of timesteps.')
            states = tuple(initial_states)  # initial_states: 4*batch_size*embedding_dim
            successive_states = []
            successive_outputs = []

            def _process_single_input_t(input_t):
                input_t = array_ops.unstack(input_t)
                if self.go_backwards:
                    input_t.reverse()
                return input_t

            if nest.is_sequence(inputs):  # inputs: max_len*batch_size*embedding_dim
                processed_input = nest.map_structure(_process_single_input_t, inputs)
            else:
                processed_input = (_process_single_input_t(inputs),)

            def _get_input_tensor(time):
                inp = [t_[time] for t_ in processed_input]
                return nest.pack_sequence_as(inputs, inp)

            if mask is not None:
                mask_list = array_ops.unstack(mask)  # mask: max_len*batch_size*1
                if self.go_backwards:
                    mask_list.reverse()

                for i in range(time_steps):
                    inp = _get_input_tensor(i)  # inp: batch_size*embedding_dim
                    mask_t = mask_list[i]  # mask_t: batch_size*1
                    dep_t = dependencies[i]  # dep_t: batch_size*recurrent_size
                    output, new_states = step(inp, tuple(states), training)  # inp: batch_size*embedding_dim, states: 4*batch_size*embedding_dim
                    # output: batch_size*embedding_dim, new_states:1*batch_size*embedding_dim
                    tiled_mask_t = _expand_mask(mask_t, output)  # tiled_mask_t: batch_size*embedding_dim

                    if not successive_outputs:
                        pre_output = array_ops.zeros_like(output)
                    else:
                        pre_output = successive_outputs[-1]

                    output = array_ops.where(tiled_mask_t, output, pre_output)  # output: batch_size*embedding_dim

                    # deal with masking
                    if not successive_states:
                        pre_states = array_ops.zeros_like(new_states)  # new_states: 1*batch_size*embedding_dim, pre_states: 1*batch_size*embedding_dim
                    else:
                        pre_states = successive_states[-1]

                    return_states = []
                    for state, new_state in zip(pre_states, new_states):
                        tiled_mask_t = _expand_mask(mask_t, new_state)
                        return_states.append(array_ops.where(tiled_mask_t, new_state, state))
                    # for state, new_state in zip(states, new_states):  # states: 4*batch_size*embedding_dim, new_states: 1*batch_size_embedding_dim
                    #     tiled_mask_t = _expand_mask(mask_t, new_state)
                    #     return_states.append(array_ops.where(tiled_mask_t, new_state, state))

                    successive_outputs.append(output)
                    successive_states.append(return_states)
                    # successive_states.append(states)

                    # get next timestep hidden input
                    states[0] = return_states
                    for k in range(self.recurrent_size - 1):
                        states[k + 1] = successive_states[dep_t[k]]

                last_output = successive_outputs[-1]  # last_output: batch_size*embedding_dim
                new_states = successive_states[-1]  # new_states: batch_size*embedding_dim
                outputs = array_ops.stack(successive_outputs)  # outputs: max_len*batch_size*embedding_dim

                if self.zero_output_for_mask:
                    last_output = array_ops.where(
                        _expand_mask(mask_list[-1], last_output),  # mask_list[-1]: batch_size*1, last_output: batch_size*embedding_dim
                        last_output,  # last_output: batch_size*embedding_dim
                        array_ops.zeros_like(last_output))
                    outputs = array_ops.where(
                        _expand_mask(mask, outputs, fixed_dim=2),  # mask: max_len*batch_size*1, outputs: max_len*batch_size*embedding_dim
                        outputs,  # outputs: max_len*batch_size*embedding_dim
                        array_ops.zeros_like(outputs))

        def set_shape(output_):
            if isinstance(output_, ops.Tensor):
                shape = output_.shape.as_list()
                shape[0] = time_steps
                shape[1] = batch
                output_.set_shape(shape)
            return output_

        outputs = nest.map_structure(set_shape, outputs)

        if not self.time_major:
            outputs = nest.map_structure(swap_batch_timestep, outputs)  # outputs: batch_size*max_len*embedding_dim

        return last_output, outputs, new_states