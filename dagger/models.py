# Copyright 2018 Francis Y. Yan, Jestin Ma
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
#     Unless required by applicable law or agreed to in writing, software
#     distributed under the License is distributed on an "AS IS" BASIS,
#     WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#     See the License for the specific language governing permissions and
#     limitations under the License.


import numpy as np
import tensorflow as tf
from tensorflow.contrib import layers, rnn

class DaggerLSTM(object):
    def __init__(self, state_dim, dwnd):
        # dummy variable used to verify that sharing variables is working
        self.cnt = tf.get_variable(
            'cnt', [], tf.float32,
            initializer=tf.constant_initializer(0.0))
        self.add_one = self.cnt.assign_add(1.0)

        # self.input: [batch_size, max_time, state_dim]
        self.input = tf.placeholder(tf.float32, [None, None, state_dim])

        self.num_layers = 1
        self.lstm_dim = 32
        self.linear_dim = 16
        self.attn_dim = 32
        stacked_lstm = rnn.MultiRNNCell([rnn.BasicLSTMCell(self.lstm_dim)
            for _ in xrange(self.num_layers)])

        self.state_in = []
        state_tuple_in = []
        for _ in xrange(self.num_layers):
            c_in = tf.placeholder(tf.float32, [None, self.lstm_dim])
            h_in = tf.placeholder(tf.float32, [None, self.lstm_dim])
            self.state_in.append((c_in, h_in))
            state_tuple_in.append(rnn.LSTMStateTuple(c_in, h_in))

        self.state_in = tuple(self.state_in)
        state_tuple_in = tuple(state_tuple_in)

        # self.output: [batch_size, max_time, lstm_dim]
        state_embedding = layers.linear(self.input, self.linear_dim)
        output, state_tuple_out = tf.nn.dynamic_rnn(
            stacked_lstm, state_embedding, initial_state=state_tuple_in)

        self.state_out = self.convert_state_out(state_tuple_out)

        # map output to scores
        u = layers.linear(output, self.attn_dim)
        u = tf.nn.tanh(u) # batch_size * max_time * attn_dim

        v = tf.get_variable('attn_v', [self.attn_dim])
        v = tf.expand_dims(tf.expand_dims(v, 1), 1)
        self.y = tf.reduce_sum(v * u, [2]) # batch_size * max_time

        attn_vec = tf.TensorArray(dtype=tf.float32, size=1, dynamic_size=True)
        i = tf.constant(0)
        num_iter = tf.shape(output)[1]
        def loop_body(i, dim, attn_vec):
            start = tf.cond(i - dwnd + 1 < 0, lambda: 0, lambda: i - dwnd + 1)
            end = i + 1
            a = tf.expand_dims(tf.nn.softmax(self.y[:, start : end]), 2)
            s = tf.reduce_sum(a * output[:, start : end, :], [1])
 
            attn_vec = attn_vec.write(i, s)
            return i + 1, dim, attn_vec

        _, _, self.attn_output = tf.while_loop(lambda i, dim, _: i < dim, 
                                            loop_body, [i, num_iter, attn_vec])
            
        # self.action_scores is still batch_size * max_time * lstm_dim
        # self.action_scores = tf.stack(self.attn_vec, 1)
        self.attn_output = self.attn_output.stack()
        self.actions = tf.nn.tanh(layers.linear(self.attn_output, 1))
        self.actions = tf.transpose(tf.squeeze(self.attn_output))  # batch_size * max_time


        self.trainable_vars = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, tf.get_variable_scope().name)

    def convert_state_out(self, state_tuple_out):
        state_out = []
        for lstm_state_tuple in state_tuple_out:
            state_out.append((lstm_state_tuple.c, lstm_state_tuple.h))

        return tuple(state_out)

    def zero_init_state(self, batch_size):
        init_state = []
        for _ in xrange(self.num_layers):
            c_init = np.zeros([batch_size, self.lstm_dim], np.float32)
            h_init = np.zeros([batch_size, self.lstm_dim], np.float32)
            init_state.append((c_init, h_init))

        return init_state
