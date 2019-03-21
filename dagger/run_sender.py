#!/usr/bin/env python

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


import argparse
import project_root
import numpy as np
import tensorflow as tf
from os import path
from env.sender import Sender
from models import DaggerLSTM
from helpers.helpers import normalize, one_hot, softmax


class Learner(object):
    def __init__(self, sender, state_dim, restore_vars):
        self.aug_state_dim = state_dim + 1#action_cnt
        self.prev_action = 0
        self.sender = sender
        with tf.variable_scope('global'):
            self.model = DaggerLSTM(
                state_dim=self.aug_state_dim, dwnd=Sender.dwnd)

        self.lstm_state = self.model.zero_init_state(1)

        self.sess = tf.Session()

        # restore saved variables
        saver = tf.train.Saver(self.model.trainable_vars)
        saver.restore(self.sess, restore_vars)

        # init the remaining vars, especially those created by optimizer
        uninit_vars = set(tf.global_variables())
        uninit_vars -= set(self.model.trainable_vars)
        self.sess.run(tf.variables_initializer(uninit_vars))

    def policy(self, state):
        """ Given a state buffer in the past step, returns an action
        to perform.

        Appends to the state/action buffers the state and the
        "correct" action to take according to the expert.
        """

        aug_state = state + [self.prev_action]
        self.sender.update_decision_window(aug_state)

        # Get probability of each action from the local network.
        pi = self.model
        feed_dict = {
            pi.input: [self.sender.decision_window],
            pi.state_in: self.lstm_state,
        }
        ops_to_run = [pi.actions, pi.state_out]
        actions, self.lstm_state = self.sess.run(ops_to_run, feed_dict)

        # Choose an action to take and update current LSTM state
        if len(self.sender.decision_window) <= 1:
            action = actions
        else:
            action = actions[-1]
        # print("actions shape:" + str(actions.shape))
        # print("in policy(): action is: " + str(action))
        self.prev_action = action

        return action


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('port', type=int)
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()

    sender = Sender(args.port, debug=args.debug)

    model_path = path.join(project_root.DIR, 'dagger', 'model', 'model')

    learner = Learner(sender=sender,
        state_dim=Sender.state_dim,
        restore_vars=model_path)

    sender.set_policy(learner.policy)

    try:
        sender.handshake()
        sender.run()
    except KeyboardInterrupt:
        pass
    finally:
        sender.cleanup()


if __name__ == '__main__':
    main()
