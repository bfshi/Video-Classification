from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

import _init_paths
from core.config import config


class ReplayBuffer():
    def __init__(self):
        """
        replay buffer
        """
        self._observation_dim = config.MODEL.LSTM_OUTDIM
        self._max_replay_buffer_size = config.TRAIN.MAX_BUFFER_SIZE
        self._observations_h = np.zeros((self._max_replay_buffer_size, self._observation_dim))
        self._observations_c = np.zeros((self._max_replay_buffer_size, self._observation_dim))
        self._next_obs_h = np.zeros((self._max_replay_buffer_size, self._observation_dim))
        self._next_obs_c = np.zeros((self._max_replay_buffer_size, self._observation_dim))
        self._actions_frame = np.zeros((self._max_replay_buffer_size, 1))
        self._actions_modality = np.zeros((self._max_replay_buffer_size, 1))
        self._rewards = np.zeros((self._max_replay_buffer_size, 1))
        self._top = 0
        self._size = 0

    def save(self, ob, act_frame, act_modality, next_ob, reward):
        """
        Save current observations, actions, rewards and next observations.
        """
        size_needed = act_frame.shape[0]
        size_left = self._max_replay_buffer_size - self._top
        size_extra = size_needed - size_left

        if size_needed <= size_left:
            self._observations_h[self._top : self._top + size_needed] = ob[0]
            self._observations_c[self._top : self._top + size_needed] = ob[1]
            self._actions_frame[self._top : self._top + size_needed] = act_frame
            self._actions_modality[self._top : self._top + size_needed] = act_modality
            self._rewards[self._top : self._top + size_needed] = reward
            self._next_obs_h[self._top : self._top + size_needed] = next_ob[0]
            self._next_obs_c[self._top : self._top + size_needed] = next_ob[1]
        else:
            self._observations_h[self._top: ] = ob[0][0 : size_left]
            self._observations_c[self._top: ] = ob[1][0 : size_left]
            self._actions_frame[self._top: ] = act_frame[0 : size_left]
            self._actions_modality[self._top: ] = act_modality[0 : size_left]
            self._rewards[self._top: ] = reward[0 : size_left]
            self._next_obs_h[self._top: ] = next_ob[0][0 : size_left]
            self._next_obs_c[self._top: ] = next_ob[1][0 : size_left]

            self._observations_h[0 : size_extra] = ob[0][-size_extra:]
            self._observations_c[0 : size_extra] = ob[1][-size_extra:]
            self._actions_frame[0 : size_extra] = act_frame[-size_extra:]
            self._actions_modality[0 : size_extra] = act_modality[-size_extra:]
            self._rewards[0 : size_extra] = reward[-size_extra:]
            self._next_obs_h[0 : size_extra] = next_ob[0][-size_extra:]
            self._next_obs_c[0 : size_extra] = next_ob[1][-size_extra:]

        self._advance(size_needed)

    def get_batch(self, batch_size):
        """
        sample a minibatch of size "batch_size"
        """
        indices = np.random.randint(0, self._size, batch_size)
        return [
            (self._observations_h[indices], self._observations_c[indices]),
            self._actions_frame[indices],
            self._actions_modality[indices],
            (self._next_obs_h[indices], self._next_obs_c[indices]),
            self._rewards[indices],
        ]

    def _advance(self, size_needed):
        self._top = (self._top + size_needed) % self._max_replay_buffer_size
        self._size = min(self._size + size_needed, self._max_replay_buffer_size)


def create_replay_buffer():

    replay_buffer = ReplayBuffer()

    return replay_buffer