from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch

import _init_paths
from core.config import config


class ReplayBuffer():
    def __init__(self):
        """
        replay buffer
        """
        self._observation_dim = config.MODEL.FEATURE_DIM
        self._framediv_num = config.MODEL.FRAMEDIV_NUM
        self._modality_num = config.MODEL.MODALITY_NUM
        self._max_replay_buffer_size = config.TRAIN.MAX_BUFFER_SIZE

        self._observations = torch.zeros((self._max_replay_buffer_size, self._observation_dim))
        self._costs = torch.zeros((self._max_replay_buffer_size))
        self._choice_his = torch.zeros((self._max_replay_buffer_size, self._modality_num, self._framediv_num))
        self._next_obs = torch.zeros((self._max_replay_buffer_size, self._observation_dim))
        self._next_costs = torch.zeros((self._max_replay_buffer_size))
        self._actions_frame = torch.zeros((self._max_replay_buffer_size))
        self._actions_modality = torch.zeros((self._max_replay_buffer_size))
        self._rewards = torch.zeros((self._max_replay_buffer_size))
        self._top = 0
        self._size = 0

    def save(self, ob, cost, choice_his, act_frame, act_modality, next_ob, next_cost, reward):
        """
        Save current observations, actions, rewards and next observations.
        """
        size_needed = act_frame.shape[0]
        size_left = self._max_replay_buffer_size - self._top
        size_extra = size_needed - size_left

        if size_needed <= size_left:
            self._observations[self._top : self._top + size_needed] = ob
            self._costs[self._top : self._top + size_needed] = cost
            self._choice_his[self._top : self._top + size_needed] = choice_his
            self._actions_frame[self._top : self._top + size_needed] = act_frame
            self._actions_modality[self._top : self._top + size_needed] = act_modality
            self._rewards[self._top : self._top + size_needed] = reward
            self._next_obs[self._top : self._top + size_needed] = next_ob
            self._next_costs[self._top : self._top + size_needed] = next_cost
        else:
            self._observations[self._top: ] = ob[0 : size_left]
            self._costs[self._top: ] = cost[0 : size_left]
            self._choice_his[self._top: ] = choice_his[0 : size_left]
            self._actions_frame[self._top: ] = act_frame[0 : size_left]
            self._actions_modality[self._top: ] = act_modality[0 : size_left]
            self._rewards[self._top: ] = reward[0 : size_left]
            self._next_obs[self._top: ] = next_ob[0 : size_left]
            self._next_costs[self._top: ] = next_cost[0 : size_left]

            self._observations[0 : size_extra] = ob[-size_extra:]
            self._costs[0 : size_extra] = cost[-size_extra:]
            self._choice_his[0 : size_extra] = choice_his[-size_extra:]
            self._actions_frame[0 : size_extra] = act_frame[-size_extra:]
            self._actions_modality[0 : size_extra] = act_modality[-size_extra:]
            self._rewards[0 : size_extra] = reward[-size_extra:]
            self._next_obs[0 : size_extra] = next_ob[-size_extra:]
            self._next_costs[0 : size_extra] = next_cost[-size_extra:]

        self._advance(size_needed)

    def get_batch(self, batch_size):
        """
        sample a minibatch of size "batch_size"
        """
        indices = np.random.randint(0, self._size, batch_size)
        return [
            self._observations[indices].cuda(),
            self._costs[indices].cuda(),
            self._choice_his[indices].cuda(),
            self._actions_frame[indices].cuda(),
            self._actions_modality[indices].cuda(),
            self._next_obs[indices].cuda(),
            self._next_costs[indices].cuda(),
            self._rewards[indices].cuda(),
        ]

    def _advance(self, size_needed):
        self._top = (self._top + size_needed) % self._max_replay_buffer_size
        self._size = min(self._size + size_needed, self._max_replay_buffer_size)


def create_replay_buffer():

    replay_buffer = ReplayBuffer()

    return replay_buffer