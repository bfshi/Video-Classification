from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging

import torch
import torch.nn as nn

import _init_paths
from core.config import config

logger = logging.getLogger(__name__)

class Loss(nn.Module):
    def __init__(self, config):
        super(Loss, self).__init__()

        self.r_plc = config.TRAIN.R_PLC
        self.r_q = config.TRAIN.R_Q
        self.r_v = config.TRAIN.R_V
        self.cross_entropy = nn.CrossEntropyLoss()
        self.q_criterion = nn.MSELoss()
        self.v_criterion = nn.MSELoss()

    def clf_loss(self, score, y):

        # logger.info(score.shape)
        # temp = torch.exp(score)
        # logger.info((temp[range(temp.shape[0]), y] / temp.sum(dim = 1)).mean())
        clf_loss = self.cross_entropy(score, y)

        return clf_loss

    def rl_loss(self, rewards, q_pred, v_pred, target_v_pred_next, log_pi, q_new_actions):

        # q-value loss
        q_target = rewards + config.MODEL.DISCOUNT * target_v_pred_next
        q_loss = self.q_criterion(q_pred, q_target.detach())

        # v-value loss
        v_target = q_new_actions - config.MODEL.ENTROPY_RATIO * log_pi
        v_loss = self.v_criterion(v_pred, v_target.detach())

        # policy loss
        log_policy_target = q_new_actions - v_pred
        policy_loss = (
                log_pi * (log_pi - log_policy_target).detach()
        ).mean()

        return self.r_plc * policy_loss + self.r_q * q_loss + self.r_v * v_loss