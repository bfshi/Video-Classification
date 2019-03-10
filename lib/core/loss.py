from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.nn as nn

class Loss(nn.Module):
    def __init__(self, config):
        super(Loss, self).__init__()

        self.r_act = config.TRAIN.R_ACT
        self.r_crt = config.TRAIN.R_CRT
        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(self, score, y, log_prob, advantages):
        """
        compute the loss

        :param score: score for each class. N * ClassNum
        :param y: ground-truth class. N
        :param log_prob: log of each action's probibility. N * stepNum
        :param advantages: reward + gamma * V_(t+1) - V_t. N * stepNum
        :return: weighted total loss
        """
        clf_loss = self.cross_entropy(score, y)
        crt_loss = advantages.pow(2).mean()
        act_loss = -(advantages.detach() * log_prob).mean()

        return clf_loss + self.r_act * act_loss + self.r_crt * crt_loss
