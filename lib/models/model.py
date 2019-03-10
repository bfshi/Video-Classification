from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn

import _init_paths
from models.backbone import getBackbone
from models.lstm import getLSTM


class VedioClfNet(nn.Module):
    def __init__(self, config):
        super(VedioClfNet, self).__init__()

        self.backbone = getBackbone()
        self.lstm = getLSTM()
        self.act_head = nn.Sequential{
            nn.Linear(config.MODEL.LSTM_OUTDIM, 1),
            nn.Sigmoid(),
        }
        self.clf_haed = nn.Linear(config.MODEL.LSTM_OUTDIM, config.MODEL.CLFDIM),
        self.crit_head = nn.Linear(config.MODEL.LSTM_OUTDIM, 1)

    def forward(self, x):
        """
        :param x: N * C * H * W
        :return:
        action: N * 1
        score: N * ClassNum (not softmaxed)
        value: N * 1
        """
        x = self.backbone(x)
        h = self.lstm(x)
        action = self.act_head(h)
        score = self.clf_haed(h)
        value = self.crit_head(h)

        return action, score, value

    def init_weights(self):



def create_model(config, is_train = True):
    """
    build a complete model.

    :param config: global configs
    :param is_train: train mode
    :return: a model
    """

    model = VedioClfNet(config)

    if (is_train):
        model.init_weights()

    return model