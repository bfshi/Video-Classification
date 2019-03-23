from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch
import torch.nn as nn

import _init_paths
from models.backbone import getBackbone
from models.lstm import getLSTM


class VedioClfNet(nn.Module):
    def __init__(self, config, is_train = True):
        super(VedioClfNet, self).__init__()

        self.backbone = getBackbone(config, is_train)
        self.lstm = getLSTM()

        #output [mean, std] of Gaussian distribution for frame selection
        self.act_head_frame = nn.Sequential(
            nn.Linear(config.MODEL.LSTM_OUTDIM, 2),
            nn.Sigmoid(),
        )

        #output modality selection probability
        self.act_head_modality = nn.Sequential(
            nn.Linear(config.MODEL.LSTM_OUTDIM, config.MODEL.MODALITY_NUM),
            nn.Softmax(config.MODEL.MODALITY_NUM)
        )

        #output classification scores (not softmaxed)
        self.clf_head = nn.Linear(config.MODEL.LSTM_OUTDIM, config.MODEL.CLFDIM)

        #output soft state value
        self.v_head = nn.Linear(config.MODEL.LSTM_OUTDIM, 1)

        #output soft state-action value
        self.q_head = nn.Linear(config.MODEL.LSTM_OUTDIM + 1 + config.MODEL.MODALITY_NUM, 1)

    def forward(self, x):
        """
        :param x: N * C * H * W
        """
        x = self.backbone(x)
        h = self.lstm(x)
        mean, std = self.act_head_frame(h)
        modality_prob = self.act_head_modality(h)
        clf_score = self.clf_haed(h)
        v_value = self.v_head(h)

        return h, mean, std, modality_prob, clf_score, v_value

    def policy(self, h):
        """
        used to replay old state using current policy
        :param h: old observation
        :return: new action and log of prob
        """
        mean, std = self.act_head_frame(h)
        modality_prob = self.act_head_modality(h)
        new_act_frame = torch.normal(mean, std)
        new_act_modality = torch.multinomial(modality_prob, num_samples=1, replacement=True)
        log_pi = -torch.log((2 * np.pi) ** 0.5 * std) - \
                 (new_act_frame - mean).pow(2) / (2 * std.pow(2)) + \
                 torch.log(modality_prob[:, new_act_modality])

        return new_act_frame, new_act_modality, log_pi

    def init_weights(self):



def create_model(config, is_train = True):
    """
    build a complete model.

    :param config: global configs
    :param is_train: train mode
    :return: a model
    """

    model = VedioClfNet(config, is_train)

    if is_train and cfg.MODEL.INIT_WEIGHTS:
        model.init_weights()

    return model