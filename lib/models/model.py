from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch
import torch.nn as nn

import _init_paths
from models.backbone import getBackbone
from models.lstm import getLSTM
from utils.utils import soft_update_from_to


class VedioClfNet(nn.Module):
    def __init__(self, config, is_train = True):
        super(VedioClfNet, self).__init__()

        self.config = config
        self.rgb_backbone = getBackbone(config, is_train)
        self.flow_backbone = getBackbone(config, is_train)
        self.backbone = [self.rgb_backbone, self.flow_backbone]
        self.lstm = getLSTM()

        # output [mean, std] of Gaussian distribution for frame selection
        self.act_head_frame = nn.Sequential(
            nn.Linear(config.MODEL.LSTM_OUTDIM, 2),
            nn.Sigmoid(),
        )

        # output modality selection probability
        self.act_head_modality = nn.Sequential(
            nn.Linear(config.MODEL.LSTM_OUTDIM, config.MODEL.MODALITY_NUM),
            nn.Softmax(config.MODEL.MODALITY_NUM)
        )

        # output classification scores (not softmaxed)
        self.clf_head = nn.Linear(config.MODEL.LSTM_OUTDIM, config.MODEL.CLFDIM)

        # output soft state value
        self.v_head = nn.Linear(config.MODEL.LSTM_OUTDIM, 1)

        # target v_head and synchronize params
        self.target_v_head = nn.Linear(config.MODEL.LSTM_OUTDIM, 1)
        soft_update_from_to(self.v_head, self.target_v_head, 1)

        # output soft state-action value
        self.q_head = nn.Linear(config.MODEL.LSTM_OUTDIM + 1 + 1, 1)

    def forward(self, x, modality):
        """
        :param x: N * C * H * W
        :param modality: N dimension array. 0-rgb_raw, 1-flow
        """
        # pick different instances for different modalities

        # positions of instances of different modalities
        pos = []
        y = torch.zeros((x.shape[0], self.config.MODEL.LSTM_INDIM)).cuda()
        for i in range(self.config.MODEL.MODALITY_NUM):
            pos.append((modality == i).nonzero().reshape((-1)))

        # feed x into different backbone
        for i in range(self.config.MODEL.MODALITY_NUM):
            # if no modality i, then continue
            if pos[i].shape[0] == 0:
                continue
            y[pos[i]] = self.backbone[i](x[pos[i]])

        h, c = self.lstm(y)
        clf_score = self.clf_haed(h)
        # mean, std = self.act_head_frame(h)
        # modality_prob = self.act_head_modality(h)
        # v_value = self.v_head(h)

        return (h, c), clf_score

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

    def init_weights(self, batch_size, h_c = None):

        self.lstm.reset(batch_size, h_c)


def create_model(config, is_train = True):
    """
    build a complete model.

    :param config: global configs
    :param is_train: train mode
    :return: a model
    """

    model = VedioClfNet(config, is_train)

    if is_train and config.MODEL.INIT_WEIGHTS:
        model.init_weights(config.TRAIN.BATCH_SIZE)

    return model
