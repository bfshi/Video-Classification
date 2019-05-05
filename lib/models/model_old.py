from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch
import torch.nn as nn
from collections import OrderedDict

import _init_paths
from models.backbone import getBackbone
from models.lstm import getLSTM
from utils.utils import soft_update_from_to
from utils.utils import torch_clip

class Backbones(nn.Module):
    def __init__(self, config, is_train = True):
        super(Backbones, self).__init__()

        self.config = config
        self.rgb_backbone = getBackbone(config, is_train)
        self.flow_backbone = getBackbone(config, is_train)

    def forward(self, x, modality):
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
            # print(x.shape, pos[i], x[pos[i]].shape)
            # print(y.device)
            # print(list(self.backbone[i].parameters())[0].device)
            # print(list(self.lstm.parameters())[0].device)
            # print(list(self.act_head_modality.parameters())[0].device)
            # y[pos[i]] = self.backbone[i](x[pos[i]])
            if i == 0:
                y[pos[i]] = self.rgb_backbone(x[pos[i]])
            elif i == 1:
                y[pos[i]] = self.flow_backbone(x[pos[i]])

        return y


class Q_head(nn.Module):
    def __init__(self, feature_dim, frame_dim, modality_dim):
        super(Q_head, self).__init__()

        self.feature_dim = feature_dim
        self.frame_dim = frame_dim
        self.modality_dim = modality_dim
        self.model = nn.Linear(feature_dim, frame_dim * modality_dim)

    def forward(self, input):
        batch_size = input.shape[0]
        return self.model(input[:, 0: self.feature_dim])[range(batch_size),
                                                         (input[:, -2] * self.frame_dim).type(torch.long) +
                                                         (self.frame_dim * input[:, -1]).type(torch.long)]


class VedioClfNet(nn.Module):
    def __init__(self, config, is_train = True):
        super(VedioClfNet, self).__init__()

        self.config = config
        self.backbones = Backbones(config, is_train=is_train)
        self.lstm = getLSTM()

        # output [mean, std] of Gaussian distribution for frame selection
        self.act_head_frame = nn.Sequential(
            nn.Linear(config.MODEL.LSTM_OUTDIM, config.MODEL.FRAMEDIV_NUM),
            nn.Softmax(dim=1),
        )

        # output modality selection probability
        self.act_head_modality = nn.Sequential(
            nn.Linear(config.MODEL.LSTM_OUTDIM, config.MODEL.MODALITY_NUM),
            nn.Softmax(dim=1)
        )

        # output classification scores (not softmaxed)
        self.clf_head = nn.Linear(config.MODEL.LSTM_OUTDIM, config.MODEL.CLFDIM)

        # output soft state value
        self.v_head = nn.Linear(config.MODEL.LSTM_OUTDIM, 1)

        # target v_head and synchronize params
        self.target_v_head = nn.Linear(config.MODEL.LSTM_OUTDIM, 1)
        soft_update_from_to(self.v_head, self.target_v_head, 1)

        # output soft state-action value
        self.q_head = Q_head(config.MODEL.LSTM_OUTDIM, config.MODEL.FRAMEDIV_NUM, config.MODEL.MODALITY_NUM)

    def forward(self, x, modality, if_backbone = True, if_lstm = True, if_return_feature = False):
        """
        :param x: N * C * H * W
        :param modality: N dimension array. 0-rgb_raw, 1-flow
        """
        if if_backbone:
            y = self.backbones(x, modality)
        else:
            y = x.clone()

        if if_lstm:
            h, c = self.lstm(y)
            clf_score = self.clf_head(h)
        else:
            h = y.clone()
            c = torch.zeros(h.shape).cuda()
            clf_score = self.clf_head(y)
        # mean, std = self.act_head_frame(h)
        # modality_prob = self.act_head_modality(h)
        # v_value = self.v_head(h)

        if if_return_feature:
            return (h, c), clf_score, y
        else:
            return (h, c), clf_score

    def policy(self, h, if_val = False):
        """
        used to replay old state using current policy
        :param h: old observation
        :return: new action and log of prob
        """
        frame_prob = self.act_head_frame(h)
        modality_prob = self.act_head_modality(h)
        if if_val:
            new_act_frame = torch.argmax(frame_prob, dim=1)
            new_act_modality = torch.argmax(modality_prob, dim=1)
        else:
            new_act_frame = torch.multinomial(frame_prob, num_samples=1, replacement=True).view(-1)
            new_act_modality = torch.multinomial(modality_prob, num_samples=1, replacement=True).view(-1)
        log_pi = torch.log(frame_prob[range(frame_prob.shape[0]), new_act_frame]) + \
                 torch.log(modality_prob[range(modality_prob.shape[0]), new_act_modality])

        # map to [0, 1]
        new_act_frame = new_act_frame.type(torch.float) / self.config.MODEL.FRAMEDIV_NUM

        print("new_act_frame = {}, \n"
              "new_act_modality = {}".format(
            new_act_frame.detach().cpu().numpy(), new_act_modality.detach().cpu().numpy()))

        return new_act_frame, new_act_modality, log_pi

    def init_weights(self, batch_size, h_c = None):

        self.lstm.reset(batch_size, h_c)

    def my_load_state_dict(self, state_dict_old, strict=True, if_lstm = True):
        state_dict = OrderedDict()
        # delete 'module.' because it is saved from DataParallel module
        for key in state_dict_old.keys():
            if not if_lstm and key.startswith('lstm'):
                continue
            state_dict[key.replace('module.', '')] = state_dict_old[key]

        self.load_state_dict(state_dict, strict=strict)

    def my_data_parallel(self, gpus):
        self.backbones = torch.nn.DataParallel(self.backbones, device_ids=gpus)
        self.clf_head = torch.nn.DataParallel(self.clf_head, device_ids=gpus)
        self.act_head_frame = torch.nn.DataParallel(self.act_head_frame, device_ids=gpus)
        self.act_head_modality = torch.nn.DataParallel(self.act_head_modality, device_ids=gpus)
        self.v_head = torch.nn.DataParallel(self.v_head, device_ids=gpus)
        self.target_v_head = torch.nn.DataParallel(self.target_v_head, device_ids=gpus)
        self.q_head = torch.nn.DataParallel(self.q_head, device_ids=gpus)



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
