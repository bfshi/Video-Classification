from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch
import torch.nn as nn
from collections import OrderedDict

import _init_paths
from models.time_conv import get_time_conv
from utils.utils import soft_update_from_to
from utils.utils import torch_clip


class Q_head(nn.Module):
    def __init__(self, feature_dim, frame_dim, modality_dim):
        super(Q_head, self).__init__()

        self.feature_dim = feature_dim
        self.frame_dim = frame_dim
        self.modality_dim = modality_dim
        self.model = nn.Sequential(
            nn.Linear(feature_dim, feature_dim - 1),
            nn.ReLU(),
            nn.Linear(feature_dim - 1, frame_dim * modality_dim)
        )

    def forward(self, input):
        batch_size = input.shape[0]
        return self.model(input[:, 0: self.feature_dim])[range(batch_size),
                                                         input[:, -2].type(torch.long) +
                                                         (self.frame_dim * input[:, -1]).type(torch.long)]


class VedioClfNet(nn.Module):
    def __init__(self, config, is_train = True):
        super(VedioClfNet, self).__init__()

        self.config = config
        self.time_conv = torch.nn.ModuleList([get_time_conv() for i in range(config.MODEL.MODALITY_NUM)])
        # self.time_conv = get_time_conv()

        # output frame selection probability
        self.act_head_frame = nn.Sequential(
            nn.Linear(config.MODEL.FEATURE_DIM * config.MODEL.MODALITY_NUM + config.MODEL.COST_DIM,
                      config.MODEL.FEATURE_DIM),
            nn.ReLU(),
            nn.Linear(config.MODEL.FEATURE_DIM, config.MODEL.FRAMEDIV_NUM),
            nn.Softmax(dim=1),
        )

        # output modality selection probability
        self.act_head_modality = nn.Sequential(
            nn.Linear(config.MODEL.FEATURE_DIM * config.MODEL.MODALITY_NUM + config.MODEL.COST_DIM,
                      config.MODEL.FEATURE_DIM),
            nn.ReLU(),
            nn.Linear(config.MODEL.FEATURE_DIM, config.MODEL.MODALITY_NUM),
            nn.Softmax(dim=1)
        )

        # output classification scores (not softmaxed)
        self.clf_head = nn.Linear(config.MODEL.FEATURE_DIM, config.MODEL.CLFDIM)

        # output soft state value
        self.v_head = nn.Sequential(
            nn.Linear(config.MODEL.FEATURE_DIM * config.MODEL.MODALITY_NUM + config.MODEL.COST_DIM,
                      config.MODEL.FEATURE_DIM),
            nn.ReLU(),
            nn.Linear(config.MODEL.FEATURE_DIM, 1)
        )

        # target v_head and synchronize params
        self.target_v_head = nn.Sequential(
            nn.Linear(config.MODEL.FEATURE_DIM * config.MODEL.MODALITY_NUM + config.MODEL.COST_DIM,
                      config.MODEL.FEATURE_DIM),
            nn.ReLU(),
            nn.Linear(config.MODEL.FEATURE_DIM, 1)
        )
        soft_update_from_to(self.v_head, self.target_v_head, 1)

        # output soft state-action value
        self.q_head = Q_head(config.MODEL.FEATURE_DIM * config.MODEL.MODALITY_NUM + config.MODEL.COST_DIM,
                             config.MODEL.FRAMEDIV_NUM, config.MODEL.MODALITY_NUM)

    def forward(self, x, if_fusion = False):
        """
        :param x: N * C * T * W
        :param if_fusion: if to merge clf_scores of different modalities
        """
        clf_score_list = []
        for i in range(self.config.MODEL.MODALITY_NUM):
            if i == 0:
                y = self.time_conv[i](x[:, i: i + 1])
                clf_score_list.append(self.clf_head(y))
            else:
                temp = self.time_conv[i](x[:, i: i + 1])
                y = torch.cat((y, temp), dim=1)
                clf_score_list.append(self.clf_head(temp))

        if if_fusion:
            clf_score = clf_score_list[0]
            for i in range(1, self.config.MODEL.MODALITY_NUM):
                clf_score += clf_score_list[i]
            return clf_score, y
        else:
            return clf_score_list, y
        # y = self.time_conv(x)
        # clf_score = self.clf_head(y)
        # return clf_score, y


    def policy(self, y, cost_his, choice_his, if_val = False, if_random = False):
        """
        used to replay old state using current policy
        :param y: old observation (N * feature_dim)
        :param cost_his: aggregated cost before (N, )
        :param choice_his: which have been chosen (N, modality_num, framediv_num)
        :return: new action and log of prob
        """
        if not if_random:
            y = torch.cat((y, cost_his.view(-1, 1)), dim=1)
            frame_prob = self.act_head_frame(y)
            modality_prob = self.act_head_modality(y)

            total_prob = torch.zeros((frame_prob.shape[0], 0)).cuda()
            for i in range(self.config.MODEL.MODALITY_NUM):
                total_prob = torch.cat((total_prob, frame_prob * modality_prob[:, i].view(-1, 1)), dim=1)
        else:
            if_val = False
            total_prob = torch.ones((y.shape[0],
                                     self.config.MODEL.FRAMEDIV_NUM * self.config.MODEL.MODALITY_NUM)).cuda()

        # cannot choose ones that have been chosen
        total_prob *= (1 - choice_his.view(total_prob.shape[0], -1))
        # normalization
        total_prob /= total_prob.sum(dim=1, keepdim=True)

        if if_val:
            new_act = torch.argmax(total_prob, dim=1)
        else:
            new_act = torch.multinomial(total_prob, num_samples=1, replacement=True).view(-1)

        new_act_frame = new_act % self.config.MODEL.FRAMEDIV_NUM
        new_act_modality = new_act / self.config.MODEL.FRAMEDIV_NUM
        log_pi = torch.log(total_prob[range(total_prob.shape[0]), new_act] + 1e-6)

        print("new_act_frame: {}".format(new_act_frame))
        print("new_act_modality: {}".format(new_act_modality))

        return new_act_frame, new_act_modality, log_pi

    def my_load_state_dict(self, state_dict_old, strict=True):
        state_dict = OrderedDict()
        # delete 'module.' because it is saved from DataParallel module
        for key in state_dict_old.keys():
            #if 'time_conv' in key or 'clf_head' in key:
            state_dict[key.replace('module.', '')] = state_dict_old[key]

        self.load_state_dict(state_dict, strict=strict)

    def my_data_parallel(self, gpus):
        self.time_conv = torch.nn.DataParallel(self.time_conv, device_ids=gpus)
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

    return model
