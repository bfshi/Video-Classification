from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from collections import OrderedDict

import _init_paths
from models.time_conv import get_time_conv
from models.time_conv import get_time_conv_sel
from utils.utils import soft_update_from_to
from utils.utils import torch_clip

class VedioClfNet(nn.Module):
    def __init__(self, config, is_train = True):
        super(VedioClfNet, self).__init__()

        self.config = config

        # self.time_conv = torch.nn.ModuleList([get_time_conv() for i in range(config.MODEL.MODALITY_NUM)])
        self.time_conv = get_time_conv()

        # output classification scores (not softmaxed)
        self.clf_head = nn.Linear(config.MODEL.FEATURE_DIM, config.MODEL.CLFDIM)

    def forward(self, x, if_fusion = False):
        """
        :param x: N * C * T * W
        :param if_fusion: if to merge clf_scores of different modalities
        """
        # clf_score_list = []
        # for i in range(self.config.MODEL.MODALITY_NUM):
        #     if i == 0:
        #         y = self.time_conv[i](x[:, i: i + 1])
        #         clf_score_list.append(self.clf_head(y))
        #     else:
        #         temp = self.time_conv[i](x[:, i: i + 1])
        #         y = torch.cat((y, temp), dim=1)
        #         clf_score_list.append(self.clf_head(temp))
        #
        # if if_fusion:
        #     clf_score = clf_score_list[0]
        #     for i in range(1, self.config.MODEL.MODALITY_NUM):
        #         clf_score += clf_score_list[i]
        #     return clf_score, y
        # else:
        #     return clf_score_list, y
        y = self.time_conv(x)
        clf_score = self.clf_head(y)
        return clf_score, y


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


class SelectiveNet(nn.Module):
    def __init__(self, config):
        super(SelectiveNet, self).__init__()

        self.config = config

        # self.time_conv = torch.nn.ModuleList([get_time_conv() for i in range(config.MODEL.MODALITY_NUM)])
        self.time_conv_sel = get_time_conv_sel()

        # output classification scores (not softmaxed)
        self.sel_head = nn.Linear(config.MODEL.FEATURE_DIM * config.MODEL.FRAMEDIV_NUM,
                                  config.MODEL.MODALITY_NUM * config.MODEL.FRAMEDIV_NUM)

    def forward(self, x, if_fusion = False):
        """
        :param x: N * C * T * W
        :param if_fusion: if to merge clf_scores of different modalities
        """
        y = self.time_conv_sel(x)
        sel_score = self.sel_head(y)
        return sel_score


    def my_load_state_dict(self, state_dict_old, strict=True):
        state_dict = OrderedDict()
        # delete 'module.' because it is saved from DataParallel module
        for key in state_dict_old.keys():
            #if 'time_conv' in key or 'clf_head' in key:
            state_dict[key.replace('module.', '')] = state_dict_old[key]

        self.load_state_dict(state_dict, strict=strict)

    def my_data_parallel(self, gpus):
        self.time_conv_sel = torch.nn.DataParallel(self.time_conv_sel, device_ids=gpus)
        self.sel_head = torch.nn.DataParallel(self.sel_head, device_ids=gpus)


class KNet(nn.Module):
    def __init__(self, config):
        super(KNet, self).__init__()

        self.config = config

        # self.time_conv = torch.nn.ModuleList([get_time_conv() for i in range(config.MODEL.MODALITY_NUM)])
        self.time_conv = get_time_conv()

        # output classification scores (not softmaxed)
        self.k_head = nn.Linear(config.MODEL.FEATURE_DIM, config.MODEL.FRAMEDIV_NUM)

    def forward(self, x, if_fusion = False):
        """
        :param x: N * C * T * W
        :param if_fusion: if to merge clf_scores of different modalities
        """
        y = self.time_conv(x)
        k_score = self.k_head(y)
        return k_score, y


    def my_load_state_dict(self, state_dict_old, strict=True):
        state_dict = OrderedDict()
        # delete 'module.' because it is saved from DataParallel module
        for key in state_dict_old.keys():
            #if 'time_conv' in key or 'clf_head' in key:
            state_dict[key.replace('module.', '')] = state_dict_old[key]

        self.load_state_dict(state_dict, strict=strict)

    def my_data_parallel(self, gpus):
        self.time_conv = torch.nn.DataParallel(self.time_conv, device_ids=gpus)
        self.k_head = torch.nn.DataParallel(self.k_head, device_ids=gpus)


def create_model(config, is_train = True):
    """
    build a complete model.

    :param config: global configs
    :param is_train: train mode
    :return: a model
    """

    model = VedioClfNet(config, is_train)

    return model

def create_sel_model(config):

    model = SelectiveNet(config)

    return model

def create_k_model(config):

    model = KNet(config)

    return model
