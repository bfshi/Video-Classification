from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import csv

import numpy as np
from easydict import EasyDict as edict

# global configuration

config = edict()

# common configs

config.GPUS = '4, 5, 6, 7'
config.GPU_NUM = 4  # number of gpus in config.GPUS
config.WORKERS = 4
config.MODE = 'train_clf'  # train / train_clf

# Cudnn related params
config.CUDNN = edict()
config.CUDNN.BENCHMARK = True
config.CUDNN.DETERMINISTIC = False
config.CUDNN.ENABLED = True

# dataset related configs

config.ActivityNet = edict()
config.ActivityNet.FLOW_H = 256
config.ActivityNet.FLOW_W = 340
config.ActivityNet.INCOMPLETE_VIDEO = 'incomplete_video.csv'
config.ActivityNet.BLANK_VIDEO = 'blank_video.csv'
config.ActivityNet.BLOCKED_VIDEO = []  # incomplete frames(loaded in L132)

config.ActivityNet_I3D = edict()
config.ActivityNet_I3D.FLOW_H = 256
config.ActivityNet_I3D.FLOW_W = 340
config.ActivityNet_I3D.INCOMPLETE_VIDEO = 'incomplete_video.csv'
config.ActivityNet_I3D.BLANK_VIDEO = 'blank_video.csv'
config.ActivityNet_I3D.BLOCKED_VIDEO = []  # incomplete frames(loaded in line 132)
config.ActivityNet_I3D.RGB_COST = 1
config.ActivityNet_I3D.FLOW_COST = 1
config.ActivityNet_I3D.COST_LIMIT = 10

# models related configs

config.MODEL = edict()

config.MODEL.DISCOUNT = 0.99
config.MODEL.ENTROPY_RATIO = 1

config.MODEL.RESNET_TYPE = 50

config.MODEL.FEATURE_DIM = 2048  # for pre-extracted feature
config.MODEL.COST_DIM = 1
config.MODEL.CLFDIM = 200
config.MODEL.FRAMEDIV_NUM = 128  # output dimension of act_head_frame
config.MODEL.MODALITY_NUM = 1

config.MODEL.COST_LIST = []

# training related configs

config.TRAIN = edict()

config.TRAIN.RESUME = False  # whether to continue previous training
config.TRAIN.STATE_DICT = 'train_clf/random_128_rgb_resnet101_model_clf_2019-07-01-19-37_0.710_mAP@R5_0.663_mAP@U5_0.674_mAP@128_0.769.pth'

config.TRAIN.SINGLE_GPU = False
config.TRAIN.GPU = '1'  # which to use when SINGLE_GPU == True

config.TRAIN.IF_TRAIN_BACKBONE = False
config.TRAIN.IF_LSTM = False
config.TRAIN.IF_BACKBONE = True

config.TRAIN.DATAROOT = '/m/shibf/video_classification/data/'
config.TRAIN.DATASET = 'ActivityNet_I3D'

config.TRAIN.LR = 0.001
config.TRAIN.LR_DECAY_RATE = 0.5
config.TRAIN.LR_MILESTONES = [30, 60, 90]  # at which epoch lr decays
config.TRAIN.SOFT_UPDATE = 0.003 #0.001  # 0.005 in SAC paper / 0.01 in rlkit

config.TRAIN.OPTIMIZER = 'sgd'  # sgd / adam
config.TRAIN.MOMENTUM = 0.9
config.TRAIN.WD = 0.0001
config.TRAIN.NESTEROV = False
config.TRAIN.R_PLC = 1  # coefficient of policy loss
config.TRAIN.R_Q = 3    # coefficient of q_value loss
config.TRAIN.R_V = 1    # coefficient of v_value loss

config.TRAIN.LAMBDA = 0  # Lagrange multiplier

config.TRAIN.TRAIN_RL_STEP = 3  # num of steps to train policy after each experience collecting

config.TRAIN.BEGIN_EPOCH = 0
config.TRAIN.END_EPOCH = 120

config.TRAIN.BATCH_SIZE = 8  # paralleled batch size per gpu
config.TRAIN.RL_BATCH_SIZE = 64
config.TRAIN.SHUFFLE = True

config.TRAIN.MAX_BUFFER_SIZE = 1000000  # replay buffer size

config.TRAIN.PRINT_EVERY = 1


# classification-only training

config.TRAIN_CLF = edict()

config.TRAIN_CLF.RANDOM_SAMPLE_NUM = True  # is number of sampled frames a random number?
config.TRAIN_CLF.SAMPLE_NUM_BOUND = 128    # upper bound of number of sampled frames
config.TRAIN_CLF.SAMPLE_NUM = 5          # if number of sampled frames is fixed

config.TRAIN_CLF.SINGLE_GPU = False
config.TRAIN_CLF.GPU = '1'  # which to use when SINGLE_GPU == True

# cyclic learning rate
config.TRAIN_CLF.CYCLE_LR = edict()
config.TRAIN_CLF.CYCLE_LR.IF_CYCLE_LR = False
config.TRAIN_CLF.CYCLE_LR.STEPSIZE = 15  # half a period
config.TRAIN_CLF.CYCLE_LR.MIN_LR = 1e-3
config.TRAIN_CLF.CYCLE_LR.MAX_LR = 1e-2


# selective training

config.TRAIN_SEL = edict()

config.TRAIN_SEL.PRECHOOSING_NUM = 5
config.TRAIN_SEL.MAX_CHOOSING_NUM = 15
config.TRAIN_SEL.ACTION_SAMPLE_NUM = 10


# testing related configs

config.TEST = edict()

config.TEST.RESUME = True
config.TEST.STATE_DICT = 'train_clf/random_128_rgb_resnet101_model_clf_2019-07-01-19-37_0.710_mAP@R5_0.663_mAP@U5_0.674_mAP@128_0.769.pth'

config.TEST.BATCH_SIZE = 32

config.TEST.TEST_EVERY = 1  # frequency of testing during training

config.TEST.PRINT_EVERY = 1

# selective testing related configs

config.TEST_SEL = edict()

config.TEST_SEL.RESUME = True
config.TEST_SEL.STATE_DICT = 'train_sel/sampling_based_rgb_resnet101_model_sel_2019-07-19-16-54_0.667.pth'

# path to each part of network

config.STATE_DICT = edict()

config.STATE_DICT.F_C = 'train_clf/random_128_rgb_resnet101_model_clf_2019-07-01-19-37_0.710_mAP@R5_0.663_mAP@U5_0.674_mAP@128_0.769.pth'
config.STATE_DICT.F_S = 'train_sel/sampling_based_rgb_resnet101_model_sel_2019-07-19-16-54_0.667.pth'
config.STATE_DICT.F_K = ''


# extra settings

def extra():
    config.OUTPUT_DIR = os.path.join('experiments/', config.TRAIN.DATASET,
                                     'resnet{}'.format(config.MODEL.RESNET_TYPE), config.MODE)

    config.TRAIN.STATE_DICT = os.path.join('experiments/', config.TRAIN.DATASET,
                                     'resnet{}'.format(config.MODEL.RESNET_TYPE), config.TRAIN.STATE_DICT)
    config.TEST.STATE_DICT = os.path.join('experiments/', config.TRAIN.DATASET,
                                           'resnet{}'.format(config.MODEL.RESNET_TYPE), config.TEST.STATE_DICT)
    config.TEST_SEL.STATE_DICT = os.path.join('experiments/', config.TRAIN.DATASET,
                                           'resnet{}'.format(config.MODEL.RESNET_TYPE), config.TEST_SEL.STATE_DICT)
    config.STATE_DICT.F_C = os.path.join('experiments/', config.TRAIN.DATASET,
                                           'resnet{}'.format(config.MODEL.RESNET_TYPE), config.STATE_DICT.F_C)
    config.STATE_DICT.F_S = os.path.join('experiments/', config.TRAIN.DATASET,
                                         'resnet{}'.format(config.MODEL.RESNET_TYPE), config.STATE_DICT.F_S)
    config.STATE_DICT.F_K = os.path.join('experiments/', config.TRAIN.DATASET,
                                         'resnet{}'.format(config.MODEL.RESNET_TYPE), config.STATE_DICT.F_K)
    if not config.TRAIN.SINGLE_GPU:
        config.TRAIN.RL_BATCH_SIZE = config.TRAIN.RL_BATCH_SIZE * config.GPU_NUM

    dataset = config.TRAIN.DATASET
    if (dataset == 'ActivityNet'):
        with open(os.path.join(config.TRAIN.DATAROOT, config.TRAIN.DATASET,
                  config[dataset].INCOMPLETE_VIDEO), 'r') as correction_file:
            reader = csv.reader(correction_file)
            config[dataset].BLOCKED_VIDEO.extend(list(reader)[0])

        with open(os.path.join(config.TRAIN.DATAROOT, config.TRAIN.DATASET,
                  config[dataset].BLANK_VIDEO), 'r') as correction_file:
            reader = csv.reader(correction_file)
            config[dataset].BLOCKED_VIDEO.extend(list(reader)[0])

    if (dataset == 'ActivityNet_I3D'):
        config.TRAIN.IF_BACKBONE = False
        config.TRAIN_CLF.IF_BACKBONE = False
        config.MODEL.INIT_WEIGHTS = False
        config.MODEL.LSTM_INDIM = 1024
        config.MODEL.LSTM_OUTDIM = 1024
        config.MODEL.COST_LIST.append(config.ActivityNet_I3D.RGB_COST)
        config.MODEL.COST_LIST.append(config.ActivityNet_I3D.FLOW_COST)
        config.MODEL.COST_LIMIT = config.ActivityNet_I3D.COST_LIMIT

    if (config.MODE == 'train'):
        config.TEST.TEST_EVERY = 1