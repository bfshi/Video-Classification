from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import csv

import numpy as np
from easydict import EasyDict as edict


config = edict()

# common configs

config.GPUS = '4, 5, 6, 7'
config.GPU_NUM = 4
config.WORKERS = 4
# config.DATASET = 'ActivityNet'
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
config.ActivityNet_I3D.BLOCKED_VIDEO = []  # incomplete frames(loaded in L132)
config.ActivityNet_I3D.RGB_COST = 1
config.ActivityNet_I3D.FLOW_COST = 1
config.ActivityNet_I3D.COST_LIMIT = 15
config.ActivityNet_I3D.RGB_MEAN = 0.118
config.ActivityNet_I3D.FLOW_MEAN = 0.144

# models related configs

config.MODEL = edict()

config.MODEL.DISCOUNT = 0.99
config.MODEL.ENTROPY_RATIO = 1

config.MODEL.BACKBONE_INDIM_H = 224
config.MODEL.BACKBONE_INDIM_W = 224
config.MODEL.LSTM_INDIM = 2048
config.MODEL.LSTM_OUTDIM = 2048

config.MODEL.FEATURE_DIM = 1024  # for pre-extracted feature
config.MODEL.COST_DIM = 1
config.MODEL.CLFDIM = 200
config.MODEL.FRAMEDIV_NUM = 128  # output dimension of act_head_frame
config.MODEL.MODALITY_NUM = 2

config.MODEL.RESNET_TYPE = 50
config.MODEL.INIT_WEIGHTS = True
config.MODEL.PRETRAINED_PATH = 'pretrained_models/resnet50-19c8e357.pth'
# config.MODEL.PRETRAINED_PATH = 'pretrained_models/resnet101-5d3b4d8f.pth'

config.MODEL.COST_LIST = []
config.MODEL.COST_LIMIT = 0


# training related configs

config.TRAIN = edict()

config.TRAIN.RESUME = True  # whether to continue previous training
config.TRAIN.STATE_DICT = 'train_clf/random_5_model_clf_2019-05-01-11-46_0.348.pth'
# config.TRAIN.STATE_DICT = 'train_clf/checkpoint.pth'

config.TRAIN.SINGLE_GPU = False
config.TRAIN.GPU = '1'  # which to use when SINGLE_GPU == True

config.TRAIN.IF_TRAIN_BACKBONE = False
config.TRAIN.IF_LSTM = False
config.TRAIN.IF_BACKBONE = True

config.TRAIN.DATAROOT = '/m/shibf/video_classification/data/'
config.TRAIN.DATASET = 'ActivityNet_I3D'

config.TRAIN.LR = 0.0005
config.TRAIN.LR_DECAY_RATE = 0.5
config.TRAIN.LR_MILESTONES = [8, 15, 30]  # at which epoch lr decays
config.TRAIN.SOFT_UPDATE = 0.001  # 0.005 in SAC paper / 0.01 in rlkit

config.TRAIN.OPTIMIZER = 'sgd'
config.TRAIN.MOMENTUM = 0.9
config.TRAIN.WD = 0.0001
config.TRAIN.NESTEROV = False
config.TRAIN.R_PLC = 1
config.TRAIN.R_Q = 3
config.TRAIN.R_V = 3

config.TRAIN.TRAIN_CLF_STEP = 5  # num of steps to train classification head on a single vedio
config.TRAIN.TRAIN_RL_STEP = 1  # num of steps to train policy after every clf_train
config.TRAIN.ROLLOUT_STEP = 1  # num of rollout steps after an action

config.TRAIN.BEGIN_EPOCH = 0
config.TRAIN.END_EPOCH = 120

config.TRAIN.BATCH_SIZE = 16  # paralleled batch size per gpu
config.TRAIN.RL_BATCH_SIZE = 64
config.TRAIN.SHUFFLE = True

config.TRAIN.MAX_BUFFER_SIZE = 1000000  # replay buffer size

config.TRAIN.PRINT_EVERY = 1

# classification-only training

config.TRAIN_CLF = edict()

config.TRAIN_CLF.RANDOM_SAMPLE_NUM = True
config.TRAIN_CLF.SAMPLE_NUM = 15
config.TRAIN_CLF.IF_TRIM = True

config.TRAIN_CLF.IF_BACKBONE = True
config.TRAIN_CLF.IF_LSTM = False

config.TRAIN_CLF.SINGLE_GPU = False
config.TRAIN_CLF.GPU = '1'  # which to use when SINGLE_GPU == True

# testing related configs

config.TEST = edict()

config.TEST.RESUME = True
config.TEST.STATE_DICT = 'train/checkpoint_0.5399503722084368.pth'

config.TEST.IF_TRIM = True

config.TEST.TEST_EVERY = 1

config.TEST.TEST_STEP = 5

config.TEST.BATCH_SIZE = 32

config.TEST.PRINT_EVERY = 1

# extra

def extra():
    config.OUTPUT_DIR = os.path.join('experiments/', config.TRAIN.DATASET,
                                     'resnet{}'.format(config.MODEL.RESNET_TYPE), config.MODE)

    config.TRAIN.STATE_DICT = os.path.join('experiments/', config.TRAIN.DATASET,
                                     'resnet{}'.format(config.MODEL.RESNET_TYPE), config.TRAIN.STATE_DICT)
    config.TEST.STATE_DICT = os.path.join('experiments/', config.TRAIN.DATASET,
                                           'resnet{}'.format(config.MODEL.RESNET_TYPE), config.TEST.STATE_DICT)
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