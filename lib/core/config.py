from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import csv

import numpy as np
from easydict import EasyDict as edict


config = edict()

# common configs

config.GPUS = '0, 1, 3'
config.GPU_NUM = 3
config.WORKERS = 0
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

# models related configs

config.MODEL = edict()

config.MODEL.DISCOUNT = 0.99
config.MODEL.ENTROPY_RATIO = 1

config.MODEL.BACKBONE_INDIM_H = 224
config.MODEL.BACKBONE_INDIM_W = 224
config.MODEL.LSTM_INDIM = 2048
config.MODEL.LSTM_OUTDIM = 2048
config.MODEL.CLFDIM = 200
config.MODEL.MODALITY_NUM = 2

config.MODEL.RESNET_TYPE = 50
config.MODEL.INIT_WEIGHTS = True
config.MODEL.PRETRAINED_PATH = 'pretrained_models/resnet50-19c8e357.pth'
# config.MODEL.PRETRAINED_PATH = 'pretrained_models/resnet101-5d3b4d8f.pth'


# training related configs

config.TRAIN = edict()

config.TRAIN.RESUME = True  # whether to continue previous training
config.TRAIN.STATE_DICT = 'train_clf/model_2019-04-21-23-59_0.606.pth'
# config.TRAIN.STATE_DICT = 'train_clf/checkpoint.pth'

config.TRAIN.SINGLE_GPU = False
config.TRAIN.GPU = '1'  # which to use when SINGLE_GPU == True

config.TRAIN.IF_TRAIN_BACKBONE = False

config.TRAIN.DATAROOT = '/m/shibf/video_classification/data/'
config.TRAIN.DATASET = 'ActivityNet'

config.TRAIN.LR = 0.001
config.TRAIN.LR_DECAY_RATE = 0.5
config.TRAIN.LR_MILESTONES = [30, 60, 90]  # at which epoch lr decays
config.TRAIN.SOFT_UPDATE = 0.005  # 0.005 in SAC paper / 0.01 in rlkit

config.TRAIN.OPTIMIZER = 'sgd'
config.TRAIN.MOMENTUM = 0.9
config.TRAIN.WD = 0.0001
config.TRAIN.NESTEROV = False
config.TRAIN.R_PLC = 1
config.TRAIN.R_Q = 1
config.TRAIN.R_V = 1

config.TRAIN.TRAIN_CLF_STEP = 5  # num of steps to train classification head on a single vedio
config.TRAIN.TRAIN_RL_STEP = 10  # num of steps to train policy after every clf_train
config.TRAIN.ROLLOUT_STEP = 1  # num of rollout steps after an action

config.TRAIN.BEGIN_EPOCH = 0
config.TRAIN.END_EPOCH = 120

config.TRAIN.BATCH_SIZE = 16  # paralleled batch size per gpu
config.TRAIN.RL_BATCH_SIZE = 32
config.TRAIN.SHUFFLE = True

config.TRAIN.MAX_BUFFER_SIZE = 1000000  # replay buffer size

config.TRAIN.PRINT_EVERY = 1

# classification-only training

config.TRAIN_CLF = edict()

config.TRAIN_CLF.SAMPLE_NUM = 5
config.TRAIN_CLF.IF_TRIM = True

config.TRAIN_CLF.IF_LSTM = False

config.TRAIN_CLF.SINGLE_GPU = False
config.TRAIN_CLF.GPU = '1'  # which to use when SINGLE_GPU == True

# testing related configs

config.TEST = edict()

config.TEST.IF_TRIM = True

config.TEST.TEST_EVERY = 5

config.TEST.TEST_STEP = 5

config.TEST.BATCH_SIZE = 32

config.TEST.PRINT_EVERY = 1

# extra

def extra():
    config.OUTPUT_DIR = os.path.join('experiments/', config.TRAIN.DATASET,
                                     'resnet{}'.format(config.MODEL.RESNET_TYPE), config.MODE)

    config.TRAIN.STATE_DICT = os.path.join('experiments/', config.TRAIN.DATASET,
                                     'resnet{}'.format(config.MODEL.RESNET_TYPE), config.TRAIN.STATE_DICT)


    dataset = config.TRAIN.DATASET
    with open(os.path.join(config.TRAIN.DATAROOT, config.TRAIN.DATASET,
              config[dataset].INCOMPLETE_VIDEO), 'r') as correction_file:
        reader = csv.reader(correction_file)
        config[dataset].BLOCKED_VIDEO.extend(list(reader)[0])

    with open(os.path.join(config.TRAIN.DATAROOT, config.TRAIN.DATASET,
              config[dataset].BLANK_VIDEO), 'r') as correction_file:
        reader = csv.reader(correction_file)
        config[dataset].BLOCKED_VIDEO.extend(list(reader)[0])