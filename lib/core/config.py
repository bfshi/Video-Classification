from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import yaml

import numpy as np
from easydict import EasyDict as edict


config = edict()

# common configs

config.GPUS = '0'
config.WORKERS = 0
config.DATASET = 'FCVID'
config.OUTPUT_DIR = 'experiments'

# models related configs

config.MODEL = edict()

config.MODEL.DISCOUNT = 0.99
config.MODEL.ENTROPY_RATIO = 1

config.MODEL.LSTM_INDIM = 2048
config.MODEL.LSTM_OUTDIM = 2048
config.MODEL.CLFDIM = 100
config.MODEL.MODALITY_NUM = 3

config.MODEL.RESNET_TYPE = 18
config.MODEL.INIT_WEIGHTS = True
config.MODEL.PRETRAINED_PATH = 'pretrained_models/?'

# training related configs

config.TRAIN = edict()

config.TRAIN.LR = 0.001
config.TRAIN.LR_DECAY_RATE = 1
config.TRAIN.LR_MILESTONES = []  # at which epoch lr decays
config.TRAIN.SOFT_UPDATE = 0.005  # 0.005 in SAC paper / 0.01 in rlkit

config.TRAIN.OPTIMIZER = 'adam'
config.TRAIN.MOMENTUM = 0.9
config.TRAIN.WD = 0.0001
config.TRAIN.NESTEROV = False
config.TRAIN.R_PLC = 1
config.TRAIN.R_Q = 1
config.TRAIN.R_V = 1

config.TRAIN.TRAIN_CLF_STEP = 10  # num of steps to train classification head on a single vedio
config.TRAIN.TRAIN_RL_STEP = 10  # num of steps to train policy after every clf_train
config.TRAIN.ROLLOUT_STEP = 1  # num of rollout steps after an action

config.TRAIN.BEGIN_EPOCH = 0
config.TRAIN.END_EPOCH = 100

config.TRAIN.BATCH_SIZE = 32
config.TRAIN.RL_BATCH_SIZE = 32
config.TRAIN.SHUFFLE = True

config.TRAIN.MAX_BUFFER_SIZE = 1000000  # replay buffer size

config.TRAIN.PRINT_EVERY = 100

# testing related ocnfigs

config.TEST = edict()

config.TEST.TEST_STEP = 10

config.TEST.BATCH_SIZE = 32

config.TEST.PRINT_EVERY = 100




