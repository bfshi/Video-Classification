from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import yaml

import numpy as np
from easydict import EasyDict as edict


config = edict()

#common configs

config.GPUS = '0'
config.WORKERS = 0

#models related configs

config.MODEL = edict()

config.MODEL.LSTM_OUTDIM =
config.MODEL.CLFDIM =



#training related configs

config.TRAIN = edict()

config.TRAIN.LR = 0.001
config.TRAIN.LR_DECAY_RATE = 1
config.TRAIN.LR_MILESTONES = []  #at which epoch lr decays

config.TRAIN.OPTIMIZER = 'adam'
config.TRAIN.MOMENTUM = 0.9
config.TRAIN.WD = 0.0001
config.TRAIN.NESTEROV = False
config.TRAIN.R_ACT =
config.TRAIN.R_CRT =

config.TRAIN.BEGIN_EPOCH = 0
config.TRAIN.END_EPOCH = 100

config.TRAIN.BATCH_SIZE = 32
config.TRAIN.SHUFFLE = True

#testing related ocnfigs

config.TEST = edict()

config.TEST.BATCH_SIZE = 32




