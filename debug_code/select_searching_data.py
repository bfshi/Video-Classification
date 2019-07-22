from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch
import csv
import glob
import os
import cv2
import json
import pprint
import time
import h5py
import copy
from PIL import Image

import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
import torch.nn.functional as F

import _init_paths
from models.model import create_model
from core.config import config
from core.config import extra
from core.loss import Loss
from dataset.dataset import get_dataset
from utils.utils import compute_acc
from utils.utils import update_input
from utils.utils import MyEncoder
from utils.utils import AverageMeter

if_selected_as_searching = (np.random.rand(4728) < 1000/4728).astype(np.int)
header = ['video', 'numFrame', 'seconds', 'fps', 'rfps', 'subset', 'featureFrame']

with open(config.TRAIN.DATAROOT + config.TRAIN.DATASET
                  + '/hhh.csv', 'r') as infofile:
    with open(config.TRAIN.DATAROOT + config.TRAIN.DATASET
              + '/hhhhh.csv', 'w') as newfile:
        dict_writer = csv.DictWriter(newfile, header)
        dict_writer.writeheader()

        lines = csv.DictReader(infofile)
        cnt = 0
        for line in lines:
            dic = dict(line)
            if line['subset'] == 'training' or line['subset'] == 'testing' or cnt >= 4728:
                dict_writer.writerow(dic)
                continue
            if if_selected_as_searching[cnt] == 1:
                dic['subset'] = 'searching'
            dict_writer.writerow(dic)
            cnt += 1
            print(cnt)





