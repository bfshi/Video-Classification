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
from PIL import Image

import torchvision.transforms as transforms

import _init_paths
from core.config import config
from models.model import create_model
from models.backbone import getBackbone


normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

transform = transforms.Compose([
        transforms.Resize((config.MODEL.BACKBONE_INDIM_H, config.MODEL.BACKBONE_INDIM_W)),
        transforms.ToTensor(),
        #normalize,
    ])

cnt = 0
with open(config.TRAIN.DATAROOT + config.TRAIN.DATASET
          + '/video_info_new.csv', 'r') as infofile:
    with open(config.TRAIN.DATAROOT + config.TRAIN.DATASET
              + '/video_numframe_correction.csv', 'r') as correction_file:
        with h5py.File(config.TRAIN.DATAROOT + config.TRAIN.DATASET
              + '/img.hdf5', 'a') as img_h5:
            lines = csv.DictReader(infofile)
            correction_lines = csv.DictReader(correction_file)
            for line, correction_line in zip(lines, correction_lines):
                if line['video'][2:] in config.ActivityNet.BLOCKED_VIDEO:
                    continue
                line['video'] = line['video'][2:]
                numframe = int(correction_line['numFrame'])
                dset = img_h5.create_dataset(line['video'], (numframe, config.ActivityNet.FLOW_H,
                                                             config.ActivityNet.FLOW_W, 3))
                for i in range(numframe):
                    # img = Image.open(
                    #             os.path.join('/m/shibf/video_classification/data/ActivityNet/frame_flow/',
                    #                          line['video'], 'img_%.5d.jpg' % (i + 1)))
                    img = cv2.imread(os.path.join('/m/shibf/video_classification/data/ActivityNet/frame_flow/',
                                              line['video'], 'img_%.5d.jpg' % (i + 1)), cv2.IMREAD_UNCHANGED | cv2.IMREAD_COLOR)
                    dset[i] = img
                cnt += 1
                print(cnt)

# for i in range(32):
#     end = time.time()
#     img = Image.open(
#         os.path.join('/m/shibf/video_classification/data/ActivityNet/frame_flow/NjlskpV3WuI', 'img_00000.jpg'))
#     print(time.time() - end)
#
#     img = transform(img)
#     print(time.time() - end)