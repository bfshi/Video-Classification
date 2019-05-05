from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch
import csv
import glob
import os

import _init_paths
from core.config import config


cnt = 0
with open(config.TRAIN.DATAROOT + config.TRAIN.DATASET + '/video_info_new.csv', 'r') as infofile:
    with open(config.TRAIN.DATAROOT + config.TRAIN.DATASET + '/video_numframe_correction.csv', 'w') as newfile:
        lines = csv.DictReader(infofile)
        writer = csv.writer(newfile)
        writer.writerow(['video', 'numFrame'])
        for line in lines:
            cnt += 1
            print(cnt)
            line['video'] = line['video'][2:]
            framenum = glob.glob(os.path.join(config.TRAIN.DATAROOT, config.TRAIN.DATASET,
                                       'frame_flow', line['video'], 'img_*.jpg')).__len__()
            writer.writerow([line['video'], str(framenum)])

