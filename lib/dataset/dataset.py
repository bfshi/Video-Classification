from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import csv
import cv2
import os
import glob

import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms


class VideoSet(Dataset):
    def __init__(self, config, if_train=True, transform=None):
        self.config = config
        self.if_train = if_train
        self.rgb_transform = transform
        self.flow_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5],
                                 std=[0.225, 0.225])
        ])
        self.mode = 'training' if if_train else 'validation'
        # video_info = [{'name': '...', 'metadata': {...}, 'label' = 'label_name'}, ...]
        self.video_info = []
        # label_dic = {'label1': 0, 'label2': 1, ...}
        self.label_dic = {}
        # label_list = ['label1', 'label2', ...]
        self.label_list = []
        # name_list = ['name1', 'name2', ...]
        self.name_list = []
        # *_feature_list = [tensor1, tensor2, ...]
        self.i3d_rgb_feature_list = []
        self.i3d_flow_feature_list = []
        self.resnet_rgb_feature_list = []

    def load_dataset(self):
        # implemented in child class
        return None

    def __len__(self):
        return self.video_info.__len__()

    def __getitem__(self, idx):
        """
        implemented in child class
        """
        return None


class ActivityNet_I3D(VideoSet):
    def __init__(self, config, if_train=True, transform=None):
        super().__init__(config, if_train=if_train, transform=transform)
        self.load_dataset()

    def load_dataset(self):
        """
        load video_info and label_dic
        """
        # load label_idx
        with open(self.config.TRAIN.DATAROOT + self.config.TRAIN.DATASET
                  + '/label_index.csv', 'r') as labelfile:
            reader = csv.DictReader(labelfile)
            for line in reader:
                self.label_dic[line['label']] = int(line['label_idx'])
                self.label_list.append(line['label'])

        # load video info from .json
        with open(self.config.TRAIN.DATAROOT + self.config.TRAIN.DATASET +
                  '/activity_net.v1-3.min.json', 'r') as annfile:
            anndata = json.load(annfile)

        video_info_dict = {}

        for video_name, v in anndata['database'].items():
            # only training or validation
            if v['subset'] != self.mode:
                continue

            video_info_dict[video_name] = {
                'name': video_name,
                'metadata': {
                    'framenum': 0,  # decided later
                    'duration': v['duration'],
                    'segment': v['annotations'][0]['segment']
                },
                'label': v['annotations'][0]['label']
            }

        # load framenum

        # using numFrame in video_numframe_correction.csv which is precise.

        cnt = 0
        with open(self.config.TRAIN.DATAROOT + self.config.TRAIN.DATASET
                  + '/video_info_new.csv', 'r') as infofile:
            with open(self.config.TRAIN.DATAROOT + self.config.TRAIN.DATASET
                      + '/video_numframe_correction.csv', 'r') as correction_file:
                lines = csv.DictReader(infofile)
                correction_lines = csv.DictReader(correction_file)
                for line, correction_line in zip(lines, correction_lines):

                    if line['subset'] != self.mode:
                        continue
                    line['video'] = line['video'][2:]

                    i3d_rgb_list = glob.glob(os.path.join(self.config.TRAIN.DATAROOT,
                                                      self.config.TRAIN.DATASET,
                                                      'tsn_i3d_rgb_skip_8',
                                                      line['video'] + '*.npy'))
                    i3d_flow_list = glob.glob(os.path.join(self.config.TRAIN.DATAROOT,
                                                       self.config.TRAIN.DATASET,
                                                       'tsn_i3d_flow_skip_8',
                                                       line['video'] + '.npy'))
                    resnet_rgb_list = glob.glob(os.path.join(self.config.TRAIN.DATAROOT,
                                                      self.config.TRAIN.DATASET,
                                                      'resnet101_i3d_rgb_skip_8',
                                                      line['video'] + '*.npy'))
                    # can't find
                    if (i3d_rgb_list.__len__() == 0
                            or i3d_flow_list.__len__() == 0
                            or resnet_rgb_list.__len__() == 0):
                        continue
                    i3d_rgb_feature = np.load(i3d_rgb_list[0])
                    i3d_flow_feature = np.load(i3d_flow_list[0])
                    resnet_rgb_feature = np.load(resnet_rgb_list[0])

                    # incomplete videos
                    if not i3d_rgb_feature.any() or not i3d_flow_feature.any() or not resnet_rgb_feature.any() \
                            or i3d_rgb_feature.shape[0] == 0 or i3d_flow_feature.shape[0] == 0 or resnet_rgb_feature.shape[0] == 0 \
                            or i3d_rgb_feature.shape[0] != i3d_flow_feature.shape[0] \
                            or i3d_rgb_feature.shape[0] != resnet_rgb_feature.shape[0]:
                        continue
                    # if not rgb_feature.any() \
                    #             or rgb_feature.shape[0] == 0:
                    #     continue
                    else:
                        cnt += 1
                        print(cnt)
                        # if cnt > 200:
                        #     break


                        self.i3d_rgb_feature_list.append(torch.FloatTensor(i3d_rgb_feature))
                        self.i3d_flow_feature_list.append(torch.FloatTensor(i3d_flow_feature))
                        self.resnet_rgb_feature_list.append(torch.FloatTensor(resnet_rgb_feature))
                        video_info_dict[line['video']]['metadata']['framenum'] = i3d_rgb_feature.shape[0]
                        self.video_info.append(video_info_dict[line['video']])
                        self.name_list.append(line['video'])



    def __getitem__(self, idx):
        """
        fetch item using idx
        :param idx: index of item
        :return: input, label_idx, meta = {'framenum': xxx, 'label': 'label_name'}
        """
        item_info = self.video_info[idx]

        label_idx = self.label_dic[item_info['label']]

        meta = item_info['metadata']
        meta['label'] = item_info['label']
        meta['name'] = item_info['name']

        sample = torch.FloatTensor(range(self.config.MODEL.FRAMEDIV_NUM)) / self.config.MODEL.FRAMEDIV_NUM + \
                 torch.rand(self.config.MODEL.FRAMEDIV_NUM) / (self.config.MODEL.FRAMEDIV_NUM + 1)
        sample = (sample * meta['framenum']).type(torch.long)

        # return torch.stack([self.i3d_rgb_feature_list[idx][sample, :],
        #                     self.i3d_flow_feature_list[idx][sample, :],
        #                     ]), \
        #        label_idx, meta

        return torch.stack([torch.cat((self.i3d_rgb_feature_list[idx][sample, :],
                                       self.i3d_rgb_feature_list[idx][sample, :]), dim=1),
                            torch.cat((self.i3d_flow_feature_list[idx][sample, :],
                                       self.i3d_flow_feature_list[idx][sample, :]), dim=1),
                            self.resnet_rgb_feature_list[idx][sample, :],
                            ]), \
               label_idx, meta


def get_dataset(config, if_train = True, transform = None):
    """
    create a train set or valid set

    :param if_train: train set or valid set
    :return: train set if if_train = True, otherwise valid set
    """
    dataset = eval(config.TRAIN.DATASET)(config, if_train=if_train, transform=transform)

    return dataset
