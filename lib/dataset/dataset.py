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


class ActivityNet(VideoSet):
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
                    'framenum': 0 # decided later
                },
                'label': v['annotations'][0]['label']
            }

        # load framenum

        # using numFrame in video_numframe_correction.csv which is precise.

        with open(self.config.TRAIN.DATAROOT + self.config.TRAIN.DATASET
                  + '/video_info_new.csv', 'r') as infofile:
            with open(self.config.TRAIN.DATAROOT + self.config.TRAIN.DATASET
                      + '/video_numframe_correction.csv', 'r') as correction_file:
                lines = csv.DictReader(infofile)
                correction_lines = csv.DictReader(correction_file)
                for line, correction_line in zip(lines, correction_lines):
                    if line['subset'] != self.mode:
                        continue
                    if line['video'][2:] in self.config.ActivityNet.BLOCKED_VIDEO:
                        continue
                    line['video'] = line['video'][2:]
                    video_info_dict[line['video']]['metadata']['framenum'] = int(correction_line['numFrame'])
                    self.video_info.append(video_info_dict[line['video']])

        # count number of frames for each video
        # video_info is sorted in alphabetic order

        # with open(self.config.TRAIN.DATAROOT + self.config.TRAIN.DATASET + '/video_info_new.csv', 'r') as infofile:
        #     lines = csv.DictReader(infofile)
        #     for line in lines:
        #         if line['subset'] != self.mode:
        #             continue
        #         line['video'] = line['video'][2:]
        #         video_info_dict[line['video']]['metadata']['framenum'] = \
        #             glob.glob(os.path.join(self.config.TRAIN.DATAROOT, self.config.TRAIN.DATASET,
        #                                    'frame_flow', line['video'], 'img_*.jpg')).__len__()
        #         self.video_info.append(video_info_dict[line['video']])
        #         # load label_dict
        #         if video_info_dict[line['video']]['label'] not in self.label_dic:
        #             self.label_dic[video_info_dict[line['video']]['label']] = label_cnt
        #             label_cnt += 1

    def __getitem__(self, idx):
        """
        fetch item using idx
        :param idx: index of item
        :return: input, label_idx, meta = {'framenum': xxx, 'label': 'label_name'}
        """
        item_info = self.video_info[idx]
        frame_flow_path = os.path.join(self.config.TRAIN.DATAROOT, self.config.TRAIN.DATASET,
                                       'frame_flow', item_info['name'])

        # # read rgb features
        # rgb_feature_list = [np.transpose(cv2.imread(os.path.join(frame_flow_path, 'img_%.5d.jpg' % i),
        #                           cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION), (2, 0, 1))
        #                for i in range(1, item_info['metadata']['framenum'] + 1)]
        # rgb_feature = np.stack(rgb_feature_list)
        #
        # # read flow features
        # # x_channel + y_channel + zero_channel (for size compatibility)
        # flow_feature_list = [np.stack([cv2.imread(os.path.join(frame_flow_path, 'flow_x_%.5d.jpg' % i),
        #                           cv2.IMREAD_GRAYSCALE | cv2.IMREAD_IGNORE_ORIENTATION),
        #                               cv2.imread(os.path.join(frame_flow_path, 'flow_y_%.5d.jpg' % i),
        #                           cv2.IMREAD_GRAYSCALE | cv2.IMREAD_IGNORE_ORIENTATION),
        #                                np.zeros((rgb_feature.shape[2], rgb_feature.shape[3]))]
        #                      ) for i in range(1, item_info['metadata']['framenum'] + 1)]
        # flow_feature = np.stack(flow_feature_list)
        #
        # # transform
        # # TODO: TOTensor only accept input of size (H, W, C)!
        # if self.rgb_transform:
        #     rgb_feature = self.rgb_transform(rgb_feature)
        #     flow_feature = self.flow_transform(flow_feature)
        #
        # input = [rgb_feature, flow_feature]

        label_idx = self.label_dic[item_info['label']]

        meta = item_info['metadata']
        meta['label'] = item_info['label']

        return frame_flow_path, label_idx, meta


def get_dataset(config, if_train = True, transform = None):
    """
    create a train set or valid set

    :param if_train: train set or valid set
    :return: train set if if_train = True, otherwise valid set
    """
    dataset = eval(config.TRAIN.DATASET)(config, if_train=if_train, transform=transform)

    return dataset
