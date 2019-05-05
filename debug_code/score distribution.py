from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import pprint
import shutil
import json
import time

import matplotlib as mpl
mpl.use('Agg')

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms

import _init_paths
from core.config import config
from core.config import extra
from core.function import train_clf
from core.function import validate_clf
from core.loss import Loss
from dataset.dataset import get_dataset
from models.model import create_model
from utils.utils import create_optimizer
from utils.utils import create_logger
from utils.replay_buffer import create_replay_buffer
from utils.utils import MyEncoder
from utils.utils import compute_acc
from utils.utils import load_frame
from utils.utils import choose_frame_randomly
from utils.utils import choose_modality_randomly

sample_size = 100


def score_distribution(config, val_loader, model, criterion, epoch = 0, transform = None, transform_gray = None,
                 output_dict = None, valid_dataset = None):
    """
    validate backbone, lstm and clf_head only.
    unsorted sampling is used for each video.

    :param config: global configs
    :param val_loader: data loader
    :param model: model to be trained
    :param criterion: loss module
    :param epoch: current epoch
    :return: None
    """
    # whether to output the result
    if_output_result = (valid_dataset != None)
    label_head = ['label' for i in range(config.MODEL.CLFDIM)]
    score_head = ['score' for i in range(config.MODEL.CLFDIM)]

    # switch to val mode
    model.eval()

    with torch.no_grad():
        for i, (video_path, target, meta) in enumerate(val_loader):

            # initialize the observation
            # ob = (h, c)
            ob = (torch.zeros((target.shape[0], config.MODEL.LSTM_OUTDIM)).cuda(),
                  torch.zeros((target.shape[0], config.MODEL.LSTM_OUTDIM)).cuda())
            model.init_weights(target.shape[0], ob)

            total_batch_size = target.shape[0]

            # unsorted sampling.
            frame_chosen = choose_frame_randomly(total_batch_size, sample_size,
                                                 meta['segment'], meta['duration'], config.TEST.IF_TRIM)

            for j in range(config.MODEL.MODALITY_NUM):
                modality_chosen = torch.LongTensor(np.zeros((1, sample_size)) + j)

                target = target.cuda()

                input_whole = load_frame(video_path, modality_chosen, frame_chosen,
                                         meta['framenum'], transform, transform_gray)[0]

                # compute output
                _, clf_score = model(input_whole.cuda(), modality_chosen.cuda(), config.TRAIN_CLF.IF_LSTM)
                clf_score = torch.nn.functional.softmax(clf_score, dim=1)

                print(clf_score[:, target[0]])

                plt.scatter(range(100), clf_score[:, target[0]].cpu().numpy())
                # plt.show()
                #
                # time.sleep(30)
            
                plt.savefig('figure_{}_{}.jpg'.format(i, j))

                plt.clf()



def main():
    # convert to train_clf mode
    config.MODE = 'val_clf'
    extra()
    config.TRAIN.STATE_DICT = os.path.join('..', config.TRAIN.STATE_DICT)
    config.TEST.IF_TRIM = False;

    # cudnn related setting
    cudnn.benchmark = config.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = config.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = config.CUDNN.ENABLED

    # create a model
    os.environ["CUDA_VISIBLE_DEVICES"] = config.GPUS
    # gpus = [int(i) for i in config.GPUS.split(',')]
    gpus = range(config.GPU_NUM)
    model = create_model(config, is_train=False)

    if config.TRAIN.RESUME:
        model.my_load_state_dict(torch.load(config.TRAIN.STATE_DICT), strict=True)

    gpus = [int(i) for i in config.TRAIN_CLF.GPU.split(',')]
    os.environ["CUDA_VISIBLE_DEVICES"] = config.TRAIN_CLF.GPU
    model = model.cuda()

    # create a Loss class
    criterion = Loss(config).cuda()

    # load data
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    transform = transforms.Compose([
            transforms.Resize((config.MODEL.BACKBONE_INDIM_H, config.MODEL.BACKBONE_INDIM_W)),
            transforms.ToTensor(),
            normalize,
        ])

    normalize_gray = transforms.Normalize(mean=[0.456], std=[0.224])

    transform_gray = transforms.Compose([
        transforms.Resize((config.MODEL.BACKBONE_INDIM_H, config.MODEL.BACKBONE_INDIM_W)),
        transforms.ToTensor(),
        normalize_gray,
    ])

    valid_dataset = get_dataset(
        config,
        if_train = False,
        transform = transform
    )

    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=1,
        shuffle=True,
        num_workers=config.WORKERS,
        pin_memory=True,
    )

    score_distribution(config, valid_loader, model, criterion, 0, transform, transform_gray)



if __name__ == '__main__':
    main()

