from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import pprint
import shutil

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


def main():
    # convert to train_clf mode
    config.MODE = 'train_clf'
    extra()

    # create a logger
    logger = create_logger(config, 'train')

    # logging configurations
    logger.info(pprint.pformat(config))

    # cudnn related setting
    cudnn.benchmark = config.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = config.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = config.CUDNN.ENABLED

    # create a model
    os.environ["CUDA_VISIBLE_DEVICES"] = config.GPUS
    # gpus = [int(i) for i in config.GPUS.split(',')]
    gpus = range(config.GPU_NUM)
    model = create_model(config, is_train=True)

    if config.TRAIN.RESUME:
        model.my_load_state_dict(torch.load(config.TRAIN.STATE_DICT), strict=False)

    if not config.TRAIN_CLF.SINGLE_GPU:  # use multi gpus in parallel
        model = model.cuda(gpus[0])
        # model.backbones = torch.nn.DataParallel(model.backbones, device_ids=gpus)
        model = torch.nn.DataParallel(model, device_ids=gpus)
    else:  # use single gpu
        gpus = [int(i) for i in config.TRAIN_CLF.GPU.split(',')]
        os.environ["CUDA_VISIBLE_DEVICES"] = config.TRAIN_CLF.GPU
        model = model.cuda()

    # create a Loss class
    criterion = Loss(config).cuda()

    # create an optimizer
    optimizer = create_optimizer(config, model)

    # create a learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, config.TRAIN.LR_MILESTONES,
        config.TRAIN.LR_DECAY_RATE
    )

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

    train_dataset = get_dataset(
        config,
        if_train = True,
        transform = transform
    )
    valid_dataset = get_dataset(
        config,
        if_train = False,
        transform = transform
    )

    mean = 0
    norm = 0
    for i in train_dataset.rgb_feature_list:
        mean += i.mean()
        norm += i.norm()
    print("rgb_mean = {}".format(mean))
    print("rgb_norm = {}".format(norm))

    mean = 0
    norm = 0
    for i in train_dataset.flow_feature_list:
        mean += i.mean()
        norm += i.norm()
    print("flow_mean = {}".format(mean))
    print("flow_norm = {}".format(norm))