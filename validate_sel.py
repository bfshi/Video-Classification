from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import pprint
import shutil
import json

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
from core.function import train_sel
from core.function import validate_sel
from core.loss import Loss
from dataset.dataset import get_dataset
from models.model import create_model
from models.model import create_sel_model
from utils.utils import create_optimizer
from utils.utils import create_logger
from utils.replay_buffer import create_replay_buffer
from utils.utils import MyEncoder


def main():
    # convert to val_clf mode
    config.MODE = 'val_sel'
    extra()

    # create a logger
    logger = create_logger(config, 'val')

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
    model = create_sel_model(config)

    if config.TEST_SEL.RESUME:
        model.my_load_state_dict(torch.load(config.TEST_SEL.STATE_DICT), strict=False)

    if not config.TRAIN_CLF.SINGLE_GPU:  # use multi gpus in parallel
        model = model.cuda(gpus[0])
        # model.backbones = torch.nn.DataParallel(model.backbones, device_ids=gpus)
        model = torch.nn.DataParallel(model, device_ids=gpus)
    else:  # use single gpu
        gpus = [int(i) for i in config.TRAIN_CLF.GPU.split(',')]
        os.environ["CUDA_VISIBLE_DEVICES"] = config.TRAIN_CLF.GPU
        model = model.cuda()

    # loading classification model
    model_clf = create_model(config, is_train=True)
    model_clf.my_load_state_dict(torch.load(config.STATE_DICT.F_C), strict=False)
    model_clf = model_clf.cuda()

    # create a Loss class
    criterion = Loss(config).cuda()

    # load data

    valid_dataset = get_dataset(
        config,
        if_train = False,
    )

    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=config.TRAIN.BATCH_SIZE * len(gpus),
        shuffle=False,
        num_workers=config.WORKERS,
        pin_memory=True,
    )

    # output result or not
    output_dict = dict()
    # complement some info
    output_dict['version'] = 'VERSION 1.3'
    output_dict['results'] = {}
    output_dict['external_data'] = {"used": False, "details": "No details."}

    # training and validating
    perf_indicator = validate_sel(config, valid_loader, model, model_clf, criterion, 0,
                                  output_dict, valid_dataset)



if __name__ == '__main__':
    main()

