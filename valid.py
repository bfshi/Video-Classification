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
from core.function import train
from core.function import validate
from core.loss import Loss
from dataset.dataset import get_dataset
from models.model import create_model
from utils.utils import create_optimizer


def main():
    #create a logger
    logger = create_logger(config, 'val')

    # cudnn related setting
    cudnn.benchmark = config.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = config.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = config.CUDNN.ENABLED

    #load a model
    model = model = create_model(config, is_train = False)


    #use multi gpus in parallel
    gpus = [int(i) for i in config.GPUS.split(',')]
    model = torch.nn.DataParallel(model, device_ids=gpus).cuda()

    #create a Loss class
    criterion = Loss(config).cuda()

    # load data
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    valid_dataset = get_dataset(if_train=False)

    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=config.TEST.BATCH_SIZE * len(gpus),
        shuffle=False,
        num_workers=config.WORKERS,
        pin_memory=True
    )

    perf = validate(config, valid_loader, model, criterion, epoch)

    # TODO: logging

if __name__ == '__main__':
    main()