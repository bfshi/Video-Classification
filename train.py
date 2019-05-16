from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import pprint

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
from core.function import train
from core.function import validate
from core.loss import Loss
from dataset.dataset import get_dataset
from models.model import create_model
from utils.utils import create_optimizer
from utils.utils import create_logger
from utils.replay_buffer import create_replay_buffer


def main():
    # convert to train mode
    config.MODE = 'train'
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
    gpus = [int(i) for i in config.GPUS.split(',')]
    gpus = range(gpus.__len__())
    model = create_model(config, is_train=True)

    if config.TRAIN.RESUME:
        # model.my_load_state_dict(torch.load(config.TRAIN.STATE_DICT), strict=False, if_lstm=config.TRAIN.IF_LSTM)
        model.my_load_state_dict(torch.load(config.TRAIN.STATE_DICT), strict=False)

    if not config.TRAIN_CLF.SINGLE_GPU:  # use multi gpus in parallel
        model = model.cuda(gpus[0])
        # model.my_data_parallel(gpus)
        model = torch.nn.DataParallel(model, device_ids=gpus)
    else:  # use single gpu
        gpus = [int(i) for i in config.TRAIN_CLF.GPU.split(',')]
        os.environ["CUDA_VISIBLE_DEVICES"] = config.TRAIN_CLF.GPU
        model = model.cuda()

    #create a Loss class
    criterion = Loss(config).cuda()

    #create an optimizer
    optimizer = create_optimizer(config, model)

    #create a learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, config.TRAIN.LR_MILESTONES,
        config.TRAIN.LR_DECAY_RATE
    )

    #create a new replay buffer
    replay_buffer = create_replay_buffer()

    # load data
    train_dataset = get_dataset(
        config,
        if_train=True,
    )
    valid_dataset = get_dataset(
        config,
        if_train=False,
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.TRAIN.BATCH_SIZE * len(gpus),
        shuffle=config.TRAIN.SHUFFLE,
        num_workers=config.WORKERS,
        pin_memory=True,
        drop_last=True
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=config.TEST.BATCH_SIZE * len(gpus),
        shuffle=False,
        num_workers=config.WORKERS,
        pin_memory=True
    )

    #training and validating
    best_perf = 0
    for epoch in range(config.TRAIN.BEGIN_EPOCH, config.TRAIN.END_EPOCH):
        lr_scheduler.step()

        # train for one epoch
        train(config, train_loader, model, criterion, optimizer, epoch, replay_buffer)

        # evaluate on validation set
        if (epoch + 1) % config.TEST.TEST_EVERY == 0:
            perf_indicator = validate(config, valid_loader, model, criterion, epoch)

            if perf_indicator > best_perf:
                logger.info("=> saving checkpoint into {}".format(os.path.join(config.OUTPUT_DIR, 'checkpoint_{}.pth'.format(perf_indicator))))
                best_perf = perf_indicator
                torch.save(model.state_dict(), os.path.join(config.OUTPUT_DIR, 'checkpoint_{}.pth'.format(perf_indicator)))

    # save the final model
    logger.info("=> saving final model into {}".format(
        os.path.join(config.OUTPUT_DIR, 'model_{}.pth'.format(perf_indicator))
    ))
    torch.save(model.state_dict(),
               os.path.join(config.OUTPUT_DIR, 'model_{}.pth'.format(perf_indicator)))


if __name__ == '__main__':
    main()

