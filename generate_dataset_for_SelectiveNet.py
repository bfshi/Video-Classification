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

def generate_dataset(config, model, train_loader, criterion):
    """
    dataset = {'video name': [[observed frames], target frame], ......}
    """
    acc_sel = AverageMeter()
    acc_rand = AverageMeter()

    dataset = dict()
    prechosing_num = 5
    max_chosing_num = 15
    prechosen_frames = ((np.array(range(prechosing_num)) + 0.5) / prechosing_num *
                        config.MODEL.FRAMEDIV_NUM).astype(np.int)

    with torch.no_grad():
        for i, (video_feature, target, meta) in enumerate(train_loader):
            print(i)

            total_batch_size = target.shape[0]

            target = target.cuda()

            input = torch.zeros((total_batch_size, config.MODEL.MODALITY_NUM,
                                 config.MODEL.FRAMEDIV_NUM, config.MODEL.FEATURE_DIM))
            # FIXME: In multi-modality setting, how to pre-choose frames?
            input[:, 0, prechosen_frames, :] = video_feature[:, 0, prechosen_frames, :]

            # record which frames have been chosen already
            chosen_frames = np.zeros((total_batch_size, config.MODEL.MODALITY_NUM, config.MODEL.FRAMEDIV_NUM))
            # FIXME: same as above
            chosen_frames[:, 0, prechosen_frames] = 1

            # random sample some frames as the chosen frames
            sample_num = np.random.randint(max_chosing_num - prechosing_num)
            if sample_num != 0:
                frame_chosen = torch.multinomial(
                    input=torch.ones((config.MODEL.FRAMEDIV_NUM)) / config.MODEL.FRAMEDIV_NUM,
                    num_samples=sample_num)
                # frame_chosen = (choose_frame_randomly(1, 5) * config.MODEL.FRAMEDIV_NUM).type(torch.long).view(-1)
                modality_chosen = torch.multinomial(
                    input=torch.ones((config.MODEL.MODALITY_NUM)) / config.MODEL.MODALITY_NUM,
                    num_samples=sample_num, replacement=True)
                chosen_frames[:, modality_chosen, frame_chosen] = 1
                input[:, modality_chosen, frame_chosen, :] = video_feature[:, modality_chosen, frame_chosen, :]

            # search the target frame
            min_loss = np.zeros((total_batch_size,))
            min_loss = 1e6
            best_modality = np.zeros((total_batch_size,))
            best_frame = np.zeros((total_batch_size,))
            for mod in range(config.MODEL.MODALITY_NUM):
                for fra in range(config.MODEL.FRAMEDIV_NUM):
                    input_temp = input.clone()
                    input_temp[:, mod, fra, :] = video_feature[:, mod, fra, :]
                    clf_score, _ = model(input_temp, if_fusion=False)
                    loss = (F.cross_entropy(clf_score, target, reduce=False, reduction='none')).cpu().numpy()

                    # updating
                    if_update = (loss < min_loss).astype(np.int)
                    if_update = if_update * (1 - chosen_frames[:, mod, fra])  # in case of one frame chosen twice
                    best_modality = (1 - if_update) * best_modality + if_update * mod
                    best_frame = (1 - if_update) * best_frame + if_update * fra
                    min_loss = (1 - if_update) * min_loss + if_update * loss

            # updating dataset
            for k, name in enumerate(meta['name']):
                dataset[name] = [np.nonzero(chosen_frames[k])[0],
                                         np.nonzero(chosen_frames[k])[1],
                                         int(best_modality[k]),
                                         int(best_frame[k])]

            # comparing selective acc v.s. random acc

            # computing selective acc
            input_temp = input.clone()
            input_temp[range(total_batch_size), best_modality.astype(np.int), best_frame.astype(np.int), :] = \
                video_feature[range(total_batch_size), best_modality.astype(np.int), best_frame.astype(np.int), :]
            clf_score, _ = model(input_temp, if_fusion=False)
            avg_acc = compute_acc(clf_score, target)
            acc_sel.update(avg_acc, total_batch_size)

            # computing random acc
            best_modality = np.random.randint(0, config.MODEL.MODALITY_NUM, size=total_batch_size)
            best_frame = np.random.randint(0, config.MODEL.FRAMEDIV_NUM, size=total_batch_size)
            input_temp = input.clone()
            input_temp[range(total_batch_size), best_modality.astype(np.int), best_frame.astype(np.int), :] = \
                video_feature[range(total_batch_size), best_modality.astype(np.int), best_frame.astype(np.int), :]
            clf_score, _ = model(input_temp, if_fusion=False)
            avg_acc = compute_acc(clf_score, target)
            acc_rand.update(avg_acc, total_batch_size)

            print("Selective accuracy: {}".format(acc_sel.avg))
            print("Random Accuracy: {}".format(acc_rand.avg))


    return dataset





def main():
    """
    Use a trained classification network to generate training set for SelectiveNet.
    Path to train network is saved in config.TRAIN.STATE_DICT
    """

    # convert to train_clf mode
    config.MODE = 'train_clf'
    extra()

    # cudnn related setting
    cudnn.benchmark = config.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = config.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = config.CUDNN.ENABLED

    # create a model
    os.environ["CUDA_VISIBLE_DEVICES"] = config.GPUS
    # gpus = [int(i) for i in config.GPUS.split(',')]
    gpus = range(config.GPU_NUM)
    model = create_model(config, is_train=True)

    model.my_load_state_dict(torch.load(config.TRAIN.STATE_DICT), strict=True)

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

    # load data
    train_dataset = get_dataset(
        config,
        if_train = True,
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.TRAIN.BATCH_SIZE * len(gpus),
        shuffle=config.TRAIN.SHUFFLE,
        num_workers=config.WORKERS,
        pin_memory=False,
        drop_last=False
    )

    dataset = generate_dataset(config, model, train_loader, criterion)

    file = open('selective_training_set.json', 'w')
    json.dump(dataset, file, cls=MyEncoder)


if __name__ == '__main__':
    main()

# a = dict()
# a['aaa'] = np.array([1, 2, 3])
# file = open('test.json', 'w')
# json.dump(a, file, cls=MyEncoder)



