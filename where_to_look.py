from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import pprint
import shutil
import json
import time

import numpy as np
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
from core.loss import Loss
from dataset.dataset import get_dataset
from models.model import create_model
from utils.utils import create_optimizer
from utils.utils import create_logger
from utils.utils import MyEncoder
from utils.utils import AverageMeter
from utils.utils import compute_acc
from utils.utils import compute_reward


def watching_sequence(config, val_loader, model, output_dict):
    """
    validating method

    :param config: global configs
    :param val_loader: data loader
    :param model: model to be trained
    :param criterion: loss module
    :param epoch: current epoch
    :return: performance indicator
    """

    # build recorders
    batch_time = AverageMeter()
    acc = AverageMeter()

    #switch to val mode
    model.eval()

    with torch.no_grad():

        end = time.time()
        for i, (video_feature, target, meta) in enumerate(val_loader):

            total_batch_size = target.shape[0]

            # initialize the observation.
            # use total batch-size instead of paralleled batch-size!!!
            ob = torch.zeros((total_batch_size, config.MODEL.FEATURE_DIM)).cuda()
            cost = torch.zeros((total_batch_size,)).cuda()
            cost_new = torch.zeros((total_batch_size,)).cuda()
            input = torch.zeros_like(video_feature)

            choice_his = torch.zeros((total_batch_size, config.MODEL.MODALITY_NUM, config.MODEL.FRAMEDIV_NUM)).cuda()
            if_finish = torch.zeros((total_batch_size)).cuda()

            clf_score = torch.zeros((total_batch_size, config.MODEL.CLFDIM)).cuda()
            clf_score_final = torch.zeros((total_batch_size, config.MODEL.CLFDIM)).cuda()

            target = target.cuda()

            finish_time = torch.zeros((total_batch_size)).type(torch.long).cuda()
            act_frame_his = torch.zeros((total_batch_size, 0)).type(torch.long).cuda()
            act_modality_his = torch.zeros((total_batch_size, 0)).type(torch.long).cuda()
            reward_his = torch.zeros((total_batch_size, 0)).cuda()

            # collect experiences until all instances have run out of cost limit.
            while ((cost <= config.MODEL.COST_LIMIT).sum() > 0):

                # decide action according to model's policy and current observation
                new_act_frame, new_act_modality, _ = model.module.policy(ob, cost, choice_his,
                                                                         if_random = False, if_val=True)
                act_frame_his = torch.cat((act_frame_his, new_act_frame.view(-1, 1)), dim=1)
                act_modality_his = torch.cat((act_modality_his, new_act_modality.view(-1, 1)), dim=1)

                # get the input
                # input = ChooseInput(video, new_act_frame, new_act_modality, meta['framenum'])
                input[range(total_batch_size), new_act_modality, new_act_frame, :] = video_feature[range(total_batch_size), new_act_modality, new_act_frame, :]

                # compute output
                clf_score_new, ob_new = model(input.cuda())

                # compute new cost
                for j in range(config.MODEL.MODALITY_NUM):
                    cost_new += (new_act_modality == j).type(torch.float) * config.MODEL.COST_LIST[j]

                reward_his = torch.cat((reward_his,
                                        compute_reward(clf_score_new, clf_score, target, cost_new).view(-1, 1)),
                                       dim=1)

                # update ob, cost and clf_score, if_finish, choice_his, finish_time
                ob = ob_new.clone()
                clf_score_final += clf_score * \
                                   ((cost_new > config.MODEL.COST_LIMIT).type(torch.float) - if_finish).view(-1, 1)
                finish_time += (cost_new <= config.MODEL.COST_LIMIT).type(torch.long)
                clf_score = clf_score_new.clone()
                cost = cost_new.clone()
                if_finish = (cost_new > config.MODEL.COST_LIMIT).type(torch.float)
                choice_his[range(total_batch_size), new_act_modality, new_act_frame] = 1

            act_frame_his = act_frame_his.detach().cpu().numpy()
            act_modality_his = act_modality_his.detach().cpu().numpy()
            # reward_his = np.around(reward_his.detach().cpu().numpy(), decimals=3)
            reward_his = reward_his.detach().cpu().numpy()

            # update acc
            avg_acc = compute_acc(clf_score_final, target)
            acc.update(avg_acc, total_batch_size)

            batch_time.update(time.time() - end)
            end = time.time()

            clf_score_final = torch.nn.functional.softmax(clf_score_final, dim=1).cpu().numpy()
            for j in range(total_batch_size):
                output_dict[meta['name'][j]] = {}
                output_dict[meta['name'][j]]['frame_sequence'] = list(act_frame_his[j, 0: finish_time[j]])
                output_dict[meta['name'][j]]['modality_sequence'] = list(act_modality_his[j, 0: finish_time[j]])
                output_dict[meta['name'][j]]['reward_sequence'] = list(reward_his[j, 0: finish_time[j]])


    return acc.avg


def main():
    """
    record the frame-watching sequence of each video.
    save as a json file with format {'name1':[(frame1, modality1), ...], ...}
    """

    # convert to val mode
    config.MODE = 'val'
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

    if config.TEST.RESUME:
        model.my_load_state_dict(torch.load(config.TEST.STATE_DICT), strict=False)

    if not config.TRAIN_CLF.SINGLE_GPU:  # use multi gpus in parallel
        model = model.cuda(gpus[0])
        # model.backbones = torch.nn.DataParallel(model.backbones, device_ids=gpus)
        model = torch.nn.DataParallel(model, device_ids=gpus)
    else:  # use single gpu
        gpus = [int(i) for i in config.TRAIN_CLF.GPU.split(',')]
        os.environ["CUDA_VISIBLE_DEVICES"] = config.TRAIN_CLF.GPU
        model = model.cuda()

    valid_dataset = get_dataset(
        config,
        if_train=False,
    )

    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=config.TEST.BATCH_SIZE * len(gpus),
        shuffle=False,
        num_workers=config.WORKERS,
        pin_memory=True
    )

    # output result or not
    output_dict = dict()

    perf_indicator = watching_sequence(config, valid_loader, model, output_dict)

    # output the result
    print("=> saving frame-watching sequence into {}".format(
        os.path.join(config.OUTPUT_DIR, 'watching_sequence_{acc_avg}.json'.format(acc_avg=perf_indicator))
    ))
    output_file = open(os.path.join(config.OUTPUT_DIR, 'watching_sequence_{acc_avg}.json'.format(acc_avg=perf_indicator)),
                       'w')
    json.dump(output_dict, output_file, cls=MyEncoder)
    output_file.close()

if __name__ == '__main__':
    main()