from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import logging
import time
from pathlib import Path
import cv2
from PIL import Image
import json

import numpy as np

import torch
import torch.optim as optim
import torch.nn as nn
from torchvision.transforms import ToTensor

import _init_paths
from core.config import config

totensor = ToTensor()

def create_optimizer(cfg, model):
    """
    create an SGD or ADAM optimizer

    :param cfg: global configs
    :param model: the model to be trained
    :return: an SGD or ADAM optimizer
    """
    optimizer = None
    if cfg.TRAIN.OPTIMIZER == 'sgd':
        optimizer = optim.SGD(
            model.parameters(),
            lr=cfg.TRAIN.LR,
            momentum=cfg.TRAIN.MOMENTUM,
            weight_decay=cfg.TRAIN.WD,
            nesterov=cfg.TRAIN.NESTEROV
        )
    elif cfg.TRAIN.OPTIMIZER == 'adam':
        optimizer = optim.Adam(
            model.parameters(),
            lr=cfg.TRAIN.LR
        )

    return optimizer


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count != 0 else 0


def ChooseInput(video, act_frame, act_modality, framenum):
    """
    choose the frame and modality indicated by action

    :param vedio: the whole batch of full videos
    :param act_frame: chosen frame position. \in [0, 1]
    :param act_modality: chosen modality
    :return: input choice for each video in the batch
    """
    # TODO: When batch_size > 1, we must use 'for loop' to choose modality
    #  and also to feed into the model because different instances may have
    #  different modalities chosen.

    act_frame = int(np.clip(act_frame * (framenum - 1), 0, framenum - 1))
    return video[act_modality][act_frame]

def choose_frame_randomly(batch_size, sample_num, segment=None, duration=None, if_trim = False):
    """
    randomly choose frames
    :param segment: [begin, end]
    """
    offset = np.random.rand(batch_size, 1) / sample_num
    # offset = np.ones((batch_size, 1), dtype=np.float32) / sample_num / 2
    frame_chosen = np.tile(np.array(range(sample_num)).astype(np.float32) / sample_num, (batch_size, 1)) + \
                   offset
    frame_chosen = torch.DoubleTensor(frame_chosen)
    if if_trim:
        duration = duration.view(-1, 1).type(dtype=torch.double)
        segment[0] = segment[0].view(-1, 1).type(dtype=torch.double)
        segment[1] = segment[1].view(-1, 1).type(dtype=torch.double)
        frame_chosen = frame_chosen * (segment[1] - segment[0]) / duration + segment[0] / duration
    # frame_chosen = np.random.rand(batch_size, sample_num) / sample_num
    # for i in range(1, sample_num):
    #     frame_chosen[:, i] = frame_chosen[:, i - 1] + 1.0 / sample_num
    return frame_chosen

def choose_modality_randomly(batch_size, modality_num, sample_num):
    """
    randomly choose modalities
    """
    # modality_chosen should be longTensor instead of np.array because it will be fed into
    # paralleled model and only Tensor can be split into paralleled pieces automatically.
    modality_chosen = torch.LongTensor(np.random.randint(0, modality_num,
                                                         size=(batch_size, sample_num)))
    return modality_chosen

def load_frame(video_path, modality_chosen, frame_chosen, framenum, transform = None, transform_gray = None):
    """
    compute input.
    :param video_path: ('path1', ..., 'pathN')
    :param modality_chosen: batch_size * sample_num
    :param frame_chosen: batch_size * sample_num (\in [0, 1])
    :param framenum: [num1, ..., numN]
    :return: batch_size * sample_num * C * H * W
    """
    frame_chosen = ((frame_chosen.type(dtype=torch.double) * (framenum.type(dtype=torch.double).view(-1, 1) - 1)).numpy()).astype(np.int16) + 1
    batch_size = frame_chosen.shape[0]
    sample_num = frame_chosen.shape[1]
    input = []

    # modality_chosen is a tensor, while frame_chosen is a np.array! They differ in transpose().
    if torch.is_tensor(video_path):

        return video_path[range(batch_size), modality_chosen.transpose(0, 1), frame_chosen.transpose(1, 0)].transpose(0, 1)


    else:

        # size of flow feature
        flow_h = config[config.TRAIN.DATASET].FLOW_H
        flow_w = config[config.TRAIN.DATASET].FLOW_W

        # load features of each video
        for i in range(batch_size):
            input_i = []
            for j in range(sample_num):
                # rgb feature
                # totensor will transpose (H, W, C) to (C, H, W) automatically
                if modality_chosen[i, j] == 0:
                    # input.append(transform(cv2.imread(os.path.join(video_path[i], 'img_%.5d.jpg' % frame_chosen[i]),
                    #                        cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)))\
                    input_i.append(transform(Image.open(os.path.join(video_path[i], 'img_%.5d.jpg' % frame_chosen[i, j]))))
                # flow feature
                # totensor must receive type np.uint8
                elif modality_chosen[i, j] == 1:
                    # input.append(np.stack([cv2.imread(os.path.join(video_path[i], 'flow_x_%.5d.jpg' % frame_chosen[i]),
                    #                       cv2.IMREAD_GRAYSCALE | cv2.IMREAD_IGNORE_ORIENTATION),
                    #                        cv2.imread(os.path.join(video_path[i], 'flow_y_%.5d.jpg' % frame_chosen[i]),
                    #                       cv2.IMREAD_GRAYSCALE | cv2.IMREAD_IGNORE_ORIENTATION),
                    #                        np.zeros((config.MODEL.BACKBONE_INDIM_H, config.MODEL.BACKBONE_INDIM_W))],
                    #                       axis = -1).astype(np.uint8)
                    #              )
                    input_i.append(torch.cat((transform_gray(Image.open(os.path.join(video_path[i],
                                                                             'flow_x_%.5d.jpg' % frame_chosen[i, j])).convert('L')),
                                           transform_gray(Image.open(os.path.join(video_path[i],
                                                                             'flow_y_%.5d.jpg' % frame_chosen[i, j])).convert('L')),
                                           torch.zeros((1, config.MODEL.BACKBONE_INDIM_H, config.MODEL.BACKBONE_INDIM_W))))
                                 )
                    # TODO: some flow.jpg have 3 channel???
                    #  "RuntimeError: invalid argument 0: Sizes of tensors must match except in dimension 0.
                    #  Got 3 and 7 in dimension 1" in "torch.stack(input)"
            input.append(torch.stack(input_i))

        return torch.stack(input)

def update_input(input, video_feature, choice_his, frame_chosen, modality_chosen):
    """
    :param input: [N, modality_num, framediv_num, feature_dim]
    :param video_feature: [N, modality_num, framediv_num, feature_dim]
    :param choice_his: [N, modality_num, framediv_num]
    :param frame_chosen: [N]
    :param modality_chosen: [N]
    """
    batch_size = input.shape[0]
    for i in range(batch_size):
        for j in range(config.MODEL.MODALITY_NUM):
            if j != modality_chosen[i]:
                continue
            start = frame_chosen[i]
            nonzero = choice_his[i, j, start:].nonzero()
            if nonzero.shape[0] == 0:
                input[i, j, start:, :] = video_feature[i, j, start, :]
            else:
                input[i, j, start: start + nonzero[0, 0], :] = video_feature[i, j, start, :]

    return input


def torch_clip(tensor, lower, upper, if_cuda=False):
    lower = torch.Tensor(lower).new_full((tensor.shape[0],), lower)
    upper = torch.Tensor(upper).new_full((tensor.shape[0],), upper)
    if if_cuda:
        lower = lower.cuda()
        upper = upper.cuda()
    return torch.max(torch.min(tensor, other=upper), other=lower)


def rollout(vedio, model, action, value):
    """
    rollout according to model's policy

    :param vedio: the whole batch of full videos
    :param model: VideoClfNet
    :param action: current action
    :param value: current state value
    :return: lists of log_prob and advantages
    """

    with torch.no_grad():
        return None

def compute_reward(clf_score_new, clf_score, target, cost = None):
    """
    compute reward according to classification score
    :param clf_score: NOT softmaxed!!!
    :param cost: aggregated cost (must below limit, otherwise reward is 0)
    :return: reward
    """
    batch_size = clf_score.shape[0]

    top2, indices = torch.topk(clf_score, k=2, dim=1)
    gap = (clf_score[range(batch_size), target] - top2[:, 0]) * (indices[:, 0] != target).type(torch.float) + \
          (clf_score[range(batch_size), target] - top2[:, 1]) * (indices[:, 0] == target).type(torch.float)

    top2, indices = torch.topk(clf_score_new, k=2, dim=1)
    gap_new = (clf_score_new[range(batch_size), target] - top2[:, 0]) * (indices[:, 0] != target).type(torch.float) + \
          (clf_score_new[range(batch_size), target] - top2[:, 1]) * (indices[:, 0] == target).type(torch.float)

    if cost is not None:
        return (gap_new - gap) * (cost <= config.MODEL.COST_LIMIT).type(torch.float)
    else:
        return gap_new - gap


def soft_update_from_to(source, target, tau):
    """
    soft update for target v_head
    :param source: v_head
    :param target: target v_head
    :param tau: update rate
    :return: None
    """
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(
            target_param.data * (1.0 - tau) + param.data * tau
        )


def compute_acc(score, target):
    """
    compute average accuracy
    :param score: classification score
    :param target: target label
    :return: avg_acc
    """
    batch_size = target.shape[0]
    avg_acc = (score.argmax(dim = 1) == target).sum().item() / batch_size
    return avg_acc



def create_logger(cfg, phase='train'):
    """
    create a logger for experiment record
    To use a logger to publish message m, just run logger.info(m)

    :param cfg: global config
    :param phase: train or val
    :return: a logger
    """
    time_str = time.strftime('%Y-%m-%d-%H-%M')
    log_file = '{}_{}_{}.log'.format(cfg.TRAIN.DATASET, time_str, phase)
    final_log_file = Path(cfg.OUTPUT_DIR) / log_file
    log_format = '%(asctime)-15s: %(message)s'
    logging.basicConfig(filename=str(final_log_file),
                        format=log_format)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    console = logging.StreamHandler()
    logging.getLogger('').addHandler(console)

    return logger


class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(MyEncoder, self).default(obj)



"""
GPU wrappers
"""

_use_gpu = False
device = None


def set_gpu_mode(mode, gpu_id=0):
    global _use_gpu
    global device
    global _gpu_id
    _gpu_id = gpu_id
    _use_gpu = mode
    device = torch.device("cuda:" + str(gpu_id) if _use_gpu else "cpu")


def gpu_enabled():
    return _use_gpu


def set_device(gpu_id):
    torch.cuda.set_device(gpu_id)


# noinspection PyPep8Naming
def FloatTensor(*args, **kwargs):
    return torch.FloatTensor(*args, **kwargs).to(device)


def from_numpy(*args, **kwargs):
    return torch.from_numpy(*args, **kwargs).float().to(device)


def get_numpy(tensor):
    return tensor.to('cpu').detach().numpy()


def ones(*sizes, torch_device=None, **kwargs):
    if torch_device is None:
        torch_device = device
    return torch.ones(*sizes, **kwargs, device=torch_device)


def ones_like(*args, torch_device=None, **kwargs):
    if torch_device is None:
        torch_device = device
    return torch.ones_like(*args, **kwargs, device=torch_device)


def randn(*args, torch_device=None, **kwargs):
    if torch_device is None:
        torch_device = device
    return torch.randn(*args, **kwargs, device=torch_device)


def zeros(*sizes, torch_device=None, **kwargs):
    if torch_device is None:
        torch_device = device
    return torch.zeros(*sizes, **kwargs, device=torch_device)


def zeros_like(*args, torch_device=None, **kwargs):
    if torch_device is None:
        torch_device = device
    return torch.zeros_like(*args, **kwargs, device=torch_device)


def tensor(*args, torch_device=None, **kwargs):
    if torch_device is None:
        torch_device = device
    return torch.tensor(*args, **kwargs, device=torch_device)


def normal(*args, **kwargs):
    return torch.normal(*args, **kwargs).to(device)

