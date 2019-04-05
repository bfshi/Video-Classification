from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import logging
import time
from pathlib import Path

import torch
import torch.optim as optim

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

def compute_reward(clf_score):
    """
    compute reward according to classification score
    :param clf_score: NOT softmaxed!!!
    :return: reward
    """
    # TODO: log(max / second_max) or log(gt / max_else) ???
    clf_score = torch.nn.functional.softmax(clf_score, dim = 1)
    maxv, maxpos = clf_score.max(dim = 1)
    minv, _ = clf_score.min(dim = 1)
    clf_score[:, maxpos] = minv
    second_maxv, _ = clf_score.max()
    return torch.log(maxv / second_maxv)


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




def create_logger(cfg, phase='train'):
    """
    create a logger for experiment record
    To use a logger to publish message m, just run logger.info(m)

    :param cfg: global config
    :param phase: train or val
    :return: a logger
    """
    time_str = time.strftime('%Y-%m-%d-%H-%M')
    log_file = '{}_{}_{}.log'.format(cfg.DATASET, time_str, phase)
    final_log_file = Path(cfg.OUTPUT_DIR) / log_file
    log_format = '%(asctime)-15s: %(message)s'
    logging.basicConfig(filename=str(final_log_file),
                        format=log_format)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    console = logging.StreamHandler()
    logging.getLogger('').addHandler(console)

    return logger





"""
GPU wrappers
"""

_use_gpu = True
device = 'gpu'


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