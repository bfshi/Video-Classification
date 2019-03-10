from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

def train(config, train_loader, model, criterion, optimizer,
          epoch):
    """
    training method

    :param config: global configs
    :param train_loader: data loader
    :param model: model to be trained
    :param criterion: loss module
    :param optimizer: SGD or ADAM
    :param epoch: current epoch
    :return: None
    """


def validate(config, val_loader, model, criterion, epoch):
    """
    validating method

    :param config: global configs
    :param val_loader: data loader
    :param model: model to be trained
    :param criterion: loss module
    :param epoch: current epoch
    :return: performance indicator
    """