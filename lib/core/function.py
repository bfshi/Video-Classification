from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import time
import os

import numpy as np
import torch

import _init_paths
from utils.utils import AverageMeter
from utils.utils import ChooseInput
from utils.utils import rollout

def Act_Init():
    """
    initialize model's action

    :return: action
    """

class Score_Updater():
    def __init__(self):
        """
        update total score after every step
        """

    def reset(self):
        """
        reset all records
        """
#TODO: train_clf and train_rl

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

    #build recorders
    batch_time = AverageMeter()
    losses = AverageMeter()
    losses_batch = AverageMeter()

    #switch to train mode
    model.train()

    end = time.time()
    for i, (vedio, target) in enumerate(train_loader):

        # initialize the action
        action = Act_Init()

        losses_batch.reset()

        for step in range(config.TRAIN.TRAIN_STEP):

            #choose input according to the action
            input = ChooseInput(vedio, action)

            #compute output
            action, score, value = model(input)

            #rollout
            log_prob, advantages = rollout(vedio, model, action, value)

            #compute loss
            loss = criterion(score, target, log_prob, advantages)

            #back prop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            #update batch loss
            losses_batch.update(loss.item(), input.shape[0])

        #update total loss
        losses.update(losses_batch.avg, losses_batch.count)

        batch_time.update(time.time() - end)
        end = time.time()

        if i % config.TRAIN.PRINT_EVERY == 0:
            #TODO: logging training record



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

    # build recorders
    batch_time = AverageMeter()
    score_updater = Score_Updater()
    losses = AverageMeter()
    losses_batch = AverageMeter()

    #switch to val mode
    model.eval()

    with torch.no_grad():
        end = time.time()

        for i, (vedio, target) in enumerate(val_loader):

            action = Act_Init()
            score_updater.reset()
            losses_batch.reset()

            #TODO: how to decide when to stop?
            for step in range(config.TRAIN.TEST_STEP):
                # choose input according to the action
                input = ChooseInput(vedio, action)

                # compute output
                action, score, value = model(input)

                #TODO: need to compute loss during test?

                # rollout
                log_prob, advantages = rollout(vedio, model, action, value)

                # compute loss
                loss = criterion(score, target, log_prob, advantages)

                # update
                losses_batch.update(loss.item(), input.shape[0])
                score_updater.update(score)

            # update total loss
            losses.update(losses_batch.avg, losses_batch.count)
            #TODO: use score_updater to update mAP

            batch_time.update(time.time() - end)
            end = time.time()

            if i % config.TRAIN.PRINT_EVERY == 0:
                # TODO: logging validating record


