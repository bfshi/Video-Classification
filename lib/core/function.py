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
from utils.utils import compute_reward
from utils.utils import soft_update_from_to


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


def train(config, train_loader, model, criterion, optimizer,
          epoch, replay_buffer):
    """
    training method

    :param config: global configs
    :param train_loader: data loader
    :param model: model to be trained
    :param criterion: loss module
    :param optimizer: SGD or ADAM
    :param epoch: current epoch
    :param replay_buffer: buffer for self replay
    :return: None
    """

    #build recorders
    batch_time = AverageMeter()
    clf_losses = AverageMeter()
    rl_losses = AverageMeter()
    losses_batch = AverageMeter()

    #switch to train mode
    model.train()

    end = time.time()
    for i, (vedio, target) in enumerate(train_loader):

        #TODO: initialize observation
        #ob = (h, c)
        ob =
        model.init_weights(vedio.shape[0], ob)

        losses_batch.reset()

        # train clf_head
        for step in range(config.TRAIN.TRAIN_CLF_STEP):

            # decide action according to model's policy and current observation
            new_act_frame, new_act_modality, _ = model.policy(ob[0])

            # get the input
            input = ChooseInput(vedio, new_act_frame, new_act_modality)

            # compute output
            next_ob, clf_score = model(input)

            # save into replay buffer
            replay_buffer.save(ob, new_act_frame, new_act_modality,
                              next_ob, compute_reward(clf_score))

            # compute loss
            loss = criterion.clf_loss(clf_score, target)

            # back prop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # update batch loss
            losses_batch.update(loss.item(), input.shape[0])

            # update ob
            ob = next_ob

        #update total clf_loss
        clf_losses.update(losses_batch.avg, losses_batch.count)

        losses_batch.reset()

        # TODO: In early stage we could only train classifier.
        # train the policy
        for step in range(config.TRAIN.TRAIN_RL_STEP):

            # sample from replay buffer a minibatch
            ob, act_frame, act_modality, next_ob, reward = \
                replay_buffer.get_batch(config.TRAIN.RL_BATCH_SIZE)

            # reset lstm's h and c
            model.init_weights(ob[0].shape[0], ob)

            # update reward and next_ob (shouldn't use the old ones)
            input = ChooseInput(vedio, act_frame, act_modality)
            next_ob, clf_score = model(input)
            reward = compute_reward(clf_score)

            # calculate new outputs
            q_pred = model.q_head(torch.cat((ob[0], act_frame, act_modality), 1))
            v_pred = model.v_head(ob[0])

            new_act_frame, new_act_modality, log_pi = model.policy(ob[0])
            q_new_actions = model.q_head(torch.cat((ob[0], new_act_frame, new_act_modality), 1))
            target_v_pred_next = model.target_v_head(ob[0])

            # compute the loss
            loss = rl_losses(reward, q_pred, v_pred, target_v_pred_next, log_pi, q_new_actions)

            # back prop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # update batch loss
            losses_batch.update(loss.item(), input.shape[0])

        # update total rl_loss
        rl_losses.update(losses_batch.avg, losses_batch.count)

        # soft update
        soft_update_from_to(model.v_head, model.target_v_head, config.TRAIN.SOFT_UPDATE)


        batch_time.update(time.time() - end)
        end = time.time()

        if i % config.TRAIN.PRINT_EVERY == 0:
            # TODO: logging training record



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


