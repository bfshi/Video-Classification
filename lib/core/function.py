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
from utils.utils import compute_acc
from utils.utils import load_frame
from utils.utils import choose_frame_randomly
from utils.utils import choose_modality_randomly


logger = logging.getLogger(__name__)

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
          epoch, replay_buffer, transform = None):
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
    data_time = AverageMeter()
    clf_losses = AverageMeter()
    rl_losses = AverageMeter()
    losses_batch = AverageMeter()
    acc = AverageMeter()

    #switch to train mode
    model.train()

    end = time.time()
    for i, (video_path, target, meta) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        # initialize the observation for lstm
        # ob = (h, c)
        # use total batch-size instead of paralleled batch-size!!!
        ob = (torch.zeros((target.shape[0], config.MODEL.LSTM_OUTDIM)).cuda(),
              torch.zeros((target.shape[0], config.MODEL.LSTM_OUTDIM)).cuda())
        model.init_weights(target.shape[0], ob)

        total_batch_size = target.shape[0]

        clf_score_sum = torch.zeros((total_batch_size, config.MODEL.CLFDIM)).cuda()

        target = target.cuda(async=True)

        losses_batch.reset()

        # train clf_head
        for step in range(config.TRAIN.TRAIN_CLF_STEP):

            # decide action according to model's policy and current observation
            new_act_frame, new_act_modality, _ = model.policy(ob[0])

            # get the input
            # input = ChooseInput(video, new_act_frame, new_act_modality, meta['framenum'])
            input = load_frame(video_path, new_act_modality, new_act_frame,
                               meta['framenum'], transform)

            # compute output
            next_ob, clf_score = model(input.cuda(), new_act_modality)
            clf_score_sum += clf_score

            # save into replay buffer (save the copy in CPU memory)
            replay_buffer.save((ob[0].cpu(), ob[1].cpu()),
                               new_act_frame.cpu(),
                               new_act_modality.cpu(),
                               input.cpu(),
                               (next_ob[0].cpu(), next_ob[1].cpu()),
                               compute_reward(clf_score).cpu())

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

        # update acc
        avg_acc = compute_acc(clf_score_sum, target)
        acc.update(avg_acc, total_batch_size)

        #update total clf_loss
        clf_losses.update(losses_batch.avg, losses_batch.count)

        losses_batch.reset()

        # TODO: In early stage we could only train classifier.
        # train the policy
        for step in range(config.TRAIN.TRAIN_RL_STEP):

            # sample from replay buffer a minibatch
            ob, act_frame, act_modality, input, next_ob, reward = \
                replay_buffer.get_batch(config.TRAIN.RL_BATCH_SIZE)

            # reset lstm's h and c
            model.module.init_weights(ob[0].shape[0], ob)

            # update reward and next_ob (shouldn't use the old ones)
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

        # update time record
        batch_time.update(time.time() - end)
        end = time.time()

        # logging
        if i % config.TRAIN.PRINT_EVERY == 0:
            msg = 'Epoch: [{0}][{1}/{2}]\t' \
                  'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                  'Speed {speed:.1f} samples/s\t' \
                  'Data {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
                  'CLF_Loss {clf_loss.val:.5f} ({clf_loss.avg:.5f})\t'\
                  'CLF_Accuracy {acc.val:.3f} ({acc.avg:.3f})\t'\
                  'RL_Loss {rl_loss.val:.5f} ({rl_loss.avg:.5f})'.format(
                epoch, i, len(train_loader), batch_time=batch_time,
                speed=total_batch_size / batch_time.val,
                data_time=data_time, clf_loss=clf_losses,
                acc=acc, rl_loss=rl_losses)
            logger.info(msg)


# TODO: update validate

def validate(config, val_loader, model, criterion, epoch, transform = None):
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
    clf_losses = AverageMeter()
    losses_batch = AverageMeter()
    acc = AverageMeter()

    #switch to val mode
    model.eval()

    with torch.no_grad():

        end = time.time()
        for i, (video_path, target, meta) in enumerate(val_loader):

            # initialize the observation
            # ob = (h, c)
            ob = (torch.zeros((target.shape[0], config.MODEL.LSTM_OUTDIM)).cuda(),
                  torch.zeros((target.shape[0], config.MODEL.LSTM_OUTDIM)).cuda())
            model.init_weights(target.shape[0], ob)

            total_batch_size = target.shape[0]

            clf_score_sum = torch.zeros((total_batch_size, config.MODEL.CLFDIM)).cuda()

            target = target.cuda()

            losses_batch.reset()

            #TODO: how to decide when to stop?
            for step in range(config.TRAIN.TEST_STEP):
                # decide action according to model's policy and current observation
                new_act_frame, new_act_modality, _ = model.policy(ob[0])

                # get the input
                # input = ChooseInput(video, new_act_frame, new_act_modality, meta['framenum'])
                input = load_frame(video_path, new_act_modality, new_act_frame,
                                   meta['framenum'], transform)

                # compute output
                next_ob, clf_score = model(input.cuda(), new_act_modality)
                clf_score_sum += clf_score

                # compute loss
                loss = criterion.clf_loss(clf_score, target)

                # update batch loss
                losses_batch.update(loss.item(), input.shape[0])

                # update ob
                ob = next_ob

            # update acc
            avg_acc = compute_acc(clf_score_sum, target)
            acc.update(avg_acc, total_batch_size)

            # update total clf_loss
            clf_losses.update(losses_batch.avg, losses_batch.count)

            batch_time.update(time.time() - end)
            end = time.time()

            # logging
            if i % config.TEST.PRINT_EVERY == 0:
                msg = 'Test: [{0}/{1}]\t' \
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' \
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t' \
                      'Accuracy {acc.val:.3f} ({acc.avg:.3f})'.format(
                    i, len(val_loader), batch_time=batch_time,
                    loss=clf_losses, acc=acc)
                logger.info(msg)

    return acc.avg


def train_clf(config, train_loader, model, criterion, optimizer, epoch, transform = None):
    """
    train backbone, lstm and clf_head only.
    unsorted sampling is used for each video.

    :param config: global configs
    :param train_loader: data loader
    :param model: model to be trained
    :param criterion: loss module
    :param optimizer: SGD or ADAM
    :param epoch: current epoch
    :return: None
    """

    # build recorders
    batch_time = AverageMeter()
    data_time = AverageMeter()
    clf_losses = AverageMeter()
    acc = AverageMeter()


    # switch to train mode
    model.train()

    end = time.time()
    for i, (video_path, target, meta) in enumerate(train_loader):
        # clear cache
        # torch.cuda.empty_cache()

        # measure data loading time
        data_time.update(time.time() - end)

        # initialize the observation for lstm
        # ob = (h, c)
        # use total batch-size instead of paralleled batch-size!!!
        ob = (torch.zeros((target.shape[0], config.MODEL.LSTM_OUTDIM)).cuda(),
              torch.zeros((target.shape[0], config.MODEL.LSTM_OUTDIM)).cuda())
        model.init_weights(target.shape[0], ob)

        total_batch_size = target.shape[0]

        # unsorted sampling.
        frame_chosen = choose_frame_randomly(total_batch_size, config.TRAIN_CLF.SAMPLE_NUM)
        modality_chosen = choose_modality_randomly(total_batch_size, config.MODEL.MODALITY_NUM,
                                                   config.TRAIN_CLF.SAMPLE_NUM)

        clf_score_sum = torch.zeros((total_batch_size, config.MODEL.CLFDIM)).cuda()

        target = target.cuda(async=True)

        for j in range(config.TRAIN_CLF.SAMPLE_NUM):
            # input = N * C * H * W (contains probably different modalities)
            input = load_frame(video_path, modality_chosen[:, j], frame_chosen[:, j],
                               meta['framenum'], transform)

            # compute output
            _, clf_score = model(input.cuda(), modality_chosen[:, j].cuda(), config.TRAIN_CLF.IF_LSTM)

            # accumulate clf_score
            clf_score_sum += clf_score
            # if j == 0:
            #     clf_score_sum = clf_score
            # else:
            #     temp = (clf_score.max(dim = 1, keepdim = True)[0] > clf_score_sum.max(dim = 1, keepdim = True)[0]).type(torch.float32)
            #     clf_score_sum = temp * clf_score + (1 - temp) * clf_score_sum

        # update acc
        avg_acc = compute_acc(clf_score_sum, target)
        acc.update(avg_acc, total_batch_size)

        # compute loss
        loss = criterion.clf_loss(clf_score_sum, target)

        # back prop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # update total clf_loss
        clf_losses.update(loss.item(), total_batch_size)

        # update time record
        batch_time.update(time.time() - end)
        end = time.time()

        # logging
        if i % config.TRAIN.PRINT_EVERY == 0:
            msg = 'Epoch: [{0}][{1}/{2}]\t' \
                  'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                  'Speed {speed:.1f} samples/s\t' \
                  'Data {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
                  'Loss {loss.val:.5f} ({loss.avg:.5f})\t' \
                  'Accuracy {acc.val:.3f} ({acc.avg:.3f})'.format(
                epoch, i, len(train_loader), batch_time=batch_time,
                speed=total_batch_size / batch_time.val,
                data_time=data_time, loss=clf_losses, acc=acc)
            logger.info(msg)


def validate_clf(config, val_loader, model, criterion, epoch, transform = None):
    """
    validate backbone, lstm and clf_head only.
    unsorted sampling is used for each video.

    :param config: global configs
    :param val_loader: data loader
    :param model: model to be trained
    :param criterion: loss module
    :param epoch: current epoch
    :return: None
    """

    # build recorders
    batch_time = AverageMeter()
    clf_losses = AverageMeter()
    acc = AverageMeter()

    # switch to val mode
    model.eval()

    with torch.no_grad():

        end = time.time()
        for i, (video_path, target, meta) in enumerate(val_loader):

            # initialize the observation
            # ob = (h, c)
            ob = (torch.zeros((target.shape[0], config.MODEL.LSTM_OUTDIM)).cuda(),
                  torch.zeros((target.shape[0], config.MODEL.LSTM_OUTDIM)).cuda())
            model.init_weights(target.shape[0], ob)

            total_batch_size = target.shape[0]

            # unsorted sampling.
            frame_chosen = np.random.rand(total_batch_size, config.TRAIN_CLF.SAMPLE_NUM)
            modality_chosen = torch.LongTensor(np.random.randint(0, config.MODEL.MODALITY_NUM,
                                                                 size=(total_batch_size, config.TRAIN_CLF.SAMPLE_NUM)))

            clf_score_sum = torch.zeros((total_batch_size, config.MODEL.CLFDIM)).cuda()

            target = target.cuda()

            # TODO: load frames when training. (refer to train_clf)
            for j in range(config.TRAIN_CLF.SAMPLE_NUM):
                # single modality feature: N * frame_num * channel * H * W
                input = load_frame(video_path, modality_chosen[:, j], frame_chosen[:, j],
                                   meta['framenum'], transform)

                # compute output
                _, clf_score = model(input.cuda(), modality_chosen[:, j].cuda(), config.TRAIN_CLF.IF_LSTM)

                # accumulate clf_score
                clf_score_sum += clf_score

            #update acc
            avg_acc = compute_acc(clf_score_sum, target)
            acc.update(avg_acc, total_batch_size)

            # compute loss
            loss = criterion.clf_loss(clf_score_sum, target)

            # update total clf_loss
            clf_losses.update(loss.item(), total_batch_size)

            # update time record
            batch_time.update(time.time() - end)
            end = time.time()

            if i % config.TEST.PRINT_EVERY == 0:
                msg = 'Test: [{0}/{1}]\t' \
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' \
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t' \
                      'Accuracy {acc.val:.3f} ({acc.avg:.3f})'.format(
                    i, len(val_loader), batch_time=batch_time,
                    loss=clf_losses, acc=acc)
                logger.info(msg)

    return acc.avg
