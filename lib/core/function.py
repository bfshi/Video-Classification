from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import time
import os

import numpy as np
import torch

import _init_paths
from core.config import config
from utils.utils import AverageMeter
from utils.utils import ChooseInput
from utils.utils import compute_reward
from utils.utils import soft_update_from_to
from utils.utils import compute_acc
from utils.utils import load_frame
from utils.utils import choose_frame_randomly
from utils.utils import choose_modality_randomly
from utils.utils import update_input


logger = logging.getLogger(__name__)


def train(config, train_loader, model, criterion, optimizer, epoch, replay_buffer):
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
    losses_batch = AverageMeter()
    clf_losses = AverageMeter()
    rl_losses = AverageMeter()
    acc = AverageMeter()

    #switch to train mode
    model.train()

    end = time.time()
    for i, (video_feature, target, meta) in enumerate(train_loader):

        # measure data loading time
        data_time.update(time.time() - end)

        total_batch_size = target.shape[0]

        # initialization
        # use total batch-size instead of paralleled batch-size!!!
        ob = torch.zeros((total_batch_size, config.MODEL.FEATURE_DIM)).cuda()
        cost = torch.zeros((total_batch_size, )).cuda()
        cost_new = torch.zeros((total_batch_size, )).cuda()
        input = torch.zeros((total_batch_size, config.MODEL.MODALITY_NUM,
                             config.MODEL.FRAMEDIV_NUM, config.MODEL.FEATURE_DIM))
        choice_his = torch.zeros((total_batch_size, config.MODEL.MODALITY_NUM, config.MODEL.FRAMEDIV_NUM)).cuda()
        if_finish = torch.zeros((total_batch_size)).cuda()

        clf_score = torch.zeros((total_batch_size, config.MODEL.CLFDIM)).cuda()
        clf_score_final = torch.zeros((total_batch_size, config.MODEL.CLFDIM)).cuda()

        target = target.cuda(async=True)


        # collect experiences and save into replay buffer
        with torch.no_grad():
            # collect experiences until all instances have run out of cost limit.
            while ((cost <= config.MODEL.COST_LIMIT).sum() > 0):

                # decide action according to model's policy and current observation
                new_act_frame, new_act_modality, _ = model.module.policy(ob, cost, choice_his,
                                                                         if_random=True)

                # update the input
                input[range(total_batch_size), new_act_modality, new_act_frame, :] = video_feature[range(total_batch_size), new_act_modality, new_act_frame, :]

                # compute output
                clf_score_new, ob_new = model(input.cuda())

                # compute new cost
                for j in range(config.MODEL.MODALITY_NUM):
                   cost_new += (new_act_modality == j).type(torch.float) * config.MODEL.COST_LIST[j]

                # save into replay buffer (save the copy in CPU memory)
                # reward is computed w.r.t clf_score_new - clf_score (if cost > limit then reward is 0!)
                replay_buffer.save(ob.detach().cpu(),
                                   cost.detach().cpu(),
                                   choice_his.detach().cpu(),
                                   new_act_frame.detach().cpu(),
                                   new_act_modality.detach().cpu(),
                                   ob_new.detach().cpu(),
                                   cost_new.detach().cpu(),
                                   compute_reward(clf_score_new.clone(), clf_score.clone(), target, cost, cost_new).detach().cpu())

                print('reward = {}'.format(compute_reward(clf_score_new.clone(), clf_score.clone(), target, cost, cost_new)))

                # update ob, cost, clf_score, if_finish, choice_his
                ob = ob_new.clone()
                clf_score_final += clf_score * \
                                   ((cost_new > config.MODEL.COST_LIMIT).type(torch.float) - if_finish).view(-1, 1)
                clf_score = clf_score_new.clone()
                cost = cost_new.clone()
                if_finish = (cost_new > config.MODEL.COST_LIMIT).type(torch.float)
                choice_his[range(total_batch_size), new_act_modality, new_act_frame] = 1

        # update acc
        avg_acc = compute_acc(clf_score_final, target)
        acc.update(avg_acc, total_batch_size)

        # update total clf_loss
        loss = criterion.clf_loss(clf_score_final, target)
        clf_losses.update(loss, total_batch_size)

        losses_batch.reset()

        # train the policy
        for step in range(config.TRAIN.TRAIN_RL_STEP):

            # during first 5 epochs, no training, just experience collecting
            if epoch < 5:
                break

            # sample from replay buffer a minibatch
            ob, cost, choice_his, act_frame, act_modality, next_ob, next_cost, reward = \
                replay_buffer.get_batch(config.TRAIN.RL_BATCH_SIZE)

            # calculate new outputs
            q_pred = model.module.q_head(torch.cat((ob, cost.view(-1, 1),
                                             act_frame.view(-1, 1), act_modality.view(-1, 1)), 1))
            v_pred = model.module.v_head(torch.cat((ob, cost.view(-1, 1)), -1))

            new_act_frame, new_act_modality, log_pi = model.module.policy(ob, cost, choice_his)
            q_new_actions = model.module.q_head(torch.cat((ob, cost.view(-1, 1),
                                                    new_act_frame.type(dtype=torch.float).view(-1, 1),
                                                    new_act_modality.type(dtype=torch.float).view(-1, 1)), 1))
            target_v_pred_next = model.module.target_v_head(torch.cat((next_ob, next_cost.view(-1, 1)), 1))

            # compute the loss
            loss = criterion.rl_loss(reward, q_pred, v_pred, target_v_pred_next, log_pi, q_new_actions)

            # back prop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # update batch loss
            losses_batch.update(loss.item(), input.shape[0])

            print('q_pred: {}'.format(q_pred))
            print('v_pred: {}'.format(v_pred))


        # update total rl_loss
        rl_losses.update(losses_batch.avg, losses_batch.count)

        # soft update target_v_head
        soft_update_from_to(model.module.v_head, model.module.target_v_head, config.TRAIN.SOFT_UPDATE)

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


def validate(config, val_loader, model, criterion, epoch, output_dict=None, valid_dataset=None):
    """
    validating method

    :param config: global configs
    :param val_loader: data loader
    :param model: model to be trained
    :param criterion: loss module
    :param epoch: current epoch
    :return: performance indicator
    """
    # whether to output the result
    if_output_result = (valid_dataset != None)
    label_head = ['label' for i in range(config.MODEL.CLFDIM)]
    score_head = ['score' for i in range(config.MODEL.CLFDIM)]

    # build recorders
    batch_time = AverageMeter()
    clf_losses = AverageMeter()
    acc = AverageMeter()

    #switch to val mode
    model.eval()

    with torch.no_grad():

        end = time.time()
        for i, (video_feature, target, meta) in enumerate(val_loader):

            total_batch_size = target.shape[0]

            # initialization
            # use total batch-size instead of paralleled batch-size!!!
            ob = torch.zeros((total_batch_size, config.MODEL.FEATURE_DIM)).cuda()
            cost = torch.zeros((total_batch_size,)).cuda()
            cost_new = torch.zeros((total_batch_size,)).cuda()
            input = torch.zeros((total_batch_size, config.MODEL.MODALITY_NUM,
                                 config.MODEL.FRAMEDIV_NUM, config.MODEL.FEATURE_DIM))
            choice_his = torch.zeros((total_batch_size, config.MODEL.MODALITY_NUM, config.MODEL.FRAMEDIV_NUM)).cuda()
            if_finish = torch.zeros((total_batch_size)).cuda()
            clf_score = torch.zeros((total_batch_size, config.MODEL.CLFDIM)).cuda()
            clf_score_final = torch.zeros((total_batch_size, config.MODEL.CLFDIM)).cuda()

            target = target.cuda()

            # simulating until all instances have run out of cost limit.
            while ((cost <= config.MODEL.COST_LIMIT).sum() > 0):

                # decide action according to model's policy and current observation
                new_act_frame, new_act_modality, _ = model.module.policy(ob, cost, choice_his,
                                                                         if_random = False, if_val=True)

                # get the input
                # input = ChooseInput(video, new_act_frame, new_act_modality, meta['framenum'])
                input[range(total_batch_size), new_act_modality, new_act_frame, :] = video_feature[range(total_batch_size), new_act_modality, new_act_frame, :]

                # compute output
                clf_score_new, ob_new = model(input.cuda())

                # compute new cost
                for j in range(config.MODEL.MODALITY_NUM):
                    cost_new += (new_act_modality == j).type(torch.float) * config.MODEL.COST_LIST[j]

                print('reward = {}'.format(compute_reward(clf_score_new, clf_score, target, cost, cost_new)))

                # update ob, cost and clf_score, if_finish, choice_his
                ob = ob_new.clone()
                clf_score_final += clf_score * \
                                   ((cost_new > config.MODEL.COST_LIMIT).type(torch.float) - if_finish).view(-1, 1)
                clf_score = clf_score_new.clone()
                cost = cost_new.clone()
                if_finish = (cost_new > config.MODEL.COST_LIMIT).type(torch.float)
                choice_his[range(total_batch_size), new_act_modality, new_act_frame] = 1

            # update acc
            avg_acc = compute_acc(clf_score_final, target)
            acc.update(avg_acc, total_batch_size)

            # update total clf_loss
            loss = criterion.clf_loss(clf_score_final, target)
            clf_losses.update(loss, total_batch_size)
            # clf_losses.update(losses_batch.avg, losses_batch.count)

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
            if if_output_result:
                clf_score_final = torch.nn.functional.softmax(clf_score_final, dim=1).cpu().numpy()
                for j in range(total_batch_size):
                    temp = list(zip(list(zip(score_head, clf_score_final[j])),
                                    list(zip(label_head, valid_dataset.label_list))))
                    output_dict['results'][meta['name'][j]] = list(map(dict, temp))


    return acc.avg


def train_clf(config, train_loader, model, criterion, optimizer, epoch, transform = None, transform_gray = None):
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
    for i, (video_feature, target, meta) in enumerate(train_loader):
        # clear cache
        # torch.cuda.empty_cache()

        # measure data loading time
        data_time.update(time.time() - end)

        total_batch_size = target.shape[0]

        # unsorted sampling.
        if config.TRAIN_CLF.RANDOM_SAMPLE_NUM:
            config.TRAIN_CLF.SAMPLE_NUM = np.random.randint(config.TRAIN_CLF.SAMPLE_NUM_BOUND) + 1
            # config.TRAIN_CLF.SAMPLE_NUM = np.random.randint(5) + 1

        frame_chosen = torch.multinomial(
            input=torch.ones((config.MODEL.FRAMEDIV_NUM)) / config.MODEL.FRAMEDIV_NUM,
            num_samples=config.TRAIN_CLF.SAMPLE_NUM)
        modality_chosen = torch.multinomial(
            input=torch.ones((config.MODEL.MODALITY_NUM)) / config.MODEL.MODALITY_NUM,
            num_samples=config.TRAIN_CLF.SAMPLE_NUM, replacement=True)

        target = target.cuda(async=True)

        input = torch.zeros((total_batch_size, config.MODEL.MODALITY_NUM,
                             config.MODEL.FRAMEDIV_NUM, config.MODEL.FEATURE_DIM))
        input[:, modality_chosen, frame_chosen, :] = video_feature[:, modality_chosen, frame_chosen, :]
        # input = video_feature[:, 0: config.MODEL.MODALITY_NUM].clone()

        # compute the output

        clf_score, _ = model(input.cuda())

        # update acc
        avg_acc = compute_acc(clf_score, target)
        acc.update(avg_acc, total_batch_size)

        # compute loss
        # loss = criterion.clf_loss(clf_score, target)
        loss = criterion.focal_loss(clf_score, target)

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


def validate_clf(config, val_loader, model, criterion, epoch = 0, transform = None, transform_gray = None,
                 output_dict = None, valid_dataset = None):
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
    # whether to output the result
    if_output_result = (valid_dataset != None)
    label_head = ['label' for i in range(config.MODEL.CLFDIM)]
    score_head = ['score' for i in range(config.MODEL.CLFDIM)]

    # build recorders
    batch_time = AverageMeter()
    clf_losses = AverageMeter()
    acc = AverageMeter()

    # switch to val mode
    model.eval()

    with torch.no_grad():

        end = time.time()
        for i, (video_feature, target, meta) in enumerate(val_loader):

            total_batch_size = target.shape[0]

            if config.TRAIN_CLF.RANDOM_SAMPLE_NUM:
                config.TRAIN_CLF.SAMPLE_NUM = np.random.randint(config.TRAIN_CLF.SAMPLE_NUM_BOUND) + 1
                # config.TRAIN_CLF.SAMPLE_NUM = np.random.randint(5) + 1

            frame_chosen = torch.multinomial(
                input=torch.ones((config.MODEL.FRAMEDIV_NUM)) / config.MODEL.FRAMEDIV_NUM,
                num_samples=config.TRAIN_CLF.SAMPLE_NUM)
            # frame_chosen = (choose_frame_randomly(1, 10) * config.MODEL.FRAMEDIV_NUM).type(torch.long).view(-1)
            modality_chosen = torch.multinomial(
                input=torch.ones((config.MODEL.MODALITY_NUM)) / config.MODEL.MODALITY_NUM,
                num_samples=config.TRAIN_CLF.SAMPLE_NUM, replacement=True)

            target = target.cuda()

            input = torch.zeros((total_batch_size, config.MODEL.MODALITY_NUM,
                                 config.MODEL.FRAMEDIV_NUM, config.MODEL.FEATURE_DIM))
            input[:, modality_chosen, frame_chosen, :] = video_feature[:, modality_chosen, frame_chosen, :]
            # input = video_feature[:, 0: config.MODEL.MODALITY_NUM].clone()


            # compute the output

            clf_score, _ = model(input.cuda())

            # update acc
            avg_acc = compute_acc(clf_score, target)
            acc.update(avg_acc, total_batch_size)

            # compute loss
            loss = criterion.clf_loss(clf_score, target)

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

            if if_output_result:
                clf_score = torch.nn.functional.softmax(clf_score, dim=1).cpu().numpy()
                for j in range(total_batch_size):
                    temp = list(zip(list(zip(score_head, clf_score[j])),
                                    list(zip(label_head, valid_dataset.label_list))))
                    output_dict['results'][meta['name'][j]] = list(map(dict, temp))


    return acc.avg
