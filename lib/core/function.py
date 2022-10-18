# ------------------------------------------------------------------------------
# Licensed under the MIT License.
# Created by Tianheng Cheng(tianhengcheng@gmail.com)
# Modified by Sierkinhane(sierkinhane@163.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import logging

import torch
import numpy as np

from .evaluation import decode_preds, compute_nme, compute_niou, decode_preds_vis

logger = logging.getLogger(__name__)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
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
        self.avg = self.sum / self.count


def train(config, train_loader, model, criterion, optimizer,
          epoch, device, writer_dict):

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    def compute_loss(output, target):
        loss = 0
        if isinstance(output, list):
            for i in range(len(output)):
                l = criterion(output[i], target)
                loss += l
        else:
            loss = criterion(output, target)

        return loss
    model.train()
    nme_count = 0
    nme_batch_sum = 0

    end = time.time()

    for i, (inp, target, _, meta) in enumerate(train_loader):
        # measure data time
        data_time.update(time.time()-end)
        inp = inp.to(device)
        target = target.to(device)

        # compute the output
        outputs = model(inp)
        loss = compute_loss(outputs, target)

        if isinstance(outputs, list):
            output = outputs[-1]
        else:
            output = outputs

        # NME
        score_map = output.data.cpu()
        preds = decode_preds(config, score_map, meta['center'], meta['scale'], [64, 64])

        nme_batch = compute_nme(config, preds, meta)
        nme_batch_sum = nme_batch_sum + np.sum(nme_batch)
        nme_count = nme_count + preds.size(0)

        # optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.update(loss.item(), inp.size(0))

        batch_time.update(time.time()-end)
        if i % config.PRINT_FREQ == 0:
            msg = 'Epoch: [{0}][{1}/{2}]\t' \
                  'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                  'Speed {speed:.1f} samples/s\t' \
                  'Data {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
                  'Loss {loss.val:.5f} ({loss.avg:.5f})\t'.format(
                      epoch, i, len(train_loader), batch_time=batch_time,
                      speed=inp.size(0)/batch_time.val,
                      data_time=data_time, loss=losses)
            logger.info(msg)

            if writer_dict:
                writer = writer_dict['writer']
                global_steps = writer_dict['train_global_steps']
                writer.add_scalar('train_loss', losses.val, global_steps)
                writer_dict['train_global_steps'] = global_steps + 1

        end = time.time()
    nme = nme_batch_sum / nme_count
    msg = 'Train Epoch {} time:{:.4f} loss:{:.4f} nme:{:.4f}'\
        .format(epoch, batch_time.avg, losses.avg, nme)
    logger.info(msg)


def validate(config, val_loader, model, criterion, epoch, device, writer_dict):
    batch_time = AverageMeter()
    data_time = AverageMeter()

    losses = AverageMeter()
    def compute_loss(output, target):
        loss = 0
        if isinstance(output, list):
            for i in range(len(output)):
                l = criterion(output[i], target)
                loss += l
        else:
            loss = criterion(output, target)

        return loss
    model.eval()

    nme_count = 0
    nme_batch_sum = 0
    count_failure_008 = 0
    count_failure_010 = 0
    end = time.time()

    with torch.no_grad():
        for i, (inp, target, _, meta) in enumerate(val_loader):
            data_time.update(time.time() - end)
            inp = inp.to(device)
            target = target.to(device)
            outputs = model(inp)
            loss = compute_loss(outputs, target)
            if isinstance(outputs, list):
                output = outputs[-1]
            else:
                output = outputs
            score_map = output.data.cpu()
            # loss
            preds = decode_preds(config, score_map, meta['center'], meta['scale'], [64, 64])
            # NME
            nme_temp = compute_nme(config, preds, meta)
            # Failure Rate under different threshold
            failure_008 = (nme_temp > 0.08).sum()
            failure_010 = (nme_temp > 0.10).sum()
            count_failure_008 += failure_008
            count_failure_010 += failure_010

            nme_batch_sum += np.sum(nme_temp)
            nme_count = nme_count + preds.size(0)
            # for n in range(score_map.size(0)):
            #     predictions[meta['index'][n], :, :] = preds[n, :, :]

            losses.update(loss.item(), inp.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

    nme = nme_batch_sum / nme_count
    failure_008_rate = count_failure_008 / nme_count
    failure_010_rate = count_failure_010 / nme_count

    msg = 'Test Epoch {} time:{:.4f} loss:{:.4f} nme:{:.4f} [008]:{:.4f} ' \
          '[010]:{:.4f}'.format(epoch, batch_time.avg, losses.avg, nme,
                                failure_008_rate, failure_010_rate)
    logger.info(msg)

    if writer_dict:
        writer = writer_dict['writer']
        global_steps = writer_dict['valid_global_steps']
        writer.add_scalar('valid_loss', losses.avg, global_steps)
        writer.add_scalar('valid_nme', nme, global_steps)
        writer_dict['valid_global_steps'] = global_steps + 1

    return nme

def train_SCBE(config, train_loader, model, criterion, optimizer,
          epoch, device, writer_dict):

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    def compute_loss(output, target):
        loss = 0
        if isinstance(output, list):
            for i in range(len(output)):
                l = criterion(output[i], target)
                loss += l
        else:
            loss = criterion(output, target)

        return loss

    model.train()
    end = time.time()
    for i, (inp, _, target, meta) in enumerate(train_loader):
        # measure data time
        data_time.update(time.time()-end)

        # compute the output
        inp = inp.to(device)
        target = target.to(device)
        output, _, _ = model(inp)

        loss = compute_loss(output, target)

        # optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.update(loss.item(), inp.size(0))

        batch_time.update(time.time()-end)
        if i % config.PRINT_FREQ == 0:
            msg = 'Epoch: [{0}][{1}/{2}]\t' \
                  'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                  'Speed {speed:.1f} samples/s\t' \
                  'Data {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
                  'Loss {loss.val:.5f} ({loss.avg:.5f})\t'.format(
                      epoch, i, len(train_loader), batch_time=batch_time,
                      speed=inp.size(0)/batch_time.val,
                      data_time=data_time, loss=losses)
            logger.info(msg)

            if writer_dict:
                writer = writer_dict['writer']
                global_steps = writer_dict['train_global_steps']
                writer.add_scalar('train_loss', losses.val, global_steps)
                writer_dict['train_global_steps'] = global_steps + 1

        end = time.time()


def validate_SCBE(val_loader, model, criterion, epoch, device, writer_dict):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    model.eval()
    end = time.time()
    with torch.no_grad():
        for i, (inp, _, target, meta) in enumerate(val_loader):
            data_time.update(time.time() - end)
            inp = inp.to(device)
            target = target.to(device)
            outputs, _, _ = model(inp)

            # loss
            loss = criterion(outputs[-1], target)
            losses.update(loss.item(), inp.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

    msg = 'Test Epoch {} time:{:.4f} loss:{:.4f} '.format(epoch, batch_time.avg, losses.avg, )
    logger.info(msg)

    if writer_dict:
        writer = writer_dict['writer']
        global_steps = writer_dict['valid_global_steps']
        writer.add_scalar('valid_loss', losses.avg, global_steps)
        writer_dict['valid_global_steps'] = global_steps + 1

    return losses.avg

def validate_Boundary_niou(config, val_loader, model, criterion, epoch, device, writer_dict):
    batch_time = AverageMeter()
    data_time = AverageMeter()

    losses = AverageMeter()
    nious = AverageMeter()

    model.eval()
    end = time.time()
    with torch.no_grad():
        for i, (inp, _, target, meta) in enumerate(val_loader):
            data_time.update(time.time() - end)
            inp = inp.to(device)
            target = target.to(device)
            outputs, _, _ = model(inp)

            # loss
            loss = criterion(outputs[-1], target)
            losses.update(loss.mean().item(), inp.size(0))

            # normalized iou
            niou = compute_niou(outputs[-1].cpu().numpy(), target.cpu().numpy(), loss.mean(dim=[1,2,3]).cpu().numpy())
            nious.update(niou, inp.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

    msg = 'Test Epoch {} time:{:.4f} loss:{:.6f} niou:{:.6f} '.format(epoch, batch_time.avg, losses.avg, nious.avg)
    logger.info(msg)

    if writer_dict:
        writer = writer_dict['writer']
        global_steps = writer_dict['valid_global_steps']
        writer.add_scalar('valid_loss', losses.avg, global_steps)
        writer_dict['valid_global_steps'] = global_steps + 1

    return losses.avg, nious.avg

def inference(config, data_loader, model, device, flag):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    model.eval()

    nme_for_auc = []
    nme_count = 0
    nme_batch_sum = 0
    count_failure_008 = 0
    count_failure_010 = 0

    end = time.time()
    with torch.no_grad():
        for i, (inp, target, _, meta) in enumerate(data_loader):
            data_time.update(time.time() - end)

            inp = inp.to(device)
            output = model(inp)
            score_map = output.data.cpu()
            preds, vis = decode_preds_vis(config, score_map, meta['center'], meta['scale'], [64, 64])

            # NME
            nme_temp = compute_nme(config, preds, meta)
            nme_for_auc += list(nme_temp)

            failure_008 = (nme_temp > 0.08).sum()
            failure_010 = (nme_temp > 0.10).sum()
            count_failure_008 += failure_008
            count_failure_010 += failure_010

            nme_batch_sum += np.sum(nme_temp)
            nme_count = nme_count + preds.size(0)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

    nme = nme_batch_sum / nme_count
    failure_008_rate = count_failure_008 / nme_count
    failure_010_rate = count_failure_010 / nme_count

    msg = 'Test Results time:{:.4f} loss:{:.4f} nme:{:.4f} [008]:{:.4f} ' \
          '[010]:{:.4f}'.format(batch_time.avg, losses.avg, nme,
                                failure_008_rate, failure_010_rate)
    logger.info(msg)

    return nme, nme_for_auc



