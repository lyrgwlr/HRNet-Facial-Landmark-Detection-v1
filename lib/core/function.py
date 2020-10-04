# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Created by Tianheng Cheng(tianhengcheng@gmail.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .evaluation import decode_preds, compute_nme, decode_preds_from_soft_argmax, compute_auc

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

def _assert_no_grad(tensor):
    assert not tensor.requires_grad, \
        "nn criterions don't compute the gradient w.r.t. targets - please " \
        "mark these tensors as not requiring gradients"



def soft_argmax_(heatmaps, joint_num):
    assert isinstance(heatmaps, torch.Tensor)
    h ,w = heatmaps.shape[-2], heatmaps.shape[-1]
    heatmaps = heatmaps.reshape((-1, joint_num, h*w))
    heatmaps = F.softmax(heatmaps, 2)
    heatmaps = heatmaps.reshape((-1, joint_num, h, w))

    accu_x = heatmaps.sum(dim=2)
    accu_y = heatmaps.sum(dim=3)
    

    accu_x = accu_x * torch.cuda.comm.broadcast(torch.arange(1,w+1).type(torch.cuda.FloatTensor), devices=[accu_x.device.index])[0]
    accu_y = accu_y * torch.cuda.comm.broadcast(torch.arange(1,h+1).type(torch.cuda.FloatTensor), devices=[accu_y.device.index])[0]
    

    accu_x = accu_x.sum(dim=2, keepdim=True)-1
    accu_y = accu_y.sum(dim=2, keepdim=True)-1
    

    coord_out = torch.cat((accu_x, accu_y), dim=2) #(B,c,2)

    return coord_out


def train(config, train_loader, model, critertion, optimizer,
          epoch, writer_dict, mse):

    batch_time = AverageMeter()
    data_time = AverageMeter()
    
    hm_losses = AverageMeter()
    l1_losses = AverageMeter()
    losses = AverageMeter()    

    model.train()
    nme_count = 0
    nme_batch_sum = 0
    nme_batchs = []

    end = time.time()

    for i, (inp, target, meta) in enumerate(train_loader):
        # measure data time
        data_time.update(time.time()-end)

        # compute the output
        output = model(inp)
        target = target.cuda(non_blocking=True)
        score_map = output.data.cpu()
        if mse:
            hm_loss = critertion(output, target)
            l1_loss = torch.zeros([1])
            loss = hm_loss
            # NME    
            preds = decode_preds(score_map, meta['center'], meta['scale'], [64, 64])
        else:
            # heatmap, preds_ = soft_argmax(output, output.shape[1])  # the coord in 64*64 resolution
            hm_loss, l1_loss, preds_ = critertion(output, target, meta['tpts_float'].cuda(non_blocking=True))
            loss = hm_loss + l1_loss
            preds = decode_preds_from_soft_argmax(preds_.data.cpu(), meta['center'], meta['scale'], [64, 64])

        nme_batch = compute_nme(preds, meta)
        nme_batchs.extend(nme_batch)

        nme_batch_sum = nme_batch_sum + np.sum(nme_batch)
        nme_count = nme_count + preds.size(0)

        # optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        
        hm_losses.update(hm_loss.item(), inp.size(0))
        l1_losses.update(l1_loss.item(), inp.size(0))
         
        losses.update(loss.item(), inp.size(0))

        batch_time.update(time.time()-end)
        if i % config.PRINT_FREQ == 0:
            msg = 'Epoch: [{0}][{1}/{2}]\t' \
                  'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                  'Speed {speed:.1f} samples/s\t' \
                  'Data {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
                  'hm_loss {hm_loss.val:.5f} ({hm_loss.avg:.5f})\t'  \
                  'l1_loss {l1_loss.val:.5f} ({l1_loss.avg:.5f})\t'  \
                  'Loss {loss.val:.5f} ({loss.avg:.5f})\t'.format(
                      epoch, i, len(train_loader), batch_time=batch_time,
                      speed=inp.size(0)/batch_time.val,
                      data_time=data_time, hm_loss=hm_losses, l1_loss=l1_losses, loss=losses)
            logger.info(msg)

            if writer_dict:
                writer = writer_dict['writer']
                global_steps = writer_dict['train_global_steps']
                writer.add_scalar('train_loss', losses.val, global_steps)
                writer_dict['train_global_steps'] = global_steps + 1

        end = time.time()
    nme = nme_batch_sum / nme_count
    auc = compute_auc(nme_batchs)
    msg = 'Train Epoch {} time:{:.4f} hm_loss:{:.4f} l1_loss:{:.4f} loss:{:.4f} nme:{:.4f} auc:{:.4f}'\
        .format(epoch, batch_time.avg, hm_losses.avg, l1_losses.avg, losses.avg, nme, auc)
    logger.info(msg)
    return nme, auc



def validate(config, val_loader, model, critertion, epoch, writer_dict, mse):
    batch_time = AverageMeter()
    data_time = AverageMeter()

    losses = AverageMeter()

    num_classes = config.MODEL.NUM_JOINTS
    predictions = torch.zeros((len(val_loader.dataset), num_classes, 2))

    model.eval()

    nme_count = 0
    nme_batch_sum = 0
    count_failure_008 = 0
    count_failure_010 = 0
    nme_batchs = []
    end = time.time()

    with torch.no_grad():
        for i, (inp, target, meta) in enumerate(val_loader):
            data_time.update(time.time() - end)
            output = model(inp)
            target = target.cuda(non_blocking=True)
            score_map = output.data.cpu()
            if mse:
                loss = critertion(output, target)
                # NME
                preds = decode_preds(score_map, meta['center'], meta['scale'], [64, 64])
            else:
                # heatmap, preds_ = soft_argmax(output, output.shape[1])  # the coord in 64*64 resolution
                loss, preds_ = critertion(output, target, meta['tpts_float'].cuda(non_blocking=True))
                preds = decode_preds_from_soft_argmax(preds_.data.cpu(), meta['center'], meta['scale'], [64, 64])


            # NME
            nme_temp = compute_nme(preds, meta)

            nme_batchs.extend(nme_temp)
            # Failure Rate under different threshold
            failure_008 = (nme_temp > 0.08).sum()
            failure_010 = (nme_temp > 0.10).sum()
            count_failure_008 += failure_008
            count_failure_010 += failure_010

            nme_batch_sum += np.sum(nme_temp)
            nme_count = nme_count + preds.size(0)
            for n in range(score_map.size(0)):
                predictions[meta['index'][n], :, :] = preds[n, :, :]

            losses.update(loss.item(), inp.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

    nme = nme_batch_sum / nme_count
    failure_008_rate = count_failure_008 / nme_count
    failure_010_rate = count_failure_010 / nme_count
    auc = compute_auc(nme_batchs)

    msg = 'Test Epoch {} time:{:.4f} loss:{:.4f} nme:{:.4f} [008]:{:.4f} ' \
          '[010]:{:.4f} auc:{:.4f}'.format(epoch, batch_time.avg, losses.avg, nme,
                                failure_008_rate, failure_010_rate, auc)
    logger.info(msg)

    if writer_dict:
        writer = writer_dict['writer']
        global_steps = writer_dict['valid_global_steps']
        writer.add_scalar('valid_loss', losses.avg, global_steps)
        writer.add_scalar('valid_nme', nme, global_steps)
        writer_dict['valid_global_steps'] = global_steps + 1

    return nme, auc, predictions

# def inference(config, data_loader, model):
#     batch_time = AverageMeter()
#     data_time = AverageMeter()
#     losses = AverageMeter()

#     num_classes = config.MODEL.NUM_JOINTS
#     predictions = torch.zeros((len(data_loader.dataset), num_classes, 2))

#     model.eval()

#     nme_count = 0
#     nme_batch_sum = 0
#     count_failure_008 = 0
#     count_failure_010 = 0
#     end = time.time()

#     with torch.no_grad():
#         for i, (inp, target, meta) in enumerate(data_loader):
#             data_time.update(time.time() - end)
#             output = model(inp)
#             score_map = output.data.cpu()
#             preds = decode_preds(score_map, meta['center'], meta['scale'], [64, 64])

#             # NME
#             nme_temp = compute_nme(preds, meta)

#             failure_008 = (nme_temp > 0.08).sum()
#             failure_010 = (nme_temp > 0.10).sum()
#             count_failure_008 += failure_008
#             count_failure_010 += failure_010

#             nme_batch_sum += np.sum(nme_temp)
#             nme_count = nme_count + preds.size(0)
#             for n in range(score_map.size(0)):
#                 predictions[meta['index'][n], :, :] = preds[n, :, :]

#             # measure elapsed time
#             batch_time.update(time.time() - end)
#             end = time.time()

#     nme = nme_batch_sum / nme_count
#     failure_008_rate = count_failure_008 / nme_count
#     failure_010_rate = count_failure_010 / nme_count

#     msg = 'Test Results time:{:.4f} loss:{:.4f} nme:{:.4f} [008]:{:.4f} ' \
#           '[010]:{:.4f}'.format(batch_time.avg, losses.avg, nme,
#                                 failure_008_rate, failure_010_rate)
#     logger.info(msg)

#     return nme, predictions

def inference(config, data_loader, model, out_path, mse):
    batch_time = AverageMeter()
    data_time = AverageMeter()

    num_classes = config.MODEL.NUM_JOINTS
 

    model.eval()
    end = time.time()
    with torch.no_grad():
        for i, (inp, meta) in enumerate(data_loader):
        
            data_time.update(time.time() - end)

            output = model(inp)
            # score_map = output.data.cpu()
            # # preds: N x L x 2
            # preds = decode_preds(score_map, meta['center'], meta['scale'], [64, 64])

            score_map = output.data.cpu()
            if mse:
                # NME
                preds = decode_preds(score_map, meta['center'], meta['scale'], [64, 64])
            else:
                preds_ = soft_argmax_(output, output.shape[1])  # the coord in 64*64 resolution  
                preds = decode_preds_from_soft_argmax(preds_.data.cpu(), meta['center'], meta['scale'], [64, 64])

            for n in range(score_map.size(0)):
                name = meta['img_name'][n].split('.')[0]+'.txt'
                per_path = os.path.join(out_path, 'res', name)

                with open(per_path, 'w') as f:
                    f.write('106\n')
                    for k in range(preds.shape[1]):
                        x, y = preds[n, k, :]
                        x, y = round(x.item()), round(y.item())
                        f.write('{} {}\n'.format(x, y))
                # predictions[meta['index'][n], :, :] = preds[n, :, :]

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

    msg = 'Test Results time:{:.4f}'.format(batch_time.avg)
    logger.info(msg)


