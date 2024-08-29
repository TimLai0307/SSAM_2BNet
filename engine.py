# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Train and eval functions used in main.py
Mostly copy-paste from DETR (https://github.com/facebookresearch/detr).
"""
import math
import os
import sys
from typing import Iterable

import torch

import util.misc as utils
from util.misc import NestedTensor
import numpy as np
import time
import torchvision.transforms as standard_transforms
import cv2
import torch.nn.functional as F

class DeNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor

# the training routine
def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0):
    model.train()
    criterion.train()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))

    for samples, targets in data_loader:
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        # forward
        outputs = model(samples)

        # calc the losses
        loss_dict = criterion(outputs, targets)

        weight_dict = criterion.weight_dict
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
        # reduce all losses
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

        loss_value = losses_reduced_scaled.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(losses))

            sys.exit(1)
        # backward
        optimizer.zero_grad()
        losses.backward()
        if max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()
        # update logger
        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

# the inference routine
@torch.no_grad()
def evaluate_crowd_no_overlap(model, data_loader, device):
    model.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    # run inference on all images to calc MAE
    maes_mean = []
    mses_mean = []
    maes_max = []
    mses_max = []
    maes_min = []
    mses_min = []

    maes1 = []
    maes2 = []

    for samples, targets in data_loader:
        samples = samples.to(device)
        b, c, h, w = samples.size()
        new_h = ((h - 1) // 128 + 1) * 128
        new_w = ((w - 1) // 128 + 1) * 128
        input = torch.zeros((b, c, new_h, new_w))
        input[:, :, : h, : w] = samples
        samples = input.to(device)

        outputs = model(samples)
        output1 = outputs[1][:, :, 0:int(h), 0:int(w)]
        output2 = outputs[2][:, :, 0:int(h), 0:int(w)]
        out1 = (output1 + output2) / 2
        out2 = torch.max(output1, output2)
        out3 = torch.min(output1, output2)

        gt_cnt = targets[0]['den_map'].sum()

        predict_cnt1 = out1.sum()
        predict_cnt2 = out2.sum()
        predict_cnt3 = out3.sum()
        predict_cnt4 = output1.sum()
        predict_cnt5 = output2.sum()

        # accumulate MAE, MSE
        mae_mean = abs(predict_cnt1 - gt_cnt)
        mse_mean = (predict_cnt1 - gt_cnt) * (predict_cnt1 - gt_cnt)
        maes_mean.append(float(mae_mean))
        mses_mean.append(float(mse_mean))

        mae_max = abs(predict_cnt2 - gt_cnt)
        mse_max = (predict_cnt2 - gt_cnt) * (predict_cnt2 - gt_cnt)
        maes_max.append(float(mae_max))
        mses_max.append(float(mse_max))

        mae_min = abs(predict_cnt3 - gt_cnt)
        mse_min = (predict_cnt3 - gt_cnt) * (predict_cnt3 - gt_cnt)
        maes_min.append(float(mae_min))
        mses_min.append(float(mse_min))

        mae1 = abs(predict_cnt4 - gt_cnt)
        mae2 = abs(predict_cnt5 - gt_cnt)
        maes1.append(float(mae1))
        maes2.append(float(mae2))

    # calc MAE, MSE
    mae_mean = np.mean(maes_mean)
    mse_mean = np.sqrt(np.mean(mses_mean))
    mae_max = np.mean(maes_max)
    mse_max = np.sqrt(np.mean(mses_max))
    mae_min = np.mean(maes_min)
    mse_min = np.sqrt(np.mean(mses_min))

    mae1 = np.mean(maes1)
    mae2 = np.mean(maes2)

    return mae_mean, mse_mean, mae_max, mse_max, mae_min, mse_min, mae1, mae2
