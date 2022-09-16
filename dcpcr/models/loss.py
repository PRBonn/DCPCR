import yaml
from collections import defaultdict
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint, Callback
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dcpcr.utils import utils


class LossHandler():
    def __init__(self,
                 w_rot=1.0,
                 w_trans=1.0,
                 w_point=0.5,
                 scale=40):
        self.w_rot = w_rot
        self.w_trans = w_trans
        self.w_point = w_point
        self.scale = scale
        self.l1_loss = nn.L1Loss()

    def getLoss(self,
                gt_pose,
                est_pose=None,
                target_points=None,
                corr_points=None,
                corr_weights=None):
        l_rot = 0 if est_pose is None else self.w_rot * \
            self.getRotloss(gt_pose, est_pose)
        l_trans = 0 if est_pose is None else self.w_trans * \
            self.l1_loss(gt_pose[..., :3, -1], est_pose[..., :3, -1])
        l_point = 0 if corr_points is None else self.w_point * \
            self.getPointLoss(gt_pose, target_points,
                              corr_points, corr_weights)
        loss = l_rot+l_trans+l_point
        return loss, {'l_rot': l_rot, 'l_trans': l_trans, 'l_point': l_point}

    def metrics():
        pass

    def getPointLoss(self, gt_pose, target, target_corr, corr_weights):
        target_corr_T = (
            gt_pose @ utils.makeHomogeneous(target_corr).transpose(-1, -2)).transpose(-1, -2)
        loss = self.l1_loss(corr_weights*target[..., :3],
                            corr_weights*target_corr_T[..., :3])
        return loss

    def getRotloss(self, x, y):
        identity = torch.eye(3, dtype=x.dtype, device=x.device).expand(
            x[..., :3, :3].shape)
        d_rot = x[..., :3, :3].transpose(-1, -2)@y[..., :3, :3]
        loss_rot = torch.diagonal(
            identity-d_rot, dim1=-2, dim2=-1).sum(-1).mean()
        return loss_rot


def arccos(x):
    if torch.is_tensor(x):
        return x.arccos()
    else:
        return np.arccos(x)


def batch_trace(x):
    return x.diagonal(0, -1, -2).sum(-1)


def pose_error(T1, T2, scale=1, dim=3):
    dt = ((T1[..., :dim, -1] - T2[..., :dim, -1])**2).sum(-1)**0.5 * scale

    dr = arccos(((batch_trace(
        T1[..., :dim, :dim].transpose(-1, -2) @ T2[..., :dim, :dim]) - 1)/2).clip(-1, 1))/math.pi*180
    return dt.mean(), dr.mean()


class ResultsSaver(Callback):
    def __init__(self, out_dir):
        self.metrics = defaultdict(list)
        self.out_dir = out_dir

    def on_test_batch_end(self,
                          trainer,
                          pl_module,
                          outputs,
                          batch,
                          batch_idx: int,
                          dataloader_idx: int):
        for key, value in outputs.items():
            self.metrics[key].append(value)

    def on_test_end(self, trainer, pl_module):
        with open(f'{self.out_dir}/results.yaml', 'w') as outfile:
            yaml.dump(dict(self.metrics), outfile)
