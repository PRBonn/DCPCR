from time import time
import open3d as o3d
import torch
import torch.nn as nn
import numpy as np
from dcpcr.utils import utils


def refine_registration(source, target, initial_guess, distance_threshold=0.8):
    kernel_threshold = 0.3 * distance_threshold
    robust_kernel = o3d.pipelines.registration.GMLoss(kernel_threshold)

    gcp = o3d.pipelines.registration.TransformationEstimationForGeneralizedICP(
        robust_kernel)
    result = o3d.pipelines.registration.registration_generalized_icp(
        source, target, distance_threshold, initial_guess, gcp)
    return result.transformation


class RegistrationTuner():
    def __init__(self,
                 distance_threshold=1,
                 scale=40,
                 compressed=True,
                 knn=25,
                 replace = None,
                 verbose=False
                 ):
        self.distance_threshold = distance_threshold
        self.scale = scale
        self.compressed = compressed
        self.knn = knn
        self.replace = replace
        
        self.verbose = verbose
        self.time = 0
        self.i = 0

    def refine_registration(self, batch, est_pose):
        if self.compressed:
            dt = self.distance_threshold / self.scale
            source = utils.torch2o3d(
                batch['source'][0, batch['mask_source'][0, :, 0]])
            target = utils.torch2o3d(
                batch['target'][0, batch['mask_target'][0, :, 0]])
            init_guess = est_pose.detach().cpu().squeeze().numpy()
            t = time()
            source.estimate_normals(
                o3d.geometry.KDTreeSearchParamKNN(self.knn))
            target.estimate_normals(
                o3d.geometry.KDTreeSearchParamKNN(self.knn))
        else:
            assert self.replace is not None
            file = batch['file_source'][0]
            for r1 in self.replace:
                file = file.replace(r1, self.replace[r1])
            source = o3d.io.read_point_cloud(file)

            file = batch['file_target'][0]
            for r1 in self.replace:
                file = file.replace(r1, self.replace[r1])
            target = o3d.io.read_point_cloud(file)

            
            init_guess = est_pose.squeeze().detach().cpu().numpy()
            init_guess[:3, -1] *= self.scale
            dt = self.distance_threshold
            t = time()

        pose = refine_registration(source,
                                   target,
                                   init_guess,
                                   distance_threshold=dt)
        self.time += time()-t
        self.i+=1
        if self.verbose:
            print(self.time/self.i)
        
        est_pose = torch.tensor(pose,
                                device=est_pose.device,
                                dtype=est_pose.dtype)
        if not self.compressed:
            est_pose[:3, -1] /= self.scale
        return est_pose
