import numpy as np
import open3d as o3d
from math import floor

import open3d as o3d
import numpy as np
from torch.utils.data import Dataset


class ScanAggregatorSet(Dataset):
    def __init__(self, scan_files, poses, parse_scan, sliding_window_size=20,  step=2, bb_size=40, voxel_res=0.1):
        super().__init__()
        self.sliding_window_size = sliding_window_size
        self.scans = scan_files
        self.poses = poses
        self.nr_scans = len(self.scans)
        self.aggregator = ScanAggregator(bb_size=bb_size, voxel_size=voxel_res,
                                         sliding_window_size=sliding_window_size)
        self.step = step
        self.parse_scan = parse_scan

    def __getitem__(self, index):

        min_id = max(0, index-self.sliding_window_size+1)
        max_id = min(self.nr_scans, index+self.sliding_window_size+1)
        poses = self.poses[min_id:index:self.step]
        poses += self.poses[index+self.step:max_id:self.step]
        poses += [self.poses[index]]
        scans = [self.parse_scan(self.scans[idx])
                 for idx in range(min_id, index, self.step)]
        scans += [self.parse_scan(self.scans[idx])
                  for idx in range(index+self.step, max_id, self.step)]
        scans += [self.parse_scan(self.scans[index])]
        scan = self.aggregator.aggregateBB(
            scans, poses, poses[-1])
        file = self.scans[index]
        return scan, file

    def __len__(self):
        return self.nr_scans


def remove_roof_points(cloud, min_depth):
    rth = min_depth / 2
    zth = 1.0
    bbox = o3d.geometry.AxisAlignedBoundingBox(
        min_bound=(-rth, -rth, -zth), max_bound=(+rth, +rth, +0)
    )
    ind = bbox.get_point_indices_within_bounding_box(cloud.points)
    far_points = cloud.select_by_index(ind, invert=True)
    near_points = cloud.select_by_index(ind)
    return far_points, near_points, bbox


def crop_cloud_with_bbox(cloud, z_level):
    bbox = o3d.geometry.AxisAlignedBoundingBox(
        min_bound=(-30, -30, -z_level), max_bound=(+30, +30, +z_level)
    )
    return cloud.crop(bbox), bbox


def preprocess_cloud(cloud, z_level=2.0, min_depth=5.5):
    cloud, _ = crop_cloud_with_bbox(cloud, z_level)
    cloud, _, _ = remove_roof_points(cloud, min_depth)
    return cloud


def preprocess_fcloud(filename, z_level=2.0, min_depth=5.5):
    cloud = o3d.io.read_point_cloud(filename)
    return preprocess_cloud(cloud, z_level, min_depth)


class ScanAggregator():
    def __init__(self, bb_size=40,
                 voxel_size=0.1,
                 sliding_window_size=5,
                 min_z=-10.5,
                 min_dist=3.0):
        self.initialized = False
        self.scans = []
        self.poses = []
        self.index = 0
        self.min_dist = min_dist
        self.sliding_window_size = sliding_window_size
        self.voxel_size = voxel_size
        self.bb_size = bb_size

        bb_min = np.array([-bb_size/2, -bb_size/2, min_z])
        self.bb = o3d.geometry.AxisAlignedBoundingBox(
            bb_min, np.full([3, 1], bb_size/2))

    def reset(self):
        self.scans = []
        self.poses = []
        self.index = 0

    def aggregateBB(self, scans: list, poses: list, target_pose):
        pcl = o3d.geometry.PointCloud()
        inv_pose = np.linalg.inv(target_pose)

        for i in range(len(scans)):
            scan = preprocess_cloud(scans[i])
            pose = inv_pose @ poses[i]
            scan.transform(pose)
            pcl += scan.crop(self.bb)
        pcl = pcl.voxel_down_sample(self.voxel_size)
        return pcl

    def aggregate(self,
                  scans=None,
                  poses=None):
        idx_mid = 0
        if scans is None:
            scans = self.scans
            poses = self.poses
            n = len(poses)
            idx_mid = (self.index+floor(n/2)) % n
        else:
            n = len(poses)
            idx_mid = floor(n/2) % n
        pcl = o3d.geometry.PointCloud()

        translate = np.zeros([4, 4])
        translate[:-1, -1] = poses[idx_mid][:-1, -1]
        for i in range(len(scans)):
            scan = o3d.geometry.PointCloud()
            valids = (np.abs(scans[i]) > self.min_dist).any(
                axis=1)  # remove points which are to near
            scan.points = o3d.utility.Vector3dVector(scans[i][valids, :])
            pose = poses[i] - translate
            scan.transform(pose)
            pcl += scan.crop(self.bb)
        pcl.voxel_down_sample(self.voxel_size)
        # pcl =pcl.crop(self.bb)
        return pcl

    def addScan(self, scan, pose):
        if len(self.poses) < (self.sliding_window_size):
            self.poses.append(pose)
            self.scans.append(scan)
        else:
            self.index += 1
            n = len(self.poses)
            idx = self.index % n
            self.scans[idx] = scan
            self.poses[idx] = pose

    def AddAndAggregate(self, scan, pose):
        self.addScan(scan, pose)
        return self.aggregate()
