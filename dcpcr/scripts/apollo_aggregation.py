from shutil import copy
import os
import glob
import natsort
import numpy as np
from dcpcr.utils.scan_aggregator import ScanAggregatorSet
import open3d as o3d
import tqdm
from scipy.spatial.transform import Rotation as R
import pathlib

def readPoses(file):
    data = np.loadtxt(file)
    id, time, t, q = np.split(data, [1, 2, 5], axis=1)
    r = R.from_quat(q).as_matrix()
    pose = np.zeros([r.shape[0], 4, 4])
    pose[:, :3, -1] = t
    pose[:, :3, :3] = r
    pose[:, -1, -1] = 1
    return pose, np.squeeze(id), np.squeeze(time)

def parse_scan(file):
    pcd = o3d.io.read_point_cloud(file)
    return pcd

if __name__ == '__main__':    
    ## Edit this, do this for all Folders!
    folder = 'path/to/your/data/apollo/MapData/ColumbiaPark/2018-09-21/1'
    out_dir = 'path/to/your/data/apollo-aggregated2/MapData/ColumbiaPark/2018-09-21/1/submaps/'
    
    ###
    scan_files = natsort.natsorted(glob.glob(f'{folder}/pcds/*.pcd'))
    pathlib.Path(out_dir).mkdir(parents=True, exist_ok=True)

    if os.path.exists(f'{folder}/poses/init_poses.txt'):
        init_poses, _, _ = readPoses(f'{folder}/poses/init_poses.txt')
        np.savetxt(out_dir+'init_poses.txt', init_poses.reshape([-1, 16]))
        copy(f'{folder}/poses/valid_range.txt', out_dir)
    poses, _, _ = readPoses(f'{folder}/poses/gt_poses.txt')
    np.savetxt(out_dir+'poses.txt', poses.reshape([-1, 16]))
    poses = list(poses)
    dataset = ScanAggregatorSet(scan_files=scan_files, poses=poses, parse_scan=parse_scan,
                                sliding_window_size=20, step=2, bb_size=40, voxel_res=0.1)
    for i in tqdm.tqdm(range(0, len(dataset))):
        scan, file = dataset[i]
        file = os.path.join(out_dir, str(i).zfill(6)+'.ply')
        o3d.io.write_point_cloud(file, scan)
