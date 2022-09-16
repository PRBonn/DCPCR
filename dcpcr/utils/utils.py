import os
import torch
import numpy as np
from scipy.spatial.transform import Rotation
import open3d as o3d

CONFIG_DIR = os.path.abspath(os.path.join(
    os.path.dirname(__file__), '../config/'))
DATA_DIR = os.path.abspath(os.path.join(
    os.path.dirname(__file__), '../data/'))
EXPERIMENT_DIR = os.path.abspath(os.path.join(
    os.path.dirname(__file__), '../experiments/'))


def dict2object(dict_):
    assert isinstance(dict_,dict)
    class_ = eval(dict_['class'])
    init_params = class_.__init__.__code__.co_varnames
    # print(f'init vars: \n {init_params}')
    params = {k: dict_[k] for k in dict_ if k in init_params}
    # print(params)
    return class_(**params)


def insideRandBB(pts, scale, xy_translation: list):
    translation = xy_translation + [0.]
    p = np.ones([pts.shape[0], 3])
    p[:, :2] = pts[:, :2]
    rot = Rotation.from_euler(
        'z', np.random.rand()*90, degrees=True).as_matrix()

    bb = np.array([
        [0, -1., 1],
        [0, 1, 1],
        [1, 0, 1],
        [-1, 0, 1]
    ]).T
    bb[-1, :] = bb[-1, :]*scale
    bb = rot@bb
    t = (np.random.rand(3)*2-1)*np.array(translation)
    inside = (p-t)@bb
    inside = np.all(inside > 0, axis=-1, keepdims=True)
    return inside


def makeHomogeneous(p):
    shape = list(p.shape)
    shape[-1] = 1
    ps_t = torch.cat([p[..., :, :3], torch.ones(
        shape, device=p.device, dtype=p.dtype)], -1)
    return ps_t


def nanstd(t: torch.Tensor, dim):
    m = torch.nanmean(t, dim, keepdim=True)
    tc = (t - m)**2

    tc[torch.isnan(t)] = 0
    w = (~torch.isnan(t)).sum(dim)-1
    v = tc.sum(dim)/w
    return v.sqrt()


def pad(array, n_points=2000, pad=True, shuffle = False):
    """ array [n x m] -> [n_points x m],
        output:
            array [*,n_points x m], padded with zeros
            mask [*,n_points x 1], 1 if valid, 0 if not valid 
    """
    if shuffle:
        sample_idx = np.random.choice(array.shape[-2], n_points, replace=False)
        array = array[...,sample_idx,:]
    if not pad:
        return array, np.ones(array.shape[:-1]+(1,), dtype='bool')
    if len(array.shape) == 2:
        size = list(array.shape)
        size[-2] = n_points
        out = np.zeros(size, dtype='float32')
        l = min(n_points, array.shape[-2])
        out[:l, :] = array[:l, :]

        size[-1] = 1
        mask = np.zeros(size, dtype='bool')
        mask[:l, :] = 1
        return out, mask
    else:
        size = list(array.shape)
        size[-2] = n_points
        out = np.zeros(size, dtype='float32')
        l = min(n_points, array.shape[-2])
        out[..., :l, :] = array[..., :l, :]
        size[-1] = 1
        mask = np.zeros(size, dtype='bool')
        mask[..., :l, :] = 1
        return out, mask


def torch2o3d(pcd, colors=None, estimate_normals=False):
    pcd = pcd.detach().cpu().squeeze().numpy() if isinstance(pcd, torch.Tensor) else pcd
    
    assert len(pcd.shape) <= 2, "Batching not implemented"
    colors = colors.detach().cpu().squeeze().numpy() if isinstance(
        colors, torch.Tensor) else colors
    pcl = o3d.geometry.PointCloud()
    pcl.points = o3d.utility.Vector3dVector(pcd[:,:3])
    if estimate_normals:
        pcl.estimate_normals()
    if colors is not None:
        pcl.colors = o3d.utility.Vector3dVector(colors)
    return pcl
