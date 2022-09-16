import open3d as o3d
from os.path import join
import tqdm
from dcpcr.utils import cache
import numpy as np
import os
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning import LightningDataModule
import glob
import dcpcr.utils.utils as utils


def dict2object(dict_):
    assert isinstance(dict_, dict)
    class_ = eval(dict_['class'])
    init_params = class_.__init__.__code__.co_varnames
    params = {k: dict_[k] for k in dict_ if k in init_params}
    return class_(**params)

###############


class DataModule(LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

    def prepare_data(self):
        # Augmentations
        pass

    def setup(self, stage=None):
        # Create datasets
        pass

    def val_dataset(self):
        return dict2object(self.cfg['val'])

    def train_dataloader(self, batch_size=None):
        batch_size = self.cfg['batch_size'] if batch_size is None else batch_size
        data_set = dict2object(self.cfg['train'])
        loader = DataLoader(data_set, batch_size=batch_size, shuffle=True,
                            num_workers=self.cfg['num_worker'])
        return loader

    def val_dataloader(self, batch_size=None):
        batch_size = self.cfg['batch_size'] if batch_size is None else batch_size
        data_set = dict2object(self.cfg['val'])
        loader = DataLoader(data_set, batch_size=batch_size,
                            num_workers=self.cfg['num_worker'])
        return loader

    def test_dataloader(self, batch_size=None):
        batch_size = self.cfg['batch_size'] if batch_size is None else batch_size
        print(self.cfg['test'])
        data_set = dict2object(self.cfg['test'])
        loader = DataLoader(data_set, batch_size=batch_size,
                            num_workers=self.cfg['num_worker'])
        return loader


#################################################
################## Data loader ##################
#################################################


class Map2Map(Dataset):
    def __init__(self,
                 map_dirs,
                 src_dirs,
                 max_pose_dist=10,
                 validation=False,
                 mask_validation=False,
                 pad=False,
                 use_cache=True,
                 file_format='.ply',
                 num_points_pad=2000,
                 scale=1,
                 shuffle=False
                 ):
        super().__init__()
        self.use_cache = use_cache
        self.cache = cache.get_cache(directory=utils.DATA_DIR)

        self.mask_validation = mask_validation
        self.validation = validation
        self.pad = lambda x: utils.pad(
            x, n_points=num_points_pad, pad=pad, shuffle=shuffle)
        self.file_format = file_format
        self.scale = scale

        self.src_dirs = src_dirs
        self.src_poses = self.loadPoses(self.src_dirs, mask=mask_validation)
        self.src_files = self.loadFiles(self.src_dirs, mask=mask_validation)

        self.map_dirs = map_dirs
        self.map_poses = self.loadPoses(self.map_dirs, mask=False)
        self.map_files = self.loadFiles(self.map_dirs, mask=False)

        self.corr, _ = self.computeMapCorrespondences(
            max_pose_dist, src_dirs, map_dirs, mask_validation, validation)

        valid = self.corr >= 0
        self.corr = self.corr[valid]
        self.src_poses = self.src_poses[valid, ...]
        self.src_files = self.src_files[valid]
        assert self.src_poses.shape[0] == self.src_files.shape[0]
        assert self.map_poses.shape[0] == self.map_files.shape[0]
        print(self.src_files.shape, self.map_files.shape)

    @cache.memoize()
    def computeMapCorrespondences(self, max_pose_dist, src_dir, map_dir, mask_validation, validation):
        correspondence = []
        dists = []
        for i, src_pose in enumerate(tqdm.tqdm(self.src_poses)):
            dist_sq = np.sum(
                (src_pose[np.newaxis, :2, -1]-self.map_poses[:, :2, -1])**2, axis=-1)
            in_range = dist_sq < max_pose_dist**2
            in_range = np.argwhere(in_range).squeeze()
            if in_range.size < 1:
                correspondence.append(-1)
                dists.append(-1)
            else:
                np.random.seed(i)
                corr_idx = np.random.choice(in_range)
                correspondence.append(corr_idx)
                dists.append(dist_sq[corr_idx]**0.5)

        correspondence = np.array(correspondence, dtype=int)
        dists = np.array(dists)
        dists = np.array(dists)
        return correspondence, dists

    def valid_range2mask(self, file, nr_files):
        mask = np.zeros([nr_files], dtype=bool)
        valid_range = np.loadtxt(file, dtype=int)
        for line in valid_range:
            mask[line[0]-1:line[1]] = True
        return ~mask if self.validation else mask

    def loadPoses(self, dirs, mask):
        if isinstance(dirs, list):
            poses = np.vstack([self.loadPoses(dir, mask) for dir in dirs])
            return poses
        else:
            poses = np.loadtxt(join(dirs, 'poses.txt')).reshape(-1, 4, 4)
            if mask:
                valid_mask = self.valid_range2mask(
                    join(dirs, 'valid_range.txt'), poses.shape[0])
                return poses[valid_mask, ...]
            else:
                return poses

    def loadFiles(self, dirs, mask):
        if isinstance(dirs, list):
            files = np.hstack([self.loadFiles(dir, mask) for dir in dirs])
            return files
        else:
            files = np.hstack(
                sorted(glob.glob(join(dirs, f'*{self.file_format}'))))
            if mask:
                valid_mask = self.valid_range2mask(
                    join(dirs, 'valid_range.txt'), files.shape[0])
                return files[valid_mask, ...]
            else:
                return files

    def __getitem__(self, index):

        p_source, mask_source = self.pad(
            self.getFile(self.src_files[index]))

        map_idx = self.corr[index]
        p_target, mask_target = self.pad(
            self.getFile(self.map_files[map_idx]))

        pose = (np.linalg.inv(self.map_poses[map_idx])
                @ self.src_poses[index])
        pose = pose.astype('float32')
        pose[:3, -1] /= self.scale

        return {'target': p_target,
                'source': p_source,
                'pose': pose,
                'mask_target': mask_target,
                'mask_source': mask_source,
                'file_source': self.src_files[index],
                'file_target': self.map_files[map_idx]
                }

    def __len__(self):
        return self.src_files.size

    def getFile(self, file: str):
        if file.endswith('.ply'):
            return np.asarray(o3d.io.read_point_cloud(file).points, dtype='float32')
        else:
            return np.load(file).astype('float32')
