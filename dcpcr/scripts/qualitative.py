from copy import copy, deepcopy
from email.policy import default
from os.path import join

import click
from urllib3 import Retry
import dcpcr.datasets.datasets as datasets
import dcpcr.utils.utils as utils
import yaml
from dcpcr.models import models, loss
from dcpcr.utils import pcd_visualizer, utils, fine_tuner
from torch.utils.data._utils import collate
import open3d as o3d
import numpy as np
from scipy.spatial.transform import rotation as R
import torch

LIGHT = 0.6
RED = [154.0/255,0,0]
BLUE = [51.0/255,102./255, 153/255]

def remove_ground(x): return pcd_visualizer.remove_ground(
    x, distance_threshold=0.3,
    num_iterations=1000,
    angle_threshold=0.5)


def colorize(ps, pos, scale=0.2):
    assert pos != 1
    color = deepcopy(np.asarray(ps.points))
    nground = normalizeColor(color[:, -1]) > scale
    # color[:, 1] = normalizeColor(color[:, -1])*scale
    color[:, 1] = 0
    color[:, 0] = 0
    color[:, 2] = 0
    color[nground, pos] = 1
    ps.colors = o3d.utility.Vector3dVector(color)

def normalizeColor(v):
    return (v-v.min())/(v.max()-v.min())

class PCD_Provider2():
    def __init__(self, batch, model) -> None:
        self.replace_pcd = {
            'apollo-compressed/': 'apollo-aggregated/', 'npy': 'ply'}
        self.model = model
        batch = collate.default_collate([batch])
        self.geoms = []
        scale = 40
        
        all =[]

        # Load Input
        file = batch['file_source'][0]
        for r1 in self.replace_pcd:
            file = file.replace(r1, self.replace_pcd[r1])
        ps = o3d.io.read_point_cloud(file)
        ps.estimate_normals()

        ps_ground, ps_nground = remove_ground(ps)
        ps_ground.paint_uniform_color(np.array([1, LIGHT, LIGHT]))
        ps_nground.paint_uniform_color(np.array(RED))

        file = batch['file_target'][0]
        for r1 in self.replace_pcd:
            file = file.replace(r1, self.replace_pcd[r1])
        pt = o3d.io.read_point_cloud(file)
        pt.estimate_normals()

        pt_ground, pt_nground = remove_ground(pt)
        pt_ground.paint_uniform_color(np.array([ LIGHT, LIGHT,1]))
        pt_nground.paint_uniform_color(np.array(BLUE))
        self.geoms.append([ps_nground + ps_ground,pt_nground + pt_ground])
        
        all.append(pt_nground + pt_ground)
        all.append(ps_nground + ps_ground)
       

        ###### Estimate transformation #####
        target = batch['target']
        source = batch['source']
        T, W, ps_corr, ww = self.model(
            target, source, batch['mask_target'], batch['mask_source'])
        T2 = T.squeeze().detach().cpu().numpy()
        T2[:3, -1] *= scale

        ps_groundt = deepcopy(ps_ground).transform(T2)
        ps_ngroundt = deepcopy(ps_nground).transform(T2)
        self.geoms.append([ps_groundt+ ps_ngroundt, pt_ground+pt_nground])
        all.append(ps_ngroundt + ps_groundt)

        gt_pose = batch['pose'][0, :, :].detach().cpu().numpy()
        gt_pose[:3, -1] *= scale
        dt, dr = loss.pose_error(
            T2, gt_pose)
        print(f'coarse: {dt:.3f}m, {dr:.3f}deg')

        # Fine registration
        T2 = fine_tuner.refine_registration(ps, pt, T2)
        ps_groundt = deepcopy(ps_ground).transform(T2)
        ps_ngroundt = deepcopy(ps_nground).transform(T2)
        self.geoms.append([ps_groundt+ ps_ngroundt, pt_ground+ pt_nground])
        all.append(ps_ngroundt + ps_groundt)
        dt, dr = loss.pose_error(
            T2, gt_pose)
        print(f'fine: {dt:.3f}m, {dr:.3f}deg')

        r = R.Rotation.from_matrix(deepcopy(gt_pose[:3, :3]))
        angle = np.linalg.norm(r.as_rotvec())/np.pi*180
        dist = np.linalg.norm(gt_pose[:3, -1])
        print(f'{angle:.2f}deg, {dist:.1f}m')
        
        self.geoms.append(all)

    def getGeometries(self, i):
        return self.geoms[i % len(self.geoms)]


@click.command()
@click.option('--data_config',
              '-dc',
              type=str,
              help='path to the config file (.yaml) for the dataloader',
              default=join(utils.CONFIG_DIR, 'data_config.yaml'))
@click.option('--checkpoint',
              '-c',
              type=str,
              help='path to checkpoint file (.ckpt)')
@click.option('--idx',
              '-i',
              type=int,
              default=100,
              help='index of the point cloud (in the validation set)')
def main(checkpoint, data_config, idx):
    data_cfg = yaml.safe_load(open(data_config))
    data_cfg['batch_size'] = 1

    data = datasets.DataModule(data_cfg)
    # model = models.StatNet(cfg)
    model = models.DCPCR.load_from_checkpoint(checkpoint)

    get_pcds = PCD_Provider2(data.val_dataset()[idx], model)
    vis = pcd_visualizer.Visualizer(get_pcds,width=1000,height=1000)
    vis.run()


if __name__ == "__main__":
    main()
