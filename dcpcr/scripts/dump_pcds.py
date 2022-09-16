import click
import dcpcr.utils.utils as utils
import yaml
import dcpcr.datasets.datasets as datasets
from dcpcr.utils import pcd_visualizer
from dcpcr.models import models
from os.path import join
from dcpcr.utils import utils
import open3d as o3d
import numpy as np
from matplotlib import pyplot as plt
from dcpcr.models import blocks
import torch
from pathlib import Path

def knn(q_pts, s_pts, k):
    dist = ((q_pts.unsqueeze(-2) - s_pts.unsqueeze(-3))**2).sum(-1)
    dist, neighb_inds = torch.topk(dist, k, dim=-1, largest=False)
    return neighb_inds, dist

class Dumper():
    def __init__(self, dataloader, model, out, replace_pcd=None):
        self.model = model
        self.dataloader = dataloader
        self.replace_pcd = replace_pcd
        self.out_dir = out

    def getGeometries(self, i):
        batch = next(self.dataloader)
        target = batch['target']
        source = batch['source']
        T_gt = batch['pose'].detach().squeeze().cpu().numpy()
        
        T, W, ps_corr,ww = self.model.model(
            target, source, batch['mask_target'], batch['mask_source'])

        ##### original Source ######
        mask = batch['mask_source'][0, :, 0]
        if self.replace_pcd is not None:
            file = batch['file_source'][0]
            for r1 in self.replace_pcd:
                file = file.replace(r1, self.replace_pcd[r1])
            ps = o3d.io.read_point_cloud(file)
            
        else:
            ps = source[0, mask, :3]
            ps = pcd_visualizer.torch2o3d(ps)
        
        ##### Target Source ######
        mask = batch['mask_target'][0, :, 0]
        if self.replace_pcd is not None:
            file = batch['file_target'][0]
            for r1 in self.replace_pcd:
                file = file.replace(r1, self.replace_pcd[r1])
            pt = o3d.io.read_point_cloud(file)
        else:
            w = W[..., mask, :].squeeze().detach().cpu().numpy()
            pt = target[0, mask, :3]
            pt = pcd_visualizer.torch2o3d(pt)
        pt.paint_uniform_color(np.array([0, 1, 0]))
        print('Target: Green')

        ##### Transformed Source ######
        mask = batch['mask_source'][0, :, 0]
        if self.replace_pcd is not None:
            file = batch['file_source'][0]
            for r1 in self.replace_pcd:
                file = file.replace(r1, self.replace_pcd[r1])
            ps_t = o3d.io.read_point_cloud(file)
            T = T.squeeze().detach().cpu().numpy()
            T[:3, -1] *= 40
            T_gt[:3, -1] *= 40
            ps_t.transform(T)
        else:
            ps_t = pcd_visualizer.transform(source, T)
            ps_t = ps_t[mask, :]
            ps_t = pcd_visualizer.torch2o3d(ps_t)
            T = T.squeeze().detach().cpu().numpy()
        ps_t.paint_uniform_color(np.array([0, 0, 1]))
        print('Transformed Source: Blue')
        
        o_dir = f"{self.out_dir}/{i}"
        print('out:',o_dir)
        Path(o_dir).mkdir(parents=True, exist_ok=True)
        
        o3d.io.write_point_cloud(f"{o_dir}/source.ply", ps)
        o3d.io.write_point_cloud(f"{o_dir}/target.ply", pt)
        if self.replace_pcd is None:
            mask = batch['mask_target'][0, :, 0]
            w= ww[0, mask, :]
            val,idx = torch.max(w,dim=-1)
            idx = torch.stack([torch.arange(idx.shape[0]),idx]).T
            print(idx.shape)
            w = W[..., mask, :]

            np.savetxt(f'{o_dir}/idx.txt',
                       idx.detach().squeeze().cpu().numpy())
            np.savetxt(f'{o_dir}/weights.txt',
                       w.detach().squeeze().cpu().numpy())

        np.savetxt(f'{o_dir}/gt.txt', T_gt)
        np.savetxt(f'{o_dir}/est.txt',T)
        geoms = [pt, ps_t]
        print()
        return geoms


@click.command()
@click.option('--data_config',
              '-dc',
              type=str,
              help='path to the config file (.yaml) for the dataloader',
              default=join(utils.CONFIG_DIR, 'data_config.yaml'))
@click.option('--checkpoint',
              '-c',
              type=str,
              help='path to checkpoint file (.ckpt) to resume training.')
@click.option('--show_gt',
              '-gt',
              type=bool,
              help='Bool if show gt transformation.',
              default=False)
def main(checkpoint, data_config, show_gt):
    data_cfg = yaml.safe_load(open(data_config))
    data_cfg['batch_size'] = 1

    data = datasets.DataModule(data_cfg)
    model = models.DCPCR.load_from_checkpoint(checkpoint)

    train = iter(data.val_dataloader(1))

    replace = {'apollo-compressed/': 'apollo-aggregated/', 'npy': 'ply'}
    dumper = Dumper(
        train, model, replace_pcd=replace, out= 'data')
    vis = pcd_visualizer.Visualizer(dumper)
    vis.run()

if __name__ == "__main__":
    main()
