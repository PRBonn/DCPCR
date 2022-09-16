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


def knn(q_pts, s_pts, k):
    dist = ((q_pts.unsqueeze(-2) - s_pts.unsqueeze(-3))**2).sum(-1)
    dist, neighb_inds = torch.topk(dist, k, dim=-1, largest=False)
    return neighb_inds, dist

class WeightVis():
    def __init__(self, dataloader, model, draw_corr_lines=False, replace_pcd=None):
        self.model = model
        self.dataloader = dataloader
        self.draw_corr_lines = draw_corr_lines
        self.replace_pcd = replace_pcd

    def getGeometries(self, i):
        batch = next(self.dataloader)
        # batch = self.dataloader.next()
        target = batch['target']
        source = batch['source']

        T, W, ps_corr,ww = self.model.model(
            target, source, batch['mask_target'], batch['mask_source'])

        ##### Target Source ######
        mask = batch['mask_target'][0, :, 0]
        w = W[..., mask, 0].squeeze().detach().cpu().numpy()
        cmap = plt.get_cmap('viridis')
        color = cmap(w.squeeze()/w.max())
        print(color)

        target = target[0, mask, :3]
        pt = pcd_visualizer.torch2o3d(target*40)
        pt.colors = o3d.utility.Vector3dVector(color[:, :3])

        geoms = [pt]
        if self.replace_pcd is not None:
            file = batch['file_target'][0]
            for r1 in self.replace_pcd:
                file = file.replace(r1, self.replace_pcd[r1])
            pt2 = o3d.io.read_point_cloud(file)
            
            target_full = torch.tensor(np.asarray(pt2.points))
            idx,dist = knn(target_full,target*40,1)
            colors = color[idx,:].mean(-2)
            pt2.colors = o3d.utility.Vector3dVector(colors[:, :3])
            pt2.estimate_normals()
            geoms.append(pt2)
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
    # replace = None
    kitti = WeightVis(
        train, model, replace_pcd=replace)
    vis = pcd_visualizer.Visualizer(kitti)
    vis.run()


if __name__ == "__main__":
    main()
