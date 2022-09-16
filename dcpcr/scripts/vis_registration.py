from email.policy import default
from os.path import join

import click
import dcpcr.datasets.datasets as datasets
import dcpcr.utils.utils as utils
import yaml
from dcpcr.models import models
from dcpcr.utils import pcd_visualizer, utils


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
@click.option('--fine_tune',
              '-ft',
              type=bool,
              help='Bool if fine tune transformation.',
              default=True)
@click.option('--show_compressed',
              '-sc',
              type=bool,
              help='Bool if fine tune transformation.',
              default=False)
def main(checkpoint, data_config, show_gt, fine_tune, show_compressed):
    data_cfg = yaml.safe_load(open(data_config))
    data_cfg['batch_size'] = 1

    # Load data and model
    data = datasets.DataModule(data_cfg)
    model = models.DCPCR.load_from_checkpoint(checkpoint)

    train = iter(data.test_dataloader(1))

    replace = None if show_compressed else {
        'apollo-compressed/': 'apollo-aggregated/', 'npy': 'ply'}
    kitti = pcd_visualizer.RegVis(
        train, model, draw_corr_points=False, show_gt=show_gt, replace_pcd=replace, ft =(fine_tune & ~show_compressed))
    vis = pcd_visualizer.Visualizer(kitti)
    vis.run()


if __name__ == "__main__":
    main()
