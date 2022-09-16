from collections import defaultdict
from gc import callbacks
import click
from os.path import join, dirname, abspath
import subprocess
from pytorch_lightning import Trainer
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint, Callback
import yaml

import dcpcr.datasets.datasets as datasets
import dcpcr.models.models as models
import torch.autograd
from dcpcr.utils.fine_tuner import RegistrationTuner
from dcpcr.models import loss

@click.command()
# Add your options here
@click.option('--data_config',
              '-dc',
              type=str,
              help='path to the config file (.yaml) for the dataloader',
              default=join(dirname(abspath(__file__)), 'config/data_config.yaml'))
@click.option('--checkpoint',
              '-ckpt',
              type=str,
              help='path to checkpoint file (.ckpt) to resume training.')
@click.option('--fine_tune',
              '-ft',
              type=bool,
              help='Whether to fine tune with icp or not.',
              default=True)
@click.option('--distance_threshold',
              '-dt',
              type=float,
              help='icp robust kernel distance threshold',
              default=1)
@click.option('--compressed',
              '-c',
              type=bool,
              help='Whether to fine tune on compressed or input data',
              default=True)
def main(checkpoint, data_config, fine_tune, distance_threshold, compressed):
    cfg = torch.load(checkpoint)['hyper_parameters']
    cfg['checkpoint'] = checkpoint
    cfg['fine_tune'] = fine_tune
    #### Eval params #####
    eval_cfg = {'distance_threshold': distance_threshold,
                'compressed': compressed,
                'replace':  {'apollo-compressed/': 'apollo-aggregated/', 'npy': 'ply'}
                }
    cfg['eval'] = eval_cfg
    fine_registrator = RegistrationTuner(**eval_cfg) if fine_tune else None
    
    data_cfg = yaml.safe_load(open(data_config))
    data = datasets.DataModule(data_cfg)
    

    model = models.DCPCR.load_from_checkpoint(
        checkpoint,
        hparams=cfg,
        fine_registrator=fine_registrator)

    tb_logger = pl_loggers.TensorBoardLogger('experiments/'+cfg['experiment']['id']+'_TESTSET',
                                             default_hp_metric=False)
    # Setup trainer
    trainer = Trainer(gpus=cfg['train']['n_gpus'],
                      logger=tb_logger,
                      resume_from_checkpoint=checkpoint,
                      max_epochs=cfg['train']['max_epoch'],
                      callbacks=[loss.ResultsSaver(tb_logger.log_dir)])

    trainer.test(model, data.test_dataloader(1))

if __name__ == "__main__":
    main()
