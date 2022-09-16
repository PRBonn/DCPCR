import click
from os.path import join, dirname, abspath
import subprocess
from pytorch_lightning import Trainer
from pytorch_lightning.tuner.tuning import Tuner
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.callbacks import ModelSummary
import yaml
import dcpcr.datasets.datasets as datasets
import dcpcr.models.models as models


@click.command()
# Add your options here
@click.option('--config',
              '-c',
              type=str,
              help='path to the config file (.yaml)',
              default=join(dirname(abspath(__file__)), 'config/config.yaml'))
@click.option('--data_config',
              '-dc',
              type=str,
              help='path to the config file (.yaml) for the dataloader',
              default=join(dirname(abspath(__file__)), 'config/data_config.yaml'))
@click.option('--weights',
              '-w',
              type=str,
              help='path to pretrained weights (.ckpt). Use this flag if you just want to load the weights from the checkpoint file without resuming training.',
              default=None)
@click.option('--checkpoint',
              '-ckpt',
              type=str,
              help='path to checkpoint file (.ckpt) to resume training.',
              default=None)
def main(config, weights, checkpoint, data_config):
    cfg = yaml.safe_load(open(config))
    data_cfg = yaml.safe_load(open(data_config))
    cfg['data_loader'] = data_cfg
    cfg['git_commit_version'] = str(subprocess.check_output(
        ['git', 'rev-parse', '--short', 'HEAD']).strip())

    print(f"!!! Starting Experiment: {cfg['experiment']['id']} !!!")

    # Load data and model
    data = datasets.DataModule(data_cfg)
    if weights is None:
        model = models.DCPCR(cfg, data_module=data)
    else:
        model = models.DCPCR.load_from_checkpoint(
            weights, hparams=cfg, data_module=data)

    # Add callbacks
    lr_monitor = LearningRateMonitor(logging_interval='step')
    checkpoint_saver = ModelCheckpoint(monitor='val/loss',
                                       filename='best',
                                       mode='min',
                                       save_last=True,
                                       save_on_train_epoch_end=False)

    tb_logger = pl_loggers.TensorBoardLogger('experiments/'+cfg['experiment']['id'],
                                             default_hp_metric=False)

    # Setup trainer
    trainer = Trainer(gpus=cfg['train']['n_gpus'],
                      logger=tb_logger,
                      resume_from_checkpoint=checkpoint,
                      max_epochs=cfg['train']['max_epoch'],
                      callbacks=[lr_monitor, checkpoint_saver, ModelSummary(max_depth=2)])

    # Train!
    trainer.fit(model)


if __name__ == "__main__":
    main()
