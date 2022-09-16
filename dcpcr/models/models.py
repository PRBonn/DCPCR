import dcpcr.models.blocks as blocks
import torch
import torch.nn as nn
from dcpcr.models import loss
from pytorch_lightning import LightningDataModule
from pytorch_lightning.core.lightning import LightningModule
import time

class DCPCR(LightningModule):
    def __init__(self, hparams: dict, data_module: LightningDataModule = None, fine_registrator = None):
        super().__init__()
        # name you hyperparameter hparams, then it will be saved automagically.
        hparams['batch_size'] = hparams['data_loader']['batch_size']
        self.save_hyperparameters(hparams)

        self.model = RegisterNet(**hparams['model'])
        self.loss_handler = loss.LossHandler(**hparams['loss'])
        self.data_module = data_module
        self.fine_registrator = fine_registrator

    def forward(self, target: torch.Tensor, source: torch.Tensor, mask_target=None, mask_source=None):
        est_pose, w, target_corr, weights = self.model(
            target, source, mask_target, mask_source)
        return est_pose, w, target_corr, weights

    def log_loss(self, losses):
        for k in losses:
            self.log(k, losses[k])

    def training_step(self, batch: dict, batch_idx):
        est_pose, w, target_corr, weights = self.forward(
            batch['target'], batch['source'], batch['mask_target'], batch['mask_source'])

        train_loss, losses = self.loss_handler.getLoss(
            gt_pose=batch['pose'],
            est_pose=est_pose,
            target_points=batch['target'],
            corr_points=target_corr,
            corr_weights=w)
        # Logging
        dt, dr = loss.pose_error(
            est_pose, batch['pose'], scale=self.hparams['loss']['scale'])

        self.log_loss(losses)
        self.log('train/loss', train_loss)
        self.log('train/dt_meter', dt)
        self.log('train/dr_degrees', dr)
        return train_loss

    def validation_step(self, batch: dict, batch_idx):
        t = time.time()
        est_pose, w, target_corr, weights = self.forward(
            batch['target'], batch['source'], batch['mask_target'], batch['mask_source'])
        t = time.time() -t

        dt, dr = loss.pose_error(
            est_pose, batch['pose'], scale=self.hparams['loss']['scale'])
        self.log('val/dt_meter', dt)
        self.log('val/dr_degrees', dr)
        self.log('val/loss', dt+dr)
        self.log('val/time', t)

    def test_step(self, batch: dict, batch_idx):
        est_pose, w, target_corr, weights = self.forward(
            batch['target'], batch['source'], batch['mask_target'], batch['mask_source'])

        if self.fine_registrator is not None: # fine registration
            est_pose = self.fine_registrator.refine_registration(batch,est_pose)

        dt, dr = loss.pose_error(
            est_pose, batch['pose'], scale=self.hparams['loss']['scale'])
        
        out = {'test/dt_meter': dt.item(),
               'test/dr_degrees': dr.item()}
        self.log_dict(out)
        return out

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.hparams['train']['lr'])
        return optimizer

    def debug_step(self, batch: dict, batch_idx):
        est_pose, w, target_corr, weights = self.forward(
            batch['target'], batch['source'], batch['mask_target'], batch['mask_source'])

        train_loss, losses = self.loss_handler.getLoss(
            gt_pose=batch['pose'],
            est_pose=est_pose,
            target_points=batch['target'],
            corr_points=target_corr,
            corr_weights=w)
        return train_loss

    def train_dataloader(self):
        return self.data_module.train_dataloader(batch_size=self.hparams['batch_size'])

    def val_dataloader(self):
        return self.data_module.val_dataloader(batch_size=self.hparams['batch_size'])

    def test_dataloader(self):
        return self.data_module.test_dataloader(batch_size=self.hparams['batch_size'])

#######################################
############# Modules #################
#######################################


class RegisterNet(nn.Module):
    def __init__(self, tau=1,
                 weighting='information_gain',
                 input_transform=False,
                 nr_attn_blocks=0,
                 attention_normalization='softmax',
                 kp_radius=0.05,
                 nr_kp_blocks=0,
                 radial=False
                 ):
        super().__init__()
        self.point_net = blocks.PointNetFeat(
            in_dim=6, out_dim=256, input_transform=input_transform, norm=True)

        self.conf = ConvNet(in_channels=256, out_channels=256,
                            radius=kp_radius, num_layer=nr_kp_blocks, radial=radial)

        self.transformer = Transformer(
            d_model=256,
            num_layer=nr_attn_blocks,
            dim_feedforward=512,
            dropout=0)

        self.attention = blocks.Attention(
            tau, attention_normalization=attention_normalization)
        self.weighting = blocks.CorrespondenceWeighter(
            weighting=weighting)
        self.registration = blocks.SVDRegistration()

    def forward(self, target, source, mask_target=None, mask_source=None):
        x_t = target[..., :3]
        x_s = source[..., :3]
        f_t = self.point_net(target)
        f_s = self.point_net(source)

        f_t = self.conf(x_t, f_t, mask_target)
        f_s = self.conf(x_s, f_s, mask_source)

        f_s, f_t = self.transformer(f_t, f_s, mask_target, mask_source)
        x_t_corr, weight = self.attention(
            f_t, f_s, x_s, mask_source, mask_target)  # q,k,v
        x_t = x_t.clone().detach()
        corr_weight = self.weighting(weight)
        transformation = self.registration(x_t, x_t_corr, corr_weight)
        return transformation, corr_weight, x_t_corr, weight


class Transformer(nn.Module):
    def __init__(self,
                 d_model,
                 num_layer=2,
                 nhead=8,
                 dim_feedforward=512,
                 dropout=0
                 ):
        super().__init__()
        enc = nn.TransformerEncoderLayer(d_model=d_model,
                                         nhead=nhead,
                                         dim_feedforward=dim_feedforward,
                                         dropout=dropout,
                                         batch_first=True,
                                         norm_first=True)
        self.enc = nn.TransformerEncoder(enc, num_layers=num_layer)

        dec = nn.TransformerDecoderLayer(d_model=d_model,
                                         nhead=nhead,
                                         dim_feedforward=dim_feedforward,
                                         dropout=dropout,
                                         batch_first=True,
                                         norm_first=True)
        self.dec = nn.TransformerDecoder(dec, num_layers=num_layer)
        self.num_layer = num_layer

    def forward(self, encoding, decoding, enc_mask=None, dec_mask=None):
        if self.num_layer > 0:
            enc_mask = enc_mask if enc_mask is None else torch.logical_not(
                enc_mask[..., 0])
            dec_mask = dec_mask if dec_mask is None else torch.logical_not(
                dec_mask[..., 0])
            encoding = self.enc(
                src=encoding, src_key_padding_mask=enc_mask)
            decoding = self.dec(tgt=decoding,
                                memory=encoding,
                                tgt_key_padding_mask=dec_mask,
                                memory_key_padding_mask=enc_mask)
        return decoding, encoding

class ConvNet(nn.Module):
    def __init__(self, in_channels,
                 out_channels,
                 radius,
                 num_layer=3,
                 num_neighbors=32,
                 kernel_size=3,
                 KP_extent=None,
                 p_dim=3,
                 radial=False,
                 f_dscale=2):
        super().__init__()
        in_c = [in_channels] + num_layer*[out_channels]
        self.blocks = nn.ModuleList([blocks.ResnetKPConv(
            in_channels,
            out_channels,
            radius,
            kernel_size,
            KP_extent,
            p_dim,
            radial,
            f_dscale) for in_channels in in_c[:num_layer]])
        self.num_neighbors = num_neighbors
        self.num_layer = num_layer

    def forward(self, coords: torch.Tensor, features: torch.Tensor, mask: torch.Tensor = None):
        if self.num_layer > 0:
            coords = coords.contiguous()
            if mask is not None:
                coords[mask.expand_as(coords) < 0.5] = 1e6
            idx = blocks.knn(coords, coords, self.num_neighbors)
            for block in self.blocks:
                features = block(coords, coords, idx, features)
        return features
