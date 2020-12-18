from model import TransferModel
from dataset import make_dataset
from eval_model import get_performance
from train_scratch import CheXpertPL

from torch.utils.data import DataLoader
import torch
import torch.optim as optim
import torch.nn.functional as F

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from argparse import ArgumentParser


class CheXpertFineTunerPL(pl.LightningModule):
    def __init__(self, backbone_path=None, task=None, num_train_samples=None,
                 lr=0.001,
                 batch_size=16,
                 freeze_backbone=True
                 ):
        super(CheXpertFineTunerPL, self).__init__()

        self.save_hyperparameters()

        self.hparams.lr = lr
        self.hparams.batch_size = batch_size
        backbone_model = CheXpertPL.load_from_checkpoint(backbone_path).model

        self.freeze_backbone = freeze_backbone
        self.model = TransferModel(backbone_model, freeze_backbone)
        self.train_dataset = make_dataset(task=task, train=True, size=num_train_samples)
        self.val_dataset = make_dataset(task=task, train=False)

    def forward(self, x):
        return self.model(x)


    def configure_optimizers(self):
        trainable = self.model.decoder if self.freeze_backbone else self.model
        optimizer = optim.Adam(trainable.parameters(), lr=self.hparams.lr)
        return {
            'optimizer': optimizer,
            'lr_scheduler': optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=10, mode='min'),
            'monitor': 'train_loss'
        }

    def train_dataloader(self):
        loader = DataLoader(self.train_dataset, batch_size=self.hparams.batch_size, shuffle=True, num_workers=8)
        return loader

    def val_dataloader(self):
        loader = DataLoader(self.val_dataset, batch_size=self.hparams.batch_size, shuffle=False, num_workers=8)
        return loader

    def training_step(self, batch, batch_nb):
        imgs, labels = batch
        outputs = self.model(imgs).squeeze()
        loss = F.binary_cross_entropy_with_logits(outputs, labels)

        self.log('train_loss', loss.detach(), on_step=True, on_epoch=False, prog_bar=True, logger=True)
        return loss

    @torch.no_grad()
    def validation_step(self, batch, batch_nb):
        imgs, labels = batch
        outputs = self.model(imgs).squeeze()
        loss = F.binary_cross_entropy_with_logits(outputs, labels)

        self.log('val_loss', loss.detach(), on_step=False, on_epoch=True, prog_bar=False, logger=True)
        return loss

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--lr", type=float, default=0.001)
        parser.add_argument("--batch_size", type=int, default=64)
        parser.add_argument("--backbone_path", type=str, default="weights/-1/cbr_large_wide/resume_checkpoint/version_14062969/checkpoints/epoch=19.ckpt")
        parser.add_argument("--num_train_samples", type=int, default=-1)
        parser.add_argument("--task", type=str, default="Edema",
                            choices=['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Pleural Effusion'])
        parser.add_argument("--freeze_backbone", action='store_true')

        return parser


if __name__ == '__main__':
    pl.seed_everything(1234)

    parser = ArgumentParser()
    parser = CheXpertFineTunerPL.add_model_specific_args(parser)
    args = parser.parse_args()

    model = CheXpertFineTunerPL(**args.__dict__)


#    bs_finder = pl.Trainer(auto_scale_batch_size=True, gpus=-1)
#    bs_finder.tune(model)
#    model.hparams.batch_size = min(model.hparams.batch_size, 64) # Avoid risking too large batch size


    #early_stop_callback = EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=20, verbose=True, mode='min')
    ckpt_callback = ModelCheckpoint(monitor='val_loss', mode='min', save_top_k=1)

    trainer = pl.Trainer(
        max_epochs=100, min_epochs=100,
        auto_lr_find=True, auto_scale_batch_size=False,
        progress_bar_refresh_rate=200, callbacks=[ckpt_callback,],
        gpus=-1,
        weights_save_path=f"./weights/freeze={args.freeze_backbone}/{args.num_train_samples}",
        default_root_dir=f"./log/freeze={args.freeze_backbone}/{args.num_train_samples}"
    )

    trainer.tune(model)
    print(f"model.lr = {model.hparams.lr} | model.batch_size = {model.hparams.batch_size}")
    print(f"args = {args}")

    trainer.fit(model)

    val_dataset = make_dataset(task=args.task, train=False)
    val_loader = DataLoader(val_dataset, model.hparams.batch_size)
    final_performance = get_performance(model, val_loader)
    print(f"\nPerformance = {final_performance}")

