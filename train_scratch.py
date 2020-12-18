from model import make_model
from dataset import make_dataset
from eval_model import get_performance

from torch.utils.data import DataLoader
import torch
import torch.optim as optim
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint

from argparse import ArgumentParser


class CheXpertPL(pl.LightningModule):
    def __init__(self, model_type=None, task=None, num_train_samples=None,
                 lr=0.001,
                 batch_size=16,
                 ):
        super(CheXpertPL, self).__init__()

        self.save_hyperparameters()

        self.hparams.lr = lr
        self.hparams.batch_size = batch_size

        self.model = make_model(model_type)
        self.train_dataset = make_dataset(task=task, train=True, size=num_train_samples)
        self.val_dataset = make_dataset(task=task, train=False)

    def forward(self, x):
        return self.model(x)


    def configure_optimizers(self):
        optimizer = optim.Adam(self.model.parameters(), lr=self.hparams.lr)
        return {
            'optimizer': optimizer,
            'lr_scheduler': optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=10, mode='min'),
            'monitor': 'train_loss'
        }

    def train_dataloader(self):
        loader = DataLoader(self.train_dataset, pin_memory=True, batch_size=self.hparams.batch_size, shuffle=True, num_workers=8)
        return loader

    def val_dataloader(self):
        loader = DataLoader(self.val_dataset, pin_memory=True, batch_size=self.hparams.batch_size, shuffle=False, num_workers=8)
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
        parser.add_argument("--model_type", type=str, default="cbr_tiny",
                            choices=["cbr_large_tall", "cbr_large_wide", "cbr_small", "cbr_tiny", "resnet18", "resnet34", "resnet50"])
        parser.add_argument("--num_train_samples", type=int, default=-1)
        parser.add_argument("--task", type=str, default="Edema",
                            choices=['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Pleural Effusion'])

        return parser


if __name__ == '__main__':
    pl.seed_everything(1234)

    parser = ArgumentParser()
    parser = CheXpertPL.add_model_specific_args(parser)
    args = parser.parse_args()

    model = CheXpertPL(**args.__dict__)


#    bs_finder = pl.Trainer(auto_scale_batch_size=True, gpus=-1)
#    bs_finder.tune(model)
#    model.hparams.batch_size = min(model.hparams.batch_size, 64) # Avoid risking too large batch size


    #early_stop_callback = EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=20, verbose=True, mode='min')

    trainer = pl.Trainer(
        max_epochs=100, min_epochs=100,
        auto_lr_find=True, auto_scale_batch_size=False,
        gpus=-1,
        weights_save_path=f"./weights/{args.num_train_samples}/{args.model_type}",
        default_root_dir=f"./log/{args.num_train_samples}/{args.model_type}",
    )
    trainer.tune(model)
    print(f"model.hparams.lr = {model.hparams.lr} | model.hparams.batch_size = {model.hparams.batch_size}")
    print(f"args = {args}")

    trainer.fit(model)

    val_dataset = make_dataset(task=args.task, train=False)
    val_loader = DataLoader(val_dataset, model.hparams.batch_size)
    final_performance = get_performance(model, val_loader)
    print(f"\nPerformance = {final_performance}")

