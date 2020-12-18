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

from train_scratch import CheXpertPL

if __name__ == '__main__':
    pl.seed_everything(1234)

    parser = ArgumentParser()
    parser.add_argument('--path', type=str)
    args = parser.parse_args()

    model = CheXpertPL.load_from_checkpoint(args.path)

    ckpt_callback = ModelCheckpoint(monitor='val_loss', mode='min', save_last=True, save_top_k=1, verbose=True)

    trainer = pl.Trainer(
        resume_from_checkpoint=args.path, gpus=-1,
        progress_bar_refresh_rate=200, callbacks=[ckpt_callback,],
        max_epochs=100, min_epochs=100
    )

    trainer.fit(model)

    val_dataset = make_dataset(task=args.task, train=False)
    val_loader = DataLoader(val_dataset, model.hparams.batch_size)
    final_performance = get_performance(model, val_loader)
    print(f"\nPerformance = {final_performance}")
