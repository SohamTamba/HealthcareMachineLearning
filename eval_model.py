import torch
from torch.utils.data import DataLoader

import pytorch_lightning as pl
from pytorch_lightning.metrics.functional.classification import auroc

from dataset import make_dataset

@torch.no_grad()
def get_performance(model, val_loader):
    pred_list = []
    target_list = []
    num_pos = 0
    num_pos_correct = 0
    num_neg = 0
    num_neg_correct = 0
    output = {}

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    for imgs, targets in val_loader:
        imgs = imgs.to(device)
        outputs = model(imgs)
        preds = torch.sigmoid(outputs).squeeze()

        pred_list.append(preds.detach().cpu())
        target_list.append(targets)

    preds = torch.cat(pred_list)
    targets = torch.cat(target_list)

    auc = auroc(preds, targets)*100
    output['aurocc'] = f"{auc:.3f}%"

    for pred, target in zip(preds, targets):
        if target == 1.0:
            num_pos += 1
            if pred > 0.5:
                num_pos_correct += 1
        elif target == 0.0:
            num_neg += 1
            if pred < 0.5:
                num_neg_correct += 1
        else:
            assert False

    output['pos'] = f'{num_pos_correct}/{num_pos}'
    output['neg'] = f'{num_neg_correct}/{num_neg}'

    return output




if __name__ == '__main__':
    from train_scratch import CheXpertPL

    pl.seed_everything(1234)
    model_type = 'cbr_large_wide'
    task = 'Edema'
    batch_size = 64



    val_dataset = make_dataset(task=task, train=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    model_paths = [
        'weights/freeze=False/100/resume_checkpoint/version_14077227/checkpoints/epoch=25.ckpt',
        'weights/freeze=False/500/resume_checkpoint/version_14077228/checkpoints/epoch=7.ckpt',
        'weights/freeze=False/1000/resume_checkpoint/version_14077230/checkpoints/epoch=3.ckpt',
    ]


    for model_path in model_paths:
        if not model_path is None:
            model = CheXpertPL.load_from_checkpoint(model_path)
        else:
            model = CheXpertPL(model_type=model_type, task=task, num_train_samples=100)

        performance = get_performance(model, val_loader)

        print(f"Model: {model_path}")
        print(f"Performance: {performance}")
        print('-'*80)
