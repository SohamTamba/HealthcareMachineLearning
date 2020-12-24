from torch.utils.data import Dataset, random_split
import torch
import os
from csv import DictReader
from PIL import Image
from torchvision import transforms

def make_dataset(task, train, root="../", size=-1):
    full_dataset = ChexpertDataset(task, train, root)
    full_size = len(full_dataset)
    if size < 0 or size > full_size:
        return full_dataset
    else:
        dataset, _ = random_split(full_dataset, [size, full_size-size], generator=torch.Generator().manual_seed(1234))
        return dataset



class ChexpertDataset(Dataset):
    def __init__(self, task, train, root="/scratch/sgt287/HCML"):
        assert task in ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Pleural Effusion']

        self.root = root

        # Get Annotations
        csv_name = 'CheXpert-v1.0-small/train.csv' if train else 'CheXpert-v1.0-small/valid.csv'
        csv_path = os.path.join(self.root, csv_name)

        annotations = []
        annotation_str_to_bool = {'': False, '0.0': False, '1.0': True}
        unknown_str = '-1.0'

        with open(csv_path, newline='') as csvfile:
            rows = DictReader(csvfile)
            for row in rows:
                annotation = row[task]
                if annotation == unknown_str:
                    continue

                annotation = annotation_str_to_bool[annotation]
                img_path = row['Path']
                annotations.append( (img_path, annotation) )

        self.annotations = annotations

        # Get Transform
        img_transforms = []
        img_transforms.append(transforms.Resize((256, 256)))
        if train: # Taken from https://github.com/jfhealthcare/Chexpert/blob/master/data/imgaug.py
            img_transforms.append(transforms.RandomAffine(
                        degrees=(-15, 15), translate=(0.05, 0.05),
                        scale=(0.95, 1.05), fillcolor=128)
            )
        img_transforms.append(transforms.ToTensor())

        self.img_transform = transforms.Compose(img_transforms)

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        img_path, annotation = self.annotations[idx]
        img_path = os.path.join(self.root, img_path)
        img = Image.open(img_path)
        img = self.img_transform(img)

        annotation = 1.0 if annotation else 0.0

        return img, torch.tensor(annotation)





if __name__ == '__main__':

    '''
    for task in ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Pleural Effusion']:
        ds = ChexpertDataset(task=task, train=True)
        print(len(ds)) #[189675, 215327, 195672, 210430, 211786]
        ds = ChexpertDataset(task=task, train=False)
        print(len(ds)) #[234, 234, 234, 234, 234]
    '''

    ds = make_dataset(task='Atelectasis', train=True)
    x, y = ds[0]

    ds = make_dataset(task='Atelectasis', train=True, size=500)
    print(f"Size of split dataset = {len(ds)}")
