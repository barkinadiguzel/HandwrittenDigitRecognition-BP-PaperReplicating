import torch
from torch.utils.data import Dataset
import torchvision.transforms as T

class HandwrittenDigitDataset(Dataset):
    def __init__(self, images, labels, augment=False):
        self.images = images  # [B,1,H,W] tensorları preprocess.py'den
        self.labels = labels
        self.augment = augment
        self.transform = T.Compose([
            T.RandomRotation(10),
            T.RandomAffine(0, translate=(0.1,0.1)),
        ]) if augment else None

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img = self.images[idx]
        label = self.labels[idx]
        if self.augment:
            img = self.transform(img)
        return img, label
