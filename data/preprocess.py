import bz2
import numpy as np
import torch
from torch.utils.data import Dataset

class USPSDataset(Dataset):
    def __init__(self, file_path):
        with bz2.open(file_path, "rt") as f:
            lines = f.readlines()

        labels = []
        images = []

        for line in lines:
            parts = line.strip().split()
            labels.append(int(parts[0]))
            pixels = np.array([float(p.split(":")[1]) for p in parts[1:]], dtype=np.float32)
            images.append((pixels + 1) / 2.0)  # Normalize [-1,1] -> [0,1]

        self.images = torch.tensor(np.array(images)).reshape(-1, 1, 16, 16)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]
