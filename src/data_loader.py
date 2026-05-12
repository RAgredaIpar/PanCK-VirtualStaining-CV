import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import os

class IHC_Benchmarking_Dataset(Dataset):
    def __init__(self, path_A, path_B, transform=None):
        self.path_A = path_A
        self.path_B = path_B
        self.images = sorted(os.listdir(path_A))
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_A = Image.open(os.path.join(self.path_A, img_name)).convert("RGB")
        img_B = Image.open(os.path.join(self.path_B, img_name)).convert("RGB")
        if self.transform:
            img_A = self.transform(img_A)
            img_B = self.transform(img_B)
        return img_A, img_B, img_name