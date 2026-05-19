from torch.utils.data import Dataset
from PIL import Image
import os
import random
import torchvision.transforms.functional as TF
from torchvision import transforms


class IHC_Benchmarking_Dataset(Dataset):
    def __init__(self, path_A, path_B, is_train=True):
        self.path_A = path_A
        self.path_B = path_B
        self.images = sorted(os.listdir(path_A))
        self.is_train = is_train

        # Variación de color para simular diferentes lentes y luces de microscopio
        self.color_jitter = transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_A = Image.open(os.path.join(self.path_A, img_name)).convert("RGB")
        img_B = Image.open(os.path.join(self.path_B, img_name)).convert("RGB")

        # 1. Redimensionado base
        img_A = img_A.resize((256, 256), Image.BILINEAR)
        img_B = img_B.resize((256, 256), Image.BILINEAR)

        # 2. Data Augmentation (Solo si estamos entrenando)
        if self.is_train:
            # Volteo Horizontal Aleatorio (Sincronizado)
            if random.random() > 0.5:
                img_A = TF.hflip(img_A)
                img_B = TF.hflip(img_B)

            # Volteo Vertical Aleatorio (Sincronizado)
            if random.random() > 0.5:
                img_A = TF.vflip(img_A)
                img_B = TF.vflip(img_B)

            # Distorsión de color SOLO en la entrada (H&E)
            # Queremos que entradas defectuosas generen IHC perfectos
            img_A = self.color_jitter(img_A)

        # 3. Conversión a Tensor y Normalización estándar [-1, 1]
        img_A = TF.normalize(TF.to_tensor(img_A), [0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        img_B = TF.normalize(TF.to_tensor(img_B), [0.5, 0.5, 0.5], [0.5, 0.5, 0.5])

        return img_A, img_B, img_name