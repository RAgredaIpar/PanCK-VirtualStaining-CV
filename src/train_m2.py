import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from data_loader import IHC_Benchmarking_Dataset
from models.unet import UNetSegmenter
import os

# --- 1. CONFIGURACIÓN ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PATH_A = r"D:\job\TESIS\data\processed_data\train_A"
PATH_B = r"D:\job\TESIS\data\processed_data\train_B"
SAVE_PATH = "../models/unet_benchmarking.pth"
EPOCHS = 20
BATCH_SIZE = 4
LEARNING_RATE = 0.0001  # LR ligeramente menor para estabilidad en U-Net

# --- 2. DATA PIPELINE ---
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

dataset = IHC_Benchmarking_Dataset(PATH_A, PATH_B, transform=transform)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# --- 3. MODELO & OPTIMIZACIÓN ---
model = UNetSegmenter().to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
criterion = nn.MSELoss()  # MSE para mejor definición de bordes estructurales

# --- 4. BUCLE DE ENTRENAMIENTO ---
print(f"Iniciando entrenamiento del Modelo 2 (U-Net) en {DEVICE}...")

for epoch in range(EPOCHS):
    epoch_loss = 0
    model.train()
    for real_A, real_B, _ in dataloader:
        real_A, real_B = real_A.to(DEVICE), real_B.to(DEVICE)

        # Forward
        preds = model(real_A)
        loss = criterion(preds, real_B)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    print(f"Epoch [{epoch + 1}/{EPOCHS}] - Loss MSE: {epoch_loss / len(dataloader):.4f}")

# --- 5. GUARDADO ---
os.makedirs("../models", exist_ok=True)
torch.save(model.state_dict(), SAVE_PATH)
print(f"Modelo guardado exitosamente en: {SAVE_PATH}")