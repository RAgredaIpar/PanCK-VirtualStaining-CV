import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from data_loader import IHC_Benchmarking_Dataset
from models.pix2pix import UNetGenerator
import os

# --- 1. CONFIGURACIÓN ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PATH_A = r"D:\job\TESIS\data\processed_data\train_A"
PATH_B = r"D:\job\TESIS\data\processed_data\train_B"
SAVE_PATH = "../models/pix2pix_benchmarking.pth"
EPOCHS = 20
BATCH_SIZE = 4
LEARNING_RATE = 0.0002

# --- 2. DATA PIPELINE ---
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

dataset = IHC_Benchmarking_Dataset(PATH_A, PATH_B, transform=transform)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# --- 3. MODELO & OPTIMIZACIÓN ---
model = UNetGenerator().to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))
criterion = nn.L1Loss()

# --- 4. BUCLE DE ENTRENAMIENTO ---
print(f"Iniciando entrenamiento del Modelo 1 (Pix2Pix) en {DEVICE}...")

for epoch in range(EPOCHS):
    epoch_loss = 0
    model.train()
    for real_A, real_B, _ in dataloader:
        real_A, real_B = real_A.to(DEVICE), real_B.to(DEVICE)

        # Forward
        fake_B = model(real_A)
        loss = criterion(fake_B, real_B)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    print(f"Epoch [{epoch + 1}/{EPOCHS}] - Loss L1: {epoch_loss / len(dataloader):.4f}")

# --- 5. GUARDADO ---
os.makedirs("../models", exist_ok=True)
torch.save(model.state_dict(), SAVE_PATH)
print(f"Modelo guardado exitosamente en: {SAVE_PATH}")