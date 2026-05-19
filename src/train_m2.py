import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from data_loader import IHC_Benchmarking_Dataset
from models.unet import UNetSegmenter
import os


def main():
    # --- 1. CONFIGURACIÓN ---
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    PATH_A = r"D:\job\TESIS\data\processed_data\train_A"
    PATH_B = r"D:\job\TESIS\data\processed_data\train_B"
    SAVE_PATH = "../models/unet_benchmarking.pth"
    EPOCHS = 100
    BATCH_SIZE = 8
    LEARNING_RATE = 0.0001

    # --- 2. DATA PIPELINE ---
    dataset = IHC_Benchmarking_Dataset(PATH_A, PATH_B, is_train=True)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)

    # --- 3. MODELO & OPTIMIZACIÓN ---
    model = UNetSegmenter().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    criterion_mse = nn.MSELoss()
    criterion_l1 = nn.L1Loss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    # Actualizado a la sintaxis moderna de PyTorch para evitar el FutureWarning
    scaler = torch.amp.GradScaler('cuda')

    # --- 4. BUCLE DE ENTRENAMIENTO ---
    print(f"Iniciando entrenamiento PRO de U-Net en {DEVICE}...")

    for epoch in range(EPOCHS):
        epoch_loss = 0
        model.train()

        for real_A, real_B, _ in dataloader:
            real_A, real_B = real_A.to(DEVICE), real_B.to(DEVICE)

            optimizer.zero_grad()

            # Actualizado a la sintaxis moderna
            with torch.amp.autocast('cuda'):
                preds = model(real_A)
                loss = criterion_mse(preds, real_B) + 0.5 * criterion_l1(preds, real_B)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()

        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        print(f"Epoch [{epoch + 1}/{EPOCHS}] - Loss: {epoch_loss / len(dataloader):.4f} - LR: {current_lr:.6f}")

    # --- 5. GUARDADO ---
    os.makedirs("../models", exist_ok=True)
    torch.save(model.state_dict(), SAVE_PATH)
    print(f"Modelo U-Net optimizado guardado exitosamente en: {SAVE_PATH}")


# Esta es la barrera de protección requerida por Windows para num_workers > 0
if __name__ == '__main__':
    main()