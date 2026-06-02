import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from data_loader import IHC_Benchmarking_Dataset
from models.pix2pix import UNetGenerator
import os
import wandb  # Inyección de Weights & Biases


def main():
    # --- 1. CONFIGURACIÓN ---
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    PATH_A = r"D:\job\TESIS\data\processed_data\train_A"
    PATH_B = r"D:\job\TESIS\data\processed_data\train_B"
    SAVE_PATH = "../models/pix2pix_benchmarking.pth"
    EPOCHS = 100
    BATCH_SIZE = 8
    LEARNING_RATE = 0.0002

    # Inicializar experimento en WandB
    wandb.init(
        project="Tesis-IHC-EsSalud",
        name="M1-Pix2Pix-Baseline",
        config={
            "learning_rate": LEARNING_RATE,
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "architecture": "Pix2Pix-UNet-Generator",
            "loss_criterion": "L1Loss"
        }
    )

    # --- 2. DATA PIPELINE ---
    dataset = IHC_Benchmarking_Dataset(PATH_A, PATH_B, is_train=True)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)

    # --- 3. MODELO & OPTIMIZACIÓN ---
    model = UNetGenerator().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))
    criterion = nn.L1Loss()

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    scaler = torch.amp.GradScaler('cuda')

    # --- 4. BUCLE DE ENTRENAMIENTO ---
    print(f"Iniciando entrenamiento PRO de Pix2Pix en {DEVICE}...")

    for epoch in range(EPOCHS):
        epoch_loss = 0
        model.train()

        for real_A, real_B, _ in dataloader:
            real_A, real_B = real_A.to(DEVICE), real_B.to(DEVICE)

            optimizer.zero_grad()

            with torch.amp.autocast('cuda'):
                fake_B = model(real_A)
                loss = criterion(fake_B, real_B)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()

        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        avg_loss = epoch_loss / len(dataloader)

        print(f"Epoch [{epoch + 1}/{EPOCHS}] - Loss L1: {avg_loss:.4f} - LR: {current_lr:.6f}")

        # Enviar métricas cuantitativas a la nube
        wandb.log({
            "epoch": epoch + 1,
            "train/loss_L1": avg_loss,
            "train/learning_rate": current_lr
        })

        # Envío de muestras visuales a la nube cada 10 épocas para control de calidad
        if (epoch + 1) % 10 == 0:
            # Desnormalizar imágenes de [-1, 1] a rango estándar [0, 1] para visualización web
            img_he = (real_A[0].detach().cpu().permute(1, 2, 0).numpy() * 0.5 + 0.5).clip(0, 1)
            img_real_ihc = (real_B[0].detach().cpu().permute(1, 2, 0).numpy() * 0.5 + 0.5).clip(0, 1)
            img_fake_ihc = (fake_B[0].detach().cpu().permute(1, 2, 0).numpy() * 0.5 + 0.5).clip(0, 1)

            wandb.log({
                "Visual_Progress/Muestra_Evolutiva": [
                    wandb.Image(img_he, caption="H&E Entrada"),
                    wandb.Image(img_real_ihc, caption="IHC Real (Ground Truth)"),
                    wandb.Image(img_fake_ihc, caption="IHC Sintética (Predicción M1)")
                ]
            }, step=epoch + 1)

    # --- 5. GUARDADO ---
    os.makedirs("../models", exist_ok=True)
    torch.save(model.state_dict(), SAVE_PATH)
    print(f"Modelo Pix2Pix optimizado guardado exitosamente en: {SAVE_PATH}")

    # Cerrar el experimento de WandB limpiamente
    wandb.finish()


if __name__ == '__main__':
    main()