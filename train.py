import os
from datetime import datetime

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F


from utils import visualize_keypoints_with_heatmaps
from model import DepthwiseSeparableConv, ResidualBlock, BlazePoseLite
from dataset import DepthKeypointDataset


def compute_losses(
    heatmap_coords,
    heatmaps,
    target_coords,
    target_heatmaps,
    alpha=0.1,   # вес для heatmap loss
    beta=10.0,   # вес для координат из soft-argmax
    heatmap_size=64,
    img_size=224,
):
    """
    heatmap_coords: [B, K, 2] - из soft-argmax (в heatmap scale)
    regressed_coords: [B, K, 2] - из прямого регресса (в image scale)
    heatmaps: [B, K, H, W] - предсказанные тепловые карты
    target_coords: [B, K, 2] - ground truth координаты (в image scale)
    target_heatmaps: [B, K, H, W] - ground truth тепловые карты
    """

    # Приводим все координаты к одному масштабу
    heatmap_coords_norm = heatmap_coords / heatmap_size
    regressed_coords_norm = regressed_coords / img_size
    target_coords_norm = target_coords / img_size

    # --- Loss тепловых карт ---
    # heatmaps = torch.sigmoid(heatmaps)  # приведение в [0,1]
    # compute_losses:
    heatmaps = torch.sigmoid(heatmaps)
    heatmap_loss = F.mse_loss(heatmaps, target_heatmaps)

    # heatmap_loss = F.mse_loss(heatmaps, target_heatmaps)

    # print("Heatmap max values (first sample):", [heatmaps[0, i].max().item() for i in range(5)])


    # --- Loss координат с soft-argmax ---
    heatmap_coord_loss = F.mse_loss(heatmap_coords_norm, target_coords_norm)

    # --- Loss координат из регрессии ---
    regressed_coord_loss = F.mse_loss(regressed_coords_norm, target_coords_norm)

    # --- Суммарный loss ---
    total_loss = (
        alpha * heatmap_loss +
        beta * heatmap_coord_loss 
        
    )

    return total_loss, {
        "heatmap_loss": alpha * heatmap_loss.item(),
        "heatmap_coord_loss": beta * heatmap_coord_loss.item(),
        "regressed_coord_loss": gamma * regressed_coord_loss.item(),
    }


def pck(preds, preds_reg, gts, img_size = 224, heatmap_size = 64, threshold=0.1):
    """
    Percentage of Correct Keypoints (PCK): процент ключей, попавших в радиус threshold.
    preds, gts: [B, N, 2]
    """

    preds = preds / heatmap_size
    preds_reg = preds_reg / img_size
    gts = gts / img_size

    
    dist = torch.norm(preds - gts, dim=2)
    dist_reg = torch.norm(preds_reg - gts, dim=2)
    correct = (dist < threshold).float()
    correct *= 100
    correct_reg = (dist_reg < threshold).float()
    correct_reg *= 100
    return correct.mean().item(), correct_reg.mean().item()





# Преобразования для изображений
transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ]
)

limb_connections = [
    (0, 1),
    (1, 2),
    (2, 3),  # левая часть лица
    (0, 4),
    (4, 5),
    (5, 6),  # правая часть лица
    (2, 7),
    (3, 7),  # левое ухо
    (5, 8),
    (6, 8),  # правое ухо
    (7, 9),
    (8, 10),  # рот
    (9, 10),  # соединение рта
    (11, 12),  # плечи
    (12, 14),
    (14, 16),  # правая рука
    (16, 18),
    (18, 20),
    (16, 20),  # правая кисть
    (11, 13),
    (13, 15),  # левая рука
    (15, 17),
    (17, 19),
    (15, 19),  # левая кисть
    (11, 23),
    (12, 24),  # туловище к бедрам
    (23, 25),
    (25, 27),
    (27, 29),
    (29, 31),  # левая нога
    (23, 24),  # соединение бедер
    (24, 26),
    (26, 28),
    (28, 30),
    (30, 32),  # правая нога
]

# Загрузка данных
csv_file = "filtered_train.csv"
img_dir = "/home/student/work/train/"
output_image_dir = "results_" + datetime.now().strftime("%Y%m%d_%H%M%S")
dataset = DepthKeypointDataset(
    csv_file, img_dir, transform=transform, limb_connections=limb_connections
)


# Разделение на обучающую и валидационную выборки
_train_dataset, _val_dataset = train_test_split(
    np.arange(len(dataset)), test_size=0.2, random_state=42, shuffle=True
)

train_dataset = Subset(dataset, _train_dataset)
val_dataset = Subset(dataset, _val_dataset)


# Создание DataLoader
train_loader = DataLoader(
    train_dataset,
    batch_size=64,
    shuffle=True,
    # num_workers=12,
    pin_memory=True,
    # persistent_workers=True,
    # prefetch_factor=10,
)
val_loader = DataLoader(
    val_dataset,
    batch_size=64,
    shuffle=False,
    # num_workers=12,
    pin_memory=True,
    # persistent_workers=True,
    # prefetch_factor=10,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# чисто ищем видеоядро
# Инициализация модели, функции потерь и оптимизатора
model = BlazePoseLite().to(device)
criterion = nn.SmoothL1Loss().to(device)  # Функция потерь для координат ключевых точек
optimizer = optim.AdamW(model.parameters(), lr=0.0001)

num_epochs = 50
idx = 0
best_val_loss = 0
best_pck = 0

if __name__ == "__main__":
    for epoch in range(num_epochs):
        model.train()
        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs} - Training"):
            images = batch["image"].to(device)  # [B, 1, 128, 128]
            target_coords = batch["keypoints"][:, :, :2].to(device)  # [B, N, 2]
            target_heatmaps = batch["heatmaps"].to(device)  # [B, N, H, W]
    
            optimizer.zero_grad()
    
            # --- Новый вызов модели ---
            pred_heatmap_coords, pred_regressed_coords, pred_heatmaps = model(images)
    
            # --- Лосс ---
            loss, loss_dict = compute_losses(
                heatmap_coords=pred_heatmap_coords,
                regressed_coords=pred_regressed_coords,
                heatmaps=pred_heatmaps,
                target_coords=target_coords,
                target_heatmaps=target_heatmaps,
                alpha=10.0,
                beta=10.0,
                gamma=0.0,
                heatmap_size=64,
                img_size=224
            )
    
            loss.backward()
            optimizer.step()
    
            idx += 1
    
        # --- Валидация ---
        with torch.no_grad():
            model.eval()
            val_batch = next(iter(val_loader))
            val_images = val_batch["image"].to(device)
            val_target_coords = val_batch["keypoints"][:, :, :2].to(device)
            val_target_heatmaps = val_batch["heatmaps"].to(device)
    
            val_heatmap_coords, val_regressed_coords, val_heatmaps = model(val_images)
    
            raw_heatmap = val_heatmaps[0, 0]  # [H, W] для 1-й точки 1-го объекта
    
            print("\n--- Heatmap stats for keypoint 0 ---")
            print("Shape:", raw_heatmap.shape)
            print("Min:", raw_heatmap.min().item())
            print("Max:", raw_heatmap.max().item())
            print("Mean:", raw_heatmap.mean().item())
            print("Std:", raw_heatmap.std().item())
    
            
            # Если ты используешь softmax-heatmaps, можешь посчитать энтропию:
            p = raw_heatmap.flatten()
            p = p - p.max()  # стабильность softmax
            p = torch.softmax(p, dim=0)
            entropy = -(p * torch.log(p + 1e-8)).sum()
            print("Entropy:", entropy.item())
        
            # Печать самой карты в текстовом виде — опционально
            print("\nHeatmap values (rounded):")
            print(torch.round(raw_heatmap * 100) / 100)  # округлим для наглядности
    
            # logits = pred_heatmaps
            # heatmap = torch.softmax(val_heatmaps.view(64, K, -1), dim=2).view(64, K, H, W)
    
    
            # PCK по координатам из тепловых карт (heatmap_coords)
            val_pck_heatmap, val_pck_regressed = pck(val_heatmap_coords, val_regressed_coords, val_target_coords)
    
                        # 💾 Сохраняем лучшую модель по total loss
            if loss.item() < best_val_loss:
                best_val_loss = loss.item()
                torch.save(model.state_dict(), "best_model.pth")
                print(f"✅ Saved best model (loss={best_val_loss:.4f})")
        
            # 💾 Сохраняем лучшую модель по PCK heatmaps
            if val_pck_heatmap > best_pck:
                best_pck = val_pck_heatmap
                torch.save(model.state_dict(), "best_model_pck2.pth")
                print(f"✅ Saved best model by PCK ({best_pck:.2f})")
    
    
    
            print(
                f"Epoch {epoch} | Total Loss: {loss.item():.4f} | "
                f"Heatmap: {loss_dict['heatmap_loss']:.4f} | "
                f"Coord (soft-argmax): {loss_dict['heatmap_coord_loss']:.4f} | "
                f"PCK Heatmaps: {val_pck_heatmap:.4f} | "
                # f"PCK Regressed: {val_pck_regressed:.4f} "
            )
    
            # --- Визуализация ---
            visualize_keypoints_with_heatmaps(
                image=val_images[0].cpu(),
                heatmaps=val_heatmaps[0].cpu(),
                pred_points=val_heatmap_coords[0].cpu().numpy(),
                gt_points=val_target_coords[0].cpu().numpy(),
                pred_points_reg=val_regressed_coords[0].cpu().numpy(),
                limb_connections=limb_connections,
                output_dir=output_image_dir,
                epoch=epoch,
                sample_idx=0,
                gt_heatmaps=val_target_heatmaps[0].cpu(),
            )
