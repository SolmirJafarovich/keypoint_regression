import json
import os

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from src.config import config


class DepthKeypointDataset(Dataset):
    def __init__(
        self,
        transform=None,
        limb_connections=None,
        visualize=False,
        output_dir=None,
    ):
        """
        Args:
            csv_file (str): путь к CSV-файлу с координатами ключевых точек.
            img_dir (str): директория с изображениями.
            transform (callable, optional): трансформация изображений.
            limb_connections (list of tuple, optional): соединения для отрисовки скелета.
            visualize (bool): если True, сохраняет изображения с наложенным скелетом.
            output_dir (str): директория для сохранения визуализаций.
        """
        self.data = pd.read_csv(config.regressor.csv_file)
        self.transform = transform
        self.limb_connections = limb_connections
        self.visualize = visualize
        self.output_dir = output_dir

        if self.visualize and self.output_dir:
            os.makedirs(self.output_dir, exist_ok=True)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Загрузка изображения
        img_path = config.regressor.img_dir / self.data.iloc[idx, 0]
        image = Image.open(img_path).convert(
            "L"
        )  # глубинные изображения — одноканальные

        # Получение координат ключевых точек
        raw = self.data.iloc[idx, 1:].values.astype(
            np.float32
        )  # [33*4 = 132] значений: x, y, z, v, ...
        keypoints = raw.reshape(33, 4)[:, :2]  # Берём только x и y → shape: (33, 2)

        img_width, img_height = 224, 224  # Явно указываем размер
        keypoints[:, 0] *= img_width
        keypoints[:, 1] *= img_height

        # Преобразования
        if self.transform:
            image = self.transform(image)

        # Генерация тепловых карт
        heatmaps = self.generate_heatmaps(keypoints, (64, 64), sigma=7.5)

        return {"image": image, "keypoints": keypoints, "heatmaps": heatmaps}

    def generate_heatmaps(self, keypoints, image_shape, sigma=1.5):
        """Создаёт гауссовы тепловые карты для каждого ключа"""
        h, w = image_shape
        num_keypoints = keypoints.shape[0]
        heatmaps = np.zeros((num_keypoints, h, w), dtype=np.float32)

        # Масштабируем координаты из 224x224 в 64x64
        scale_x = w / 224.0
        scale_y = h / 224.0

        for i, (x, y) in enumerate(keypoints):
            if x < 0 or y < 0:
                continue

            # Масштабирование координат
            x_scaled = x * scale_x
            y_scaled = y * scale_y

            xv, yv = np.meshgrid(np.arange(w), np.arange(h))
            heatmaps[i] = np.exp(
                -((xv - x_scaled) ** 2 + (yv - y_scaled) ** 2) / (2 * sigma**2)
            )

        return torch.tensor(heatmaps, dtype=torch.float32)

    def save_visualization(self, image_tensor, keypoints, idx):
        """Сохраняет изображение с наложенными ключевыми точками и соединениями"""
        image_np = image_tensor.squeeze().numpy() * 255
        image_np = np.clip(image_np, 0, 255).astype(np.uint8)
        image_color = cv2.cvtColor(image_np, cv2.COLOR_GRAY2BGR)

        # Отрисовка ключевых точек
        for x, y in keypoints:
            if x >= 0 and y >= 0:
                cv2.circle(image_color, (int(x), int(y)), 2, (0, 255, 0), -1)

        # Отрисовка соединений
        if self.limb_connections:
            for i, j in self.limb_connections:
                x1, y1 = keypoints[i]
                x2, y2 = keypoints[j]
                if min(x1, y1, x2, y2) >= 0:
                    cv2.line(
                        image_color,
                        (int(x1), int(y1)),
                        (int(x2), int(y2)),
                        (0, 0, 255),
                        1,
                    )

        save_path = os.path.join(self.output_dir, f"sample_{idx}.png")
        cv2.imwrite(save_path, image_color)


class CachedPoseDataset(Dataset):
    def init(self, cache_file, transform=None):
        with open(cache_file, "r") as f:
            self.samples = json.load(f)
        self.transform = transform or transforms.ToTensor()

    def len(self):
        return len(self.samples)

    def getitem(self, idx):
        sample = self.samples[idx]
        image = Image.open(sample["path"]).convert("L")
        image_tensor = self.transform(image)

        return {
            "image": image_tensor,
            "keypoints": torch.tensor(sample["keypoints"], dtype=torch.float),
            "label": torch.tensor(sample["label"], dtype=torch.long),
        }
