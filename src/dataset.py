import os

import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image

from src.config import config


class DepthKeypointDataset:
    def __init__(self, data: pd.DataFrame):
        # Load csv with filenames and keypoints
        self.data = data

    def __len__(self):
        return len(self.data)

    def generator(self):
        for idx in range(len(self.data)):
            # Load image and resize to 224x224 grayscale
            img_path = config.regressor.img_dir / str(self.data.iloc[idx, 0])
            image = (
                Image.open(img_path)
                .convert("L")
                .resize((config.img_size, config.img_size))
            )
            image_np = np.array(image, dtype=np.float32) / 255.0
            image_np = np.expand_dims(image_np, axis=-1)  # shape: (224, 224, 1)

            # Load keypoints: 33 ключевые точки, берем только x и y (первые 2)
            raw_keypoints = self.data.iloc[idx, 1:].values.astype(np.float32)
            keypoints = raw_keypoints.reshape(33, 4)[:, :2]

            keypoints[:, 0] *= config.img_size
            keypoints[:, 1] *= config.img_size

            # Генерируем тепловые карты (heatmaps) размером 64x64 для каждого ключевого пункта
            heatmaps = self.generate_heatmaps(keypoints, (64, 64), sigma=7.5)

            yield image_np, keypoints, heatmaps

    def get_dataset(self, batch_size=32, shuffle=True):
        output_signature = (
            tf.TensorSpec(shape=(224, 224, 1), dtype=tf.float32),
            tf.TensorSpec(shape=(33, 2), dtype=tf.float32),
            tf.TensorSpec(shape=(33, 64, 64), dtype=tf.float32),
        )
        dataset = tf.data.Dataset.from_generator(
            self.generator, output_signature=output_signature
        )
        if shuffle:
            dataset = dataset.shuffle(buffer_size=len(self.data))
        dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
        return dataset

    def generate_heatmaps(self, keypoints, shape, sigma=7.5):
        h, w = shape
        num_keypoints = keypoints.shape[0]
        heatmaps = np.zeros((num_keypoints, h, w), dtype=np.float32)

        # Precompute meshgrid once
        xv, yv = np.meshgrid(np.arange(w), np.arange(h))

        scale_x = w / config.img_size
        scale_y = h / config.img_size

        for i, (x, y) in enumerate(keypoints):
            if x < 0 or y < 0:
                continue

            x_scaled = x * scale_x
            y_scaled = y * scale_y

            # Gaussian heatmap
            heatmaps[i] = np.exp(
                -((xv - x_scaled) ** 2 + (yv - y_scaled) ** 2) / (2 * sigma**2)
            )

        return heatmaps.astype(np.float32)


class KeypointPrecomputedDataset:
    def __init__(self):
        self.class_to_label = {"Belly": 0, "Back": 1, "Right_side": 2, "Left_side": 3}
        self.samples = []
        self._prepare_samples()

    def _prepare_samples(self):
        for class_path in config.eval_dataset.iterdir():
            if not class_path.is_dir():
                continue

            label = self.class_to_label[class_path.name]
            for ext in ("*.jpg", "*.png"):
                for fname in class_path.glob(ext):
                    image = (
                        Image.open(fname)
                        .convert("L")
                        .resize((config.img_size, config.img_size))
                    )
                    image_np = np.array(image, dtype=np.float32) / 255.0
                    image_np = np.expand_dims(image_np, axis=-1)  # shape: (H, W, 1)
                    self.samples.append((image_np, label))

    def generator(self):
        for image_np, label in self.samples:
            yield image_np, label

    def get_dataset(self):
        output_types = (tf.float32, tf.int32)
        output_shapes = ((config.img_size, config.img_size, 1), ())
        return tf.data.Dataset.from_generator(
            self.generator, output_types, output_shapes
        )
