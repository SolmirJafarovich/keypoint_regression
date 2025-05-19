import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import tensorflow as tf
from tensorflow.keras.callbacks import Callback
from tqdm import tqdm

from src.config import config
from src.dataset_cl import KeypointPrecomputedDatasetTF
from src.models import build_combined_classifier

# Параметры
img_size = config.img_size
batch_size = 64
num_epochs = 20
output_dir = "./data/classifier_tf/output"
os.makedirs(output_dir, exist_ok=True)

# Загрузка датасета
dataset = KeypointPrecomputedDatasetTF(
    root_dir="../depth",
    keypoint_file="./data/raw/cached_keypoints1_flattened.json",
    img_size=config.img_size
)


# Разделение вручную (если хочешь контролировать баланс классов, перемешивание и т.п.)
samples = dataset.samples

print(f"[DEBUG] Кол-во samples: {len(samples)}")
print(f"[DEBUG] Пример: {samples[:1]}")

np.random.seed(42)
np.random.shuffle(samples)

split_index = int(0.7 * len(samples))
train_samples = samples[:split_index]
val_samples = samples[split_index:]

# Создание tf.data.Dataset
train_ds = dataset.get_tf_dataset_from_samples(train_samples, batch_size=64, shuffle=True)
val_ds = dataset.get_tf_dataset_from_samples(val_samples, batch_size=64, shuffle=False)

# Получим размерность координат ключевых точек
keypoint_dim = len(train_samples[0][1])  # [img_path, keypoints, label]

# Создание модели
model = build_combined_classifier(
    input_shape_img=(img_size, img_size, 1),
    keypoint_dim=keypoint_dim,
    num_classes=4,
)
model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=["accuracy"]
)

# Кастомный callback для оценки F1, precision, recall
class MetricsCallback(Callback):
    def __init__(self, val_ds):
        super().__init__()
        self.val_ds = val_ds
        self.best_acc = 0.0

    def on_epoch_end(self, epoch, logs=None):
        y_true = []
        y_pred = []

        for (img_batch, kp_batch), y_batch in self.val_ds:
            preds = self.model.predict_on_batch([img_batch, kp_batch])
            y_pred.extend(np.argmax(preds, axis=1))
            y_true.extend(y_batch.numpy())

        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, average="weighted", zero_division=0)
        rec = recall_score(y_true, y_pred, average="weighted", zero_division=0)
        f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

        print(f"\nEpoch {epoch+1}")
        print(f"Accuracy: {acc:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f} | F1: {f1:.4f}")

        if acc > self.best_acc:
            self.best_acc = acc
            self.model.save(os.path.join(output_dir, "best_classifier.h5"))
            print(f"✅ Saved best model by accuracy ({self.best_acc:.2f})")


# Обучение
model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=num_epochs,
    callbacks=[MetricsCallback(val_ds)],
    verbose=1  # можно заменить на tqdm отдельно
)
