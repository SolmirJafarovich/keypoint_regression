import os

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from tqdm import tqdm

from src.config import config, device
from src.dataset import CachedPoseDataset
from src.models import CombinedClassifier

# Настройка
transform = transforms.Compose(
    [transforms.Resize((config.img_size, config.img_size)), transforms.ToTensor()]
)

# Датасеты и загрузчики
dataset = CachedPoseDataset("cached_keypoints1_flattened.json", transform=transform)
train_idx, val_idx = train_test_split(
    list(range(len(dataset))), test_size=0.3, random_state=42
)
train_loader = DataLoader(Subset(dataset, train_idx), batch_size=64, shuffle=True)
val_loader = DataLoader(Subset(dataset, val_idx), batch_size=64)
config.init_checkpoint("classifier")

# Модель, оптимизатор, лосс
model = CombinedClassifier().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

# --- Классы ---
class_to_label = {"Belly": 0, "Back": 1, "Right_side": 2, "Left_side": 3}

# --- Путь к данным (эмулируем структуру папок) ---
root_dir = "depth"

# Инициализация параметров сохранения
best_acc = 0.0
output_image_dir = "output"

# Тренировка и валидация
num_epochs = 20
for epoch in range(num_epochs):
    model.train()

    for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs} - Training"):
        images = batch["image"].to(device)

        keypoints = batch["keypoints"].to(device)
        # print("Keypoints shape:", batch["keypoints"].shape)  # Должно быть (batch_size, 66)

        labels = batch["label"].to(device)

        outputs = model(images, keypoints)

        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    model.eval()
    preds, targets = [], []

    with torch.no_grad():
        for batch in tqdm(
            val_loader, desc=f"Epoch {epoch + 1}/{num_epochs} - Validation"
        ):
            images = batch["image"].to(device)
            keypoints = batch["keypoints"].to(device)
            labels = batch["label"].to(device)

            predictions = model(images, keypoints).argmax(dim=1).cpu()
            preds += predictions.tolist()
            targets += labels.cpu().tolist()

    acc = accuracy_score(targets, preds)
    prec = precision_score(targets, preds, average="weighted", zero_division=0)
    rec = recall_score(targets, preds, average="weighted", zero_division=0)
    f1 = f1_score(targets, preds, average="weighted", zero_division=0)

    if acc > best_acc:
        os.makedirs(output_image_dir, exist_ok=True)
        best_acc = acc
        torch.save(
            model.state_dict(), os.path.join(output_image_dir, "best_classifier.pth")
        )
        print(f"✅ Saved best model by accuracy ({best_acc:.2f})")

    print(f"\nEpoch {epoch + 1}")
    print(
        f"Accuracy: {acc:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f} | F1: {f1:.4f}"
    )
