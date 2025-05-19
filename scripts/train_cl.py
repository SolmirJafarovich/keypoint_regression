import numpy as np
import tensorflow as tf
from rich.console import Console
from rich.live import Live
from rich.text import Text
from sklearn.model_selection import train_test_split

from src.models.classifier import build_combined_classifier
from src.dataset_cl import KeypointPrecomputedDatasetTF
from src.config import config

console = Console()

# --- Метрики ---
def compute_accuracy(preds, labels):
    correct = tf.equal(tf.argmax(preds, axis=1), tf.argmax(labels, axis=1))
    return tf.reduce_mean(tf.cast(correct, tf.float32)) * 100.0

@tf.function
def train_step(images, keypoints, labels):
    with tf.GradientTape() as tape:
        preds = model([images, keypoints], training=True)
        loss = tf.keras.losses.categorical_crossentropy(labels, preds)
        loss = tf.reduce_mean(loss)
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    acc = compute_accuracy(preds, labels)
    return loss, acc

@tf.function
def val_step(images, keypoints, labels):
    preds = model([images, keypoints], training=False)
    loss = tf.keras.losses.categorical_crossentropy(labels, preds)
    loss = tf.reduce_mean(loss)
    acc = compute_accuracy(preds, labels)
    return loss, acc

if __name__ == "__main__":
    config.init_checkpoint("classifier")

    print(tf.test.gpu_device_name() or 'Используется CPU')

    # Параметры
    BATCH_SIZE = 64
    IMG_SIZE = 224
    EPOCHS = 100

    # Загружаем датасет
    dataset = KeypointPrecomputedDatasetTF(
        root_dir=config.classifier.image_root,
        keypoint_file=config.classifier.keypoints_json,
        img_size=IMG_SIZE,
    )

    # Получаем все данные
    images, keypoints, labels = dataset.get_all_data()

    # One-hot
    labels = tf.keras.utils.to_categorical(labels, num_classes=4)

    # Сплит
    (train_images, val_images,
     train_kps, val_kps,
     train_labels, val_labels) = train_test_split(
        images, keypoints, labels,
        test_size=0.2, random_state=42, shuffle=True
    )

    # Сборка датасетов
    def make_ds(images, kps, labels, shuffle):
        ds = tf.data.Dataset.from_tensor_slices(((images, kps), labels))
        if shuffle:
            ds = ds.shuffle(buffer_size=len(images))
        return ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    train_dataset = make_ds(train_images, train_kps, train_labels, shuffle=True)
    val_dataset = make_ds(val_images, val_kps, val_labels, shuffle=False)

    # Модель
    model = build_combined_classifier(
        input_shape_img=(IMG_SIZE, IMG_SIZE, 1),
        keypoint_dim=66,
        num_classes=4
    )

    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
    best_acc = 0.0

    with Live(refresh_per_second=4) as live:
        for epoch in range(EPOCHS):
            train_losses, train_accuracies = [], []
            for (images, keypoints), labels in train_dataset:
                loss, acc = train_step(images, keypoints, labels)
                train_losses.append(loss.numpy())
                train_accuracies.append(acc.numpy())

            val_losses, val_accuracies = [], []
            for (images, keypoints), labels in val_dataset:
                loss, acc = val_step(images, keypoints, labels)
                val_losses.append(loss.numpy())
                val_accuracies.append(acc.numpy())

            avg_train_loss = np.mean(train_losses)
            avg_train_acc = np.mean(train_accuracies)
            avg_val_loss = np.mean(val_losses)
            avg_val_acc = np.mean(val_accuracies)

            saved = ""
            if avg_val_acc > best_acc:
                best_acc = avg_val_acc
                model.save(config.checkpoint / "classifier.keras")
                saved = "🏆"

            status = Text(
                f"Epoch {epoch + 1}/{EPOCHS} | "
                f"Train Loss: {avg_train_loss:.4f}, Acc: {avg_train_acc:.2f}% | "
                f"Val Loss: {avg_val_loss:.4f}, Acc: {avg_val_acc:.2f}% {saved}",
                style="bold green"
            )
            live.update(status)

    console.print("[bold green]Training complete.[/]")
