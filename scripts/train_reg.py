import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from rich.console import Console
from rich.live import Live
from rich.text import Text
from sklearn.model_selection import train_test_split

from src.config import config
from src.dataset import DepthKeypointDataset
from src.models.regressor import build_blazepose_lite, soft_argmax_2d
from src.utils import visualize

console = Console()


def compute_losses_tf(
    pred_heatmaps, target_coords, target_heatmaps, alpha=0.1, beta=10.0
):
    target_heatmaps = tf.transpose(target_heatmaps, perm=[0, 2, 3, 1])
    heatmap_coords = soft_argmax_2d(pred_heatmaps)
    heatmap_coords_norm = heatmap_coords / config.regressor.heatmap_size
    target_coords_norm = target_coords / config.img_size

    pred_heatmaps_sigmoid = tf.sigmoid(pred_heatmaps)
    heatmap_loss = tf.reduce_mean(tf.square(pred_heatmaps_sigmoid - target_heatmaps))
    heatmap_coord_loss = tf.reduce_mean(
        tf.square(heatmap_coords_norm - target_coords_norm)
    )

    total_loss = alpha * heatmap_loss + beta * heatmap_coord_loss
    return total_loss, {
        "heatmap_loss": heatmap_loss,
        "heatmap_coord_loss": heatmap_coord_loss,
    }


def pck_tf(preds, gts, threshold=0.1):
    # preds, gts: (batch_size, 33, 2), normalized coords [0..1]
    dist = tf.norm(preds - gts, axis=-1)
    correct = tf.cast(dist < threshold, tf.float32) * 100.0
    return tf.reduce_mean(correct)


@tf.function
def train_step(images, target_coords, target_heatmaps):
    with tf.GradientTape() as tape:
        pred_heatmaps = model(images, training=True)
        loss, loss_dict = compute_losses_tf(
            pred_heatmaps, target_coords, target_heatmaps, alpha=10.0, beta=10.0
        )
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss, loss_dict


@tf.function
def val_step(images):
    pred_heatmaps = model(images, training=False)
    return pred_heatmaps


if __name__ == "__main__":
    config.init_checkpoint("regressor")

    # Загрузка исходного датасета (csv)
    full_df = pd.read_csv(config.regressor.csv_file)

    # Разбиение на train/val
    train_df, val_df = train_test_split(
        full_df, test_size=0.2, random_state=42, shuffle=True
    )

    train_dataset = DepthKeypointDataset(train_df).get_dataset(
        batch_size=32, shuffle=True
    )
    val_dataset = DepthKeypointDataset(val_df).get_dataset(batch_size=32, shuffle=True)

    model = build_blazepose_lite()
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)

    num_epochs = 150
    best_pck = 0.0

    with Live(refresh_per_second=4) as live:
        for epoch in range(num_epochs):
            train_losses = []
            for batch in train_dataset:
                images, target_coords, target_heatmaps = batch
                loss, _ = train_step(images, target_coords, target_heatmaps)
                train_losses.append(loss.numpy())

            avg_train_loss = np.mean(train_losses)

            val_pcks = []
            val_losses = []
            for i, batch in enumerate(val_dataset):
                images, keypoints, heatmaps = batch
                heatmaps_pred = val_step(images)
                keypoints_pred = soft_argmax_2d(heatmaps_pred)

                loss, _ = compute_losses_tf(
                    pred_heatmaps=heatmaps_pred,
                    target_coords=keypoints,
                    target_heatmaps=heatmaps,
                )
                val_losses.append(loss.numpy())

                val_pck = pck_tf(preds=keypoints_pred, gts=keypoints)
                val_pcks.append(val_pck)

                if i == 0:
                    visualize(
                        image=images[0],
                        heatmaps_pred=heatmaps_pred[0],
                        pred_points=keypoints_pred[0],
                        gt_points=keypoints[0],
                        gt_heatmaps=heatmaps[0],
                    )

            avg_val_pck = np.mean(val_pcks)

            avg_val_loss = np.mean(val_losses)

            saved = ""
            if avg_val_pck > best_pck:
                best_pck = avg_val_pck
                model.save_weights(str(config.checkpoint / "best_model_tf.weights.h5"))
                saved = "🏆"

            status_text = Text(
                f"Epoch {epoch + 1}/{num_epochs} | "
                f"Train Loss: {avg_train_loss:.4f} | "
                f"Val Loss: {avg_val_loss:.4f} | "
                f"Val PCK: {avg_val_pck:.2f}% {saved}",
                style="bold green",
            )

            # Обновляем отображение в live
            live.update(status_text)

    console.print("[bold green]Training completed.[/]")
