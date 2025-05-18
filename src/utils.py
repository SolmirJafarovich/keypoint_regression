import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from src.config import config

# connections for drawing skeleton
limb_connections = [
    (0, 1),
    (1, 2),
    (2, 3),
    (0, 4),
    (4, 5),
    (5, 6),
    (2, 7),
    (3, 7),
    (5, 8),
    (6, 8),
    (7, 9),
    (8, 10),
    (9, 10),
    (11, 12),
    (12, 14),
    (14, 16),
    (16, 18),
    (18, 20),
    (16, 20),
    (11, 13),
    (13, 15),
    (15, 17),
    (17, 19),
    (15, 19),
    (11, 23),
    (12, 24),
    (23, 25),
    (25, 27),
    (27, 29),
    (29, 31),
    (23, 24),
    (24, 26),
    (26, 28),
    (28, 30),
    (30, 32),
]


def to_numpy(tensor, apply_sigmoid=False):
    if isinstance(tensor, tf.Tensor):
        tensor = tf.sigmoid(tensor) if apply_sigmoid else tensor
        return tensor.numpy()
    return tensor


def draw_points(img, points, color):
    h, w = img.shape[:2]
    for x, y in points:
        x, y = int(x * w), int(y * h)
        if 0 <= x < w and 0 <= y < h:
            cv2.circle(img, (x, y), 3, color, -1)


def draw_connections(img, points, color):
    h, w = img.shape[:2]
    for i, j in limb_connections:
        if i < len(points) and j < len(points):
            x1, y1 = int(points[i][0] * w), int(points[i][1] * h)
            x2, y2 = int(points[j][0] * w), int(points[j][1] * h)
            cv2.line(img, (x1, y1), (x2, y2), color, 1)


def normalize_image(image):
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.ndim == 3 and image.shape[-1] == 1:
        image = cv2.cvtColor(image.squeeze(), cv2.COLOR_GRAY2BGR)
    elif image.ndim == 3 and image.shape[-1] == 3:
        image = image
    else:
        raise ValueError(f"Unsupported image shape: {image.shape}")
    if image.dtype != np.uint8:
        image = (image * 255).clip(0, 255).astype(np.uint8)
    return image


def visualize(
    image,
    heatmaps_pred,
    pred_points=None,
    gt_points=None,
    epoch=0,
    sample_idx=0,
    figsize=(18, 6),
    gt_heatmaps=None,
):
    config.checkpoint.mkdir(exist_ok=True, parents=True)

    try:
        image = normalize_image(to_numpy(image))
        heatmaps_pred = to_numpy(heatmaps_pred, apply_sigmoid=True)
        gt_heatmaps = (
            to_numpy(gt_heatmaps, apply_sigmoid=True)
            if gt_heatmaps is not None
            else None
        )
        pred_points = to_numpy(pred_points) if pred_points is not None else None
        gt_points = to_numpy(gt_points) if gt_points is not None else None

        vis_img = image.copy()

        if gt_points is not None:
            draw_points(vis_img, gt_points, (255, 0, 0))  # Blue
            draw_connections(vis_img, gt_points, (0, 255, 0))  # Green
        if pred_points is not None:
            draw_points(vis_img, pred_points, (0, 0, 255))  # Red
            draw_connections(vis_img, pred_points, (255, 255, 0))  # Yellow

        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=figsize)
        ax1.imshow(cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB))
        ax1.set_title("Keypoints (Red=Pred, Blue=GT)")
        ax1.axis("off")

        if heatmaps_pred is not None:
            hm = np.max(heatmaps_pred, axis=0)
            hm = (hm - hm.min()) / (hm.max() + 1e-6)
            hm = cv2.resize(hm, (image.shape[1], image.shape[0]))
            im2 = ax2.imshow(hm, cmap="jet", vmin=0, vmax=1)
            plt.colorbar(im2, ax=ax2)
            ax2.set_title("Predicted Heatmap")
            ax2.axis("off")

        if gt_heatmaps is not None:
            gt_hm = (
                np.max(gt_heatmaps, axis=0) if gt_heatmaps.ndim == 3 else gt_heatmaps
            )
            im3 = ax3.imshow(gt_hm, cmap="jet", vmin=0, vmax=1)
            plt.colorbar(im3, ax=ax3)
            ax3.set_title("GT Heatmap")
            ax3.axis("off")

        plt.tight_layout()
        save_path = config.checkpoint / "viz"
        save_path.mkdir(exist_ok=True, parents=True)
        plt.savefig(
            save_path / f"epoch_{epoch}_sample_{sample_idx}.png",
            bbox_inches="tight",
            dpi=100,
        )
        plt.close(fig)

    except Exception as e:
        print(f"Visualization error: {e}")
        if "fig" in locals():
            plt.close(fig)
