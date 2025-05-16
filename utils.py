import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch


def visualize_keypoints_with_heatmaps(
    image,
    heatmaps,
    pred_points=None,
    gt_points=None,
    limb_connections=None,
    output_dir=None,
    epoch=0,
    sample_idx=0,
    figsize=(18, 6),
    gt_heatmaps=None,
):
    """
    Визуализация: ключевые точки, предсказанные heatmap и GT heatmap (три картинки рядом)
    """
    # Создаем директорию если нужно
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    try:
        # Конвертация данных
        if isinstance(image, torch.Tensor):
            image = image.detach().cpu().numpy()
            if image.ndim == 3 and image.shape[0] in [1, 3]:  # CHW -> HWC
                image = np.transpose(image, (1, 2, 0))
            image = (image * 255).clip(0, 255).astype(np.uint8)

        if isinstance(heatmaps, torch.Tensor):
            heatmaps = heatmaps.detach().cpu().numpy()

        if isinstance(gt_heatmaps, torch.Tensor):
            gt_heatmaps = gt_heatmaps.detach().cpu().numpy()

        # Преобразование точек в numpy array если они тензоры
        if pred_points is not None and isinstance(pred_points, torch.Tensor):
            pred_points = pred_points.detach().cpu().numpy()

        if gt_points is not None and isinstance(gt_points, torch.Tensor):
            gt_points = gt_points.detach().cpu().numpy()

        # Создаем фигуру с тремя subplots
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=figsize)

        # 1. Изображение с ключевыми точками
        img_with_points = image.copy()
        if image.ndim == 2 or (image.ndim == 3 and image.shape[-1] == 1):
            img_with_points = cv2.cvtColor(img_with_points, cv2.COLOR_GRAY2BGR)

        pred_points = pred_points / 64 * 224

        # Рисуем GT точки (синие)
        if gt_points is not None:
            for point in gt_points:
                if len(point) >= 2:
                    x, y = point[0], point[1]
                    if (
                        0 <= x < img_with_points.shape[1]
                        and 0 <= y < img_with_points.shape[0]
                    ):
                        cv2.circle(
                            img_with_points, (int(x), int(y)), 3, (255, 0, 0), -1
                        )

        # Рисуем предсказанные точки (красные)
        if pred_points is not None:
            for point in pred_points:
                if len(point) >= 2:
                    x, y = point[0], point[1]
                    if (
                        0 <= x < img_with_points.shape[1]
                        and 0 <= y < img_with_points.shape[0]
                    ):
                        cv2.circle(
                            img_with_points, (int(x), int(y)), 3, (0, 0, 255), -1
                        )

        # Рисуем соединения (зеленые)
        if limb_connections is not None and gt_points is not None:
            for i, j in limb_connections:
                if i < len(gt_points) and j < len(gt_points):
                    x1, y1 = gt_points[i][0], gt_points[i][1]
                    x2, y2 = gt_points[j][0], gt_points[j][1]
                    if all(0 <= x < img_with_points.shape[1] for x in [x1, x2]) and all(
                        0 <= y < img_with_points.shape[0] for y in [y1, y2]
                    ):
                        cv2.line(
                            img_with_points,
                            (int(x1), int(y1)),
                            (int(x2), int(y2)),
                            (0, 255, 0),
                            1,
                        )

        if limb_connections is not None and pred_points is not None:
            for i, j in limb_connections:
                if i < len(pred_points) and j < len(pred_points):
                    x1, y1 = pred_points[i][0], pred_points[i][1]
                    x2, y2 = pred_points[j][0], pred_points[j][1]
                    if all(0 <= x < img_with_points.shape[1] for x in [x1, x2]) and all(
                        0 <= y < img_with_points.shape[0] for y in [y1, y2]
                    ):
                        cv2.line(
                            img_with_points,
                            (int(x1), int(y1)),
                            (int(x2), int(y2)),
                            (0, 255, 255),
                            1,
                        )

        ax1.imshow(cv2.cvtColor(img_with_points, cv2.COLOR_BGR2RGB))
        ax1.set_title("Keypoints (Red=Pred, Blue=GT)")
        ax1.axis("off")

        if heatmaps is not None:
            heatmap_vis = heatmaps.max(axis=0)
            heatmap_vis -= heatmap_vis.min()
            heatmap_vis /= heatmap_vis.max() + 1e-6
            heatmap_vis = cv2.resize(heatmap_vis, (image.shape[1], image.shape[0]))

            im = ax2.imshow(heatmap_vis, cmap="jet", vmin=0, vmax=1)
            plt.colorbar(im, ax=ax2)
            ax2.set_title("Predicted Heatmap")
            ax2.axis("off")

        # 3. GT heatmap
        if gt_heatmaps is not None:
            max_gt_heatmap = (
                gt_heatmaps.max(axis=0) if gt_heatmaps.ndim == 3 else gt_heatmaps
            )
            im = ax3.imshow(max_gt_heatmap, cmap="jet", vmin=0, vmax=1)
            plt.colorbar(im, ax=ax3)
            ax3.set_title("GT Heatmap")
            ax3.axis("off")

        plt.tight_layout()

        # Сохранение и закрытие
        if output_dir:
            save_path = os.path.join(
                output_dir, f"epoch_{epoch}_sample_{sample_idx}.png"
            )
            plt.savefig(save_path, bbox_inches="tight", dpi=100)
            plt.close(fig)
        else:
            plt.show()

    except Exception as e:
        print(f"Visualization error: {str(e)}")
        if "fig" in locals():
            plt.close(fig)
