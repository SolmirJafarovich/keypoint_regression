import json
import time
from pathlib import Path
from statistics import mean
from typing import Annotated

import torch
import typer
from pydantic import BaseModel, ConfigDict

from src.config import config, device
from src.models import ClassifierWrapper, RegressorWrapper, BlazePoseLite, CombinedClassifier
from src.utils import soft_argmax_2d

from torchmetrics import Accuracy, F1Score, Precision, Recall
from rich.progress import track

app = typer.Typer()


class InferenceStep(BaseModel):
    preds: torch.Tensor
    targets: torch.Tensor
    time: float  # seconds
    extra: dict = {}

    model_config = ConfigDict(arbitrary_types_allowed=True)


class PytorchInferenceGenerator:
    def __init__(self, regressor_path: Path, classifier_path: Path):
        self.regressor = BlazePoseLite()
        state_dict = torch.load(regressor_path / "weights.pth", weights_only=True)
        self.regressor.load_state_dict(state_dict)

        self.classifier = CombinedClassifier()
        state_dict = torch.load(classifier_path / "weights.pth", weights_only=True)
        self.classifier.load_state_dict(state_dict)

        self.regressor.eval()
        self.classifier.eval()

    def __call__(self, num_images=100) -> list[InferenceStep]:
        steps = []
        with torch.no_grad():
            for _ in range(num_images):
                # Случайное изображение-одноканальный шум
                image = torch.randint(0, 256, (1, 1, 224, 224), dtype=torch.uint8)
                image = image.float() / 255.0  # float32 для модели

                target = torch.tensor([0])  # фиктивная метка

                start_reg = time.perf_counter()
                heatmaps = self.regressor(image)
                keypoints = soft_argmax_2d(heatmaps)
                keypoints = keypoints.view(keypoints.shape[0], -1)
                end_reg = time.perf_counter()

                start_clf = time.perf_counter()
                preds = self.classifier(image, keypoints)
                end_clf = time.perf_counter()

                steps.append(
                    InferenceStep(
                        preds=preds,
                        targets=target,
                        time=end_clf - start_reg,
                        extra={
                            "reg_time": end_reg - start_reg,
                            "clf_time": end_clf - start_clf,
                        },
                    )
                )
        return steps


class InferenceGenerator:
    def __init__(self, regressor_path: Path, classifier_path: Path):
        self.regressor = (
            RegressorWrapper(regressor_path / "weights_quant.tflite").eval().to(device)
        )
        self.classifier = (
            ClassifierWrapper(classifier_path / "weights_quant.tflite").eval().to(device)
        )

    def __call__(self, num_images=100) -> list[InferenceStep]:
        steps = []
        with torch.no_grad():
            for _ in range(num_images):
                image = torch.randint(0, 256, (1, 1, 224, 224), dtype=torch.uint8).to(device)
                image_float = image.float() / 255.0

                target = torch.tensor([0])  # фиктивная метка

                start_reg = time.perf_counter()
                heatmaps = self.regressor(image_float)
                keypoints = soft_argmax_2d(heatmaps)
                keypoints = (keypoints * 255).to(torch.uint8)
                end_reg = time.perf_counter()

                start_clf = time.perf_counter()
                preds = self.classifier(image, keypoints)
                preds = (preds / 255).to(torch.float)
                end_clf = time.perf_counter()

                steps.append(
                    InferenceStep(
                        preds=preds,
                        targets=target,
                        time=end_clf - start_reg,
                        extra={
                            "reg_time": end_reg - start_reg,
                            "clf_time": end_clf - start_clf,
                        },
                    )
                )
        return steps


class Metrics:
    def __init__(self):
        self.metrics = {
            "Accuracy": Accuracy(
                task="multiclass",
                num_classes=config.classifier.num_classes,
                average="weighted",
            ),
            "Precision": Precision(
                task="multiclass",
                num_classes=config.classifier.num_classes,
                average="weighted",
            ),
            "Recall": Recall(
                task="multiclass",
                num_classes=config.classifier.num_classes,
                average="weighted",
            ),
            "F1": F1Score(
                task="multiclass",
                num_classes=config.classifier.num_classes,
                average="weighted",
            ),
        }

    def update(self, preds, targets):
        for _, metric in self.metrics.items():
            metric.update(preds, targets)

    def compute(self):
        result = {}
        for name, metric in self.metrics.items():
            result[name] = metric.compute().cpu().tolist()
        return result


@app.command()
def evaluate(
    reg: Path = typer.Option(..., help="Path to regressor folder"),
    cl: Path = typer.Option(..., help="Path to classifier folder"),
    pytorch: Annotated[bool, typer.Option("--pytorch")] = False,
):
    config.init_checkpoint("metrics")
    metrics = Metrics()

    if pytorch:
        generator = PytorchInferenceGenerator(regressor_path=reg, classifier_path=cl)
    else:
        generator = InferenceGenerator(regressor_path=reg, classifier_path=cl)

    steps = list(track(generator(), description="Evaluating"))

    # метрики (условные, т.к. метки фиктивны)
    for step in steps:
        metrics.update(preds=step.preds, targets=step.targets)

    computed = metrics.compute()

    times = [s.time for s in steps]
    reg_times = [s.extra["reg_time"] for s in steps]
    clf_times = [s.extra["clf_time"] for s in steps]

    computed["Average Total Time"] = mean(times)
    computed["Average Regressor Time"] = mean(reg_times)
    computed["Average Classifier Time"] = mean(clf_times)

    typer.echo("=== METRICS ===")
    for k, v in computed.items():
        typer.echo(f"{k:25s}: {v:.6f}")

    with open(config.checkpoint / "metrics.json", "w") as f:
        json.dump(computed, f, indent=4)


if __name__ == "__main__":
    app()
