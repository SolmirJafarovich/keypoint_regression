import json
import time
from pathlib import Path
from statistics import mean

import torch
import typer
from pydantic import BaseModel
from torch.utils.data import DataLoader
from torchmetrics import Accuracy, F1Score, Precision, Recall

from config import config, device
from src.dataset import CachedPoseDataset
from src.models import ClassifierWrapper, RegressorWrapper
from src.utils import soft_argmax_2d

app = typer.Typer()


class InferenceStep(BaseModel):
    preds: torch.Tensor
    targets: torch.Tensor
    time: float  # seconds


class InferenceGenerator:
    def __init__(self, regressor_path: Path, classifier_path: Path):
        self.regressor = (
            RegressorWrapper(regressor_path / "weights.tflite").eval().to(device)
        )
        self.classifier = (
            ClassifierWrapper(classifier_path / "weights.tflite").eval().to(device)
        )

        self.dataset = CachedPoseDataset(config.eval_dataset)

        self.loader = DataLoader(
            self.dataset,
            batch_size=64,
            shuffle=True,
            pin_memory=True,
        )

    def step(self) -> InferenceStep:
        images, targets = next(iter(self.loader))

        _start_time = time.perf_counter()

        heatmaps = self.regressor(images)
        keypoints = soft_argmax_2d(heatmaps)

        _end_time = time.perf_counter()

        preds = self.classifier(images, keypoints)

        return InferenceStep(preds=preds, targets=targets, time=_end_time - _start_time)

    def __call__(self):
        with torch.no_grad():
            yield self.step()


class Metrics:
    def __init__(self):
        self.metrics = {
            "Accuracy": Accuracy(
                task="multiclass", num_classes=config.num_classes, average="weighted"
            ),
            "Precision": Precision(
                task="multiclass", num_classes=config.num_classes, average="weighted"
            ),
            "Recall": Recall(
                task="multiclass", num_classes=config.num_classes, average="weighted"
            ),
            "F1": F1Score(
                task="multiclass", num_classes=config.num_classes, average="weighted"
            ),
        }

    def update(self, preds, targets):
        for _, metric in self.metrics.items():
            metric.update(preds, targets)

    def compute(self):
        result = {}
        for name, metric in self.metrics.items():
            result[name] = metric.compute()
        return result


app = typer.Typer()


@app.command()
def convert(
    regressor_path: Path = typer.Option("--regressor"),
    classifier_path: Path = typer.Option("--classifier"),
):
    config.init_checkpoint("metrics")
    metrics = Metrics()

    times = []
    generator = InferenceGenerator(
        regressor_path=regressor_path, classifier_path=classifier_path
    )
    for step in generator():
        metrics.update(preds=step.preds, targets=step.targets)
        times.append(step.time)

    computed = metrics.compute()
    computed["Average Time"] = mean(times)

    typer.echo("=== METRICS ===")
    for k, v in computed.items():
        typer.echo(f"{k:10s}: {v:.4f}")

    with open(config.checkpoint / "metrics.json", "w") as f:
        json.dump(computed, f, indent=4)
