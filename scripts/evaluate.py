import json
import time
from pathlib import Path
from statistics import mean
from typing import Annotated
import torch
import typer
from pydantic import BaseModel, ConfigDict
from torch.utils.data import DataLoader
from torchmetrics import Accuracy, F1Score, Precision, Recall

from src.config import config, device
from src.dataset import KeypointPrecomputedDataset
from src.models import ClassifierWrapper, RegressorWrapper, BlazePoseLite, CombinedClassifier
from src.utils import soft_argmax_2d

from rich.progress import track

app = typer.Typer()


class InferenceStep(BaseModel):
    preds: torch.Tensor
    targets: torch.Tensor
    time: float  # seconds

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

        self.dataset = KeypointPrecomputedDataset()

        self.loader = DataLoader(
            self.dataset,
            batch_size=1,
            shuffle=True,
        )

    def __call__(self) -> InferenceStep:
        with torch.no_grad():
            for images, targets in self.loader:
                _start_time = time.perf_counter()

                heatmaps = self .regressor(images)
                keypoints = soft_argmax_2d(heatmaps)
                keypoints = keypoints.view(keypoints.shape[0], -1)

                _end_time = time.perf_counter()
                preds = self.classifier(images, keypoints)

                yield InferenceStep(preds=preds, targets=targets, time=_end_time - _start_time)


class InferenceGenerator:
    def __init__(self, regressor_path: Path, classifier_path: Path):
        self.regressor = (
            RegressorWrapper(regressor_path / "weights.tflite").eval().to(device)
        )
        self.classifier = (
            ClassifierWrapper(classifier_path / "weights.tflite").eval().to(device)
        )

        self.dataset = KeypointPrecomputedDataset()

        self.loader = DataLoader(
            self.dataset,
            batch_size=1,
            shuffle=True,
        )

    def __call__(self) -> InferenceStep:
        with torch.no_grad():
            for images, targets in self.loader:
                _start_time = time.perf_counter()

                heatmaps = self.regressor(images)
                keypoints = soft_argmax_2d(heatmaps)
                keypoints = (keypoints * 255).to(torch.uint8)

                _end_time = time.perf_counter()

                preds = self.classifier(images, keypoints)

                preds = (preds / 255).to(torch.float)

                yield InferenceStep(preds=preds, targets=targets, time=_end_time - _start_time)


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


app = typer.Typer()


@app.command()
def evaluate(
    reg: Path = typer.Option("--reg"),
    cl: Path = typer.Option("--cl"),
    pytorch: Annotated[bool, typer.Option("--pytorch")] = False,
):
    config.init_checkpoint("metrics")
    metrics = Metrics()

    times = []
    if pytorch:
        generator = PytorchInferenceGenerator(regressor_path=reg, classifier_path=cl)
    else:
        generator = InferenceGenerator(regressor_path=reg, classifier_path=cl)

    for step in track(generator(), description="Evaluating"):
        metrics.update(preds=step.preds, targets=step.targets)
        times.append(step.time)

    computed = metrics.compute()
    computed["Average Time"] = mean(times)

    typer.echo("=== METRICS ===")
    for k, v in computed.items():
        typer.echo(f"{k:10s}: {v:.4f}")

    with open(config.checkpoint / "metrics.json", "w") as f:
        json.dump(computed, f, indent=4)


if __name__ == "__main__":
    app()
