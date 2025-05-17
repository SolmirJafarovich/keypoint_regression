import os
from pathlib import Path
from typing import Annotated

import onnx
import tensorflow as tf
import torch
import typer
import numpy as np
from onnx_tf.backend import prepare
from rich import print
from rich.progress import track
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from rich.console import Console
from rich.progress import track
from src.config import config
from src.dataset import DepthKeypointDataset
from src.models import BlazePoseLite, CombinedClassifier

app = typer.Typer(pretty_exceptions_enable=False, pretty_exceptions_short=True)
console = Console()


def _init_loader():
    transform = transforms.Compose(
        [
            transforms.Resize((config.img_size, config.img_size)),
            transforms.ToTensor(),
            # Don't normalize here — quantization expects 0-255 scale
        ]
    )
    dataset = DepthKeypointDataset(transform=transform)
    return DataLoader(
        dataset=Subset(dataset, list(range(100))),
        batch_size=1,
        shuffle=False,
    )


def _regressor_gen():
    loader = _init_loader()

    for i, batch in enumerate(track(loader, description="Generating representative dataset")):
        image = batch["image"].numpy()  # shape: [1, 1, H, W]
        yield [image]


def _classifier_gen():
    loader = _init_loader()

    for i, batch in enumerate(track(loader, description="Generating representative dataset")):
        image = batch["image"].numpy()   # shape: [1, 1, H, W]
        keypoints = np.random.randint(0, 223, size=(1, 66)).astype(np.float32)

        yield {
            'image': image,
            'keypoints': keypoints,
        }


@app.command()
def convert(
    checkpoint: Path = typer.Option("--checkpoint"),
    is_classifier: Annotated[bool, typer.Option("--is-classifier")] = False,
):
    # === Load and initialize model ===
    console.rule("[bold blue]Этап 1: Загрузка модели")

    if is_classifier:
        model_fp32 = CombinedClassifier()
        sample_input = (
            torch.randn(1, 1, config.img_size, config.img_size),
            torch.randn(1, 66),
        )
        input_names = ["image", "keypoints"]
        output_names=["class"]
    else:
        model_fp32 = BlazePoseLite()
        sample_input = (
            torch.randn(1, 1, config.img_size, config.img_size),
        )
        input_names = ["image"]
        output_names=["heatmap"]

    state_dict = torch.load(checkpoint / "weights.pth", weights_only=True)
    model_fp32.load_state_dict(state_dict)
    model_fp32.eval()

    # === .pth -> ONNX ===
    console.rule("[bold blue]Этап 2: Экспорт в ONNX")
    
    torch.onnx.export(
        model=model_fp32,
        args=sample_input,
        f=checkpoint / "weights.onnx",  # where should it be saved
        verbose=False,
        export_params=True,
        do_constant_folding=True,  # fold constant values for optimization
        input_names=input_names,
        output_names=output_names,
    )
    onnx_model = onnx.load(checkpoint / "weights.onnx")
    onnx.checker.check_model(onnx_model)

    # === ONNX -> TensorFlow SavedModel ===
    console.rule("[bold blue]Этап 3: ONNX → TensorFlow SavedModel")

    tf_rep = prepare(onnx_model)
    saved_model_dir = checkpoint / "saved_model"
    tf_rep.export_graph(str(saved_model_dir))

    model = tf.saved_model.load(saved_model_dir)
    infer = model.signatures["serving_default"]
    console.log(f"[TF] Input signature: {infer.structured_input_signature}")
    console.log(f"[TF] Output signature: {infer.structured_outputs}")

    # === TensorFlow SavedModel -> TFLite (with quantization) ===
    console.rule("[bold blue]Этап 4: TensorFlow → TFLite INT8")

    converter = tf.lite.TFLiteConverter.from_saved_model(str(saved_model_dir))
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    if is_classifier:
        converter.representative_dataset = _classifier_gen
    else:
        converter.representative_dataset = _regressor_gen
    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.uint8

    try:
        console.log("Начинается квантование...")
        tf_lite_model = converter.convert()
        with open(checkpoint / "weights_quant.tflite", "wb") as f:
            f.write(tf_lite_model)
        console.log("Квантованная модель сохранена!")
    except Exception as e:
        console.log(f"[red]Ошибка при конвертации: {e}[/red]")


if __name__ == "__main__":
    app()
