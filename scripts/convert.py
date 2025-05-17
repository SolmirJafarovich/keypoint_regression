import os
from pathlib import Path
from typing import Annotated

import onnx
import tensorflow as tf
import torch
import typer
from onnx_tf.backend import prepare
from rich import print
from rich.progress import track
from torch.utils.data import DataLoader, Subset
from torchvision import transforms

from src.config import config
from src.dataset import DepthKeypointDataset
from src.models import BlazePoseLite, CombinedClassifier

app = typer.Typer(pretty_exceptions_enable=False)


def representative_dataset_gen():
    transform = transforms.Compose(
        [
            transforms.Resize((config.img_size, config.img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )
    dataset = DepthKeypointDataset(transform=transform)
    data_loader = DataLoader(
        dataset=Subset(
            dataset, list(range(100))
        ),  # 100 изображений достаточно для калибровки
        batch_size=1,
        shuffle=False,
    )

    for batch in track(data_loader, description="Generating representative dataset"):
        image = batch["image"].numpy()  # shape: [1, 1, H, W]
        # TFLite expects NHWC format and float32 values
        yield [image.transpose(0, 2, 3, 1).astype("float32")]


@app.command()
def convert(
    checkpoint: Path = typer.Option("--checkpoint"),
    is_classifier: Annotated[bool, typer.Option("--is-classifier")] = False,
):
    # === Load and initialize model ===
    typer.echo("Loading model")

    if is_classifier:
        model_fp32 = CombinedClassifier()
        sample_input = (
            torch.randn(1, 1, config.img_size, config.img_size, dtype=torch.float),
            torch.randn(1, 66, dtype=torch.float),
        )
    else:
        model_fp32 = BlazePoseLite()
        sample_input = (
            torch.randn(1, 1, config.img_size, config.img_size, dtype=torch.float),
        )

    state_dict = torch.load(checkpoint / "weights.pth", weights_only=True)
    model_fp32.load_state_dict(state_dict)
    model_fp32.eval()

    # === .pth -> ONNX ===
    typer.echo(".pth -> ONNX")

    torch.onnx.export(
        model=model_fp32,
        args=sample_input,
        f=checkpoint / "weights.onnx",  # where should it be saved
        verbose=False,
        export_params=True,
        do_constant_folding=True,  # fold constant values for optimization
        # do_constant_folding=True,   # fold constant values for optimization
        input_names=["input"],
        output_names=["output"],
    )
    onnx_model = onnx.load(checkpoint / "weights.onnx")
    onnx.checker.check_model(onnx_model)

    # === ONNX -> TensorFlow SavedModel ===
    typer.echo("ONNX -> tensorflow")

    tf_rep = prepare(onnx_model)
    saved_model_dir = checkpoint / "saved_model"
    tf_rep.export_graph(str(saved_model_dir))

    # === TensorFlow SavedModel -> TFLite (with quantization) ===
    typer.echo("tensorflow -> tflite (with int8 quantization)")

    converter = tf.lite.TFLiteConverter.from_saved_model(str(saved_model_dir))
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset_gen
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.uint8

    tf_lite_model = converter.convert()
    with open(checkpoint / "weights.tflite", "wb") as f:
        f.write(tf_lite_model)

    typer.echo(f"Готово!")


if __name__ == "__main__":
    app()
