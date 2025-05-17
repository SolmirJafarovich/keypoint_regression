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
from torch.ao.quantization.quantize_pt2e import convert_pt2e, prepare_pt2e
from torch.ao.quantization.quantizer.xnnpack_quantizer import (
    XNNPACKQuantizer,
    get_symmetric_quantization_config,
)
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torch.ao.quantization import move_exported_model_to_eval

from src.config import config
from src.dataset import DepthKeypointDataset
from src.models import BlazePoseLite, CombinedClassifier

app = typer.Typer(pretty_exceptions_enable=False)


def calibrate(model):
    transform = transforms.Compose(
        [
            transforms.Resize((config.img_size, config.img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )
    dataset = DepthKeypointDataset(transform=transform)
    data_loader = DataLoader(
        dataset=Subset(dataset, list(range(500))),
        batch_size=config.batch,
        shuffle=True,
    )
    with torch.no_grad():
        for batch in track(data_loader, description="Calibration"):
            images = batch["image"]
            _ = model(images.to(torch.float32))


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

    # === Program capture ===

    model_fp32 = torch.export.export_for_training(model_fp32, sample_input).module()

    # === Configure quantization ===

    quant_config = get_symmetric_quantization_config()
    quantizer = XNNPACKQuantizer().set_global(quant_config)

    # === Quantize ===

    model_fp32 = prepare_pt2e(model_fp32, quantizer)

    calibrate(model_fp32)

    model_i8 = convert_pt2e(model_fp32, fold_quantize=True)

    move_exported_model_to_eval(model_i8)
    # === .pth -> ONNX ===
    typer.echo(".pth -> ONNX")

    torch.onnx.export(
        model=model_i8,
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

    # TensorFlow SavedModel -> TFLite
    typer.echo("tensorflow -> tflite")
    converter = tf.lite.TFLiteConverter.from_saved_model(str(saved_model_dir))
    converter.experimental_new_converter = True
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,
        tf.lite.OpsSet.SELECT_TF_OPS,
    ]
    tf_lite_model = converter.convert()
    with open(checkpoint / "weights.tflite", "wb") as f:
        f.write(tf_lite_model)

    typer.echo(f"Готово!")


if __name__ == "__main__":
    app()
