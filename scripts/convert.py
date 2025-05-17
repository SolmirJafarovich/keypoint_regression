from pathlib import Path
from typing import Annotated

import onnx
import tensorflow as tf
import torch
import typer
from onnx_tf.backend import prepare

from src.config import config
from src.models import BlazePoseLite, CombinedClassifier

app = typer.Typer(pretty_exceptions_enable=False)


@app.command()
def convert(
    checkpoint: Path = typer.Option("--checkpoint"),
    is_classifier: Annotated[bool, typer.Option("--is-classifier")] = False,
):
    # === Load and initialize model ===

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

    torch.onnx.export(
        model=model_fp32,
        args=sample_input,
        f=checkpoint / "weights.onnx",  # where should it be saved
        verbose=False,
        export_params=True,
        do_constant_folding=False,  # fold constant values for optimization
        # do_constant_folding=True,   # fold constant values for optimization
        input_names=["input"],
        output_names=["output"],
    )
    onnx_model = onnx.load(checkpoint / "weights.onnx")
    onnx.checker.check_model(onnx_model)

    # === ONNX -> tensorflow ===

    tf_rep = prepare(onnx_model)  # creating TensorflowRep object

    tf_rep.export_graph(checkpoint / "weights.pb")

    # === tensorflow -> tflite

    converter = tf.compat.v1.lite.TFLiteConverter.from_frozen_graph(
        checkpoint / "weights.pb",
        input_arrays=["input"],
        output_arrays=["output"],
    )
    converter.experimental_new_converter = True

    converter.target_spec.supported_ops = [
        tf.compat.v1.lite.OpsSet.TFLITE_BUILTINS,
        tf.compat.v1.lite.OpsSet.SELECT_TF_OPS,
    ]

    tf_lite_model = converter.convert()
    with open(checkpoint / "weights.tflite", "wb") as f:
        f.write(tf_lite_model)

    typer.echo(f"Готово!")


if __name__ == "__main__":
    app()
