from pathlib import Path
from typing import Annotated

import numpy as np
import pandas as pd
import tensorflow as tf
import typer
from rich import print
from rich.console import Console
from rich.progress import track

from src.config import config
from src.dataset import DepthKeypointDataset
from src.dataset_cl import KeypointPrecomputedDatasetTF

app = typer.Typer(pretty_exceptions_enable=False, pretty_exceptions_short=True)
console = Console()


def _regressor_gen():
    full_df = pd.read_csv(config.regressor.csv_file)

    dataset = DepthKeypointDataset(full_df).get_dataset(batch_size=1)

    for batch in track(dataset, description="Generating representative dataset"):
        images, _, _ = batch
        yield [images]


def _classifier_gen():
    full_df = pd.read_csv(config.regressor.csv_file)
    full_df = full_df[:100]

    dataset = DepthKeypointDataset(full_df).get_dataset(batch_size=1)

    for batch in track(dataset, description="Generating representative dataset"):
        # Ожидаем структуру: (images, keypoints, labels)
        images, keypoints, _ = batch
        yield [images, keypoints]



@app.command()
def convert(
    checkpoint: Path = typer.Option("--checkpoint"),
    is_classifier: Annotated[bool, typer.Option("--is-classifier")] = False,
):
    # === TensorFlow SavedModel -> TFLite (with quantization) ===
    console.rule("[bold blue]TensorFlow → TFLite INT8")
    model = tf.keras.models.load_model(str(checkpoint / "classifier.keras"))
    print("Original input shape:", model.input)

    # Создаем новый Input с фиксированным batch size = 1
    if is_classifier:
        new_input = [tf.keras.Input(shape=model.input[0].shape[1:], batch_size=1), tf.keras.Input(shape=model.input[1].shape[1:], batch_size=1)]
    else:
        new_input = tf.keras.Input(shape=model.input.shape[1:], batch_size=1)

    # Получаем выходы модели на новом входе
    new_outputs = model(new_input)

    # Создаем новую модель с фиксированным batch size
    model_fixed = tf.keras.Model(inputs=new_input, outputs=new_outputs)

    print("New input shape:", model_fixed.input)
    converter = tf.lite.TFLiteConverter.from_keras_model(model_fixed)
    # https://ai.google.dev/edge/litert/models/post_training_quantization
    converter.experimental_enable_resource_variables = False
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.uint8
    if is_classifier:
        converter.representative_dataset = _classifier_gen
    else:
        converter.representative_dataset = _regressor_gen

    console.log("Начинается квантование...")
    tf_lite_model = converter.convert()
    with open(checkpoint / "weights.tflite", "wb") as f:
        f.write(tf_lite_model)
    console.log("Квантованная модель сохранена!")


if __name__ == "__main__":
    app()
