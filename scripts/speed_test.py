import time
from pathlib import Path
from statistics import mean
from typing import Annotated

import numpy as np
import typer
from PIL import Image

from src.config import config

app = typer.Typer(pretty_exceptions_enable=False, pretty_exceptions_short=True)


def load_and_preprocess_image(image_path: Path):
    # Load image as grayscale
    image = (
        Image.open(image_path).convert("L").resize((config.img_size, config.img_size))
    )
    image_np = np.array(image, dtype=np.float32) / 255.0
    image_np = np.expand_dims(image_np, axis=-1)  # shape: (224, 224, 1)
    return image_np


def _regressor_gen():
    for ext in ("*.jpg", "*.png"):
        for fname in config.regressor.img_dir.glob(ext):
            yield load_and_preprocess_image(fname)


def inference_cpu(tflite: Path):
    import tensorflow as tf

    interpreter = tf.lite.Interpreter(model_path=str(tflite))
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    scale, zero_point = input_details[0]["quantization"]
    times = []
    for image in _regressor_gen():
        image = np.expand_dims(image, axis=0)
        image = image / scale + zero_point
        image = np.clip(image, 0, 255).astype(input_details[0]["dtype"])
        interpreter.set_tensor(input_details[0]["index"], image)
        start = time.perf_counter()
        interpreter.invoke()
        end = time.perf_counter()
        times.append(end - start)

    return mean(times)


def inference_tpu(tflite: Path = Path("weights_edgetpu.tflite")):
    from pycoral.adapters.common import set_input
    from pycoral.utils.edgetpu import make_interpreter

    interpreter = make_interpreter(tflite)
    interpreter.allocate_tensors()

    input_index = interpreter.get_input_details()[0]["index"]
    times = []
    for image in _regressor_gen():
        set_input(interpreter, image, input_index=input_index)
        start = time.perf_counter()
        interpreter.invoke()
        end = time.perf_counter()
        times.append(end - start)

    return mean(times)


@app.command()
def speed_test(
    tflite: Path = typer.Option("--tflite"),
    cpu: Annotated[bool, typer.Option("--cpu")] = False,
):
    if cpu:
        _time = inference_cpu(tflite)
    else:
        _time = inference_tpu(tflite)

    print("  Regressor:", _time)


if __name__ == "__main__":
    app()
