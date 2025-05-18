import numpy as np
import time
import json
from statistics import mean
from pycoral.utils.edgetpu import make_interpreter, run_inference
from pycoral.adapters.common import set_input
from typing import List

NUM_IMAGES = 100

def load_model(model_path: str):
    interpreter = make_interpreter(model_path)
    interpreter.allocate_tensors()
    return interpreter

def get_image_input_shape(interpreter, image_input_index: int):
    input_details = interpreter.get_input_details()
    shape = input_details[image_input_index]['shape']
    if len(shape) == 4:
        return tuple(shape[1:3])  # (H, W)
    elif len(shape) == 3:
        return tuple(shape[0:2])  # (H, W)
    else:
        raise ValueError(f"Unexpected input shape: {shape}")

def generate_images(input_shape, num_images):
    h, w = input_shape
    return [np.random.randint(0, 256, (h, w, 1), dtype=np.uint8) for _ in range(num_images)]

def generate_extra_inputs(num_images, feature_size=66):
    return [np.random.randint(0, 256, (feature_size,), dtype=np.uint8) for _ in range(num_images)]

def run_inference_loop_single_input(interpreter, inputs: List[np.ndarray], input_index: int):
    times = []
    for image in inputs:
        set_input(interpreter, image, input_index=input_index)
        start = time.perf_counter()
        interpreter.invoke()
        end = time.perf_counter()
        times.append(end - start)
    return times

def run_inference_loop_dual_input(interpreter, images: List[np.ndarray], extras: List[np.ndarray], image_index: int, extra_index: int):
    input_details = interpreter.get_input_details()
    times = []
    for image, extra in zip(images, extras):
        set_input(interpreter, image, input_index=image_index)
        interpreter.set_tensor(input_details[extra_index]['index'], np.expand_dims(extra, axis=0))
        start = time.perf_counter()
        interpreter.invoke()
        end = time.perf_counter()
        times.append(end - start)
    return times

def main():
    model1_path = "weights_regressor_edgetpu.tflite"
    model2_path = "weights_classifier_edgetpu.tflite"

    interpreter1 = load_model(model1_path)
    interpreter2 = load_model(model2_path)

    # Явно указываем входы
    input_details1 = interpreter1.get_input_details()
    input_details2 = interpreter2.get_input_details()

    image_input_index1 = 0  # Модель 1 — только изображение
    image_input_index2 = 0  # Модель 2 — первый вход — изображение
    extra_input_index2 = 1  # Модель 2 — второй вход — 66 чисел

    # Получаем размер входного изображения
    input_shape1 = get_image_input_shape(interpreter1, image_input_index1)
    input_shape2 = get_image_input_shape(interpreter2, image_input_index2)

    if input_shape1 != input_shape2:
        raise ValueError("Модели имеют разные размеры входных изображений.")

    # Генерация тестовых данных
    images = generate_images(input_shape1, NUM_IMAGES)
    extras = generate_extra_inputs(NUM_IMAGES, feature_size=66)

    # Инференс первой модели (только изображение)
    times1 = run_inference_loop_single_input(interpreter1, images, input_index=image_input_index1)

    # Инференс второй модели (изображение + 66 чисел)
    times2 = run_inference_loop_dual_input(
        interpreter2, images, extras,
        image_index=image_input_index2,
        extra_index=extra_input_index2
    )

    result = {
        "model_1_avg_time": mean(times1),
        "model_2_avg_time": mean(times2),
        "model_1_total_time": sum(times1),
        "model_2_total_time": sum(times2),
        "images_processed": NUM_IMAGES
    }

    print("Среднее время инференса:")
    print("  Модель 1:", result["model_1_avg_time"])
    print("  Модель 2:", result["model_2_avg_time"])

if __name__ == "__main__":
    main()
