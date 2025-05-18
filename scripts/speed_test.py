import numpy as np
import time
import json
from statistics import mean
from pycoral.utils.edgetpu import make_interpreter, run_inference
from pycoral.adapters.common import input_size

NUM_IMAGES = 100

def load_model(model_path: str):
    interpreter = make_interpreter(model_path)
    interpreter.allocate_tensors()
    return interpreter

def generate_images(input_shape, num_images):
    h, w = input_shape
    return [np.random.randint(0, 256, (h, w, 1), dtype=np.uint8) for _ in range(num_images)]

def quantize_uint8(array, scale, zero_point):
    return np.clip(np.round(array / scale + zero_point), 0, 255).astype(np.uint8)

def run_dual_inference(interpreter1, interpreter2, inputs):
    times1, times2 = [], []

    input_details2 = interpreter2.get_input_details()
    output_details1 = interpreter1.get_output_details()[0]

    for image in inputs:
        # --- Модель 1 (регрессор) ---
        start1 = time.perf_counter()
        run_inference(interpreter1, image)
        end1 = time.perf_counter()
        times1.append(end1 - start1)

        # Получение выходных данных (предсказанные ключевые точки в float32)
        output1 = interpreter1.get_tensor(output_details1['index'])[0]

        # --- Модель 2 (классификатор) ---
        start2 = time.perf_counter()

        # Вход 0: изображение
        interpreter2.set_tensor(input_details2[0]['index'], np.expand_dims(image, axis=0))

        # Вход 1: предсказанные keypoints → квантованные в uint8
        keypoint_input_details = input_details2[1]
        kp_scale, kp_zero_point = keypoint_input_details['quantization']
        keypoints_uint8 = quantize_uint8(output1, kp_scale, kp_zero_point)
        keypoints_uint8 = np.expand_dims(keypoints_uint8, axis=0)  # (1, 66)

        interpreter2.set_tensor(keypoint_input_details['index'], keypoints_uint8)

        interpreter2.invoke()
        end2 = time.perf_counter()
        times2.append(end2 - start2)

    return times1, times2

def main():
    model1_path = "data/checkpoints/regressor/weights_quant_edgetpu.tflite"
    model2_path = "data/checkpoints/classifier/weights_quant_edgetpu.tflite"

    interpreter1 = load_model(model1_path)
    interpreter2 = load_model(model2_path)

    input_shape1 = input_size(interpreter1)
    input_shape2 = input_size(interpreter2)

    if input_shape1 != input_shape2:
        raise ValueError("Модели имеют разные размеры входных изображений.")

    inputs = generate_images(input_shape1, NUM_IMAGES)

    times1, times2 = run_dual_inference(interpreter1, interpreter2, inputs)

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

    with open("dual_model_metrics.json", "w") as f:
        json.dump(result, f, indent=2)

if __name__ == "__main__":
    main()
