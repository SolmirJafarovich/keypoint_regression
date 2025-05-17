import os
import subprocess

# Путь к ONNX-модели
onnx_model_path = "model1.onnx"

# Папка, куда сохранить промежуточный TensorFlow и TFLite
output_dir = "onnx2tf_out"

# Команда onnx2tf: квантование под int8 и совместимость с Edge TPU
command = [
    "onnx2tf",
    "-i", onnx_model_path,
    "--output_folder", output_dir,
    "--output_format", "tflite",
    "--vital",                   # режим совместимости с Edge TPU
    "--quant_type", "int8",      # квантование в INT8
    "--overwrite_output"         # перезаписывать если есть
]

# Запуск
print("🔄 Конвертация ONNX → TFLite (Edge TPU)...")
subprocess.run(command, check=True)

# Путь к сгенерированной модели
tflite_model_path = os.path.join(output_dir, "model_float32_full_integer_quant.tflite")

# Проверка наличия tflite-файла
if os.path.exists(tflite_model_path):
    print(f"✅ Успешно: модель сохранена в {tflite_model_path}")
else:
    print("❌ Ошибка: модель не найдена. Проверь логи.")
