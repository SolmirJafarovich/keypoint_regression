import onnx
from onnx_tf.backend import prepare
import tensorflow as tf
import numpy as np
import os

# === 1. Конвертация ONNX → TensorFlow SavedModel ===
onnx_model_path = "model1.onnx"
saved_model_dir = "model_tf"

onnx_model = onnx.load(onnx_model_path)
tf_rep = prepare(onnx_model)
tf_rep.export_graph(saved_model_dir)

print(f"✅ SavedModel экспортирован в {saved_model_dir}")