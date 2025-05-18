import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2

def build_combined_classifier(input_shape_img=(224, 224, 1), keypoint_dim=66, num_classes=4):
    # Вход изображения
    image_input = layers.Input(shape=input_shape_img, name="image_input")

    # Подключаем MobileNetV2 с входом (224, 224, 3), так как веса и структура завязаны на это
    base_model = MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights=None)

    # Поднимаем grayscale в RGB, дублируя канал 3 раза
    x = layers.Concatenate()([image_input, image_input, image_input])  # (H, W, 1) -> (H, W, 3)

    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)

    # Вход для ключевых точек
    kps_input = layers.Input(shape=(keypoint_dim,), name="keypoints_input")
    k = layers.Dense(64, activation=None)(kps_input)

    # Объединение и классификация
    combined = layers.Concatenate()([x, k])
    combined = layers.Dense(128, activation="relu")(combined)
    output = layers.Dense(num_classes, activation="softmax")(combined)

    model = models.Model(inputs=[image_input, kps_input], outputs=output)
    return model
