import tensorflow as tf
from tensorflow.keras import layers, Model

def build_combined_classifier(input_shape_img=(224, 224, 1), keypoint_dim=66, num_classes=4):
    # Вход изображения
    image_input = layers.Input(shape=input_shape_img, name="image_input")

    # Упрощённый CNN-блок (совместим с Coral)
    x = layers.Conv2D(16, 3, strides=2, padding="same", activation="relu")(image_input)
    x = layers.Conv2D(32, 3, strides=2, padding="same", activation="relu")(x)
    x = layers.Conv2D(64, 3, strides=2, padding="same", activation="relu")(x)
    x = layers.GlobalAveragePooling2D()(x)  # [B, 64]

    # Вход ключевых точек
    kps_input = layers.Input(shape=(keypoint_dim,), name="keypoints_input")
    k = layers.Dense(64, activation="relu")(kps_input)  # Обязательно activation для квантизации

    # Объединение
    combined = layers.Concatenate()([x, k])
    combined = layers.Dense(128, activation="relu")(combined)
    output = layers.Dense(num_classes, activation="softmax")(combined)

    model = Model(inputs=[image_input, kps_input], outputs=output)
    return model
