import tensorflow as tf
from tensorflow.keras import Model, layers

from src.config import config


# --- Depthwise Separable Convolution ---
def DepthwiseSeparableConv(x, out_channels, stride=1):
    x = layers.DepthwiseConv2D(
        kernel_size=3, strides=stride, padding="same", use_bias=False
    )(x)
    x = layers.Conv2D(out_channels, kernel_size=1, use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    return layers.ReLU()(x)


# --- Residual Block ---
def ResidualBlock(x, channels):
    shortcut = x
    out = layers.Conv2D(channels, 3, padding="same", use_bias=False)(x)
    out = layers.BatchNormalization()(out)
    out = layers.ReLU()(out)
    out = layers.Conv2D(channels, 3, padding="same", use_bias=False)(out)
    out = layers.BatchNormalization()(out)
    out = layers.Add()([shortcut, out])
    return layers.ReLU()(out)


# --- BlazePoseLite Model ---
def build_blazepose_lite(
    input_shape=(config.img_size, config.img_size, 1), heatmap_size=64, num_keypoints=33
):
    inputs = layers.Input(shape=input_shape, batch_size=1)

    # --- Encoder ---
    e1 = DepthwiseSeparableConv(inputs, 16, stride=2)  # 128 -> 64
    e2 = DepthwiseSeparableConv(e1, 32, stride=2)  # 64 -> 32
    e3 = DepthwiseSeparableConv(e2, 64, stride=2)  # 32 -> 16
    e4 = DepthwiseSeparableConv(e3, 128, stride=2)  # 16 -> 8
    e5 = DepthwiseSeparableConv(e4, 192, stride=2)  # 8 -> 4

    # --- Bottleneck ---
    b = ResidualBlock(e5, 192)
    b = ResidualBlock(b, 192)
    b = ResidualBlock(b, 192)

    # --- Decoder ---
    b_up = layers.UpSampling2D(size=(2, 2), interpolation="bilinear")(b)

    d4 = layers.Concatenate()([b_up, e4])
    d4 = layers.Conv2DTranspose(128, 2, strides=2)(d4)  # 4 -> 8

    d3 = layers.Concatenate()([d4, e3])
    d3 = layers.Conv2DTranspose(64, 2, strides=2)(d3)  # 8 -> 16

    d2 = layers.Concatenate()([d3, e2])
    d2 = layers.Conv2DTranspose(32, 2, strides=2)(d2)  # 16 -> 32

    d1 = layers.Concatenate()([d2, e1])
    d1 = layers.Conv2DTranspose(32, 2, strides=2)(d1)  # 32 -> 64

    # --- Heatmap Output ---
    heatmaps = layers.Conv2D(num_keypoints, kernel_size=1)(d1)
    heatmaps = layers.Resizing(64, 64, interpolation="bilinear")(heatmaps)

    return Model(inputs=inputs, outputs=heatmaps)


def soft_argmax_2d(heatmaps):
    shape = tf.shape(heatmaps)
    batch_size, height, width, num_kpoints = shape[0], shape[1], shape[2], shape[3]

    # reshape heatmaps to [B, H*W, K]
    heatmaps_reshaped = tf.nn.softmax(
        tf.reshape(heatmaps, [batch_size, height * width, num_kpoints]),
        axis=1,
    )

    # create coordinate grids
    pos_x, pos_y = tf.meshgrid(
        tf.linspace(0.0, 1.0, width), tf.linspace(0.0, 1.0, height)
    )
    pos_x = tf.reshape(pos_x, [-1])  # [H*W]
    pos_y = tf.reshape(pos_y, [-1])  # [H*W]

    # expand dims for broadcasting
    pos_x = pos_x[None, :, None]  # [1, H*W, 1]
    pos_y = pos_y[None, :, None]  # [1, H*W, 1]

    # weighted sum to get expected coordinates
    exp_x = tf.reduce_sum(heatmaps_reshaped * pos_x, axis=1)  # [B, K]
    exp_y = tf.reduce_sum(heatmaps_reshaped * pos_y, axis=1)  # [B, K]

    coords = tf.stack([exp_x, exp_y], axis=-1)  # [B, K, 2]
    return coords
