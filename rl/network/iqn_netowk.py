import tensorflow as tf
import numpy as np
from rl.network.noisy_dense import IndependentDense
from rl.network.cbam import cbam, ChannelGlobalAvgPool1D, ChannelGlobalMaxPool1D
import rl.network.convnet as conv

custom_objects = {"IndependentDense": IndependentDense,
                  "ChannelGlobalAvgPool1D": ChannelGlobalAvgPool1D,
                  "ChannelGlobalMaxPool1D": ChannelGlobalMaxPool1D}

tau_ = tf.keras.layers.Input((32,), name="t")
position_value = tf.keras.layers.Input((1,), name="p")


def quantile(x):
    tau, x = x
    r = tf.range(1, 33, dtype=tf.float32)
    pi = tf.constant(np.pi)
    tau = tf.cos(pi * r * tau)
    tau = tf.reshape(tau, (1, -1, 32))
    tau = tf.tile(tau, (64, 1, 1))
    tau = tf.transpose(tau, (1, 2, 0))

    shape = x.get_shape().as_list()[-1]
    x = tf.reshape(x, (1, -1, shape))
    x = tf.tile(x, (32, 1, 1))
    x = tf.transpose(x, (1, 0, 2))

    return tau, x


def dueling_network(x):
    a, v = x
    return v + (a - tf.reduce_mean(a, 1, keepdims=True))


def output(x, noisy, dueling, action_size):
    """
    :param x: keras layer
    :param noisy: bool
    :param dueling: bool
    :param action_size: int
    :return: keras.layer
    """
    shape = x.get_shape().as_list()[-1]
    tau, x = tf.keras.layers.Lambda(quantile)([tau_, x])
    tau = tf.keras.layers.Dense(shape, "relu", kernel_initializer="he_normal")(tau)
    x *= tau

    dense = IndependentDense if noisy else tf.keras.layers.Dense

    if dueling:
        v = tf.keras.layers.Dense(256, "selu", kernel_initializer="lecun_normal")(x)
        v = tf.keras.layers.AlphaDropout(0.1)(v)
        v = dense(1)(v)
        v = tf.transpose(v, (0, 2, 1))

        add = []
        for _ in range(action_size):
            a = tf.keras.layers.Dense(256, "selu", kernel_initializer="lecun_normal")(x)
            a = tf.keras.layers.AlphaDropout(0.1)(a)
            a = dense(1)(a)
            add.append(a)
        a = tf.keras.layers.Concatenate()(add)
        a = tf.transpose(a, (0, 2, 1))

        x = tf.keras.layers.Lambda(dueling_network, name="q")([a, v])
    else:
        x = tf.keras.layers.Dense(256, "selu", kernel_initializer="lecun_normal")(x)
        x = tf.keras.layers.AlphaDropout(0.1)(x)
        x = dense(action_size)(x)

        # add = []
        # for _ in range(action_size):
        #     a = tf.keras.layers.Dense(256, "selu", kernel_initializer="lecun_normal")(x)
        #     a = tf.keras.layers.AlphaDropout(0.1)(a)
        #     a = dense(1)(a)
        #     add.append(a)
        # x = tf.keras.layers.Concatenate()(add)
        x = tf.keras.layers.Lambda(lambda x: tf.transpose(x, (0, 2, 1)), name="q")(x)

    return x


def position_net(x=position_value):
    for _ in range(3):
        x = tf.keras.layers.Dense(32, "selu", kernel_initializer="lecun_normal")(x)
        x = tf.keras.layers.AlphaDropout(0.1)(x)

    return x


def snn(x: tf.keras.layers.Layer, l=[128, 128, 256, 256, 512, 512]) -> tf.keras.layers.Layer:
    for l in l:
        x = tf.keras.layers.Dense(l, "selu", kernel_initializer="lecun_normal")(x)
        x = tf.keras.layers.AlphaDropout(0.1)(x)

    return x


def model1(dim: tuple, action_size: int, dueling: bool, noisy: bool) -> tf.keras.Model:
    i = tf.keras.layers.Input(dim, name="i")
    x = tf.keras.layers.Flatten()(i)

    x = snn(x)
    p = position_net(position_value)
    x = tf.keras.layers.Concatenate()([x, p])

    x = output(x, noisy, dueling, action_size)

    return tf.keras.Model([i, tau_, position_value], x)

