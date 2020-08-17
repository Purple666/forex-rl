import tensorflow as tf
import numpy as np
from rl.network.noisy_dense import IndependentDense

custom_objects = {"IndependentDense": IndependentDense}

inputs = tf.keras.layers.Input
tau_ = tf.keras.layers.Input((32,), name="t")


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
        v = dense(256, "relu", kernel_initializer="he_normal")(x)
        v = dense(1)(v)
        v = tf.transpose(v, (0, 2, 1))

        a = dense(256, "relu", kernel_initializer="he_normal")(x)
        a = dense(action_size)(a)
        a = tf.transpose(a, (0, 2, 1))

        x = tf.keras.layers.Lambda(dueling_network, name="q")([a, v])
    else:
        x = dense(256, "relu", kernel_initializer="he_normal")(x)
        x = dense(action_size, name="q")(x)

    return x


def model1(dim: tuple, action_size: int, dueling: bool, noisy: bool) -> tf.keras.Model:
    i = inputs(dim, name="i")

    x = tf.keras.layers.Conv1D(512, 3, 2, "same", kernel_initializer="he_normal")(i)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Conv1D(256, 3, 2, "same", kernel_initializer="he_normal")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Conv1D(128, 3, 2, "same", kernel_initializer="he_normal")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)

    x = tf.keras.layers.GlobalAveragePooling1D()(x)

    x = output(x, noisy, dueling, action_size)

    return tf.keras.Model([i, tau_], x)

