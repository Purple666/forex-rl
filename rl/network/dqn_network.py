import tensorflow as tf

from rl.network.noisy_dense import IndependentDense
from rl.network.cbam import cbam, ChannelGlobalAvgPool1D, ChannelGlobalMaxPool1D

custom_objects = {"IndependentDense": IndependentDense,
                  "ChannelGlobalAvgPool1D": ChannelGlobalAvgPool1D,
                  "ChannelGlobalMaxPool1D": ChannelGlobalMaxPool1D}


def dueling_network(x):
    return x


def output(x, noisy, dueling, action_size):
    """
    :param x: keras layer
    :param noisy: bool
    :param dueling: bool
    :param action_size: int
    :return: keras.layer
    """
    dense = IndependentDense if noisy else tf.keras.layers.Dense

    if dueling:
        v = tf.keras.layers.Dense(256, "elu", kernel_initializer="he_normal")(x)
        v = dense(1)(v)

        a = tf.keras.layers.Dense(256, "elu", kernel_initializer="he_normal")(x)
        a = dense(action_size)(a)

        x = v + (a - tf.reduce_mean(a, -1, keepdims=True))
        x = tf.keras.layers.Lambda(dueling_network, name="q")(x)
    else:
        add = []
        for _ in range(action_size):
            a = tf.keras.layers.Dense(256, "elu", kernel_initializer="he_normal")(x)
            add.append(dense(1)(a))
        x = tf.keras.layers.Concatenate(name="q")(add)
    return x


def model1(dim: tuple, action_size: int, dueling: bool, noisy: bool) -> tf.keras.Model:
    i = tf.keras.layers.Input(dim, name="i")

    x = tf.keras.layers.Conv1D(128, 5, 1, "same", kernel_initializer="he_normal")(i)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ELU()(x)
    # x = tf.keras.layers.AvgPool1D()(x)
    x = tf.keras.layers.Conv1D(256, 3, 1, "same", kernel_initializer="he_normal")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ELU()(x)
    # x = tf.keras.layers.AvgPool1D()(x)
    # x = cbam(x)
    #
    # x = tf.keras.layers.GRU(512)(x)
    x = tf.keras.layers.Flatten()(x)

    x = output(x, noisy, dueling, action_size)

    return tf.keras.Model(i, x)