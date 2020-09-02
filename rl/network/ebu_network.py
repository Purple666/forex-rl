import tensorflow as tf

from rl.network.noisy_dense import IndependentDense
from rl.network.cbam import cbam, ChannelGlobalAvgPool1D, ChannelGlobalMaxPool1D

custom_objects = {"IndependentDense": IndependentDense,
                  "ChannelGlobalAvgPool1D": ChannelGlobalAvgPool1D,
                  "ChannelGlobalMaxPool1D": ChannelGlobalMaxPool1D}

position_value = tf.keras.layers.Input((1,), name="p")


def position_net(x=position_value):
    for _ in range(3):
        x = tf.keras.layers.Dense(32, "selu", kernel_initializer="lecun_normal")(x)
        x = tf.keras.layers.AlphaDropout(0.1)(x)

    return x


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
        v = tf.keras.layers.Dense(256, "selu", kernel_initializer="lecun_normal")(x)
        v = tf.keras.layers.AlphaDropout(0.05)(v)
        v = dense(1)(v)

        a = tf.keras.layers.Dense(256, "elu", kernel_initializer="lecun_normal")(x)
        a = tf.keras.layers.AlphaDropout(0.05)(a)
        a = dense(action_size)(a)

        # add = []
        # for _ in range(action_size):
        #     a = tf.keras.layers.Dense(256, "selu", kernel_initializer="lecun_normal")(x)
        #     a = tf.keras.layers.AlphaDropout(0.1)(a)
        #     # a = tf.keras.layers.BatchNormalization()(a)
        #     # a = tf.keras.layers.Dropout(0.3)(a)
        #     add.append(dense(1)(a))
        # a = tf.keras.layers.Concatenate()(add)

        x = v + (a - tf.reduce_mean(a, -1, keepdims=True))
        x = tf.keras.layers.Lambda(dueling_network, name="q")(x)
    else:
        x = tf.keras.layers.Dense(256, "selu", kernel_initializer="lecun_normal")(x)
        x = tf.keras.layers.AlphaDropout(0.05)(x)
        x = dense(action_size, name="q")(x)

        # add = []
        # for _ in range(action_size):
        #     a = tf.keras.layers.Dense(256, "selu", kernel_initializer="lecun_normal")(x)
        #     a = tf.keras.layers.AlphaDropout(0.1)(a)
        #     # a = tf.keras.layers.BatchNormalization()(a)
        #     # a = tf.keras.layers.Dropout(0.3)(a)
        #     add.append(dense(1)(a))
        # x = tf.keras.layers.Concatenate(name="q")(add)
    return x


def snn(x: tf.keras.layers.Layer, l=[256, 256, 256, 256]) -> tf.keras.layers.Layer:
    for l in l:
        x = tf.keras.layers.Dense(l, "selu", kernel_initializer="lecun_normal", kernel_regularizer=tf.keras.regularizers.l1_l2())(x)
        x = tf.keras.layers.AlphaDropout(0.05)(x)

    return x


def model1(dim: tuple, action_size: int, dueling: bool, noisy: bool) -> tf.keras.Model:
    i = tf.keras.layers.Input(dim, name="i")

    # x = tf.keras.layers.Flatten()(i)
    #
    def conv(x):
        for i in range(3):
            x = tf.keras.layers.Conv1D(256*(i+1), 3, 1, "causal", activation="selu", kernel_initializer="lecun_normal", dilation_rate=2**i)(x)
            # x = tf.keras.layers.AveragePooling1D()(x)
        x = cbam(x)
        # x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.GlobalAveragePooling1D()(x)
        return x

    x = tf.keras.layers.Flatten()(i)
    x = snn(x)
    p = position_net(position_value)
    x = tf.keras.layers.Concatenate()([x, p])

    x = output(x, noisy, dueling, action_size)

    return tf.keras.Model([i, position_value], x)

