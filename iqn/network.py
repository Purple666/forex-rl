import tensorflow as tf
import numpy as np
from cbam import cbam
from noisy_dense import IndependentDense
import tensorflow_addons as tfa

dim = (30, 3)
action_size = 3


def quantile(x):
    tau, x = x
    r = tf.range(1, 33, dtype=tf.float32)
    pi = tf.constant(np.pi)
    t = tf.cos(pi * r * tau)
    t = tf.reshape(t, (1, -1, 32))
    t = tf.tile(t, (64, 1, 1))
    t = tf.transpose(t, (1, 2, 0))

    x = tf.reshape(x, (1, -1, x.get_shape().as_list()[-1]))
    x = tf.tile(x, (32, 1, 1))
    x = tf.transpose(x, (1, 0, 2))

    return t, x

def iqn_output(x):
    # advantage, value = x
    # out = value + tf.subtract(advantage, tf.reduce_mean(advantage, axis=1, keepdims=True))
    return tf.transpose(x, (0, 2, 1))


def noisy_out(x, tau, action_size = action_size):
    q, x_tile = tf.keras.layers.Lambda(quantile)([tau, x])
    q = tf.keras.layers.Dense(x.get_shape().as_list()[-1], "relu", kernel_initializer="he_normal")(q)
    x = tf.keras.layers.Multiply()([x_tile, q])

    id1 = IndependentDense(128, "elu", kernel_initializer="he_normal", name="id1")
    id2 = IndependentDense(action_size, name="id2")

    advantage = id1(x)
    advantage = tf.keras.layers.Dropout(0.3)(advantage)
    advantage = id2(advantage)

    return tf.keras.layers.Lambda(iqn_output, name="q")(advantage)

def out(x, tau, action_size = action_size):
    q, x_tile = tf.keras.layers.Lambda(quantile)([tau, x])
    q = tf.keras.layers.Dense(x.get_shape().as_list()[-1], "relu", kernel_initializer="he_normal")(q)
    x = tf.keras.layers.Multiply()([x_tile, q])

    advantage = tf.keras.layers.Dense(256, "elu", kernel_initializer="he_normal")(x)
    # advantage = tf.keras.layers.AlphaDropout(0.1)(advantage)
    advantage = tf.keras.layers.Dense(action_size)(advantage)

    return tf.keras.layers.Lambda(iqn_output, name="q")(advantage)


t = tf.keras.layers.Input((32,), name="t")


def model1(dimension=dim, action_size=action_size):
    def conv(x, k):
        return tf.keras.layers.Conv1D(128, k, 1, "same", kernel_initializer="he_normal")(x)

    def netowrk(i):
        x = []
        k = [1, 3, 5, 7, 9]
        for k in k:
            x.append(conv(i, k))
        x = tf.keras.layers.concatenate(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ELU()(x)
        x = tf.keras.layers.Conv1D(128, 1, 1, "same", kernel_initializer="he_normal")(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ELU()(x)
        return cbam(x)

    i = tf.keras.layers.Input(dimension, name="i")

    x = netowrk(i)
    x = netowrk(x)
    x = tf.keras.layers.GlobalAvgPool1D()(x)
    x =  noisy_out(x, t, action_size)

    return tf.keras.Model([i, t], x)


def model2(dim, action_size):
    def network(i):
        b = tf.keras.layers.Conv1D(128, 3, 1, "same", kernel_initializer="he_normal")(i)
        b = tf.keras.layers.BatchNormalization()(b)
        b = tf.keras.layers.ELU()(b)
        b = tf.keras.layers.Conv1D(128, 1, 1, "same", kernel_initializer="he_normal")(b)
        b = tf.keras.layers.BatchNormalization()(b)

        x = tf.keras.layers.Conv1D(128, 3, 1, "same", kernel_initializer="he_normal")(i)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ELU()(x)
        x = tf.keras.layers.Conv1D(128, 3, 1, "same", dilation_rate=2, kernel_initializer="he_normal")(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ELU()(x)
        x = tf.keras.layers.Conv1D(128, 3, 1, "same", dilation_rate=3, kernel_initializer="he_normal")(x)
        x = tf.keras.layers.BatchNormalization()(x)

        x = tf.keras.layers.Add()([x, b])
        x = tf.keras.layers.ELU()(x)
        return cbam(x)

    i = tf.keras.layers.Input(dim, name="i")
    x = network(i)
    x = tf.keras.layers.Flatten()(x)
    # x = tf.keras.layers.LSTM(128)(x)
    # x = tf.keras.layers.Dropout(0.3)(x)
    x = noisy_out(x, t, action_size)

    return tf.keras.Model([i, t], x)


def model3(dim, action_size):
    def network(i):
        b = tf.keras.layers.Conv1D(128, 1, 1, "same", kernel_initializer="he_normal")(i)

        x = tf.keras.layers.Conv1D(128, 3, 1, "same", kernel_initializer="he_normal")(i)
        x = tf.keras.layers.LayerNormalization()(x)
        x = tf.keras.layers.ELU()(x)
        x = tf.keras.layers.Conv1D(128, 3, 1, "same", dilation_rate=2, kernel_initializer="he_normal")(x)
        x = tf.keras.layers.LayerNormalization()(x)
        x = tf.keras.layers.ELU()(x)
        x = tf.keras.layers.Conv1D(128, 3, 1, "same", dilation_rate=3, kernel_initializer="he_normal")(x)

        x = tf.keras.layers.Add()([x, b])
        x = tf.keras.layers.LayerNormalization()(x)
        x = tf.keras.layers.ELU()(x)
        return cbam(x)

    i = tf.keras.layers.Input(dim, name="i")
    x = network(i)
    x = tf.keras.layers.Flatten()(x)
    # x = tf.keras.layers.Dropout(0.3)(x)
    x = out(x, t, action_size)

    return tf.keras.Model([i, t], x)


def model4(dim, action_size):
    i = tf.keras.layers.Input(dim, name="i")
    # x = tf.keras.layers.Dropout(0.1)(i)
    # x = tf.keras.layers.GaussianNoise(0.2)(i)

    x = tf.keras.layers.LSTM(128, return_sequences=True)(i)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.LSTM(128)(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = noisy_out(x, t, action_size)

    return tf.keras.Model([i, t], x)

def model5(dim, action_size):
    def network(i, d):
        b = tfa.layers.WeightNormalization(tf.keras.layers.Conv1D(128, 1, 1, "causal", kernel_initializer="he_normal"))(i)

        x = tfa.layers.WeightNormalization(tf.keras.layers.Conv1D(128, 3, 1, "causal", dilation_rate=d, kernel_initializer="he_normal"))(i)
        x = tf.keras.layers.ELU()(x)
        x = tfa.layers.WeightNormalization(tf.keras.layers.Conv1D(128, 3, 1, "causal", dilation_rate=d, kernel_initializer="he_normal"))(x)
        x = tf.keras.layers.Add()([b, x])
        x = tf.keras.layers.ELU()(x)
        return cbam(x)

    i = tf.keras.layers.Input(dim, name="i")

    x = network(i, 1)
    x = network(x, 2)
    x = network(x, 4)
    x = tf.keras.layers.Flatten()(x)
    x = noisy_out(x, t, action_size)

    return tf.keras.Model([i, t], x)

def model6(dim, action_size):
    i = tf.keras.layers.Input(dim, name="i")
    x = tf.keras.layers.Flatten()(i)

    x = tf.keras.layers.Dense(128, activation="elu", kernel_initializer="he_normal")(x)
    x = tf.keras.layers.Dense(128, activation="elu", kernel_initializer="he_normal")(x)
    x = tf.keras.layers.Dense(128, activation="elu", kernel_initializer="he_normal")(x)
    x = noisy_out(x, t, action_size)

    return tf.keras.Model([i, t], x)