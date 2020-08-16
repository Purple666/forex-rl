import tensorflow as tf
from rl.network.noisy_dense import IndependentDense

custom_objects = {"IndependentDense" : IndependentDense}


def dueling_network(x):
    a = x[0]
    v = x[1]
    return v + (a - tf.reduce_mean(a, 1, keepdims=True))


def model(dim:tuple, action_size:int, dueling:bool, noisy:bool) -> tf.keras.Model:
    i = tf.keras.layers.Input(dim, name="i")
    layer_size = [128, 256, 128]

    for e, l in enumerate(layer_size[:-1]):
        if e == 0:
            x = tf.keras.layers.LSTM(l, return_sequences=True)(i)
        else:
            x = tf.keras.layers.LSTM(l, return_sequences=True)(x)
    x = tf.keras.layers.LSTM(layer_size[-1])(x)

    dense = IndependentDense if noisy else tf.keras.layers.Dense

    if dueling:
        v = dense(128, "elu", kernel_initializer="he_normal", kernel_regularizer=tf.keras.regularizers.l1())(x)
        v = dense(1)(v)

        a = dense(128, "elu", kernel_initializer="he_normal", kernel_regularizer=tf.keras.regularizers.l1())(x)
        a = dense(action_size)(a)

        x = tf.keras.layers.Lambda(dueling_network, name="q")([a, v])
    else:
        x = dense(action_size, name="q")

    return tf.keras.Model(i, x)