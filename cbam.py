import tensorflow as tf
from tensorflow.python.keras import backend
from tensorflow.python.keras.layers.pooling import GlobalPooling1D
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops


class ChannelGlobalMaxPool1D(GlobalPooling1D):
    def call(self, inputs):
        steps_axis = 2 if self.data_format == 'channels_last' else 1
        return backend.max(inputs, axis=steps_axis)


class ChannelGlobalAvgPool1D(GlobalPooling1D):
    def call(self, inputs, mask=None):
        steps_axis = 2 if self.data_format == 'channels_last' else 1
        if mask is not None:
            mask = math_ops.cast(mask, backend.floatx())
            mask = array_ops.expand_dims(
                mask, 2 if self.data_format == 'channels_last' else 1)
            inputs *= mask
            return backend.sum(inputs, axis=steps_axis) / math_ops.reduce_sum(
                mask, axis=steps_axis)
        else:
            return backend.mean(inputs, axis=steps_axis)


def cbam(x, r=16):
    # channel attention module
    ###############################################
    a = tf.keras.layers.GlobalAveragePooling1D()(x)
    a = tf.keras.layers.Reshape((1, a.shape[-1]))(a)
    m = tf.keras.layers.GlobalMaxPool1D()(x)
    m = tf.keras.layers.Reshape((1, m.shape[-1]))(m)

    w0 = tf.keras.layers.Dense(a.shape[-1] // r, "elu")
    w1 = tf.keras.layers.Dense(a.shape[-1])

    a = w1(w0(a))
    m = w1(w0(m))
    mc = tf.keras.layers.Activation("sigmoid")(a + m)
    x *= mc
    ###############################################
    # spatial attention module
    a = ChannelGlobalMaxPool1D()(x)
    a = tf.keras.layers.Reshape((a.shape[1], 1))(a)
    m = ChannelGlobalAvgPool1D()(x)
    m = tf.keras.layers.Reshape((m.shape[1], 1))(m)
    ms = tf.keras.layers.Concatenate()([a, m])
    ms = tf.keras.layers.Conv1D(1, 7, 1, "same", activation="sigmoid")(ms)
    x *= ms

    return x
