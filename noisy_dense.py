from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.python.eager import context
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.keras import activations
from tensorflow.python.keras import constraints
from tensorflow.python.keras import initializers
from tensorflow.python.keras import regularizers
from tensorflow.python.keras.engine.input_spec import InputSpec
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import standard_ops


class IndependentDense(tf.keras.layers.Dense):
    noise = True
    # def __init__(self,
    #              units,
    #              activation=None,
    #              use_bias=True,
    #              noise=True,
    #              kernel_initializer='glorot_uniform',
    #              bias_initializer='zeros',
    #              kernel_regularizer=None,
    #              bias_regularizer=None,
    #              activity_regularizer=None,
    #              kernel_constraint=None,
    #              bias_constraint=None,
    #              **kwargs):
    #     if 'input_shape' not in kwargs and 'input_dim' in kwargs:
    #         kwargs['input_shape'] = (kwargs.pop('input_dim'),)
    #
    #     super(IndependentDense, self).__init__(
    #         activity_regularizer=regularizers.get(activity_regularizer), **kwargs)
    #
    #     self.units = int(units) if not isinstance(units, int) else units
    #     self.activation = activations.get(activation)
    #     self.use_bias = use_bias
    #     self.noise = noise
    #     self.kernel_initializer = initializers.get(kernel_initializer)
    #     self.bias_initializer = initializers.get(bias_initializer)
    #     self.kernel_regularizer = regularizers.get(kernel_regularizer)
    #     self.bias_regularizer = regularizers.get(bias_regularizer)
    #     self.kernel_constraint = constraints.get(kernel_constraint)
    #     self.bias_constraint = constraints.get(bias_constraint)
    #
    #     self.supports_masking = True
    #     self.input_spec = InputSpec(min_ndim=2)

    def build(self, input_shape):
        # super(IndependentDense, self).build(input_shape)
        self.last_dim = tensor_shape.dimension_value(input_shape[-1])
        self.input_spec = InputSpec(min_ndim=2,
                                    axes={-1: self.last_dim})

        self.mu_init = tf.random_uniform_initializer(-((3 / self.last_dim) ** 0.5), (3 / self.last_dim) ** 0.5)
        self.sigma_init = tf.constant_initializer(0.017)

        self.w_mu = self.add_weight("w_mu", [self.last_dim, self.units], initializer=self.mu_init, dtype=self.dtype,
                                    trainable=True)
        self.w_sigma = self.add_weight("w_sigma", [self.last_dim, self.units], initializer=self.sigma_init,
                                       dtype=self.dtype, trainable=True)

        if self.use_bias:
            self.b_mu = self.add_weight("b_mu", [self.units, ], initializer=self.mu_init, dtype=self.dtype,
                                        trainable=True)
            self.b_sigma = self.add_weight("b_sigma", [self.units, ], initializer=self.sigma_init, dtype=self.dtype,
                                           trainable=True)
        self.built = True

    def call(self, inputs):
        def rank(tensor):
            """Return a rank if it is a tensor, else return None."""
            if isinstance(tensor, ops.Tensor):
                return tensor._rank()  # pylint: disable=protected-access
            return None

        inputs = ops.convert_to_tensor(inputs)
        rank = rank(inputs)
        w_epsilon = tf.random.normal((self.last_dim, self.units)) if self.noise else 0
        w = self.w_mu + self.w_sigma * w_epsilon

        if rank > 2:
            # Broadcasting is required for the inputs.
            outputs = standard_ops.tensordot(inputs, w, [[rank - 1], [0]])
            # Reshape the output back to the original ndim of the input.
            if not context.executing_eagerly():
                shape = inputs.get_shape().as_list()
                output_shape = shape[:-1] + [self.units]
                outputs.set_shape(output_shape)
        else:
            outputs = gen_math_ops.mat_mul(inputs, w)
        if self.use_bias:
            b_epsilon = tf.random.normal([self.units, ]) if self.noise else 0
            b = self.b_mu + self.b_sigma * b_epsilon
            outputs = nn.bias_add(outputs, b)
        if self.activation is not None:
            return self.activation(outputs)  # pylint: disable=not-callable
        return outputs


class FactorisedDense(tf.keras.layers.Dense):
    def build(self, input_shape):
        # super(FactorisedDense, self).build(input_shape)
        self.last_dim = tensor_shape.dimension_value(input_shape[-1])
        self.input_spec = InputSpec(min_ndim=2,
                                    axes={-1: self.last_dim})
        mu = 1 / self.last_dim ** 0.5
        self.mu_init = tf.random_uniform_initializer(-mu, mu)
        self.sigma_init = tf.constant_initializer(0.5 / self.last_dim ** 0.5)

        self.w_mu = self.add_weight("w_mu", [self.last_dim, self.units], initializer=self.mu_init, dtype=self.dtype,
                                    trainable=True)
        self.w_sigma = self.add_weight("w_sigma", [self.last_dim, self.units], initializer=self.sigma_init,
                                       dtype=self.dtype, trainable=True)

        if self.use_bias:
            self.b_mu = self.add_weight("b_mu", [self.units, ], initializer=self.mu_init, dtype=self.dtype,
                                        trainable=True)
            self.b_sigma = self.add_weight("b_sigma", [self.units, ], initializer=self.sigma_init, dtype=self.dtype,
                                           trainable=True)

        self.built = True

    def call(self, inputs):
        def rank(tensor):
            """Return a rank if it is a tensor, else return None."""
            if isinstance(tensor, ops.Tensor):
                return tensor._rank()  # pylint: disable=protected-access
            return None

        inputs = ops.convert_to_tensor(inputs)
        rank = rank(inputs)

        p, q = tf.random.normal((self.last_dim, 1)), tf.random.normal((1, self.units))
        p, q = (tf.math.sign(p) * tf.abs(p) ** 0.5), (tf.math.sign(q) * tf.abs(q) ** 0.5)
        w_epsilon = p * q
        w = self.w_mu + self.w_sigma * w_epsilon

        if rank > 2:
            # Broadcasting is required for the inputs.
            outputs = standard_ops.tensordot(inputs, w, [[rank - 1], [0]])
            # Reshape the output back to the original ndim of the input.
            if not context.executing_eagerly():
                shape = inputs.get_shape().as_list()
                output_shape = shape[:-1] + [self.units]
                outputs.set_shape(output_shape)
        else:
            outputs = gen_math_ops.mat_mul(inputs, w)
        if self.use_bias:
            b_epsilon = tf.squeeze(q)
            b = self.b_mu + self.b_sigma * b_epsilon
            outputs = nn.bias_add(outputs, b)
        if self.activation is not None:
            return self.activation(outputs)  # pylint: disable=not-callable
        return outputs
