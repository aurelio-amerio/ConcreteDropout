import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Wrapper, InputSpec

# IMPORTANT: these layers perform dropout BEFORE the wrapped operation.

# base class


class ConcreteDropout(Wrapper):
    """
    Base class for ConcreteDropout. It allows to learn the dropout probability for a given layer.

    # Arguments
        layer: a layer instance.
        weight_regularizer:
            A positive number which satisfies
                $weight_regularizer = l**2 / (\tau * N)$
            with prior lengthscale l, model precision $\tau$ (inverse observation noise),
            and N the number of instances in the dataset.
            Note that kernel_regularizer is not needed.
        dropout_regularizer:
            A positive number which satisfies
                $dropout_regularizer = 2 / (\tau * N)$
            with model precision $\tau$ (inverse observation noise) and N the number of
            instances in the dataset.
            Note the relation between dropout_regularizer and weight_regularizer:
                $weight_regularizer / dropout_regularizer = l**2 / 2$
            with prior lengthscale l. Note also that the factor of two should be
            ignored for cross-entropy loss, and used only for the eculedian loss.
        temperature:
            The temperature of the Concrete Distribution. 
            Must be low enough in order to have the correct KL approximation
            For:
                $t \rightarrow 0$ we obtain the discrete distribution.
                $t \rightarrow \infty$ we obtain a uniform distribution
    """

    def __init__(self, layer, weight_regularizer=1e-6, dropout_regularizer=1e-5,
                 init_min=0.1, init_max=0.1, is_mc_dropout=False, data_format=None, temperature=0.1, **kwargs):
        assert 'kernel_regularizer' not in kwargs, "Must not provide a kernel regularizer."
        super(ConcreteDropout, self).__init__(layer, **kwargs)
        self.temperature = temperature
        self.weight_regularizer = weight_regularizer
        self.dropout_regularizer = dropout_regularizer
        self.is_mc_dropout = is_mc_dropout
        self.supports_masking = True
        self.p_logit = None
        self.init_min = tf.math.log(init_min) - tf.math.log(1. - init_min)
        self.init_max = tf.math.log(init_max) - tf.math.log(1. - init_max)
        self.data_format = 'channels_last' if data_format is None else 'channels_first'

    def build(self, input_shape=None):
        self.input_spec = InputSpec(shape=input_shape)
        # build the superclass layers
        if not self.layer.built:
            self.layer.build(input_shape)
            self.layer.built = True
        super(ConcreteDropout, self).build()

        # initialise p (learnable dropout probability)
        self.p_logit = self.add_weight(name='p_logit',
                                       shape=(1,),
                                       initializer=tf.random_uniform_initializer(
                                           self.init_min, self.init_max),
                                       dtype=tf.dtypes.float32,
                                       trainable=True)

    def _get_noise_shape(self, inputs):
        raise NotImplementedError(
            "Subclasses of ConcreteDropout must implement the noise shape")

    def compute_output_shape(self, input_shape):
        return self.layer.compute_output_shape(input_shape)

    def spatial_concrete_dropout(self, x, p):
        '''
        Concrete dropout - used at training time (gradients can be propagated)
        :param x: input
        :return:  approx. dropped out input
        '''
        eps = K.cast_to_floatx(
            K.epsilon())  # machine precision epsilon for numerical stability inside the log

        # this is the shape of the dropout noise, same as tf.nn.dropout
        noise_shape = self._get_noise_shape(x)

        unif_noise = K.random_uniform(shape=noise_shape)  # uniform noise
        # bracket inside equation 5, where u=uniform_noise
        drop_prob = (
            tf.math.log(p + eps)
            - tf.math.log1p(eps - p)
            + tf.math.log(unif_noise + eps)
            - tf.math.log1p(eps - unif_noise)
        )
        drop_prob = tf.math.sigmoid(drop_prob / self.temperature)  # z of eq 5
        random_tensor = 1. - drop_prob

        retain_prob = 1. - p
        x *= random_tensor  # we multiply the input by the concrete dropout mask
        x /= retain_prob  # we normalise by the probability to retain

        return x

    def call(self, inputs, training=None):
        p = tf.nn.sigmoid(self.p_logit)
        # We will now compute the KL terms following eq.3 of 1705.07832
        weight = self.layer.kernel
        # The kernel regularizer corresponds to the first term
        # Note: we  divide by (1 - p) because  we  scaled  layer  output  by(1 - p)
        kernel_regularizer = self.weight_regularizer * tf.reduce_sum(tf.square(
            weight)) / (1. - p)
        # the dropout regularizer corresponds to the second term
        dropout_regularizer = p * tf.math.log(p)
        dropout_regularizer += (1. - p) * tf.math.log1p(- p)
        dropout_regularizer *= self.dropout_regularizer * self.input_dim
        # this is the KL term to be added as a loss
        regularizer = tf.reduce_sum(kernel_regularizer + dropout_regularizer)

        self.layer.add_loss(regularizer)

        if self.is_mc_dropout:
            return self.layer.call(self.spatial_concrete_dropout(inputs, p))
        else:
            def relaxed_dropped_inputs():
                return self.layer.call(self.spatial_concrete_dropout(inputs, p))
            return K.in_train_phase(relaxed_dropped_inputs,
                                    self.layer.call(inputs),
                                    training=training)

# dense layers


class ConcreteDenseDropout(ConcreteDropout):
    """
    This wrapper allows to learn the dropout probability for any given Dense input layer.
    ```python
        x = # some input layer
        x = ConcreteDenseDropout(Dense(64))(x)
    ```
    # Arguments
        layer: a layer instance.
        weight_regularizer:
            A positive number which satisfies
                $weight_regularizer = l**2 / (\tau * N)$
            with prior lengthscale l, model precision $\tau$ (inverse observation noise),
            and N the number of instances in the dataset.
            Note that kernel_regularizer is not needed.
        dropout_regularizer:
            A positive number which satisfies
                $dropout_regularizer = 2 / (\tau * N)$
            with model precision $\tau$ (inverse observation noise) and N the number of
            instances in the dataset.
            Note the relation between dropout_regularizer and weight_regularizer:
                $weight_regularizer / dropout_regularizer = l**2 / 2$
            with prior lengthscale l. Note also that the factor of two should be
            ignored for cross-entropy loss, and used only for the eculedian loss.
        temperature:
            The temperature of the Concrete Distribution. 
            Must be low enough in order to have the correct KL approximation.
            For:
                $t \rightarrow 0$ we obtain the discrete distribution
                $t \rightarrow \infty$ we obtain a uniform distribution
    """

    def __init__(self, layer, temperature=0.1, **kwargs):
        super(ConcreteDenseDropout, self).__init__(
            layer, temperature=temperature, **kwargs)

    # implement the noise shape for regular dropout
    def _get_noise_shape(self, inputs):
        input_shape = tf.shape(inputs)
        return input_shape

    def build(self, input_shape=None):
        self.input_spec = InputSpec(shape=input_shape)

        super(ConcreteDenseDropout, self).build(input_shape=input_shape)

        # initialise regulariser / prior KL term
        assert len(input_shape) == 2, 'this wrapper only supports Dense layers'
        self.input_dim = input_shape[1]  # we drop only channels

    def call(self, inputs, training=None):
        return super().call(inputs, training=training)

# convolution layers


class ConcreteSpatialDropout1D(ConcreteDropout):
    """
    This wrapper allows to learn the dropout probability for any given Conv1D input layer. 
    It is the Concrete Dropout implementation of `SpatialDropout1D`.

    ```python
        x = # some input layer
        x = ConcreteSpatialDropout1D(Conv1D(64, 3))(x)
    ```
    # Arguments
        layer: a layer instance.
        weight_regularizer:
            A positive number which satisfies
                $weight_regularizer = l**2 / (\tau * N)$
            with prior lengthscale l, model precision $\tau$ (inverse observation noise),
            and N the number of instances in the dataset.
            Note that kernel_regularizer is not needed.
        dropout_regularizer:
            A positive number which satisfies
                $dropout_regularizer = 2 / (\tau * N)$
            with model precision $\tau$ (inverse observation noise) and N the number of
            instances in the dataset.
            Note the relation between dropout_regularizer and weight_regularizer:
                $weight_regularizer / dropout_regularizer = l**2 / 2$
            with prior lengthscale l. Note also that the factor of two should be
            ignored for cross-entropy loss, and used only for the eculedian loss.
        temperature:
            The temperature of the Concrete Distribution. 
            Must be low enough in order to have the correct KL approximation.
            For:
                $t \rightarrow 0$ we obtain the discrete distribution
                $t \rightarrow \infty$ we obtain a uniform distribution
    """

    def __init__(self, layer, temperature=2./3., **kwargs):
        super(ConcreteSpatialDropout1D, self).__init__(
            layer, temperature=temperature, **kwargs)

    # implement the noise shape for spatial dropout
    def _get_noise_shape(self, inputs):
        input_shape = tf.shape(inputs)
        if self.data_format == 'channels_first':
            return (input_shape[0], input_shape[1], 1)
        elif self.data_format == 'channels_last':
            return (input_shape[0], 1, input_shape[2])

    def build(self, input_shape=None):
        self.input_spec = InputSpec(shape=input_shape)

        super(ConcreteSpatialDropout1D, self).build(input_shape=input_shape)

        # initialise regulariser / prior KL term
        assert len(input_shape) == 3, 'this wrapper only supports Conv1D layers'
        if self.data_format == 'channels_first':
            self.input_dim = input_shape[1]  # we drop only channels
        else:
            self.input_dim = input_shape[2]

    def call(self, inputs, training=None):
        return super().call(inputs, training=training)


class ConcreteSpatialDropout2D(ConcreteDropout):
    """
    This wrapper allows to learn the dropout probability for any given Conv2D input layer. 
    It is the Concrete Dropout implementation of `SpatialDropout2D`.

    ```python
        x = # some input layer
        x = ConcreteSpatialDropout2D(Conv2D(64, (3,3))(x)
    ```
    # Arguments
        layer: a layer instance.
        weight_regularizer:
            A positive number which satisfies
                $weight_regularizer = l**2 / (\tau * N)$
            with prior lengthscale l, model precision $\tau$ (inverse observation noise),
            and N the number of instances in the dataset.
            Note that kernel_regularizer is not needed.
        dropout_regularizer:
            A positive number which satisfies
                $dropout_regularizer = 2 / (\tau * N)$
            with model precision $\tau$ (inverse observation noise) and N the number of
            instances in the dataset.
            Note the relation between dropout_regularizer and weight_regularizer:
                $weight_regularizer / dropout_regularizer = l**2 / 2$
            with prior lengthscale l. Note also that the factor of two should be
            ignored for cross-entropy loss, and used only for the eculedian loss.
        temperature:
            The temperature of the Concrete Distribution. 
            Must be low enough in order to have the correct KL approximation.
            For:
                $t \rightarrow 0$ we obtain the discrete distribution
                $t \rightarrow \infty$ we obtain a uniform distribution
    """

    def __init__(self, layer, temperature=2./3., **kwargs):
        super(ConcreteSpatialDropout2D, self).__init__(
            layer, temperature=temperature, **kwargs)

    # implement the noise shape for spatial dropout
    def _get_noise_shape(self, inputs):
        input_shape = tf.shape(inputs)
        if self.data_format == 'channels_first':
            return (input_shape[0], input_shape[1], 1, 1)
        elif self.data_format == 'channels_last':
            return (input_shape[0], 1, 1, input_shape[3])

    def build(self, input_shape=None):
        self.input_spec = InputSpec(shape=input_shape)

        super(ConcreteSpatialDropout2D, self).build(input_shape=input_shape)

        # initialise regulariser / prior KL term
        assert len(input_shape) == 4, 'this wrapper only supports Conv2D layers'
        if self.data_format == 'channels_first':
            self.input_dim = input_shape[1]  # we drop only channels
        else:
            self.input_dim = input_shape[3]

    def call(self, inputs, training=None):
        return super().call(inputs, training=training)


class ConcreteSpatialDropout3D(ConcreteDropout):
    """
    This wrapper allows to learn the dropout probability for any given Conv3D input layer. 
    It is the Concrete Dropout implementation of `SpatialDropout3D`.

    ```python
        x = # some input layer
        x = ConcreteSpatialDropout3D(Conv3D(64, (3,3,3))(x)
    ```
    # Arguments
        layer: a layer instance.
        weight_regularizer:
            A positive number which satisfies
                $weight_regularizer = l**2 / (\tau * N)$
            with prior lengthscale l, model precision $\tau$ (inverse observation noise),
            and N the number of instances in the dataset.
            Note that kernel_regularizer is not needed.
        dropout_regularizer:
            A positive number which satisfies
                $dropout_regularizer = 2 / (\tau * N)$
            with model precision $\tau$ (inverse observation noise) and N the number of
            instances in the dataset.
            Note the relation between dropout_regularizer and weight_regularizer:
                $weight_regularizer / dropout_regularizer = l**2 / 2$
            with prior lengthscale l. Note also that the factor of two should be
            ignored for cross-entropy loss, and used only for the eculedian loss.
        temperature:
            The temperature of the Concrete Distribution. 
            Must be low enough in order to have the correct KL approximation
            For:
                $t \rightarrow 0$ we obtain the discrete distribution
                $t \rightarrow \infty$ we obtain a uniform distribution
    """

    def __init__(self, layer, temperature=2./3., **kwargs):
        super(ConcreteSpatialDropout3D, self).__init__(
            layer, temperature=temperature, **kwargs)

    # implement the noise shape for spatial dropout
    def _get_noise_shape(self, inputs):
        input_shape = tf.shape(inputs)
        if self.data_format == 'channels_first':
            return (input_shape[0], input_shape[1], 1, 1, 1)
        elif self.data_format == 'channels_last':
            return (input_shape[0], 1, 1, 1, input_shape[4])

    def build(self, input_shape=None):
        self.input_spec = InputSpec(shape=input_shape)

        super(ConcreteSpatialDropout3D, self).build(input_shape=input_shape)

        # initialise regulariser / prior KL term
        assert len(input_shape) == 5, 'this wrapper only supports Conv3D layers'
        if self.data_format == 'channels_first':
            self.input_dim = input_shape[1]  # we drop only channels
        else:
            self.input_dim = input_shape[4]

    def call(self, inputs, training=None):
        return super().call(inputs, training=training)

# depthwise conv


class ConcreteDropoutDepthwise(ConcreteDropout):
    """
    Base class for ConcreteDropout. It allows to learn the dropout probability for a given layer.

    # Arguments
        layer: a layer instance.
        weight_regularizer:
            A positive number which satisfies
                $weight_regularizer = l**2 / (\tau * N)$
            with prior lengthscale l, model precision $\tau$ (inverse observation noise),
            and N the number of instances in the dataset.
            Note that kernel_regularizer is not needed.
        dropout_regularizer:
            A positive number which satisfies
                $dropout_regularizer = 2 / (\tau * N)$
            with model precision $\tau$ (inverse observation noise) and N the number of
            instances in the dataset.
            Note the relation between dropout_regularizer and weight_regularizer:
                $weight_regularizer / dropout_regularizer = l**2 / 2$
            with prior lengthscale l. Note also that the factor of two should be
            ignored for cross-entropy loss, and used only for the eculedian loss.
        temperature:
            The temperature of the Concrete Distribution. 
            Must be low enough in order to have the correct KL approximation
            For:
                $t \rightarrow 0$ we obtain the discrete distribution.
                $t \rightarrow \infty$ we obtain a uniform distribution
    """

    def __init__(self, layer, **kwargs):
        assert 'depthwise_regularizer' not in kwargs, "Must not provide a kernel regularizer."
        super(ConcreteDropoutDepthwise, self).__init__(layer, **kwargs)

    # implement the noise shape for spatial dropout
    def build(self, input_shape=None):
        self.input_spec = InputSpec(shape=input_shape)
        super(ConcreteDropoutDepthwise, self).build(input_shape=input_shape)

    def call(self, inputs, training=None):
        p = tf.nn.sigmoid(self.p_logit)
        # We will now compute the KL terms following eq.3 of 1705.07832
        weight = self.layer.depthwise_kernel
        # The kernel regularizer corresponds to the first term
        # Note: we  divide by (1 - p) because  we  scaled  layer  output  by(1 - p)
        kernel_regularizer = self.weight_regularizer * tf.reduce_sum(tf.square(
            weight)) / (1. - p)
        # the dropout regularizer corresponds to the second term
        dropout_regularizer = p * tf.math.log(p)
        dropout_regularizer += (1. - p) * tf.math.log1p(- p)
        dropout_regularizer *= self.dropout_regularizer * self.input_dim
        # this is the KL term to be added as a loss
        regularizer = tf.reduce_sum(kernel_regularizer + dropout_regularizer)

        self.layer.add_loss(regularizer)

        if self.is_mc_dropout:
            return self.layer.call(self.spatial_concrete_dropout(inputs, p))
        else:
            def relaxed_dropped_inputs():
                return self.layer.call(self.spatial_concrete_dropout(inputs, p))
            return K.in_train_phase(relaxed_dropped_inputs,
                                    self.layer.call(inputs),
                                    training=training)


class ConcreteSpatialDropoutDepthwise1D(ConcreteDropoutDepthwise):
    """
    This wrapper allows to learn the dropout probability for any given Conv1D input layer. 
    It is the Concrete Dropout implementation of `SpatialDropout1D`.

    ```python
        x = # some input layer
        x = ConcreteSpatialDropout1D(Conv1D(64, 3))(x)
    ```
    # Arguments
        layer: a layer instance.
        weight_regularizer:
            A positive number which satisfies
                $weight_regularizer = l**2 / (\tau * N)$
            with prior lengthscale l, model precision $\tau$ (inverse observation noise),
            and N the number of instances in the dataset.
            Note that kernel_regularizer is not needed.
        dropout_regularizer:
            A positive number which satisfies
                $dropout_regularizer = 2 / (\tau * N)$
            with model precision $\tau$ (inverse observation noise) and N the number of
            instances in the dataset.
            Note the relation between dropout_regularizer and weight_regularizer:
                $weight_regularizer / dropout_regularizer = l**2 / 2$
            with prior lengthscale l. Note also that the factor of two should be
            ignored for cross-entropy loss, and used only for the eculedian loss.
        temperature:
            The temperature of the Concrete Distribution. 
            Must be low enough in order to have the correct KL approximation.
            For:
                $t \rightarrow 0$ we obtain the discrete distribution
                $t \rightarrow \infty$ we obtain a uniform distribution
    """

    def __init__(self, layer, temperature=2./3., **kwargs):
        super(ConcreteSpatialDropoutDepthwise1D, self).__init__(
            layer, temperature=temperature, **kwargs)

    def _get_noise_shape(self, inputs):
        input_shape = tf.shape(inputs)
        if self.data_format == 'channels_first':
            return (input_shape[0], input_shape[1], 1)
        elif self.data_format == 'channels_last':
            return (input_shape[0], 1, input_shape[2])

    # implement the noise shape for spatial dropout
    def build(self, input_shape=None):
        self.input_spec = InputSpec(shape=input_shape)

        super(ConcreteSpatialDropoutDepthwise1D,
              self).build(input_shape=input_shape)

        # initialise regulariser / prior KL term
        assert len(input_shape) == 3, 'this wrapper only supports Conv1D layers'
        if self.data_format == 'channels_first':
            self.input_dim = input_shape[1]  # we drop only channels
        else:
            self.input_dim = input_shape[2]

    def call(self, inputs, training=None):
        return super().call(inputs, training=training)


class ConcreteSpatialDropoutDepthwise2D(ConcreteDropoutDepthwise):
    """
    This wrapper allows to learn the dropout probability for any given Conv2D input layer. 
    It is the Concrete Dropout implementation of `SpatialDropout2D`.

    ```python
        x = # some input layer
        x = ConcreteSpatialDropout2D(Conv2D(64, (3,3))(x)
    ```
    # Arguments
        layer: a layer instance.
        weight_regularizer:
            A positive number which satisfies
                $weight_regularizer = l**2 / (\tau * N)$
            with prior lengthscale l, model precision $\tau$ (inverse observation noise),
            and N the number of instances in the dataset.
            Note that kernel_regularizer is not needed.
        dropout_regularizer:
            A positive number which satisfies
                $dropout_regularizer = 2 / (\tau * N)$
            with model precision $\tau$ (inverse observation noise) and N the number of
            instances in the dataset.
            Note the relation between dropout_regularizer and weight_regularizer:
                $weight_regularizer / dropout_regularizer = l**2 / 2$
            with prior lengthscale l. Note also that the factor of two should be
            ignored for cross-entropy loss, and used only for the eculedian loss.
        temperature:
            The temperature of the Concrete Distribution. 
            Must be low enough in order to have the correct KL approximation.
            For:
                $t \rightarrow 0$ we obtain the discrete distribution
                $t \rightarrow \infty$ we obtain a uniform distribution
    """

    def __init__(self, layer, temperature=2./3., **kwargs):
        super(ConcreteSpatialDropoutDepthwise2D, self).__init__(
            layer, temperature=temperature, **kwargs)

    # implement the noise shape for spatial dropout
    def _get_noise_shape(self, inputs):
        input_shape = tf.shape(inputs)
        if self.data_format == 'channels_first':
            return (input_shape[0], input_shape[1], 1, 1)
        elif self.data_format == 'channels_last':
            return (input_shape[0], 1, 1, input_shape[3])

    def build(self, input_shape=None):
        self.input_spec = InputSpec(shape=input_shape)

        super(ConcreteSpatialDropoutDepthwise2D,
              self).build(input_shape=input_shape)

        # initialise regulariser / prior KL term
        assert len(input_shape) == 4, 'this wrapper only supports Conv2D layers'
        if self.data_format == 'channels_first':
            self.input_dim = input_shape[1]  # we drop only channels
        else:
            self.input_dim = input_shape[3]

    def call(self, inputs, training=None):
        return super().call(inputs, training=training)

# utils


def get_weight_regularizer(N, l=1e-2, tau=0.1):
    return l**2 / (tau * N)


def get_dropout_regularizer(N, tau=0.1, cross_entropy_loss=False):
    reg = 1 / (tau * N)
    if not cross_entropy_loss:
        reg *= 2
    return reg
