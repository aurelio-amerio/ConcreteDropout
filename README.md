# ConcreteDropout-TF2
Concrete Dropout updated implementation for Tensorflow 2.0 following the [original code](https://github.com/yaringal/ConcreteDropout) from the paper.

# Introduction
Concrete dropout allows for the dropout probability of a layer to become a trainable parameter. For more information, see the original paper: [https://arxiv.org/abs/1705.07832](https://arxiv.org/abs/1705.07832)

This package implements Concrete Dropout for the following layers:
- Dense - `ConcreteDenseDropout`
- Conv1D - `ConcreteSpatialDropout1D`
- Conv2D - `ConcreteSpatialDropout2D`
- Conv3D - `ConcreteSpatialDropout2D`
- DepthwiseConv1D - `ConcreteSpatialDropoutDepthwise1D`
- DepthwiseConv2D - `ConcreteSpatialDropoutDepthwise2D`
- DepthwiseConv3D - `ConcreteSpatialDropoutDepthwise3D`

Please notice that the dropout layer will be applied **before** the chosen layer.

# Arguments
Each concrete dropout layer supports the following arguments:
- `layer`: 
    an instance of the layer to which concrete dropout will be applied
- `weight_regularizer=1e-6`:
    A positive number which satisfies
    $$
    \text{weight\_regularizer} = l^2 / (\tau * N)
    $$
    with prior lengthscale l, model precision $\tau$ (inverse observation noise),
    and N the number of instances in the dataset.
    Note that kernel_regularizer is not needed.
    The appropriate weight_regularizer value can be computed with the utility function `get_weight_regularizer(N, l, tau)`
- `dropout_regularizer=1e-5`:
    A positive number which satisfies
    $$
        \text{dropout\_regularizer} = 2 / (\tau * N)
    $$
    with model precision $\tau$ (inverse observation noise) and N the number of
    instances in the dataset.
    Note the relation between dropout_regularizer and weight_regularizer:
        $weight_regularizer / dropout_regularizer = l**2 / 2$
    with prior lengthscale l. Note also that the factor of two should be
    ignored for cross-entropy loss, and used only for the eculedian loss.
    The appropriate dropout_regularizer value can be computed with the utility function `get_dropout_regularizer(N, tau, cross_entropy_loss=False)`. By default, a regression problem will be assumed. 
- `init_min=0.1`: minimum value for the random initial dropout probability
- `init_max=0.1`: maximum value for the random initial dropout probability
- `is_mc_dropout=False`: enables Monte Carlo Dropout (i.e. dropout will remain active also at prediction time). Default: False. 
- `data_format=None`: channels_last or channels_first. Defaults to channels_last.
- `temperature`: temperature of the concrete distribution. For more information see [arXiv:1611.00712](https://arxiv.org/abs/1611.00712). Defaults to `0.1` for dense layers, and `2/3` for convolution layers.

# Example
The suggested way to employ concrete dropout layers is the following:
```python
import tensorflow as tf
from concretedropout import ConcreteDenseDropout 

#... import the dataset
Ns = x_train.shape[0]
# get the regularizers
wr = get_weight_regularizer(Ns, l=1e-2, tau=1.0) # tau is the inverse 
dr = get_dropout_regularizer(Ns, tau=1.0, cross_entropy_loss=True)

# ... a neural network with output x
dense1 = tf.keras.layers.Dense(N_neurons, weight_regularizer=wr, dropout_regularizer=dr)
x = ConcreteDenseDropout(dense1)(x)
```

For a practical example on how to use concrete dropout for the mnist dataset, see this [example](https://github.com/aurelio-amerio/ConcreteDropout-TF2/blob/main/examples/mnist_convnet_concrete_dropout.ipynb).

# Bayesian neural network with MCDropout
You can find [here](https://github.com/aurelio-amerio/ConcreteDropout-TF2/blob/main/examples/regression_MCDropout.ipynb) an example on how to use MCDropout and Concrete Dropout to implement a Bayesian Neural Network with MCDropout. For more information, see [arXiv:1506.02142](https://arxiv.org/abs/1506.02142).




