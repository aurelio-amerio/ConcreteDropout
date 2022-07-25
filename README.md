# ConcreteDropout
[![Downloads](https://pepy.tech/badge/concretedropout)](https://pepy.tech/project/concretedropout)
[![](https://img.shields.io/pypi/v/concretedropout.svg?maxAge=3600)](https://pypi.org/project/concretedropout/)

Concrete Dropout updated implementation for Tensorflow 2.0 and PyTorch, following the [original code](https://github.com/yaringal/ConcreteDropout) from the paper.
# Installation
To install this package, please use:
```bash
pip install concretedropout
```

# Introduction
Concrete dropout allows for the dropout probability of a layer to become a trainable parameter. For more information, see the original paper: [https://arxiv.org/abs/1705.07832](https://arxiv.org/abs/1705.07832)

This package implements Concrete Dropout for the following layers:
Tensorflow:
- Dense - `tensorflow.ConcreteDenseDropout`
- Conv1D - `tensorflow.ConcreteSpatialDropout1D`
- Conv2D - `tensorflow.ConcreteSpatialDropout2D`
- Conv3D - `tensorflow.ConcreteSpatialDropout3D`
- DepthwiseConv1D - `tensorflow.ConcreteSpatialDropoutDepthwise1D`
- DepthwiseConv2D - `tensorflow.ConcreteSpatialDropoutDepthwise2D`

PyTorch:
- Linear - `pytorch.ConcreteLinearDropout`
- Conv1d - `pytorch.ConcreteDropout1d`
- Conv2d - `pytorch.ConcreteDropout2d`
- Conv3d - `pytorch.ConcreteDropout3d`

Please notice that the dropout layer will be applied **before** the chosen layer.

# Arguments
Each concrete dropout layer supports the following arguments:
- `layer`: 
    an instance of the layer to which concrete dropout will be applied. Only required for Tensorflow.
- `weight_regularizer=1e-6`:
    A positive number which satisfies weight_regularizer = $l^2 / (\tau * N)$ with prior lengthscale l, model precision τ (inverse observation noise), and N the number of instances in the dataset.
    Note that kernel_regularizer is not needed.
    The appropriate weight_regularizer value can be computed with the utility function `get_weight_regularizer(N, l, tau)`
- `dropout_regularizer=1e-5`:
    A positive number which satisfies dropout_regularizer = $2 / (\tau * N)$ with model precision τ (inverse observation noise) and N the number of instances in the dataset.
    Note the relation between dropout_regularizer and weight_regularizer: weight_regularizer / dropout_regularizer = $l^2 / 2$ with prior lengthscale l. Note also that the factor of two should be
    ignored for cross-entropy loss, and used only for the eculedian loss.
    The appropriate dropout_regularizer value can be computed with the utility function `get_dropout_regularizer(N, tau, cross_entropy_loss=False)`. By default, a regression problem will be assumed. 
- `init_min=0.1`: minimum value for the random initial dropout probability
- `init_max=0.1`: maximum value for the random initial dropout probability
- `is_mc_dropout=False`: enables Monte Carlo Dropout (i.e. dropout will remain active also at prediction time). Default: False. 
- `data_format=None`: channels_last or channels_first (only for Tensorflow). Defaults to channels_last for Tensorflow. PyTorch defaults to channel_first.
- `temperature`: temperature of the concrete distribution. For more information see [arXiv:1611.00712](https://arxiv.org/abs/1611.00712). Defaults to `0.1` for dense layers, and `2/3` for convolution layers.

# Example
The suggested way to employ concrete dropout layers is the following.
Tensorflow:
```python
import tensorflow as tf
from concretedropout.tensorflow import ConcreteDenseDropout 

#... import the dataset
Ns = x_train.shape[0]
# get the regularizers
wr = get_weight_regularizer(Ns, l=1e-2, tau=1.0) # tau is the inverse 
dr = get_dropout_regularizer(Ns, tau=1.0, cross_entropy_loss=True)

# ... a neural network with output x
dense1 = tf.keras.layers.Dense(N_neurons)
x = ConcreteDenseDropout(dense1, weight_regularizer=wr, dropout_regularizer=dr)(x)
```
PyTorch:
```python
import torch 
from concretedropout.pytorch import ConcreteLinearDropout 

#... import the dataset
Ns = x_train.shape[0]
# get the regularizers
wr = get_weight_regularizer(Ns, l=1e-2, tau=1.0) # tau is the inverse 
dr = get_dropout_regularizer(Ns, tau=1.0, cross_entropy_loss=True)

# ... a neural network with output x
linear = torch.nn.Linear(n_input, N_neurons)
x = ConcreteLinearDropout(weight_regularizer=wr, dropout_regularizer=dr)(x, linear)

# inside the train step of your model, you need to add a new regularization term, which is due to the concrete dropout:
def training_step(self, batch, batch_nb):
    x, y = batch
    output = self(x)
    
    reg = torch.zeros(1) # get the regularization term
    for module in filter(lambda x: isinstance(x, ConcreteDropout), self.modules()):
        reg += module.regularization

    loss = self.loss(output, y) + reg # add the reg term
    return loss
```
For a practical example on how to use concrete dropout for the mnist dataset, see this [Tensorflow example](https://github.com/aurelio-amerio/ConcreteDropout-TF2/blob/main/examples/Tensorflowmnist_convnet_concrete_dropout.ipynb) and this [PyTorch example](https://github.com/aurelio-amerio/ConcreteDropout-TF2/blob/main/examples/PyTorch/MNIST_pytorch.ipynb).

# Bayesian neural network with MCDropout
You can find [here](https://github.com/aurelio-amerio/ConcreteDropout-TF2/blob/main/examples/regression_MCDropout.ipynb) an example on how to use MCDropout and Concrete Dropout to implement a Bayesian Neural Network with MCDropout on Tensorflow. For more information, see [arXiv:1506.02142](https://arxiv.org/abs/1506.02142).

# Known issues
Due to the way the additional dropout loss term is added to the main loss term, during training and evaluation the **model loss** might become a **negative number**. This has no impact on the actual optimisation of the model. If you desire to track your loss function separately, as a work around it is advised to add it to the list of metrics. 
