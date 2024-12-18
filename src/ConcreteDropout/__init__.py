import os 
from .util import get_weight_regularizer, get_dropout_regularizer

# get backend from environment variable
backend_name = os.getenv("CONCRETEDROPOUT_BACKEND", "none").lower()

def import_packages():
    global ConcreteDenseDropout, ConcreteSpatialDropout1D, ConcreteSpatialDropout2D, ConcreteSpatialDropout3D, ConcreteDropoutDepthwise, ConcreteSpatialDropoutDepthwise1D, ConcreteSpatialDropoutDepthwise2D
    global ConcreteLinearDropout, ConcreteDropout1D, ConcreteDropout2D, ConcreteDropout3D
    global ConcreteDropout, CDRegLoss
    if backend_name == "tensorflow":
        try:
            import tensorflow as tf
            from .tensorflow_backend import ConcreteDenseDropout, ConcreteSpatialDropout1D, ConcreteSpatialDropout2D, ConcreteSpatialDropout3D, ConcreteDropoutDepthwise, ConcreteSpatialDropoutDepthwise1D, ConcreteSpatialDropoutDepthwise2D
        except ImportError:
            raise ImportError("TensorFlow not found. Install with 'pip install concretedropout[tensorflow]'")
    elif backend_name == "jax":
        import jax.numpy as jnp
        from .jax_backend import ConcreteDropout, CDRegLoss
        # try:
        # except ImportError:
        #     raise ImportError("JAX not found. Install with 'pip install concretedropout[jax]'")
    elif backend_name == "torch":
        try:
            import torch
            from .torch_backend import ConcreteLinearDropout, ConcreteDropout1D, ConcreteDropout2D, ConcreteDropout3D
        except ImportError:
            raise ImportError("PyTorch not found. Install with 'pip install concretedropout[torch]'")
    elif backend_name == "none":
        print("No backend selected. Functionality will be limited. Set MYPACKAGE_BACKEND to 'tensorflow', 'jax', or 'torch'.")
    else:
        raise ValueError(f"Invalid backend: {backend_name}. Use 'tensorflow', 'jax', or 'torch'.")

def set_backend(new_backend):
    global backend_name
    backend_name = new_backend
    import_packages()
    return

set_backend(backend_name)

