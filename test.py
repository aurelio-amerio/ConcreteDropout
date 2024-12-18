#%%
import sys
sys.path.append('src')
import os 
os.environ["CONCRETEDROPOUT_BACKEND"] = "jax"
#%%
from ConcreteDropout import set_backend
# set_backend("torch")
from  ConcreteDropout import ConcreteDropout, CDRegLoss, get_dropout_regularizer, get_weight_regularizer

# %%
# %%
from flax import nnx
import jax 
import jax.numpy as jnp
# %%
rngs = nnx.Rngs(0)
# %%
conv1d = nnx.Conv(3,4, kernel_size=(3,), strides=(1,), padding='SAME', rngs=rngs)
conv1dCD = ConcreteDropout(conv1d, broadcast_dims=(1,),weight_regularizer=1e-6, dropout_regularizer=1e-5, init_min=0.1, init_max=0.1, temperature=0.1, deterministic=False, rngs=rngs)

conv1d_2 = nnx.Conv(4,5, kernel_size=(3,), strides=(1,), padding='SAME', rngs=rngs)
conv1dCD_2 = ConcreteDropout(conv1d_2, broadcast_dims=(1,),weight_regularizer=1e-6, dropout_regularizer=1e-5, init_min=0.1, init_max=0.1, temperature=0.1, deterministic=False, rngs=rngs)
# %%
x = jnp.ones((20,64,64,3))
# %%
class testmodel(nnx.Module):
    def __init__(self, din: int, dout: int, *, rngs: nnx.Rngs):
        self.conv1 = nnx.Conv(din,32, kernel_size=(3,3), strides=(2,2), padding='SAME', rngs=rngs)
        self.conv1_cd = ConcreteDropout(self.conv1, broadcast_dims=(1,2),weight_regularizer=1e-6, dropout_regularizer=1e-5, init_min=0.1, init_max=0.1, temperature=0.1, deterministic=False, rngs=rngs)
        self.conv2 = nnx.Conv(32,dout, kernel_size=(3,3), strides=(2,2), padding='SAME', rngs=rngs)
        self.conv2_cd = ConcreteDropout(self.conv2, broadcast_dims=(1,2),weight_regularizer=1e-6, dropout_regularizer=1e-5, init_min=0.1, init_max=0.1, temperature=0.1, deterministic=False, rngs=rngs)
        self.din, self.dout = din, dout

    def get_regularization_loss(self):
        return self.conv1_cd.reg_loss + self.conv2_cd.reg_loss

    def __call__(self, x: jax.Array):
        y = self.conv1_cd(x)
        y = jax.nn.relu(y)
        y = self.conv2_cd(y)
        return y
# %%
model = testmodel(3,64, rngs=rngs)
# %%
model(x)
# %%
model.get_regularization_loss()
# # %%
# intermediates["conv1_cd"].regularizer.value
# # %%
# total_regularizer_value = sum(layer.regularizer.value for layer in intermediates.values())

# # %%
# jnp.concatenate([layer.regularizer.value[0] for layer in intermediates.values()]).sum()
# %%
intermediates = nnx.state(model, CDRegLoss)
# %%
intermediates
# %%
