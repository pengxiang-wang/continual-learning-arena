r"""

# Backbone Networks for Continual Learning

This submodule provides the **backbone neural network architectures for continual learning**. 

Please note that this is an API documentation. Please refer to the main documentation pages for more information about the backbone networks and how to 
configure and implement them:

- [**Configure Backbone Network**](https://pengxiang-wang.com/projects/continual-learning-arena/docs/configure-your-experiment/backbone-network)
- [**Implement Your CL Backbone Class**](https://pengxiang-wang.com/projects/continual-learning-arena/docs/implement-your-CL-modules/backbone-network)



The backbones are implemented as subclasses of `CLBackbone` classes, which are the base class for all continual learning backbones in CLArena.

- `CLBackbone`: The base class for continual learning backbones.
- `HATMaskBackbone`: The base class for backbones used in [HAT (Hard Attention to the Task) algorithm](http://proceedings.mlr.press/v80/serra18a). A child class of `CLBackbone`.


"""

from .base import CLBackbone, HATMaskBackbone
from .mlp import MLP, HATMaskMLP
from .resnet import (
    HATMaskResNet18,
    HATMaskResNet34,
    HATMaskResNet50,
    HATMaskResNet101,
    HATMaskResNet152,
    ResNet18,
    ResNet34,
    ResNet50,
    ResNet101,
    ResNet152,
)

__all__ = ["CLBackbone", "HATMaskBackbone", "mlp", "resnet"]
