r"""

# Backbone Networks

This submodule provides the **backbone neural network architectures** for various machine learning paradigms in CLArena.

Here are the base classes for backbone networks, which inherit from PyTorch `nn.Module`:

- `Backbone`: The base class for backbones.
- `CLBackbone`: The base class for continual learning backbones.
- `HATMaskBackbone`: The base class for backbones used in [HAT (Hard Attention to the Task)](http://proceedings.mlr.press/v80/serra18a) CL algorithm.
- `WSNMaskBackbone`: The base class for backbones used in [WSN (Winning Subnetworks)](https://proceedings.mlr.press/v162/kang22b/kang22b.pdf) CL algorithm.

Please note that this is an API documentation. Please refer to the main documentation pages for more information about how to configure and implement backbone networks:

- [**Configure Backbone Network (CL Main)**](https://pengxiang-wang.com/projects/continual-learning-arena/docs/continual-learning/configure-main-experiment/backbone-network)
- [**Configure Backbone Network (MTL)**](https://pengxiang-wang.com/projects/continual-learning-arena/docs/multi-task-learning-arena/configure-experiment/backbone-network)
- [**Configure Backbone Network (STL)**](https://pengxiang-wang.com/projects/continual-learning-arena/docs/single-task-learning-arena/configure-experiment/backbone-network)
- [**Implement Custom Backbone Network**](https://pengxiang-wang.com/projects/continual-learning-arena/docs/custom-implementation/backbone-network)


"""

from .base import (
    Backbone,
    CLBackbone,
    HATMaskBackbone,
    WSNMaskBackbone,
    NISPAMaskBackbone,
)
from .mlp import MLP, CLMLP, HATMaskMLP, WSNMaskMLP
from .resnet import (
    ResNet18,
    ResNet34,
    ResNet50,
    ResNet101,
    ResNet152,
    CLResNet18,
    CLResNet34,
    CLResNet50,
    CLResNet101,
    CLResNet152,
    HATMaskResNet18,
    HATMaskResNet34,
    HATMaskResNet50,
    HATMaskResNet101,
    HATMaskResNet152,
)


__all__ = [
    "Backbone",
    "CLBackbone",
    "HATMaskBackbone",
    "WSNMaskBackbone",
    "NISPAMaskBackbone",
    "mlp",
    "resnet",
]
