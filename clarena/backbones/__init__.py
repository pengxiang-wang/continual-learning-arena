r"""

# Backbone Networks

This submodule provides the **backbone neural network architectures** for all paradigms in CLArena.

Here are the base classes for backbone networks, which inherit from PyTorch `nn.Module`:

- `Backbone`: the base class for all backbone networks. Multi-task and single-task learning can use this class directly.
-   `CLBackbone`: the base class for continual learning backbone networks, which incorporates mechanisms for managing continual learning tasks.
    - `HATMaskBackbone`: the base class for backbones used in [HAT (Hard Attention to the Task)](http://proceedings.mlr.press/v80/serra18a) CL algorithm.
    - `WSNMaskBackbone`: The base class for backbones used in [WSN (Winning Subnetworks)](https://proceedings.mlr.press/v162/kang22b/kang22b.pdf) CL algorithm.

Please note that this is an API documentation. Please refer to the main documentation pages for more information about how to configure and implement backbone networks:

- [**Configure Backbone Network**](https://pengxiang-wang.com/projects/continual-learning-arena/docs/components/backbone-network)
- [**Implement Custom Backbone Network**](https://pengxiang-wang.com/projects/continual-learning-arena/docs/custom-implementation/backbone-network)


"""

from .base import (
    Backbone,
    CLBackbone,
    HATMaskBackbone,
    WSNMaskBackbone,
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
    "mlp",
    "resnet",
]
