r"""

# Backbone Networks for Continual Learning

This submodule provides the **neural network architectures** for continual learning** that can be used in CLArena. 

Please note that this is an API documentation. Please refer to the main documentation page for more information about the backbone networks and how to use and customize them:

- **Configure your backbone network:** [https://pengxiang-wang.com/projects/continual-learning-arena/docs/configure-your-experiments/backbone-network](https://pengxiang-wang.com/projects/continual-learning-arena/docs/configure-your-experiments/backbone-network)
- **Implement your backbone network:** [https://pengxiang-wang.com/projects/continual-learning-arena/docs/implement-your-cl-modules/backbone-network](https://pengxiang-wang.com/projects/continual-learning-arena/docs/implement-your-cl-modules/backbone-network)

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
