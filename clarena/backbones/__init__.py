"""

# Backbone Networks for Continual Learning

This submodule provides the **neural network architectures** for continual learning** that can be used in CLArena. 

Please note that this is an API documantation. Please refer to the main documentation page for more information about the backbone networks and how to use and customize them:

- **Configure your backbone network:** [https://pengxiang-wang.com/projects/continual-learning-arena/docs/configure-your-experiments/backbone-network](https://pengxiang-wang.com/projects/continual-learning-arena/docs/configure-your-experiments/backbone-network)
- **Implement your backbone network:** [https://pengxiang-wang.com/projects/continual-learning-arena/docs/implement-your-cl-modules/backbone-network](https://pengxiang-wang.com/projects/continual-learning-arena/docs/implement-your-cl-modules/backbone-network)

"""

from .base import CLBackbone
from .mlp import MLP

__all__ = ["CLBackbone", "mlp"]
