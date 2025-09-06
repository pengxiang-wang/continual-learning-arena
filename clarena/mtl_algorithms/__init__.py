r"""

# Multi-Task Learning Algorithms

This submodule provides the **multi-task learning algorithms** in CLArena.

Here are the base classes for MTL algorithms, which inherit from PyTorch Lightning `LightningModule`:

- `MTLAlgorithm`: the base class for all multi-task learning algorithms.


Please note that this is an API documantation. Please refer to the main documentation pages for more information about how to configure and implement MTL algorithms:

- [**Configure MTL Algorithm**](https://pengxiang-wang.com/projects/continual-learning-arena/docs/components/mtl-algorithm)
- [**Implement Custom MTL Algorithm**](https://pengxiang-wang.com/projects/continual-learning-arena/docs/custom-implementation/mtl-algorithm)

"""

from .base import MTLAlgorithm

from .joint_learning import JointLearning

__all__ = ["MTLAlgorithm", "joint_learning"]
