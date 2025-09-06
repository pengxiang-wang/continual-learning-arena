r"""

# Single-Task Learning Algorithms

This submodule provides the **single-task learning algorithms** in CLArena.

Here are the base classes for STL algorithms, which inherit from PyTorch Lightning `LightningModule`:

- `STLAlgorithm`: the base class for all single-task learning algorithms.

Please note that this is an API documantation. Please refer to the main documentation pages for more information about how to configure and implement STL algorithms:

- [**Configure STL Algorithm**](https://pengxiang-wang.com/projects/continual-learning-arena/docs/components/stl-algorithm)
- [**Implement Custom STL Algorithm**](https://pengxiang-wang.com/projects/continual-learning-arena/docs/custom-implementation/stl-algorithm)

"""

from .base import STLAlgorithm

from .single_learning import SingleLearning

__all__ = ["STLAlgorithm", "single_learning"]
