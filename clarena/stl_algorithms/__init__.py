r"""

# Multi-Task Learning Algorithms

This submodule provides the **multi-task learning algorithms** in CLArena.

The algorithms are implemented as subclasses of `MTLAlgorithm`.

Please note that this is an API documantation. Please refer to the main documentation pages for more information about the backbone networks and how to configure and implement them:

- [**Configure MTL Algorithm**](https://pengxiang-wang.com/projects/continual-learning-arena/docs/configure-your-experiment/cl-algorithm)
- [**Implement Your CL Algorithm Class**](https://pengxiang-wang.com/projects/continual-learning-arena/docs/implement-your-cl-modules/cl-algorithm)
- [**A Beginners' Guide to Continual Learning (Methodology Overview)**](https://pengxiang-wang.com/posts/continual-learning-beginners-guide#sec-methodology)
"""

from .base import STLAlgorithm

#
from .single_learning import SingleLearning
