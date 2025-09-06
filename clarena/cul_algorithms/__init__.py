r"""

# Continual Unlearning Algorithms

This submodule provides the **continual unlearning algorithms** in CLArena.

Here are the base classes for CUL algorithms:

- `CULAlgorithm`: the base class for all continual unlearning algorithms.

Please note that this is an API documantation. Please refer to the main documentation pages for more information about how to configure and implement CUL algorithms:

- [**Configure CUL Algorithm**](https://pengxiang-wang.com/projects/continual-learning-arena/docs/components/cul-algorithm)
- [**Implement Custom CUL Algorithm**](https://pengxiang-wang.com/projects/continual-learning-arena/docs/custom-implementation/cul-algorithm)



"""

from .base import CULAlgorithm
from .independent_unlearn import IndependentUnlearn
from .amnesiac_hat_unlearn import AmnesiacHATUnlearn


__all__ = ["CULAlgorithm", "independent_unlearn", "amnesiac_hat_unlearn"]
