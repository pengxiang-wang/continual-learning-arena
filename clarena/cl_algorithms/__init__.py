r"""

# Continual Learning Algorithms

This submodule provides the **continual learning algorithms** in CLArena. 

Please note that this is an API documantation. Please refer to the main documentation pages for more information about the backbone networks and how to configure and implement them:

- [**Configure CL Algorithm**](https://pengxiang-wang.com/projects/continual-learning-arena/docs/configure-your-experiment/cl-algorithm)
- [**Implement Your CL Algorithm Class**](https://pengxiang-wang.com/projects/continual-learning-arena/docs/implement-your-cl-modules/cl-algorithm)
- [**A Beginners' Guide to Continual Learning (Methodology Overview)**](https://pengxiang-wang.com/posts/continual-learning-beginners-guide#sec-methodology)


The algorithms are implemented as subclasses of `CLAlgorithm`.

"""

from .base import CLAlgorithm

# finetuning first
from .finetuning import Finetuning
from .fix import Fix

from .lwf import LwF
from .ewc import EWC
from .cbp import CBP

from .hat import HAT
from .adahat import AdaHAT
from .cbphat import CBPHAT


__all__ = [
    "CLAlgorithm",
    "regularisers",
    "finetuning",
    "fix",
    "lwf",
    "ewc",
    "hat",
    "cbp",
    "adahat",
    "cbphat",
]
