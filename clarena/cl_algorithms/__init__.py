r"""

# Continual Learning Algorithms

This submodule provides the **continual learning algorithms** in CLArena.

Please note that this is an API documantation. Please refer to the main documentation pages for more information about the backbone networks and how to configure and implement them:

- [**Configure CL Algorithm**](https://pengxiang-wang.com/projects/continual-learning-arena/docs/configure-your-experiment/cl-algorithm)
- [**Implement Your CL Algorithm Class**](https://pengxiang-wang.com/projects/continual-learning-arena/docs/implement-your-cl-modules/cl-algorithm)
- [**A Beginners' Guide to Continual Learning (Methodology Overview)**](https://pengxiang-wang.com/posts/continual-learning-beginners-guide#sec-methodology)


The algorithms are implemented as subclasses of `CLAlgorithm`.

"""

from .base import CLAlgorithm, JointLearning

# finetuning first
from .finetuning import Finetuning
from .independent import Independent
from .fix import Fix
from .random import Random

from .lwf import LwF
from .ewc import EWC
from .cbp import CBP

from .hat import HAT
from .adahat import AdaHAT
from .fgadahat import FGAdaHAT
from .wsn import WSN


__all__ = [
    "CLAlgorithm",
    "JointLearning",
    "regularisers",
    "finetuning",
    "independent",
    "fix",
    "random",
    "lwf",
    "ewc",
    "hat",
    "cbp",
    "adahat",
    "fgadahat",
    "wsn",
]
