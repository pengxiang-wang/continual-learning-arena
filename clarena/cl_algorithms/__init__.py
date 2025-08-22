r"""

# Continual Learning Algorithms

This submodule provides the **continual learning algorithms** in CLArena.

The algorithms are implemented as subclasses of `CLAlgorithm`.

Please note that this is an API documentation. Please refer to the main documentation pages for more information about and how to configure and implement CL algorithms:

- [**Configure CL Algorithm (CL Main)**](https://pengxiang-wang.com/projects/continual-learning-arena/docs/continual-learning/configure-main-experiment/cl-algorithm)
- [**Implement Custom CL Algorithm**](https://pengxiang-wang.com/projects/continual-learning-arena/docs/custom-implementation/cl-algorithm)
- [**A Beginners' Guide to Continual Learning (Methodology Overview)**](https://pengxiang-wang.com/posts/continual-learning-beginners-guide#sec-methodology)


"""

from .base import CLAlgorithm, UnlearnableCLAlgorithm

# finetuning first
from .finetuning import Finetuning
from .independent import Independent, UnlearnableIndependent
from .fix import Fix
from .random import Random

from .lwf import LwF
from .ewc import EWC
from .cbp import CBP

from .hat import HAT
from .adahat import AdaHAT
from .fgadahat import FGAdaHAT
from .wsn import WSN
from .nispa import NISPA


__all__ = [
    "CLAlgorithm",
    "UnlearnableCLAlgorithm",
    "regularizers",
    "finetuning",
    "independent",
    "fix",
    "random",
    "lwf",
    "ewc",
    "cbp",
    "hat",
    "adahat",
    "fgadahat",
    "wsn",
    "nispa",
]
