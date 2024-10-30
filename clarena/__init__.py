"""

# Welcome to CLArena

**CLArena (Continual Learning Arena)** is a open-source Python package for Continual Learning (CL) research, solely developed by myself. In this package, I provide a integrated environment and various APIs to conduct CL experiments for research purposes, as well as implemented CL algorithms and datasets that you can give it a spin immediately.

Please note that this is an API documantation providing detailed information about the available classes, functions, and modules in CLArena. Please refer to the main documentation and my beginners' guide to continual learning for more intuitive tutorials, examples, and guides on how to use CLArena.

- **Main documentation:** [https://pengxiang-wang.com/projects/continual-learning-arena/](https://pengxiang-wang.com/projects/continual-learning-arena/)
- **A beginners' guide to continual learning:** [https://pengxiang-wang.com/posts/continual-learning-beginners-guide](https://pengxiang-wang.com/posts/continual-learning-beginners-guide)

We provide various components of continual learning system in the submodules:

- `clarena.cl_datasets`: Continual learning datasets.
- `clarena.backbones`: Neural network architectures used as backbones for CL algorithms.
- `clarena.cl_heads`: Multi-head classifiers for continual learning outputs. Task-Incremental Learning (TIL) head and Class-Incremental Learning (CIL) head are included.
- `clarena.cl_algorithms`: Implementations of various continual learning algorithms.
- `clarena.optimizers`: Custom optimizers tailored for continual learning.
- `clarena.callbacks`: Extra functionality or action added in the continual learning process.

As well as the base class in the outmost directory of the package:

- Class `CLExperiment`: The base class for continual learning experiments.

"""

from .base import CLExperiment

__all__ = [
    "CLExperiment",
    "cl_datasets",
    "backbones",
    "cl_heads",
    "cl_algorithms",
    "optimizers",
    "callbacks",
]
