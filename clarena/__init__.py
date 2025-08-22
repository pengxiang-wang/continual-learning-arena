r"""

# Welcome to CLArena

**CLArena (Continual Learning Arena)** is a open-source Python package for Continual Learning (CL) research. In this package, we provide a integrated environment and various APIs to conduct CL experiments for research purposes, as well as implemented CL algorithms and datasets that you can give it a spin immediately. We also developed an environment for Continual Unlearning (CUL), where you can conduct CUL experiments and use various unlearning algorithms.

Please note that this is an API documantation providing detailed information about the available classes, functions, and modules in CLArena. Please refer to the main documentation and my beginners' guide to continual learning for more intuitive tutorials, examples, and guides on how to use CLArena:

- [**Main Documentation**](https://pengxiang-wang.com/projects/continual-learning-arena)
- [**A Beginners' Guide to Continual Learning**](https://pengxiang-wang.com/posts/continual-learning-beginners-guide)

We provide various components of continual learning and unlearning system in the submodules:

- `clarena.experiments`: Continual learning and unlearning experiment objects, as well as other reference experiments.
- `clarena.cl_datasets`: Continual learning datasets.
- `clarena.backbones`: Neural network architectures used as backbones for CL algorithms.
- `clarena.heads`: Multi-head classifiers for outputs. Task-Incremental Learning (TIL) head, Class-Incremental Learning (CIL) head, and Multi-Task Learning (MTL) head are included.
- `clarena.cl_algorithms`: Implementation of various continual learning algorithms.
- `clarena.cul_algorithms`: Implementation of various unlearning algorithms on top of continual learning.
- `clarena.callbacks`: Extra actions added to the continual learning process.
- `clarena.utils`: Utility functions for continual learning experiments.
"""

__all__ = [
    "experiments",
    "cl_datasets",
    "mtl_datasets",
    "stl_datasets",
    "backbones",
    "heads",
    "unlearning_algorithms",
    "cl_algorithms",
    "mtl_algorithms",
    "stl_algorithms",
    "metrics",
    "callbacks",
    "utils",
]
