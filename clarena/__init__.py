r"""

# Continual Learning Arena (CLArena)

**CLArena (Continual Learning Arena)** is an open-source Python package designed for **Continual Learning (CL)** research. This package provides an integrated environment with extensive APIs for conducting CL experiments, along with pre-implemented algorithms and datasets that you can start using immediately. This package also supports **Continual Unlearning (CUL)**, **Multi-Task Learning (MTL)** and **Single-Task Learning (STL)**.

Please note this is the API reference providing detailed information about the available classes, functions, and modules in CLArena. Please refer to the main documentation for tutorials, examples, and guides on how to use CLArena:

- [**Main Documentation**](https://pengxiang-wang.com/projects/continual-learning-arena)
- [**A Beginners' Guide to Continual Learning**](https://pengxiang-wang.com/posts/continual-learning-beginners-guide)

We provide various components in the submodules:

- `clarena.pipelines`: Pre-defined experiment and evaluation pipelines for different paradigms.
- `clarena.cl_datasets`: Continual learning datasets.
- `clarena.mtl_datasets`: Multi-task learning datasets.
- `clarena.stl_datasets`: Single-task learning datasets.
- `clarena.backbones`: Neural network architectures used as backbones networks.
- `clarena.heads`: Output heads.
- `clarena.cl_algorithms`: Continual learning algorithms.
- `clarena.cul_algorithms`: Continual unlearning algorithms.
- `clarena.mtl_algorithms`: Multi-task learning algorithms.
- `clarena.stl_algorithms`: Single-task learning algorithms.
- `clarena.metrics`: Metrics for evaluation.
- `clarena.callbacks`: Extra actions added to the pipelines.
- `clarena.utils`: Utilities for the package.
"""

__all__ = [
    "pipelines",
    "cl_datasets",
    "mtl_datasets",
    "stl_datasets",
    "backbones",
    "heads",
    "cl_algorithms",
    "cul_algorithms",
    "mtl_algorithms",
    "stl_algorithms",
    "metrics",
    "callbacks",
    "utils",
]
