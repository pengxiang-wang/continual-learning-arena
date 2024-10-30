"""

# Continual Learning Datasets

This submodule provides the **continual learning datasets** that can be used in CLArena. 

Please note that this is an API documantation. Please refer to the main documentation page for more information about the CL datasets and how to use and customize them:

- **Configure your CL dataset:** [https://pengxiang-wang.com/projects/continual-learning-arena/docs/configure-your-experiments/cl-dataset](https://pengxiang-wang.com/projects/continual-learning-arena/docs/configure-your-experiments/cl-dataset)
- **Implement your CL dataset:** [https://pengxiang-wang.com/projects/continual-learning-arena/docs/implement-your-cl-modules/cl-dataset](https://pengxiang-wang.com/projects/continual-learning-arena/docs/implement-your-cl-modules/cl-dataset)
- **A beginners' guide to continual learning (CL dataset):** [https://pengxiang-wang.com/posts/continual-learning-beginners-guide#CL-dataset](https://pengxiang-wang.com/posts/continual-learning-beginners-guide#CL-dataset)

The datasets are implemented as subclasses of `CLDataset` classes, which are the base class for all continual learning datasets in CLArena.

- `CLDataset`: The base class for continual learning datasets.
- `CLPermutedDataset`: The base class for permuted continual learning datasets. A child class of `CLDataset`.

"""

from .base import CLClassMapping, CLDataset, CLPermutedDataset, Permute
from .permuted_mnist import PermutedMNIST

__all__ = [
    "CLDataset",
    "CLPermutedDataset",
    "CLClassMapping",
    "Permute",
    "permuted_mnist",
]
