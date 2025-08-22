r"""

# Single-Task Learning Datasets

This submodule provides the **single-task learning datasets** that can be used in CLArena.

The datasets are implemented as subclasses of `STLDataset` classes, which are the base class for all single-task learning datasets in CLArena.

Please note that this is an API documantation. Please refer to the main documentation pages for more information about the STL datasets and how to configure and implement them:

- [**Configure STL Dataset**](https://pengxiang-wang.com/projects/continual-learning-arena/single-task-learning/configure-main-experiment/STL-dataset.qmd)
- [**Implement Your CL Dataset Class**](https://pengxiang-wang.com/projects/continual-lear ning-arena/docs/implement-your-cl-modules/cl-dataset)


"""

from .base import STLDataset, STLDatasetFromRaw

from .mnist import STLMNIST
