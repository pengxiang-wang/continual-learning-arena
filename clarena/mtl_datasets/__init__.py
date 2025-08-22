r"""

# Continual Learning Datasets

This submodule provides the **continual learning datasets** that can be used in CLArena.

The datasets are implemented as subclasses of `CLDataset` classes, which are the base class for all continual learning datasets in CLArena.

- `CLDataset`: The base class for continual learning datasets.
- `CLPermutedDataset`: The base class for permuted continual learning datasets. A child class of `CLDataset`.
- `CLSplitDataset`: The base class for split continual learning datasets. A child class of `CLDataset`.
- `CLCombinedDataset`: The base class for combined continual learning datasets. A child class of `CLDataset`.

Please note that this is an API documantation. Please refer to the main documentation pages for more information about the CL datasets and how to configure and implement them:

- [**Configure CL Dataset**](https://pengxiang-wang.com/projects/continual-learning-arena/docs/configure-your-experiment/cl-dataset)
- [**Implement Your CL Dataset Class**](https://pengxiang-wang.com/projects/continual-learning-arena/docs/implement-your-cl-modules/cl-dataset)
- [**A Beginners' Guide to Continual Learning (CL Dataset)**](https://pengxiang-wang.com/posts/continual-learning-beginners-guide#sec-CL-dataset)



"""

from .base import MTLDataset, MTLDatasetFromRaw, MTLDatasetFromCL

from .multi_domain_sentiment import MultiDomainSentiment
