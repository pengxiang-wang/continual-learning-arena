r"""

# Multi-Task Learning Datasets

This submodule provides the **multi-task learning datasets** that can be used in CLArena.

Here are the base classes for multi-task learning datasets, which inherit from Lightning `LightningDataModule`:

- `MTLDataset`: The base class for all multi-task learning datasets.
    - `MTLCombinedDataset`: The base class for combined multi-task learning datasets. A child class of `MTLDataset`.
    - `MTLDatasetFromCL`: The base class for constructing multi-task learning datasets from continual learning datasets. A child class of `MTLDataset`.

Please note that this is an API documantation. Please refer to the main documentation pages for more information about how to configure and implement MTL datasets:

- [**Configure MTL Dataset**](https://pengxiang-wang.com/projects/continual-learning-arena/docs/components/mtl-dataset)
- [**Implement Custom MTL Dataset**](https://pengxiang-wang.com/projects/continual-learning-arena/docs/custom-implementation/mtl_dataset)



"""

from .base import MTLDataset, MTLCombinedDataset, MTLDatasetFromCL

from .combined import Combined


__all__ = ["MTLDataset", "MTLCombinedDataset", "MTLDatasetFromCL", "combined"]
