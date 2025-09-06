r"""
The submodule in `stl_datasets` for Rendered SST2 dataset.
"""

__all__ = ["RenderedSST2"]

import logging
from typing import Callable

import torch
from torch.utils.data import Dataset, random_split
from torchvision.datasets import RenderedSST2 as RenderedSST2Raw
from torchvision.transforms import transforms

from clarena.stl_datasets.base import STLDatasetFromRaw

# always get logger for built-in logging in each module
pylogger = logging.getLogger(__name__)


class RenderedSST2(STLDatasetFromRaw):
    r"""Rendered SST2 dataset. The [Rendered SST2 dataset](https://github.com/openai/CLIP/blob/main/data/rendered-sst2.md) is a collection of optical character recognition images. It consists of 9,613 images in 2 classes (positive and negative sentiment), each 448x448 color image."""

    original_dataset_python_class: type[Dataset] = RenderedSST2Raw
    r"""The original dataset class."""

    def __init__(
        self,
        root: str,
        batch_size: int = 1,
        num_workers: int = 0,
        custom_transforms: Callable | transforms.Compose | None = None,
        repeat_channels: int | None = None,
        to_tensor: bool = True,
        resize: tuple[int, int] | None = None,
    ) -> None:
        r"""
        **Args:**
        - **root** (`str`): the root directory where the original Rendered SST2 data 'RenderedSST2/' live.
        - **batch_size** (`int`): The batch size in train, val, test dataloader.
        - **num_workers** (`int`): the number of workers for dataloaders.
        - **custom_transforms** (`transform` or `transforms.Compose` or `None`): the custom transforms to apply to ONLY TRAIN dataset. Can be a single transform, composed transforms or no transform. `ToTensor()`, normalize and so on are not included.
        - **repeat_channels** (`int` | `None`): the number of channels to repeat. Default is None, which means no repeat. If not None, it should be an integer.
        - **to_tensor** (`bool`): whether to include `ToTensor()` transform. Default is True.
        - **resize** (`tuple[int, int]` | `None` or list of them): the size to resize the images to. Default is None, which means no resize. If not None, it should be a tuple of two integers.
        """
        super().__init__(
            root=root,
            batch_size=batch_size,
            num_workers=num_workers,
            custom_transforms=custom_transforms,
            repeat_channels=repeat_channels,
            to_tensor=to_tensor,
            resize=resize,
        )

    def prepare_data(self) -> None:
        r"""Download the original Rendered SST2 dataset if haven't."""

        RenderedSST2Raw(root=self.root, split="train", download=True)
        RenderedSST2Raw(root=self.root, split="val", download=True)
        RenderedSST2Raw(root=self.root, split="test", download=True)

        pylogger.debug(
            "The original Rendered SST2 dataset has been downloaded to %s.",
            self.root,
        )

    def train_and_val_dataset(self) -> tuple[Dataset, Dataset]:
        """Get the training and validation dataset.

        **Returns:**
        - **train_and_val_dataset** (`tuple[Dataset, Dataset]`): the train and validation dataset.
        """
        dataset_train = RenderedSST2Raw(
            root=self.root,
            split="train",
            transform=self.train_and_val_transforms(),
            target_transform=self.target_transform(),
            download=False,
        )

        dataset_val = RenderedSST2Raw(
            root=self.root,
            split="val",
            transform=self.train_and_val_transforms(),
            target_transform=self.target_transform(),
            download=False,
        )

        return dataset_train, dataset_val

    def test_dataset(self) -> Dataset:
        r"""Get the test dataset.

        **Returns:**
        - **test_dataset** (`Dataset`): the test dataset.
        """
        dataset_test = RenderedSST2Raw(
            root=self.root,
            split="test",
            transform=self.test_transforms(),
            target_transform=self.target_transform(),
            download=False,
        )

        return dataset_test
