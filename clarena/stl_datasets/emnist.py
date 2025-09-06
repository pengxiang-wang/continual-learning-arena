r"""
The submodule in `stl_datasets` for EMNIST dataset.
"""

__all__ = ["EMNIST"]

import logging
from typing import Callable

import torch
from torch.utils.data import Dataset, random_split
from torchvision.datasets import EMNIST as EMNISTRaw
from torchvision.transforms import transforms

from clarena.stl_datasets.base import STLDatasetFromRaw
from clarena.stl_datasets.raw import (
    EMNISTBalanced,
    EMNISTByClass,
    EMNISTByMerge,
    EMNISTDigits,
    EMNISTLetters,
)

# always get logger for built-in logging in each module
pylogger = logging.getLogger(__name__)


class EMNIST(STLDatasetFromRaw):
    r"""EMNIST dataset. The [EMNIST dataset](https://www.nist.gov/itl/products-and-services/emnist-dataset/) is a collection of handwritten letters and digits (including A-Z, a-z, 0-9). It consists of 814,255 images in 62 classes, each 28x28 grayscale image.

    EMNIST has 6 different splits: `byclass`, `bymerge`, `balanced`, `letters`, `digits` and `mnist`, each containing a different subset of the original collection. We support all of them in Permuted EMNIST.
    """

    def __init__(
        self,
        root: str,
        split: str,
        validation_percentage: float,
        batch_size: int = 1,
        num_workers: int = 0,
        custom_transforms: Callable | transforms.Compose | None = None,
        repeat_channels: int | None = None,
        to_tensor: bool = True,
        resize: tuple[int, int] | None = None,
    ) -> None:
        r"""
        **Args:**
        - **root** (`str`): the root directory where the original EMNIST data 'EMNIST/' live.
        - **split** (`str`): the original EMNIST dataset has 6 different splits: `byclass`, `bymerge`, `balanced`, `letters`, `digits` and `mnist`. This argument specifies which one to use.
        - **validation_percentage** (`float`): the percentage to randomly split some training data into validation data.
        - **batch_size** (`int`): The batch size in train, val, test dataloader.
        - **num_workers** (`int`): the number of workers for dataloaders.
        - **custom_transforms** (`transform` or `transforms.Compose` or `None`): the custom transforms to apply to ONLY TRAIN dataset. Can be a single transform, composed transforms or no transform. `ToTensor()`, normalize and so on are not included.
        - **repeat_channels** (`int` | `None`): the number of channels to repeat. Default is None, which means no repeat. If not None, it should be an integer.
        - **to_tensor** (`bool`): whether to include `ToTensor()` transform. Default is True.
        - **resize** (`tuple[int, int]` | `None` or list of them): the size to resize the images to. Default is None, which means no resize. If not None, it should be a tuple of two integers.
        """
        if split == "byclass":
            self.original_dataset_python_class: type[Dataset] = EMNISTByClass
        elif split == "bymerge":
            self.original_dataset_python_class: type[Dataset] = EMNISTByMerge
        elif split == "balanced":
            self.original_dataset_python_class: type[Dataset] = EMNISTBalanced
        elif split == "letters":
            self.original_dataset_python_class: type[Dataset] = EMNISTLetters
        elif split == "digits":
            self.original_dataset_python_class: type[Dataset] = EMNISTDigits
            r"""The original dataset class."""

        super().__init__(
            root=root,
            batch_size=batch_size,
            num_workers=num_workers,
            custom_transforms=custom_transforms,
            repeat_channels=repeat_channels,
            to_tensor=to_tensor,
            resize=resize,
        )

        self.split: str = split
        r"""The split of the original EMNIST dataset. It can be `byclass`, `bymerge`, `balanced`, `letters`, `digits` or `mnist`."""

        self.validation_percentage: float = validation_percentage
        r"""The percentage to randomly split some training data into validation data."""

    def prepare_data(self) -> None:
        r"""Download the original EMNIST dataset if haven't."""

        EMNISTRaw(root=self.root, split=self.split, train=True, download=True)
        EMNISTRaw(root=self.root, split=self.split, train=False, download=True)

        pylogger.debug(
            "The original EMNIST dataset has been downloaded to %s.", self.root
        )

    def train_and_val_dataset(self) -> tuple[Dataset, Dataset]:
        """Get the training and validation dataset.

        **Returns:**
        - **train_and_val_dataset** (`tuple[Dataset, Dataset]`): the train and validation dataset.
        """
        dataset_train_and_val = EMNISTRaw(
            root=self.root,
            split=self.split,
            train=True,
            transform=self.train_and_val_transforms(),
            target_transform=self.target_transform(),
            download=False,
        )

        return random_split(
            dataset_train_and_val,
            lengths=[1 - self.validation_percentage, self.validation_percentage],
            generator=torch.Generator().manual_seed(
                42
            ),  # this must be set fixed to make sure the datasets across experiments are the same. Don't handle it to global seed as it might vary across experiments
        )

    def test_dataset(self) -> Dataset:
        r"""Get the test dataset.

        **Returns:**
        - **test_dataset** (`Dataset`): the test dataset.
        """
        dataset_test = EMNISTRaw(
            root=self.root,
            split=self.split,
            train=False,
            transform=self.test_transforms(),
            target_transform=self.target_transform(),
            download=False,
        )

        return dataset_test
