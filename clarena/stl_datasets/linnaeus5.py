r"""
The submodule in `stl_datasets` for Linnaeus 5 dataset.
"""

__all__ = ["Linnaeus5"]

import logging
from typing import Callable

import torch
from torch.utils.data import Dataset, random_split
from torchvision.transforms import transforms

from clarena.stl_datasets.base import STLDatasetFromRaw
from clarena.stl_datasets.raw import Linnaeus5 as Linnaeus5Raw
from clarena.stl_datasets.raw import (
    Linnaeus5_32,
    Linnaeus5_64,
    Linnaeus5_128,
    Linnaeus5_256,
)

# always get logger for built-in logging in each module
pylogger = logging.getLogger(__name__)


class Linnaeus5(STLDatasetFromRaw):
    r"""Linnaeus 5 dataset. The [Linnaeus 5 dataset](https://chaladze.com/l5/) is a collection of flower images. It consists of 8,000 images of 5 flower species (classes). It provides 256x256, 128x128, 64x64, and 32x32 color images. We support all of them in Permuted Linnaeus 5."""

    def __init__(
        self,
        root: str,
        resolution: str,
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
        - **root** (`str`): the root directory where the original Linnaeus 5 data 'Linnaeus5/' live.
        - **resolution** (`str`): Image resolution, one of ["256", "128", "64", "32"].
        - **validation_percentage** (`float`): the percentage to randomly split some training data into validation data.
        - **batch_size** (`int`): The batch size in train, val, test dataloader.
        - **num_workers** (`int`): the number of workers for dataloaders.
        - **custom_transforms** (`transform` or `transforms.Compose` or `None`): the custom transforms to apply to ONLY TRAIN dataset. Can be a single transform, composed transforms or no transform. `ToTensor()`, normalize and so on are not included.
        - **repeat_channels** (`int` | `None`): the number of channels to repeat. Default is None, which means no repeat. If not None, it should be an integer.
        - **to_tensor** (`bool`): whether to include `ToTensor()` transform. Default is True.
        - **resize** (`tuple[int, int]` | `None` or list of them): the size to resize the images to. Default is None, which means no resize. If not None, it should be a tuple of two integers.
        """

        if resolution == "32":
            self.original_dataset_python_class: type[Dataset] = Linnaeus5_32
        elif resolution == "64":
            self.original_dataset_python_class: type[Dataset] = Linnaeus5_64
        elif resolution == "128":
            self.original_dataset_python_class: type[Dataset] = Linnaeus5_128
        elif resolution == "256":
            self.original_dataset_python_class: type[Dataset] = Linnaeus5_256
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

        self.resolution: str = resolution
        r"""Store the resolution of the original dataset."""

        self.validation_percentage: float = validation_percentage
        r"""The percentage to randomly split some training data into validation data."""

    def prepare_data(self) -> None:
        r"""Download the original Linnaeus 5 dataset if haven't."""

        Linnaeus5Raw(
            root=self.root, resolution=self.resolution, train=True, download=True
        )
        Linnaeus5Raw(
            root=self.root, resolution=self.resolution, train=False, download=True
        )

        pylogger.debug(
            "The original Linnaeus 5 dataset has been downloaded to %s.",
            self.root,
        )

    def train_and_val_dataset(self) -> tuple[Dataset, Dataset]:
        """Get the training and validation dataset.

        **Returns:**
        - **train_and_val_dataset** (`tuple[Dataset, Dataset]`): the train and validation dataset.
        """
        dataset_train_and_val = Linnaeus5Raw(
            root=self.root,
            resolution=self.resolution,
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
        dataset_test = Linnaeus5Raw(
            root=self.root,
            resolution=self.resolution,
            train=False,
            transform=self.test_transforms(),
            target_transform=self.target_transform(),
            download=False,
        )

        return dataset_test
