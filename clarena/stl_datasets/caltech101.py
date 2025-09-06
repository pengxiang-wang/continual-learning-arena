r"""
The submodule in `stl_datasets` for Caltech 101 dataset.
"""

__all__ = ["Caltech101"]

import logging
from typing import Callable

import torch
from torch.utils.data import Dataset, random_split
from torchvision.datasets import Caltech101 as Caltech101Raw
from torchvision.transforms import transforms

from clarena.stl_datasets.base import STLDatasetFromRaw

# always get logger for built-in logging in each module
pylogger = logging.getLogger(__name__)


class Caltech101(STLDatasetFromRaw):
    r"""Caltech 101 dataset. The [Caltech 101 dataset](https://data.caltech.edu/records/mzrjq-6wc02) is a collection of pictures of objects. It consists of 9,146 images of 101 classes, each color image."""

    original_dataset_python_class: type[Dataset] = Caltech101Raw
    r"""The original dataset class."""

    def __init__(
        self,
        root: str,
        test_percentage: float,
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
        - **root** (`str`): the root directory where the original Caltech data 'Caltech/' live.
        - **test_percentage** (`float`): the percentage to randomly split some data into test data.
        - **validation_percentage** (`float`): the percentage to randomly split some training data into validation data.
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

        self.test_percentage: float = test_percentage
        r"""The percentage to randomly split some data into test data."""
        self.validation_percentage: float = validation_percentage
        r"""The percentage to randomly split some training data into validation data."""

    def prepare_data(self) -> None:
        r"""Download the original Caltech 101 dataset if haven't."""

        Caltech101Raw(root=self.root, download=True)

        pylogger.debug(
            "The original Caltech 101 dataset has been downloaded to %s.",
            self.root,
        )

    def train_and_val_dataset(self) -> tuple[Dataset, Dataset]:
        """Get the training and validation dataset.

        **Returns:**
        - **train_and_val_dataset** (`tuple[Dataset, Dataset]`): the train and validation dataset.
        """
        dataset_all = Caltech101Raw(
            root=self.root,
            transform=self.train_and_val_transforms(),
            target_transform=self.target_transform(),
            download=False,
        )

        dataset_train_and_val, _ = random_split(
            dataset_all,
            lengths=[
                1 - self.test_percentage,
                self.test_percentage,
            ],
            generator=torch.Generator().manual_seed(
                42
            ),  # this must be set fixed to make sure the datasets across experiments are the same. Don't handle it to global seed as it might vary across experiments
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
        dataset_all = Caltech101Raw(
            root=self.root,
            transform=self.train_and_val_transforms(),
            target_transform=self.target_transform(),
            download=False,
        )

        _, dataset_test = random_split(
            dataset_all,
            lengths=[1 - self.test_percentage, self.test_percentage],
            generator=torch.Generator().manual_seed(42),
        )

        return dataset_test
