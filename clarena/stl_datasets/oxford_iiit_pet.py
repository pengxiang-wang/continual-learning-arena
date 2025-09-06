r"""
The submodule in `stl_datasets` for Oxford-IIIT Pet dataset.
"""

__all__ = ["OxfordIIITPet"]

import logging
from typing import Callable

import torch
from torch.utils.data import Dataset, random_split
from torchvision.datasets import OxfordIIITPet as OxfordIIITPetRaw
from torchvision.transforms import transforms

from clarena.stl_datasets.base import STLDatasetFromRaw
from clarena.stl_datasets.raw import OxfordIIITPet2, OxfordIIITPet37

# always get logger for built-in logging in each module
pylogger = logging.getLogger(__name__)


class OxfordIIITPet(STLDatasetFromRaw):
    r"""Permuted Oxford-IIIT Pet dataset. The [Oxford-IIIT Pet dataset](https://www.robots.ox.ac.uk/~vgg/data/pets/) is a collection of cat and dog pictures. It consists of 7,349 images of 37 breeds (classes), each color image. It also provides a binary classification version with 2 classes (cat or dog). We support both versions in Permuted Oxford-IIIT Pet."""

    def __init__(
        self,
        root: str,
        target_type: str,
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
        - **root** (`str`): the root directory where the original Oxford-IIIT Pet data 'OxfordIIITPet/' live.
        - **target_type** (`str`): the target type; one of:
            1. 'category': Label for one of the 37 pet categories.
            2. 'binary-category': Binary label for cat or dog.
        - **validation_percentage** (`float`): the percentage to randomly split some training data into validation data.
        - **batch_size** (`int`): The batch size in train, val, test dataloader.
        - **num_workers** (`int`): the number of workers for dataloaders.
        - **custom_transforms** (`transform` or `transforms.Compose` or `None`): the custom transforms to apply to ONLY TRAIN dataset. Can be a single transform, composed transforms or no transform. `ToTensor()`, normalize and so on are not included.
        - **repeat_channels** (`int` | `None`): the number of channels to repeat. Default is None, which means no repeat. If not None, it should be an integer.
        - **to_tensor** (`bool`): whether to include `ToTensor()` transform. Default is True.
        - **resize** (`tuple[int, int]` | `None` or list of them): the size to resize the images to. Default is None, which means no resize. If not None, it should be a tuple of two integers.
        """

        if target_type == "category":
            self.original_dataset_python_class: type[Dataset] = OxfordIIITPet37
        elif target_type == "binary-category":
            self.original_dataset_python_class: type[Dataset] = OxfordIIITPet2
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

        self.target_type: str = target_type
        r"""The target type. """

        self.validation_percentage: float = validation_percentage
        r"""The percentage to randomly split some training data into validation data."""

    def prepare_data(self) -> None:
        r"""Download the original Oxford-IIIT Pet dataset if haven't."""

        OxfordIIITPetRaw(
            root=self.root,
            split="trainval",
            target_types=self.target_type,
            download=True,
        )
        OxfordIIITPetRaw(
            root=self.root,
            split="test",
            target_types=self.target_type,
            download=True,
        )

        pylogger.debug(
            "The original Oxford-IIIT Pet dataset has been downloaded to %s.",
            self.root,
        )

    def train_and_val_dataset(self) -> tuple[Dataset, Dataset]:
        """Get the training and validation dataset.

        **Returns:**
        - **train_and_val_dataset** (`tuple[Dataset, Dataset]`): the train and validation dataset.
        """
        dataset_train_and_val = OxfordIIITPetRaw(
            root=self.root,
            split="trainval",
            target_types=self.target_type,
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
        dataset_test = OxfordIIITPetRaw(
            root=self.root,
            split="test",
            target_types=self.target_type,
            transform=self.test_transforms(),
            target_transform=self.target_transform(),
            download=False,
        )

        return dataset_test
