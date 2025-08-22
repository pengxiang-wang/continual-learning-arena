r"""
The submodule in `stl_datasets` for MNIST dataset.
"""

__all__ = ["STLMNIST"]

import logging
from typing import Callable

import torch
from torch.utils.data import Dataset, random_split
from torchvision.datasets import MNIST
from torchvision.transforms import transforms

from clarena.stl_datasets.base import STLDatasetFromRaw
from clarena.utils.transforms import ClassMapping

# always get logger for built-in logging in each module
pylogger = logging.getLogger(__name__)


class STLMNIST(STLDatasetFromRaw):
    r"""MNIST dataset. The [MNIST dataset](http://yann.lecun.com/exdb/mnist/) is a collection of handwritten digits. It consists of 60,000 training and 10,000 test images of handwritten digit images (10 classes), each 28x28 grayscale image."""

    original_dataset_python_class: type[Dataset] = MNIST
    r"""The original dataset class."""

    def __init__(
        self,
        root: str,
        validation_percentage: float,
        batch_size: int = 1,
        num_workers: int = 0,
        custom_transforms: Callable | transforms.Compose | None = None,
        repeat_channels: int | None = None,
        to_tensor: bool = True,
        resize: tuple[int, int] | None = None,
    ) -> None:
        r"""Initialize the dataset object providing the root where data files live.

        **Args:**
        - **root** (`str`): the root directory where the original MNIST data 'MNIST/' live.
        - **validation_percentage** (`float`): the percentage to randomly split some training data into validation data.
        - **batch_size** (`int`): The batch size in train, val, test dataloader.
        - **num_workers** (`int`): the number of workers for dataloaders.
        - **custom_transforms** (`transform` or `transforms.Compose` or `None`): the custom transforms to apply to ONLY TRAIN dataset. Can be a single transform, composed transforms or no transform. `ToTensor()`, normalize, permute and so on are not included.
        - **repeat_channels** (`int` | `None`): the number of channels to repeat for each task. Default is None, which means no repeat. If not None, it should be an integer.
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

        self.validation_percentage: float = validation_percentage
        """The percentage to randomly split some training data into validation data."""

    def prepare_data(self) -> None:
        r"""Download the original MNIST dataset if haven't."""

        MNIST(root=self.root, train=True, download=True)
        MNIST(root=self.root, train=False, download=True)

        pylogger.debug(
            "The original MNIST dataset has been downloaded to %s.", self.root
        )

    def train_and_val_dataset(self) -> tuple[Dataset, Dataset]:
        """Get the training and validation dataset.

        **Returns:**
        - **train_and_val_dataset** (`tuple[Dataset, Dataset]`): the train and validation dataset.
        """
        dataset_train_and_val = MNIST(
            root=self.root,
            train=True,
            transform=self.train_and_val_transforms(),
            target_transform=ClassMapping(self.get_class_map()),
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
        dataset_test = MNIST(
            root=self.root,
            train=False,
            transform=self.test_transforms(),
            target_transform=ClassMapping(self.get_class_map()),
            download=False,
        )

        return dataset_test
