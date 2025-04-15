r"""
The submodule in `cl_datasets` for conbined torchvision datasets.
"""

__all__ = ["CombinedTorchvision"]

import logging
from typing import Callable

import torch
from torch.utils.data import Dataset, random_split
from torchvision import transforms

from clarena.cl_datasets import CLCombinedDataset

# always get logger for built-in logging in each module
pylogger = logging.getLogger(__name__)


class CombinedTorchvision(CLCombinedDataset):
    r"""The base class of continual learning datasets which are constructed as combinations of several original datasets (one dataset for one task) from [torchvision](https://pytorch.org/vision/0.8/datasets.html), inherited from `CLCombinedDataset`."""

    def __init__(
        self,
        datasets: list[str],
        root: list[str],
        num_classes: list[int],
        validation_percentage: float,
        batch_size: int | list[int] = 1,
        num_workers: int | list[int] = 0,
        custom_transforms: (
            Callable
            | transforms.Compose
            | None
            | list[Callable | transforms.Compose | None]
        ) = None,
        repeat_channels: int | None | list[int | None] = None,
        to_tensor: bool | list[bool] = True,
        resize: tuple[int, int] | None | list[tuple[int, int] | None] = None,
        custom_target_transforms: (
            Callable
            | transforms.Compose
            | None
            | list[Callable | transforms.Compose | None]
        ) = None,
    ) -> None:
        r"""Initialise the Combined Torchvision dataset object providing the root where data files live.

        **Args:**
        - **datasets** (`list[str]`): the list of dataset class paths for each task. Each element in the list must be a string referring to a valid PyTorch Dataset class.
        - **root** (`list[str]`): the list of root directory where the original data files for constructing the CL dataset physically live.
        - **num_classes** (`list[int]`): the list of number of classes for each task. Each element in the list is an integer.
        - **validation_percentage** (`float`): the percentage to randomly split some of the training data into validation data.
        - **batch_size** (`int` | `list[int]`): The batch size in train, val, test dataloader. If `list[str]`, it should be a list of integers, each integer is the batch size for each task.
        - **num_workers** (`int` | `list[int]`): the number of workers for dataloaders. If `list[str]`, it should be a list of integers, each integer is the num of workers for each task.
        - **custom_transforms** (`transform` or `transforms.Compose` or `None` or list of them): the custom transforms to apply to ONLY TRAIN dataset. Can be a single transform, composed transforms or no transform. `ToTensor()`, normalise, permute and so on are not included. If it is a list, each item is the custom transforms for each task.
        - **repeat_channels** (`int` | `None` | list of them): the number of channels to repeat for each task. Default is None, which means no repeat. If not None, it should be an integer. If it is a list, each item is the number of channels to repeat for each task.
        - **to_tensor** (`bool` | `list[bool]`): whether to include `ToTensor()` transform. Default is True.
        - **resize** (`tuple[int, int]` | `None` or list of them): the size to resize the images to. Default is None, which means no resize. If not None, it should be a tuple of two integers. If it is a list, each item is the size to resize for each task.
        - **custom_target_transforms** (`transform` or `transforms.Compose` or `None` or list of them): the custom target transforms to apply to dataset labels. Can be a single transform, composed transforms or no transform. CL class mapping is not included. If it is a list, each item is the custom transforms for each task.
        """
        CLCombinedDataset.__init__(
            self,
            datasets=datasets,
            root=root,
            num_classes=num_classes,
            batch_size=batch_size,
            num_workers=num_workers,
            custom_transforms=custom_transforms,
            repeat_channels=repeat_channels,
            to_tensor=to_tensor,
            resize=resize,
            custom_target_transforms=custom_target_transforms,
        )

        self.validation_percentage: float = validation_percentage
        """Store the percentage to randomly split some of the training data into validation data."""

    def prepare_data(self) -> None:
        r"""Download the original datasets if haven't."""
        # torchvision datasets have same APIs
        self.original_dataset_python_class_t(
            root=self.root_t, train=True, download=True
        )
        self.original_dataset_python_class_t(
            root=self.root_t, train=False, download=True
        )

        pylogger.debug(
            "The original %s dataset has been downloaded to %s.",
            self.original_dataset_python_class_t,
            self.root_t,
        )

    def train_and_val_dataset(self) -> tuple[Dataset, Dataset]:
        """Get the training and validation dataset of task `self.task_id`.

        **Returns:**
        - **train_and_val_dataset** (`tuple[Dataset, Dataset]`): the train and validation dataset of task `self.task_id`.
        """

        dataset_train_and_val = self.original_dataset_python_class_t(
            root=self.root_t,
            train=True,
            transform=self.train_and_val_transforms(),
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
        r"""Get the test dataset of task `self.task_id`.

        **Returns:**
        - **test_dataset** (`Dataset`): the test dataset of task `self.task_id`.
        """

        return self.original_dataset_python_class_t(
            root=self.root_t,
            train=False,
            transform=self.test_transforms(),
            download=False,
        )
