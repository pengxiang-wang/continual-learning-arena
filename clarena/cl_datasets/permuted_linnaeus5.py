r"""
The submodule in `cl_datasets` for Permuted Linnaeus 5 dataset.
"""

__all__ = ["PermutedLinnaeus5"]

import logging
from typing import Callable

import torch
from torch.utils.data import Dataset, random_split
from torchvision.transforms import transforms

from clarena.cl_datasets import CLPermutedDataset
from clarena.cl_datasets.original import (
    Linnaeus5,
    Linnaeus5_32,
    Linnaeus5_64,
    Linnaeus5_128,
    Linnaeus5_256,
)

# always get logger for built-in logging in each module
pylogger = logging.getLogger(__name__)


class PermutedLinnaeus5(CLPermutedDataset):
    r"""Permuted Linnaeus 5 dataset. The [original Linnaeus 5 dataset](https://chaladze.com/l5/) is a collection of flower images across 5 classes. It includes four versions with resized resolutions: 256x256, 128x128, 64x64, and 32x32."""

    def __init__(
        self,
        root: str,
        resolution: str,
        num_tasks: int,
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
        permutation_mode: str = "first_channel_only",
        permutation_seeds: list[int] | None = None,
    ) -> None:
        r"""Initialise the Permuted Linnaeus 5 dataset object providing the root where data files live.

        **Args:**
        - **root** (`str`): the root directory where the original Linnaeus 5 data 'Linnaeus5/' live.
        - **resolution** (`str`): Image resolution, one of ["256", "128", "64", "32"].
        - **num_tasks** (`int`): the maximum number of tasks supported by the CL dataset.
        - **validation_percentage** (`float`): the percentage to randomly split some of the training data into validation data.
        - **batch_size** (`int` | `list[int]`): The batch size in train, val, test dataloader. If `list[str]`, it should be a list of integers, each integer is the batch size for each task.
        - **num_workers** (`int` | `list[int]`): the number of workers for dataloaders. If `list[str]`, it should be a list of integers, each integer is the num of workers for each task.
        - **custom_transforms** (`transform` or `transforms.Compose` or `None` or list of them): the custom transforms to apply to ONLY TRAIN dataset. Can be a single transform, composed transforms or no transform. `ToTensor()`, normalise, permute and so on are not included. If it is a list, each item is the custom transforms for each task.
        - **repeat_channels** (`int` | `None` | list of them): the number of channels to repeat for each task. Default is None, which means no repeat. If not None, it should be an integer. If it is a list, each item is the number of channels to repeat for each task.
        - **to_tensor** (`bool` | `list[bool]`): whether to include `ToTensor()` transform. Default is True.
        - **resize** (`tuple[int, int]` | `None` or list of them): the size to resize the images to. Default is None, which means no resize. If not None, it should be a tuple of two integers. If it is a list, each item is the size to resize for each task.
        - **custom_target_transforms** (`transform` or `transforms.Compose` or `None` or list of them): the custom target transforms to apply to dataset labels. Can be a single transform, composed transforms or no transform. CL class mapping is not included. If it is a list, each item is the custom transforms for each task.
        - **permutation_mode** (`str`): the mode of permutation, should be one of the following:
            1. 'all': permute all pixels.
            2. 'by_channel': permute channel by channel separately. All channels are applied the same permutation order.
            3. 'first_channel_only': permute only the first channel.
        - **permutation_seeds** (`list[int]` or `None`): the seeds for permutation operations used to construct tasks. Make sure it has the same number of seeds as `num_tasks`. Default is None, which creates a list of seeds from 1 to `num_tasks`.
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

        CLPermutedDataset.__init__(
            self,
            root=root,
            num_tasks=num_tasks,
            batch_size=batch_size,
            num_workers=num_workers,
            custom_transforms=custom_transforms,
            repeat_channels=repeat_channels,
            to_tensor=to_tensor,
            resize=resize,
            custom_target_transforms=custom_target_transforms,
            permutation_mode=permutation_mode,
            permutation_seeds=permutation_seeds,
        )

        self.resolution: str = resolution
        r"""Store the resolution of the original dataset."""

        self.validation_percentage: float = validation_percentage
        r"""Store the percentage to randomly split some of the training data into validation data."""

    def prepare_data(self) -> None:
        r"""Download the original Linnaeus 5 dataset if haven't."""
        if self.task_id == 1:
            # just download the original dataset once
            Linnaeus5(
                root=self.root_t, resolution=self.resolution, train=True, download=True
            )
            Linnaeus5(
                root=self.root_t, resolution=self.resolution, train=False, download=True
            )

            pylogger.debug(
                "The original Linnaeus 5 dataset has been downloaded to %s.",
                self.root_t,
            )

    def train_and_val_dataset(self) -> tuple[Dataset, Dataset]:
        """Get the training and validation dataset of task `self.task_id`.

        **Returns:**
        - **train_and_val_dataset** (`tuple[Dataset, Dataset]`): the train and validation dataset of task `self.task_id`.
        """
        dataset_train_and_val = Linnaeus5(
            root=self.root_t,
            resolution=self.resolution,
            train=True,
            transform=self.train_and_val_transforms(),
            download=False,
        )
        dataset_train_and_val.target_transform = self.target_transforms()

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
        dataset_test = Linnaeus5(
            root=self.root_t,
            resolution=self.resolution,
            train=False,
            transform=self.test_transforms(),
            download=False,
        )
        dataset_test.target_transform = self.target_transforms()

        return dataset_test
