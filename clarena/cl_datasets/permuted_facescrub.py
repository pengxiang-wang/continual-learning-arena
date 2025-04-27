r"""
The submodule in `cl_datasets` for Permuted FaceScrub dataset.
"""

__all__ = ["PermutedFaceScrub"]

import logging
from typing import Callable

import torch
from torch.utils.data import Dataset, random_split
from torchvision.transforms import transforms

from clarena.cl_datasets import CLPermutedDataset
from clarena.cl_datasets.original import (
    FaceScrub10,
    FaceScrub20,
    FaceScrub50,
    FaceScrub100,
)

# always get logger for built-in logging in each module
pylogger = logging.getLogger(__name__)


class PermutedFaceScrub(CLPermutedDataset):
    r"""Permuted FaceScrub dataset. The [original FaceScrub dataset](https://vintage.winklerbros.net/facescrub.html) is a collection of 106,863 images of 530 different people. It consists of 530 classes (correspond to 530 people), with 200-300 images per class. [This version](https://github.com/nkundiushuti/facescrub_subset) uses subset of the official [Megaface Facescrub challenge](http://megaface.cs.washington.edu/participate/challenge.html), cropped and resized to 38x38 pixels. We have FaceScrub-10, FaceScrub-20, FaceScrub-50, FaceScrub-100 datasets where the number of classes are 10, 20, 50 and 100 respectively."""

    def __init__(
        self,
        root: str,
        size: str,
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
        r"""Initialise the Permuted FaceScrub dataset object providing the root where data files live.

        **Args:**
        - **root** (`str`): the root directory where the original FaceScrub data 'FaceScrub/' live.
        - **size** (`str`): the size of the dataset, should be one of the following:
            1. '10': 10 classes (10 people).
            2. '20': 20 classes (20 people).
            3. '50': 50 classes (50 people).
            4. '100': 100 classes (100 people).
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

        if size == "10":
            self.original_dataset_python_class: type[Dataset] = FaceScrub10
        elif size == "20":
            self.original_dataset_python_class: type[Dataset] = FaceScrub20
        elif size == "50":
            self.original_dataset_python_class: type[Dataset] = FaceScrub50
        elif size == "100":
            self.original_dataset_python_class: type[Dataset] = FaceScrub100
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

        self.validation_percentage: float = validation_percentage
        """Store the percentage to randomly split some of the training data into validation data."""

    def prepare_data(self) -> None:
        r"""Download the original FaceScrub dataset if haven't."""
        if self.task_id == 1:
            # just download the original dataset once
            self.original_dataset_python_class(
                root=self.root_t, train=True, download=True
            )
            self.original_dataset_python_class(
                root=self.root_t, train=False, download=True
            )

            pylogger.debug(
                "The original FaceScrub dataset has been downloaded to %s.", self.root_t
            )

    def train_and_val_dataset(self) -> tuple[Dataset, Dataset]:
        """Get the training and validation dataset of task `self.task_id`.

        **Returns:**
        - **train_and_val_dataset** (`tuple[Dataset, Dataset]`): the train and validation dataset of task `self.task_id`.
        """
        dataset_train_and_val = self.original_dataset_python_class(
            root=self.root_t,
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
        dataset_test = self.original_dataset_python_class(
            root=self.root_t,
            train=False,
            transform=self.test_transforms(),
            download=False,
        )

        dataset_test.target_transform = self.target_transforms()

        return dataset_test
