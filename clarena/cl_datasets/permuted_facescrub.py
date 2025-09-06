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
from clarena.stl_datasets.raw import FaceScrub10, FaceScrub20, FaceScrub50, FaceScrub100

# always get logger for built-in logging in each module
pylogger = logging.getLogger(__name__)


class PermutedFaceScrub(CLPermutedDataset):
    r"""Permuted FaceScrub dataset. The [original FaceScrub dataset](https://vintage.winklerbros.net/facescrub.html) is a collection of human face images. It consists 106,863 images of 530 people (classes), each high resolution color image.

    To make it simple, [this version](https://github.com/nkundiushuti/facescrub_subset) uses subset of the official [Megaface FaceScrub challenge](http://megaface.cs.washington.edu/participate/challenge.html), cropped and resized to 32x32. We have [FaceScrub-10](https://github.com/nkundiushuti/facescrub_subset/blob/master/data/facescrub_10.zip), [FaceScrub-20](https://github.com/nkundiushuti/facescrub_subset/blob/master/data/facescrub_20.zip), [FaceScrub-50](https://github.com/nkundiushuti/facescrub_subset/blob/master/data/facescrub_50.zip), [FaceScrub-100](https://github.com/nkundiushuti/facescrub_subset/blob/master/data/facescrub_100.zip) datasets where the number of classes are 10, 20, 50 and 100 respectively.
    """

    def __init__(
        self,
        root: str,
        size: str,
        num_tasks: int,
        validation_percentage: float,
        batch_size: int | dict[int, int] = 1,
        num_workers: int | dict[int, int] = 0,
        custom_transforms: (
            Callable
            | transforms.Compose
            | None
            | dict[int, Callable | transforms.Compose | None]
        ) = None,
        repeat_channels: int | None | dict[int, int | None] = None,
        to_tensor: bool | dict[int, bool] = True,
        resize: tuple[int, int] | None | dict[int, tuple[int, int] | None] = None,
        permutation_mode: str = "first_channel_only",
        permutation_seeds: dict[int, int] | None = None,
    ) -> None:
        r"""
        **Args:**
        - **root** (`str`): the root directory where the original FaceScrub data 'FaceScrub/' live.
        - **size** (`str`): the size of the dataset; one of:
            1. '10': 10 classes (10 people).
            2. '20': 20 classes (20 people).
            3. '50': 50 classes (50 people).
            4. '100': 100 classes (100 people).
        - **num_tasks** (`int`): the maximum number of tasks supported by the CL dataset. This decides the valid task IDs from 1 to `num_tasks`.
        - **validation_percentage** (`float`): the percentage to randomly split some training data into validation data.
        - **batch_size** (`int` | `dict[int, int]`): the batch size for train, val, and test dataloaders.
        If it is a dict, the keys are task IDs and the values are the batch sizes for each task. If it is an `int`, it is the same batch size for all tasks.
        - **num_workers** (`int` | `dict[int, int]`): the number of workers for dataloaders.
        If it is a dict, the keys are task IDs and the values are the number of workers for each task. If it is an `int`, it is the same number of workers for all tasks.
        - **custom_transforms** (`transform` or `transforms.Compose` or `None` or dict of them): the custom transforms to apply ONLY to the TRAIN dataset. Can be a single transform, composed transforms, or no transform. `ToTensor()`, normalization, permute, and so on are not included.
        If it is a dict, the keys are task IDs and the values are the custom transforms for each task. If it is a single transform or composed transforms, it is applied to all tasks. If it is `None`, no custom transforms are applied.
        - **repeat_channels** (`int` | `None` | dict of them): the number of channels to repeat for each task. Default is `None`, which means no repeat.
        If it is a dict, the keys are task IDs and the values are the number of channels to repeat for each task. If it is an `int`, it is the same number of channels to repeat for all tasks. If it is `None`, no repeat is applied.
        - **to_tensor** (`bool` | `dict[int, bool]`): whether to include the `ToTensor()` transform. Default is `True`.
        If it is a dict, the keys are task IDs and the values are whether to include the `ToTensor()` transform for each task. If it is a single boolean value, it is applied to all tasks.
        - **resize** (`tuple[int, int]` | `None` or dict of them): the size to resize the images to. Default is `None`, which means no resize.
        If it is a dict, the keys are task IDs and the values are the sizes to resize for each task. If it is a single tuple of two integers, it is applied to all tasks. If it is `None`, no resize is applied.
        - **permutation_mode** (`str`): the mode of permutation; one of:
            1. 'all': permute all pixels.
            2. 'by_channel': permute channel by channel separately. All channels are applied the same permutation order.
            3. 'first_channel_only': permute only the first channel.
        - **permutation_seeds** (`dict[int, int]` | `None`): the dict of seeds for permutation operations used to construct each task. Keys are task IDs and the values are permutation seeds for each task. Default is `None`, which creates a dict of seeds from 0 to `num_tasks`-1.
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

        super().__init__(
            root=root,
            num_tasks=num_tasks,
            batch_size=batch_size,
            num_workers=num_workers,
            custom_transforms=custom_transforms,
            repeat_channels=repeat_channels,
            to_tensor=to_tensor,
            resize=resize,
            permutation_mode=permutation_mode,
            permutation_seeds=permutation_seeds,
        )

        self.validation_percentage: float = validation_percentage
        r"""The percentage to randomly split some training data into validation data."""

    def prepare_data(self) -> None:
        r"""Download the original FaceScrub dataset if haven't."""

        if self.task_id != 1:
            return  # download all original datasets only at the beginning of first task

        self.original_dataset_python_class(root=self.root_t, train=True, download=True)
        self.original_dataset_python_class(root=self.root_t, train=False, download=True)

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
        r"""Get the test dataset of task `self.task_id`.

        **Returns:**
        - **test_dataset** (`Dataset`): the test dataset of task `self.task_id`.
        """
        dataset_test = self.original_dataset_python_class(
            root=self.root_t,
            train=False,
            transform=self.test_transforms(),
            target_transform=self.target_transform(),
            download=False,
        )

        return dataset_test
