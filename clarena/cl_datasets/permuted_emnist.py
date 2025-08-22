r"""
The submodule in `cl_datasets` for Permuted EMNIST dataset.
"""

__all__ = ["PermutedEMNIST"]

import logging
from typing import Callable

import torch
from torch.utils.data import Dataset, random_split
from torchvision.datasets import EMNIST
from torchvision.transforms import transforms

from clarena.cl_datasets import CLPermutedDataset
from clarena.stl_datasets.raw import (
    EMNISTBalanced,
    EMNISTByClass,
    EMNISTByMerge,
    EMNISTDigits,
    EMNISTLetters,
)
from clarena.utils.transforms import ClassMapping

# always get logger for built-in logging in each module
pylogger = logging.getLogger(__name__)


class PermutedEMNIST(CLPermutedDataset):
    r"""Permuted EMNIST dataset. The [EMNIST dataset](https://www.nist.gov/itl/products-and-services/emnist-dataset/) is a collection of handwritten letters and digits (including A-Z, a-z, 0-9). It consists of 814,255 images in 62 classes, each 28x28 grayscale image.

    EMNIST has 6 different splits: `byclass`, `bymerge`, `balanced`, `letters`, `digits` and `mnist`, each containing a different subset of the original collection. We support all of them in Permuted EMNIST.
    """

    def __init__(
        self,
        root: str,
        split: str,
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
        r"""Initialize the dataset object providing the root where data files live.

        **Args:**
        - **root** (`str`): the root directory where the original EMNIST data 'EMNIST/' live.
        - **split** (`str`): the original EMNIST dataset has 6 different splits: `byclass`, `bymerge`, `balanced`, `letters`, `digits` and `mnist`. This argument specifies which one to use.
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
        - **permutation_mode** (`str`): the mode of permutation, should be one of the following:
            1. 'all': permute all pixels.
            2. 'by_channel': permute channel by channel separately. All channels are applied the same permutation order.
            3. 'first_channel_only': permute only the first channel.
        - **permutation_seeds** (`dict[int, int]` | `None`): the dict of seeds for permutation operations used to construct each task. Keys are task IDs and the values are permutation seeds for each task. Default is `None`, which creates a dict of seeds from 0 to `num_tasks`-1.
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

        self.split: str = split
        r"""Store the split of the original EMNIST dataset. It can be `byclass`, `bymerge`, `balanced`, `letters`, `digits` or `mnist`."""

        self.validation_percentage: float = validation_percentage
        r"""The percentage to randomly split some training data into validation data."""

    def prepare_data(self) -> None:
        r"""Download the original EMNIST dataset if haven't."""

        if self.task_id != 1:
            return  # download all original datasets only at the beginning of first task

        EMNIST(root=self.root_t, split=self.split, train=True, download=True)
        EMNIST(root=self.root_t, split=self.split, train=False, download=True)

        pylogger.debug(
            "The original EMNIST dataset has been downloaded to %s.", self.root_t
        )

    def train_and_val_dataset(self) -> tuple[Dataset, Dataset]:
        """Get the training and validation dataset of task `self.task_id`.

        **Returns:**
        - **train_and_val_dataset** (`tuple[Dataset, Dataset]`): the train and validation dataset of task `self.task_id`.
        """
        dataset_train_and_val = EMNIST(
            root=self.root_t,
            split=self.split,
            train=True,
            transform=self.train_and_val_transforms(),
            target_transform=ClassMapping(self.get_cl_class_map(self.task_id)),
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

        dataset_test = EMNIST(
            root=self.root_t,
            split=self.split,
            train=False,
            transform=self.test_transforms(),
            target_transform=ClassMapping(self.get_cl_class_map(self.task_id)),
            download=False,
        )

        return dataset_test
