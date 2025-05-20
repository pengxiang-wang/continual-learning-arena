r"""
The submodule in `cl_datasets` for Permuted SEMEION dataset.
"""

__all__ = ["PermutedSEMEION"]

import logging
from typing import Callable

import torch
from torch.utils.data import Dataset, random_split
from torchvision.datasets import SEMEION
from torchvision.transforms import transforms

from clarena.cl_datasets import CLClassMapping, CLPermutedDataset

# always get logger for built-in logging in each module
pylogger = logging.getLogger(__name__)


class PermutedSEMEION(CLPermutedDataset):
    r"""Permuted SEMEION dataset. The [original SEMEION dataset](https://archive.ics.uci.edu/dataset/178/semeion+handwritten+digit) is a collection of handwritten digits. It consists of 1,593 16x16 grayscale images in 10 classes (correspond to 10 digits)."""

    original_dataset_python_class: type[Dataset] = SEMEION
    r"""The original dataset class."""

    def __init__(
        self,
        root: str,
        num_tasks: int,
        validation_percentage: float,
        test_percentage: float,
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
        permutation_mode: str = "first_channel_only",
        permutation_seeds: list[int] | None = None,
    ) -> None:
        r"""Initialise the Permuted SEMEION dataset object providing the root where data files live.

        **Args:**
        - **root** (`str`): the root directory where the original SEMEION data 'SEMEION/' live.
        - **num_tasks** (`int`): the maximum number of tasks supported by the CL dataset.
        - **validation_percentage** (`float`): the percentage to randomly split some of the training data into validation data.
        - **test_percentage** (`float`): the percentage to randomly split some of the entire data into test data.
        - **batch_size** (`int` | `list[int]`): The batch size in train, val, test dataloader. If `list[str]`, it should be a list of integers, each integer is the batch size for each task.
        - **num_workers** (`int` | `list[int]`): the number of workers for dataloaders. If `list[str]`, it should be a list of integers, each integer is the num of workers for each task.
        - **custom_transforms** (`transform` or `transforms.Compose` or `None` or list of them): the custom transforms to apply to ONLY TRAIN dataset. Can be a single transform, composed transforms or no transform. `ToTensor()`, normalise, permute and so on are not included. If it is a list, each item is the custom transforms for each task.
        - **repeat_channels** (`int` | `None` | list of them): the number of channels to repeat for each task. Default is None, which means no repeat. If not None, it should be an integer. If it is a list, each item is the number of channels to repeat for each task.
        - **to_tensor** (`bool` | `list[bool]`): whether to include `ToTensor()` transform. Default is True.
        - **resize** (`tuple[int, int]` | `None` or list of them): the size to resize the images to. Default is None, which means no resize. If not None, it should be a tuple of two integers. If it is a list, each item is the size to resize for each task.
        - **permutation_mode** (`str`): the mode of permutation, should be one of the following:
            1. 'all': permute all pixels.
            2. 'by_channel': permute channel by channel separately. All channels are applied the same permutation order.
            3. 'first_channel_only': permute only the first channel.
        - **permutation_seeds** (`list[int]` or `None`): the seeds for permutation operations used to construct tasks. Make sure it has the same number of seeds as `num_tasks`. Default is None, which creates a list of seeds from 0 to `num_tasks`-1.
        """
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
            permutation_mode=permutation_mode,
            permutation_seeds=permutation_seeds,
        )

        self.validation_percentage: float = validation_percentage
        """Store the percentage to randomly split some of the training data into validation data."""
        self.test_percentage: float = test_percentage
        """Store the percentage to randomly split some of the entire data into test data."""

    def prepare_data(self) -> None:
        r"""Download the original SEMEION dataset if haven't."""

        if self.task_id != 1:
            return  # download all original datasets only at the beginning of first task

        SEMEION(root=self.root_t, download=True)

        pylogger.debug(
            "The original SEMEION dataset has been downloaded to %s.", self.root_t
        )

    def train_and_val_dataset(self) -> tuple[Dataset, Dataset]:
        """Get the training and validation dataset of task `self.task_id`.

        **Returns:**
        - **train_and_val_dataset** (`tuple[Dataset, Dataset]`): the train and validation dataset of task `self.task_id`.
        """
        dataset_all = SEMEION(
            root=self.root_t,
            transform=self.train_and_val_transforms(),
            target_transform=CLClassMapping(self.cl_class_map_t),
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
        r"""Get the test dataset of task `self.task_id`.

        **Returns:**
        - **test_dataset** (`Dataset`): the test dataset of task `self.task_id`.
        """
        dataset_all = SEMEION(
            root=self.root_t,
            transform=self.train_and_val_transforms(),
            target_transform=CLClassMapping(self.cl_class_map_t),
            download=False,
        )

        _, dataset_test = random_split(
            dataset_all,
            lengths=[1 - self.test_percentage, self.test_percentage],
            generator=torch.Generator().manual_seed(42),
        )

        return dataset_test
