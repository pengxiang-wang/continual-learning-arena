r"""
The submodule in `cl_datasets` for Split CIFAR-10 dataset.
"""

__all__ = ["SplitCIFAR10"]

import logging
from typing import Callable

import torch
from torch.utils.data import Dataset, random_split
from torchvision.datasets import CIFAR10
from torchvision.transforms import transforms

from clarena.cl_datasets import CLSplitDataset

# always get logger for built-in logging in each module
pylogger = logging.getLogger(__name__)


class SplitCIFAR10(CLSplitDataset):
    r"""Split CIFAR-10 dataset. The [CIFAR-10 dataset](https://www.cs.toronto.edu/~kriz/cifar.html) is a subset of the [80 million tiny images dataset](https://people.csail.mit.edu/torralba/tinyimages/). It consists of 50,000 training and 10,000 test images of 10 classes, each 32x32 color image."""

    original_dataset_python_class: type[Dataset] = CIFAR10
    r"""The original dataset class."""

    def __init__(
        self,
        root: str,
        class_split: dict[int, list[int]],
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
    ) -> None:
        r"""
        **Args:**
        - **root** (`str`): the root directory where the original CIFAR-10 data 'cifar-10-python/' live.
        - **class_split** (`dict[int, list[int]]`): the dict of classes for each task. The keys are task IDs ane the values are lists of class labels (integers starting from 0) to split for each task.
        - **validation_percentage** (`float`): The percentage to randomly split some training data into validation data.
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
        """
        super().__init__(
            root=root,
            class_split=class_split,
            batch_size=batch_size,
            num_workers=num_workers,
            custom_transforms=custom_transforms,
            repeat_channels=repeat_channels,
            to_tensor=to_tensor,
            resize=resize,
        )

        self.validation_percentage: float = validation_percentage
        r"""The percentage to randomly split some training data into validation data."""

    def prepare_data(self) -> None:
        r"""Download the original CIFAR-10 dataset if haven't."""

        if self.task_id != 1:
            return  # download all original datasets only at the beginning of first task

        CIFAR10(root=self.root_t, train=True, download=True)
        CIFAR10(root=self.root_t, train=False, download=True)

        pylogger.debug(
            "The original CIFAR-10 dataset has been downloaded to %s.", self.root
        )

    def get_subset_of_classes(self, dataset: Dataset) -> Dataset:
        r"""Get a subset of classes from the dataset of current classes of `self.task_id`. It is used when constructing the split. It must be implemented by subclasses.

        **Args:**
        - **dataset** (`Dataset`): the dataset to retrieve subset from.

        **Returns:**
        - **subset** (`Dataset`): the subset of classes from the dataset.
        """
        classes = self.class_split[self.task_id]

        # get the indices of the dataset that belong to the classes
        idx = [i for i, (_, target) in enumerate(dataset) if target in classes]

        # subset the dataset by the indices, in-place operation
        dataset.data = dataset.data[idx]  # data is a Numpy ndarray
        dataset.targets = [dataset.targets[i] for i in idx]  # targets is a list

        dataset.target_transform = self.target_transform()  # cl class mapping should be applied after the split

        return dataset

    def train_and_val_dataset(self) -> tuple[Dataset, Dataset]:
        r"""Get the training and validation dataset of task `self.task_id`.

        **Returns:**
        - **train_and_val_dataset** (`tuple[Dataset, Dataset]`): the train and validation dataset of task `self.task_id`.
        """
        dataset_train_and_val = self.get_subset_of_classes(
            CIFAR10(
                root=self.root_t,
                train=True,
                transform=self.train_and_val_transforms(),
                # cl class mapping should be applied after the split
                download=False,
            )
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
        dataset_test = self.get_subset_of_classes(
            CIFAR10(
                root=self.root_t,
                train=False,
                transform=self.test_transforms(),
                # cl class mapping should be applied after the split
                download=False,
            )
        )

        return dataset_test
