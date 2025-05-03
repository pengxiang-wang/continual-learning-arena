r"""
The submodule in `cl_datasets` for Split MNIST dataset.
"""

__all__ = ["SplitMNIST"]

import logging
from typing import Callable

import torch
from torch.utils.data import Dataset, random_split
from torchvision.datasets import MNIST
from torchvision.transforms import transforms

from clarena.cl_datasets import CLClassMapping, CLSplitDataset

# always get logger for built-in logging in each module
pylogger = logging.getLogger(__name__)


class SplitMNIST(CLSplitDataset):
    r"""Split MNIST dataset. The [original MNIST dataset](http://yann.lecun.com/exdb/mnist/) is a collection of handwritten digits. It consists of 70,000 28x28 B&W images in 10 classes (correspond to 10 digits), with 7000 images per class. There are 60,000 training examples and 10,000 test examples."""

    original_dataset_python_class: type[Dataset] = MNIST
    r"""The original dataset class."""

    def __init__(
        self,
        root: str,
        class_split: list[list[int]],
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
    ) -> None:
        r"""Initialise the Split MNIST dataset object providing the root where data files live.

        **Args:**
        - **root** (`str`): the root directory where the original MNIST data 'MNIST/' live.
        - **class_split** (`list[list[int]]`): the class split for each task. Each element in the list is a list of class labels (integers starting from 0) to split for a task.
        - **validation_percentage** (`float`): the percentage to randomly split some of the training data into validation data.

        - **batch_size** (`int` | `list[int]`): The batch size in train, val, test dataloader. If `list[str]`, it should be a list of integers, each integer is the batch size for each task.
        - **num_workers** (`int` | `list[int]`): the number of workers for dataloaders. If `list[str]`, it should be a list of integers, each integer is the num of workers for each task.
        - **custom_transforms** (`transform` or `transforms.Compose` or `None` or list of them): the custom transforms to apply to ONLY TRAIN dataset. Can be a single transform, composed transforms or no transform. `ToTensor()`, normalise, permute and so on are not included. If it is a list, each item is the custom transforms for each task.
        - **repeat_channels** (`int` | `None` | list of them): the number of channels to repeat for each task. Default is None, which means no repeat. If not None, it should be an integer. If it is a list, each item is the number of channels to repeat for each task.
        - **to_tensor** (`bool` | `list[bool]`): whether to include `ToTensor()` transform. Default is True.
        - **resize** (`tuple[int, int]` | `None` or list of them): the size to resize the images to. Default is None, which means no resize. If not None, it should be a tuple of two integers. If it is a list, each item is the size to resize for each task.
        """
        CLSplitDataset.__init__(
            self,
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
        """Store the percentage to randomly split some of the training data into validation data."""

    def prepare_data(self) -> None:
        r"""Download the original MNIST dataset if haven't."""

        if self.task_id != 1:
            return  # download all original datasets only at the beginning of first task

        MNIST(root=self.root_t, train=True, download=True)
        MNIST(root=self.root_t, train=False, download=True)

        pylogger.debug(
            "The original MNIST dataset has been downloaded to %s.", self.root
        )

    def get_subset_of_classes(self, dataset: Dataset) -> Dataset:
        r"""Get a subset of classes from the dataset of current classes of `self.task_id`. It is used when constructing the split. It must be implemented by subclasses.

        **Args:**
        - **dataset** (`Dataset`): the dataset to retrieve subset from.

        **Returns:**
        - **subset** (`Dataset`): the subset of classes from the dataset.
        """
        classes = self.class_split[self.task_id - 1]

        # get the indices of the dataset that belong to the classes
        idx = [i for i, (_, target) in enumerate(dataset) if target in classes]

        # subset the dataset by the indices, in-place operation
        dataset.data = dataset.data[idx]  # data is a Numpy ndarray
        dataset.targets = [dataset.targets[i] for i in idx]  # targets is a list

        dataset.target_transform = CLClassMapping(
            self.cl_class_map_t
        )  # cl class mapping should be applied after the split

        return dataset

    def train_and_val_dataset(self) -> tuple[Dataset, Dataset]:
        r"""Get the training and validation dataset of task `self.task_id`.

        **Returns:**
        - **train_and_val_dataset** (`tuple[Dataset, Dataset]`): the train and validation dataset of task `self.task_id`.
        """
        dataset_train_and_val = self.get_subset_of_classes(
            MNIST(
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
            MNIST(
                root=self.root_t,
                train=False,
                transform=self.test_transforms(),
                # cl class mapping should be applied after the split
                download=False,
            )
        )

        return dataset_test
