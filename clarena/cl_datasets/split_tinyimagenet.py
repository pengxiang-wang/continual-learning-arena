r"""
The submodule in `cl_datasets` for Split TinyImageNet dataset.
"""

__all__ = ["SplitTinyImageNet"]

import logging
from typing import Callable

import torch
from tinyimagenet import TinyImageNet
from torch.utils.data import Dataset, random_split
from torchvision import transforms
from torchvision.datasets import ImageFolder

from clarena.cl_datasets import CLSplitDataset

# always get logger for built-in logging in each module
pylogger = logging.getLogger(__name__)


class SplitTinyImageNet(CLSplitDataset):
    r"""Split TinyImageNet dataset. [TinyImageNet](http://vision.stanford.edu/teaching/cs231n/reports/2015/pdfs/yle_project.pdf) is smaller, more manageable version of the [larger ImageNet dataset](https://www.image-net.org). It consists of 120,000 64x64 colour images in 200 classes, with 500 training, 50 validation and 50 test examples per class."""

    num_classes: int = 200
    r"""The number of classes in TinyImageNet dataset."""

    mean_original: tuple[float] = (0.4802, 0.4481, 0.3975)
    r"""The mean values for normalisation."""

    std_original: tuple[float] = (0.2302, 0.2265, 0.2262)
    r"""The standard deviation values for normalisation."""

    def __init__(
        self,
        root: str,
        num_tasks: int,
        class_split: list[list[int]],
        validation_percentage: float,
        batch_size: int = 1,
        num_workers: int = 0,
        custom_transforms: Callable | transforms.Compose | None = None,
        to_tensor: bool = True,
        resize: tuple[int, int] | None = None,
        custom_target_transforms: Callable | transforms.Compose | None = None,
    ) -> None:
        r"""Initialise the Split TinyImageNet dataset.

        **Args:**
        - **root** (`str`): the root directory where the original TinyImageNet data 'tiny-imagenet-200/' live.
        - **num_tasks** (`int`): the maximum number of tasks supported by the CL dataset.
        - **class_split** (`list[list[int]]`): the class split for each task. Each element in the list is a list of class labels (integers starting from 0) to split for a task.
        - **validation_percentage** (`float`): the percentage to randomly split some of the training data into validation data.
        - **batch_size** (`int`): The batch size in train, val, test dataloader.
        - **num_workers** (`int`): the number of workers for dataloaders.
        - **custom_transforms** (`transform` or `transforms.Compose` or `None`): the custom transforms to apply to ONLY TRAIN dataset. Can be a single transform, composed transforms or no transform.
        `ToTensor()`, normalise, permute and so on are not included.
        - **to_tensor** (`bool`): whether to include `ToTensor()` transform. Default is True.
        - **resize** (`tuple[int, int]` | `None`): the size to resize the images to. Default is None, which means no resize. If not None, it should be a tuple of two integers.
        - **custom_target_transforms** (`transform` or `transforms.Compose` or `None`): the custom target transforms to apply to dataset labels. Can be a single transform, composed transforms or no transform. CL class mapping is not included.
        - **permutation_mode** (`str`): the mode of permutation, should be one of the following:
            1. 'all': permute all pixels.
            2. 'by_channel': permute channel by channel separately. All channels are applied the same permutation order.
            3. 'first_channel_only': permute only the first channel.
        """
        CLSplitDataset.__init__(
            self,
            root=root,
            num_tasks=num_tasks,
            class_split=class_split,
            batch_size=batch_size,
            num_workers=num_workers,
            custom_transforms=custom_transforms,
            to_tensor=to_tensor,
            resize=resize,
            custom_target_transforms=custom_target_transforms,
        )

        self.validation_percentage: float = validation_percentage
        """Store the percentage to randomly split some of the training data into validation data."""

    def prepare_data(self) -> None:
        r"""Download the original TinyImagenet dataset if haven't."""
        TinyImageNet(self.root)

    def get_class_subset(self, dataset: ImageFolder) -> ImageFolder:
        r"""Provide a util method here to retrieve a subset from PyTorch ImageFolder of current classes of `self.task_id`. It could be useful when you constructing the split CL dataset.

        **Args:**
        - **dataset** (`ImageFolder`): the original dataset to retrieve subset from.

        **Returns:**
        - **subset** (`ImageFolder`): subset of original dataset in classes.
        """
        classes = self.class_split[self.task_id - 1]

        # get the indices of the dataset that belong to the classes
        idx = [i for i, (_, target) in enumerate(dataset) if target in classes]

        # subset the dataset by the indices, in-place operation
        dataset.samples = [dataset.samples[i] for i in idx]  # samples is a list
        dataset.targets = [dataset.targets[i] for i in idx]  # targets is a list

        return dataset

    def train_and_val_dataset(self) -> Dataset:
        r"""Get the training and validation dataset of task `self.task_id`.

        **Returns:**
        - **train_dataset** (`Dataset`): the training dataset of task `self.task_id`.
        - **val_dataset** (`Dataset`): the validation dataset of task `self.task_id`.
        """
        dataset_train_and_val = self.get_class_subset(
            TinyImageNet(
                root=self.root,
                split="train",
                transform=self.train_and_val_transforms(),
            )
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
        dataset_test = self.get_class_subset(
            TinyImageNet(
                root=self.root,
                split="val",
                transform=self.train_and_val_transforms(),
            )
        )

        dataset_test.target_transform = self.target_transforms()

        return dataset_test
