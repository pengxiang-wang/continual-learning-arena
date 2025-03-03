r"""
The submodule in `cl_datasets` for Split CIFAR-100 dataset.
"""

__all__ = ["SplitCIFAR100"]

import logging
from typing import Callable

from torch.utils.data import Dataset, random_split
from torchvision.datasets import CIFAR100
from torchvision.transforms import transforms

from clarena.cl_datasets import CLSplitDataset

# always get logger for built-in logging in each module
pylogger = logging.getLogger(__name__)


class SplitCIFAR100(CLSplitDataset):
    r"""Split CIFAR-100 dataset. The [original CIFAR100 dataset](https://www.cs.toronto.edu/~kriz/cifar.html) is a subset of the 80 million tiny images dataset. It consists of 60,000 32x32 colour images in 100 classes, with 600 images per class. There are 50,000 training examples and 10,000 test examples."""

    num_classes: int = 100
    r"""The number of classes in CIFAR-100 dataset."""

    mean_original: tuple[float] = (0.5074, 0.4867, 0.4411)
    r"""The mean values for normalisation."""

    std_original: tuple[float] = (0.2011, 0.1987, 0.2025)
    r"""The standard deviation values for normalisation."""

    def __init__(
        self,
        root: str,
        num_tasks: int,
        class_split: list[list[int]],
        validation_percentage: float,
        batch_size: int = 1,
        num_workers: int = 8,
        custom_transforms: Callable | transforms.Compose | None = None,
        custom_target_transforms: Callable | transforms.Compose | None = None,
    ) -> None:
        r"""Initialise the Split CIFAR-100 dataset.

        **Args:**
        - **root** (`str`): the root directory where the original CIFAR-100 data 'cifar-100-python/' live.
        - **num_tasks** (`int`): the maximum number of tasks supported by the CL dataset.
        - **class_split** (`list[list[int]]`): the class split for each task. Each element in the list is a list of class labels (integers starting from 0) to split for a task.
        - **validation_percentage** (`float`): the percentage to randomly split some of the training data into validation data.
        - **batch_size** (`int`): The batch size in train, val, test dataloader.
        - **num_workers** (`int`): the number of workers for dataloaders.
        - **custom_transforms** (`transform` or `transforms.Compose` or `None`): the custom transforms to apply to ONLY TRAIN dataset. Can be a single transform, composed transforms or no transform.
        `ToTensor()`, normalise, permute and so on are not included.
        - **custom_target_transforms** (`transform` or `transforms.Compose` or `None`): the custom target transforms to apply to dataset labels. Can be a single transform, composed transforms or no transform. CL class mapping is not included.
        - **permutation_mode** (`str`): the mode of permutation, should be one of the following:
            1. 'all': permute all pixels.
            2. 'by_channel': permute channel by channel separately. All channels are applied the same permutation order.
            3. 'first_channel_only': permute only the first channel.
        - **permutation_seeds** (`list[int]` or `None`): the seeds for permutation operations used to construct tasks. Make sure it has the same number of seeds as `num_tasks`. Default is None, which creates a list of seeds from 1 to `num_tasks`.
        """
        CLSplitDataset.__init__(
            self,
            root=root,
            num_tasks=num_tasks,
            class_split=class_split,
            validation_percentage=validation_percentage,
            batch_size=batch_size,
            num_workers=num_workers,
            custom_transforms=custom_transforms,
            custom_target_transforms=custom_target_transforms,
        )

    def prepare_data(self) -> None:
        r"""Download the original CIFAR-100 dataset if haven't."""
        # just download
        CIFAR100(root=self.root, train=True, download=True)
        CIFAR100(root=self.root, train=False, download=True)

        pylogger.debug(
            "The original CIFAR-100 dataset has been downloaded to %s.", self.root
        )

    def train_and_val_dataset(self) -> tuple[Dataset, Dataset]:
        r"""Get the training and validation dataset of task `self.task_id`.

        **Returns:**
        - **train_and_val_dataset** (`tuple[Dataset, Dataset]`): the train and validation dataset of task `self.task_id`.
        """
        dataset_train_and_val = self.get_class_subset(
            CIFAR100(
                root=self.root,
                train=True,
                transform=self.train_and_val_transforms(to_tensor=True),
                download=False,
            )
        )
        dataset_train_and_val.target_transform = self.target_transforms()

        return random_split(
            dataset_train_and_val,
            lengths=[1 - self.validation_percentage, self.validation_percentage],
        )

    def test_dataset(self) -> Dataset:
        r"""Get the test dataset of task `self.task_id`.

        **Returns:**
        - **test_dataset** (`Dataset`): the test dataset of task `self.task_id`.
        """
        dataset_test = self.get_class_subset(
            CIFAR100(
                root=self.root,
                train=False,
                transform=self.test_transforms(to_tensor=True),
                download=False,
            )
        )
        dataset_test.target_transform = self.target_transforms()

        return dataset_test
