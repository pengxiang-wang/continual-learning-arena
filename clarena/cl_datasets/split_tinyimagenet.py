r"""
The submodule in `cl_datasets` for Split TinyImageNet dataset.
"""

__all__ = ["SplitTinyImageNet"]

import logging
import os
import urllib.request
import zipfile
from typing import Callable

from torch.utils.data import Dataset
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
        batch_size: int = 1,
        num_workers: int = 0,
        custom_transforms: Callable | transforms.Compose | None = None,
        custom_target_transforms: Callable | transforms.Compose | None = None,
    ) -> None:
        r"""Initialise the Split TinyImageNet dataset.

        **Args:**
        - **root** (`str`): the root directory where the original TinyImageNet data 'tiny-imagenet-200/' live.
        - **num_tasks** (`int`): the maximum number of tasks supported by the CL dataset.
        - **class_split** (`list[list[int]]`): the class split for each task. Each element in the list is a list of class labels (integers starting from 0) to split for a task.
        - **batch_size** (`int`): The batch size in train, val, test dataloader.
        - **num_workers** (`int`): the number of workers for dataloaders.
        - **custom_transforms** (`transform` or `transforms.Compose` or `None`): the custom transforms to apply to ONLY TRAIN dataset. Can be a single transform, composed transforms or no transform.
        `ToTensor()`, normalise, permute and so on are not included.
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
            custom_target_transforms=custom_target_transforms,
        )

    def prepare_data(self) -> None:
        r"""Download the original CIFAR-100 dataset if haven't."""

        url = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
        zip_path = os.path.join(self.root, "tiny-imagenet-200.zip")
        data_dir = os.path.join(self.root, "tiny-imagenet-200")

        # check if the root directory exists. If it doesn't, create it
        if not os.path.exists(self.root):
            os.makedirs(self.root)
            pylogger.info("Created directory %s", self.root)

        # check if the dataset folder already exists. If it does, skip download
        if os.path.exists(data_dir):
            pylogger.info(
                "TinyImageNet dataset already exists at %s. Skipping download.",
                data_dir,
            )
            return

        # download the zip file if it doesn't exist
        if not os.path.exists(zip_path):
            pylogger.info("Downloading TinyImageNet dataset from %s...", url)
            urllib.request.urlretrieve(url, zip_path)
            pylogger.info("Download complete: %s", self.root)

        # Extract the zip file
        pylogger.info("Extracting TinyImageNet dataset to %s...", self.root)
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(self.root)
        os.remove(zip_path)
        pylogger.info("Extraction complete.")

    def train_dataset(self) -> Dataset:
        r"""Get the training dataset of task `self.task_id`.

        **Returns:**
        - **train_dataset** (`Dataset`): the train dataset of task `self.task_id`.
        """
        dataset_train = self.get_class_subset(
            ImageFolder(
                root=os.path.join(self.root, "tiny-imagenet-200", "train"),
                transform=self.train_and_val_transforms(to_tensor=True),
            )
        )

        dataset_train.target_transform = self.target_transforms()

        return dataset_train

    def val_dataset(self) -> Dataset:
        r"""Get the validation dataset of task `self.task_id`.

        **Returns:**
        - **val_dataset** (`Dataset`): the validation dataset of task `self.task_id`.
        """
        dataset_val = self.get_class_subset(
            ImageFolder(
                root=os.path.join(self.root, "tiny-imagenet-200", "val"),
                transform=self.train_and_val_transforms(to_tensor=True),
            )
        )

        dataset_val.target_transform = self.target_transforms()

        return dataset_val

    def test_dataset(self) -> Dataset:
        r"""Get the test dataset of task `self.task_id`.

        **Returns:**
        - **test_dataset** (`Dataset`): the test dataset of task `self.task_id`.
        """
        dataset_test = self.get_class_subset(
            ImageFolder(
                root=os.path.join(self.root, "tiny-imagenet-200", "val"),
                transform=self.test_transforms(to_tensor=True),
            )
        )

        dataset_test.target_transform = self.target_transforms()

        return dataset_test
