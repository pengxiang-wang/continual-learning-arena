r"""
The submodule in `stl_datasets` for FaceScrub dataset.
"""

__all__ = ["FaceScrub"]

import logging
from typing import Callable

import torch
from torch.utils.data import Dataset, random_split
from torchvision.transforms import transforms

from clarena.stl_datasets.base import STLDatasetFromRaw
from clarena.stl_datasets.raw import FaceScrub10, FaceScrub20, FaceScrub50, FaceScrub100

# always get logger for built-in logging in each module
pylogger = logging.getLogger(__name__)


class FaceScrub(STLDatasetFromRaw):
    r"""FaceScrub dataset. The [original FaceScrub dataset](https://vintage.winklerbros.net/facescrub.html) is a collection of human face images. It consists 106,863 images of 530 people (classes), each high resolution color image.

    To make it simple, [this version](https://github.com/nkundiushuti/facescrub_subset) uses subset of the official [Megaface FaceScrub challenge](http://megaface.cs.washington.edu/participate/challenge.html), cropped and resized to 32x32. We have [FaceScrub-10](https://github.com/nkundiushuti/facescrub_subset/blob/master/data/facescrub_10.zip), [FaceScrub-20](https://github.com/nkundiushuti/facescrub_subset/blob/master/data/facescrub_20.zip), [FaceScrub-50](https://github.com/nkundiushuti/facescrub_subset/blob/master/data/facescrub_50.zip), [FaceScrub-100](https://github.com/nkundiushuti/facescrub_subset/blob/master/data/facescrub_100.zip) datasets where the number of classes are 10, 20, 50 and 100 respectively.
    """

    def __init__(
        self,
        root: str,
        size: str,
        validation_percentage: float,
        batch_size: int = 1,
        num_workers: int = 0,
        custom_transforms: Callable | transforms.Compose | None = None,
        repeat_channels: int | None = None,
        to_tensor: bool = True,
        resize: tuple[int, int] | None = None,
    ) -> None:
        r"""
        **Args:**
        - **root** (`str`): the root directory where the original FaceScrub data 'FaceScrub/' live.
        - **size** (`str`): the size of the dataset; one of:
            1. '10': 10 classes (10 people).
            2. '20': 20 classes (20 people).
            3. '50': 50 classes (50 people).
            4. '100': 100 classes (100 people).
        - **validation_percentage** (`float`): the percentage to randomly split some training data into validation data.
        - **batch_size** (`int`): The batch size in train, val, test dataloader.
        - **num_workers** (`int`): the number of workers for dataloaders.
        - **custom_transforms** (`transform` or `transforms.Compose` or `None`): the custom transforms to apply to ONLY TRAIN dataset. Can be a single transform, composed transforms or no transform. `ToTensor()`, normalize and so on are not included.
        - **repeat_channels** (`int` | `None`): the number of channels to repeat. Default is None, which means no repeat. If not None, it should be an integer.
        - **to_tensor** (`bool`): whether to include `ToTensor()` transform. Default is True.
        - **resize** (`tuple[int, int]` | `None` or list of them): the size to resize the images to. Default is None, which means no resize. If not None, it should be a tuple of two integers.
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
        r"""Download the original FaceScrub dataset if haven't."""

        self.original_dataset_python_class(root=self.root, train=True, download=True)
        self.original_dataset_python_class(root=self.root, train=False, download=True)

        pylogger.debug(
            "The original FaceScrub dataset has been downloaded to %s.", self.root
        )

    def train_and_val_dataset(self) -> tuple[Dataset, Dataset]:
        """Get the training and validation dataset.

        **Returns:**
        - **train_and_val_dataset** (`tuple[Dataset, Dataset]`): the train and validation dataset.
        """
        dataset_train_and_val = self.original_dataset_python_class(
            root=self.root,
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
        r"""Get the test dataset.

        **Returns:**
        - **test_dataset** (`Dataset`): the test dataset.
        """
        dataset_test = self.original_dataset_python_class(
            root=self.root,
            train=False,
            transform=self.test_transforms(),
            target_transform=self.target_transform(),
            download=False,
        )

        return dataset_test
