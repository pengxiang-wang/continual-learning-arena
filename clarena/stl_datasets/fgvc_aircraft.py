r"""
The submodule in `stl_datasets` for FGVC-Aircraft dataset.
"""

__all__ = ["FGVCAircraft"]

import logging
from typing import Callable

from torch.utils.data import Dataset
from torchvision.datasets import FGVCAircraft as FGVCAircraftRaw
from torchvision.transforms import transforms

from clarena.stl_datasets.base import STLDatasetFromRaw
from clarena.stl_datasets.raw import (
    FGVCAircraftFamily,
    FGVCAircraftManufacturer,
    FGVCAircraftVariant,
)

# always get logger for built-in logging in each module
pylogger = logging.getLogger(__name__)


class FGVCAircraft(STLDatasetFromRaw):
    r"""FGVC-Aircraft dataset. The [FGVC-Aircraft dataset](https://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/) is a collection of aircraft images. It consists of 10,200 images, each color image.

    FGVC-Aircraft has 3 different class labels by variant, family and manufacturer, which has 102, 70, 41 classes respectively. We support all of them in Permuted FGVC-Aircraft.
    """

    def __init__(
        self,
        root: str,
        annotation_level: str,
        batch_size: int = 1,
        num_workers: int = 0,
        custom_transforms: Callable | transforms.Compose | None = None,
        repeat_channels: int | None = None,
        to_tensor: bool = True,
        resize: tuple[int, int] | None = None,
    ) -> None:
        r"""
        **Args:**
        - **root** (`str`): the root directory where the original FGVCAircraft data 'FGVCAircraft/' live.
        - **annotation_level** (`str`): The annotation level, supports 'variant', 'family' and 'manufacturer'.
        - **batch_size** (`int`): The batch size in train, val, test dataloader.
        - **num_workers** (`int`): the number of workers for dataloaders.
        - **custom_transforms** (`transform` or `transforms.Compose` or `None`): the custom transforms to apply to ONLY TRAIN dataset. Can be a single transform, composed transforms or no transform. `ToTensor()`, normalize and so on are not included.
        - **repeat_channels** (`int` | `None`): the number of channels to repeat. Default is None, which means no repeat. If not None, it should be an integer.
        - **to_tensor** (`bool`): whether to include `ToTensor()` transform. Default is True.
        - **resize** (`tuple[int, int]` | `None` or list of them): the size to resize the images to. Default is None, which means no resize. If not None, it should be a tuple of two integers.
        """

        if annotation_level == "variant":
            self.original_dataset_python_class: type[Dataset] = FGVCAircraftVariant
        elif annotation_level == "family":
            self.original_dataset_python_class: type[Dataset] = FGVCAircraftFamily
        elif annotation_level == "manufacturer":
            self.original_dataset_python_class: type[Dataset] = FGVCAircraftManufacturer
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

        self.annotation_level: str = annotation_level
        r"""The annotation level, supports 'variant', 'family' and 'manufacturer'."""

    def prepare_data(self) -> None:
        r"""Download the original FGVC-Aircraft dataset if haven't."""

        FGVCAircraftRaw(root=self.root, split="train", download=True)
        FGVCAircraftRaw(root=self.root, split="val", download=True)
        FGVCAircraftRaw(root=self.root, split="test", download=True)

        pylogger.debug(
            "The original FGVC-Aircraft dataset has been downloaded to %s.",
            self.root,
        )

    def train_and_val_dataset(self) -> tuple[Dataset, Dataset]:
        """Get the training and validation dataset.

        **Returns:**
        - **train_and_val_dataset** (`tuple[Dataset, Dataset]`): the train and validation dataset.
        """
        dataset_train = FGVCAircraftRaw(
            root=self.root,
            split="train",
            annotation_level=self.annotation_level,
            transform=self.train_and_val_transforms(),
            target_transform=self.target_transform(),
            download=False,
        )

        dataset_val = FGVCAircraftRaw(
            root=self.root,
            split="val",
            annotation_level=self.annotation_level,
            transform=self.train_and_val_transforms(),
            target_transform=self.target_transform(),
            download=False,
        )

        return dataset_train, dataset_val

    def test_dataset(self) -> Dataset:
        r"""Get the test dataset.

        **Returns:**
        - **test_dataset** (`Dataset`): the test dataset.
        """
        dataset_test = FGVCAircraftRaw(
            root=self.root,
            split="test",
            annotation_level=self.annotation_level,
            transform=self.test_transforms(),
            target_transform=self.target_transform(),
            download=False,
        )

        return dataset_test
