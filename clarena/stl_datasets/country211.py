r"""
The submodule in `stl_datasets` for Country211 dataset.
"""

__all__ = ["Country211"]

import logging
from typing import Callable

from torch.utils.data import Dataset
from torchvision.datasets import Country211 as Country211Raw
from torchvision.transforms import transforms

from clarena.stl_datasets.base import STLDatasetFromRaw

# always get logger for built-in logging in each module
pylogger = logging.getLogger(__name__)


class Country211(STLDatasetFromRaw):
    r"""Country211 dataset. The [Country211 dataset](https://github.com/openai/CLIP/blob/main/data/country211.md) is a collection of geolocation pictures of different countries. It consists of 31,650 training, 10,550 validation, and 21,100 test images of 211 countries (classes), each 256x256 color image."""

    original_dataset_python_class: type[Dataset] = Country211Raw
    r"""The original dataset class."""

    def __init__(
        self,
        root: str,
        batch_size: int = 1,
        num_workers: int = 0,
        custom_transforms: Callable | transforms.Compose | None = None,
        repeat_channels: int | None = None,
        to_tensor: bool = True,
        resize: tuple[int, int] | None = None,
    ) -> None:
        r"""
        **Args:**
        - **root** (`str`): the root directory where the original Country211 data 'Country211/' live.
        - **validation_percentage** (`float`): the percentage to randomly split some training data into validation data.
        - **batch_size** (`int`): The batch size in train, val, test dataloader.
        - **num_workers** (`int`): the number of workers for dataloaders.
        - **custom_transforms** (`transform` or `transforms.Compose` or `None`): the custom transforms to apply to ONLY TRAIN dataset. Can be a single transform, composed transforms or no transform. `ToTensor()`, normalize and so on are not included.
        - **repeat_channels** (`int` | `None`): the number of channels to repeat. Default is None, which means no repeat. If not None, it should be an integer.
        - **to_tensor** (`bool`): whether to include `ToTensor()` transform. Default is True.
        - **resize** (`tuple[int, int]` | `None` or list of them): the size to resize the images to. Default is None, which means no resize. If not None, it should be a tuple of two integers.
        """
        super().__init__(
            root=root,
            batch_size=batch_size,
            num_workers=num_workers,
            custom_transforms=custom_transforms,
            repeat_channels=repeat_channels,
            to_tensor=to_tensor,
            resize=resize,
        )

    def prepare_data(self) -> None:
        r"""Download the original Country211 dataset if haven't."""

        Country211Raw(root=self.root, split="train", download=True)
        Country211Raw(root=self.root, split="valid", download=True)
        Country211Raw(root=self.root, split="test", download=True)

        pylogger.debug(
            "The original Country211 dataset has been downloaded to %s.",
            self.root,
        )

    def train_and_val_dataset(self) -> tuple[Dataset, Dataset]:
        """Get the training and validation dataset.

        **Returns:**
        - **train_and_val_dataset** (`tuple[Dataset, Dataset]`): the train and validation dataset.
        """
        dataset_train = Country211Raw(
            root=self.root,
            split="train",
            transform=self.train_and_val_transforms(),
            target_transform=self.target_transform(),
            download=False,
        )

        dataset_val = Country211Raw(
            root=self.root,
            split="valid",
            transform=self.train_and_val_transforms(),
            download=False,
        )

        return dataset_train, dataset_val

    def test_dataset(self) -> Dataset:
        r"""Get the test dataset.

        **Returns:**
        - **test_dataset** (`Dataset`): the test dataset.
        """
        dataset_test = Country211Raw(
            root=self.root,
            split="test",
            transform=self.test_transforms(),
            target_transform=self.target_transform(),
            download=False,
        )

        return dataset_test
