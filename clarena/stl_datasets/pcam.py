r"""
The submodule in `stl_datasets` for PCAM dataset.
"""

__all__ = ["PCAM"]

import logging
from typing import Callable

from torch.utils.data import Dataset
from torchvision.datasets import PCAM as PCAMRaw
from torchvision.transforms import transforms

from clarena.stl_datasets.base import STLDatasetFromRaw

# always get logger for built-in logging in each module
pylogger = logging.getLogger(__name__)


class PCAM(STLDatasetFromRaw):
    r"""PCAM dataset. The [PCAM dataset](https://github.com/basveeling/pcam) is a collection of medical images of breast cancer. It consists of 327,680 images in 2 classes (benign and malignant), each 96x96 color image."""

    original_dataset_python_class: type[Dataset] = PCAMRaw
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
        - **root** (`str`): the root directory where the original PCAM data 'PCAM/' live.
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
        r"""Download the original PCAM dataset if haven't."""

        PCAMRaw(root=self.root, split="train", download=True)
        PCAMRaw(root=self.root, split="val", download=True)
        PCAMRaw(root=self.root, split="test", download=True)

        pylogger.debug(
            "The original PCAM dataset has been downloaded to %s.", self.root
        )

    def train_and_val_dataset(self) -> tuple[Dataset, Dataset]:
        """Get the training and validation dataset.

        **Returns:**
        - **train_and_val_dataset** (`tuple[Dataset, Dataset]`): the train and validation dataset.
        """
        dataset_train = PCAMRaw(
            root=self.root,
            split="train",
            transform=self.train_and_val_transforms(),
            target_transform=self.target_transform(),
            download=False,
        )

        dataset_val = PCAMRaw(
            root=self.root,
            split="val",
            transform=self.train_and_val_transforms(),
            download=False,
        )

        return dataset_train, dataset_val

    def test_dataset(self) -> Dataset:
        r"""Get the test dataset.

        **Returns:**
        - **test_dataset** (`Dataset`): the test dataset.
        """
        dataset_test = PCAMRaw(
            root=self.root,
            split="test",
            transform=self.test_transforms(),
            target_transform=self.target_transform(),
            download=False,
        )

        return dataset_test
