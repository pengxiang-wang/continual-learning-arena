r"""
The submodule in `stl_datasets` for CelebA dataset.
"""

__all__ = ["CelebA"]

import logging
from typing import Callable

from torch.utils.data import Dataset
from torchvision.datasets import CelebA as CelebARaw
from torchvision.transforms import transforms

from clarena.stl_datasets.base import STLDatasetFromRaw

# always get logger for built-in logging in each module
pylogger = logging.getLogger(__name__)


class CelebA(STLDatasetFromRaw):
    r"""CelebA dataset. The [CelebFaces Attributes Dataset (CelebA)](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) is a large-scale celebrity faces dataset. It consists of 202,599 face images of 10,177 celebrity identities (classes), each 178x218 color image.

    Note that the original CelebA dataset is not a classification dataset but an attributes dataset. We only use the identity of each face as the class label for classification.
    """

    original_dataset_python_class: type[Dataset] = CelebARaw
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
        - **root** (`str`): the root directory where the original CelebA data 'CelebA/' live.
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
        r"""Download the original CelebA dataset if haven't."""

        CelebARaw(root=self.root, split="train", target_type="identity", download=True)
        CelebARaw(root=self.root, split="valid", target_type="identity", download=True)
        CelebARaw(root=self.root, split="test", target_type="identity", download=True)

        pylogger.debug(
            "The original CelebA dataset has been downloaded to %s.", self.root
        )

    def train_and_val_dataset(self) -> tuple[Dataset, Dataset]:
        """Get the training and validation dataset.

        **Returns:**
        - **train_and_val_dataset** (`tuple[Dataset, Dataset]`): the train and validation dataset.
        """
        dataset_train = CelebARaw(
            root=self.root,
            split="train",
            target_type="identity",
            transform=self.train_and_val_transforms(),
            target_transform=self.target_transform(),
            download=False,
        )

        dataset_val = CelebARaw(
            root=self.root,
            split="valid",
            target_type="identity",
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
        dataset_test = CelebARaw(
            root=self.root,
            split="test",
            target_type="identity",
            transform=self.test_transforms(),
            target_transform=self.target_transform(),
            download=False,
        )

        return dataset_test
