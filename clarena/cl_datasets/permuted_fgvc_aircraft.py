r"""
The submodule in `cl_datasets` for Permuted FGVC-Aircraft dataset.
"""

__all__ = ["PermutedFGVCAircraft"]

import logging
from typing import Callable

from torch.utils.data import Dataset
from torchvision.datasets import FGVCAircraft
from torchvision.transforms import transforms

from clarena.cl_datasets import CLPermutedDataset
from clarena.stl_datasets.raw import (
    FGVCAircraftFamily,
    FGVCAircraftManufacturer,
    FGVCAircraftVariant,
)

# always get logger for built-in logging in each module
pylogger = logging.getLogger(__name__)


class PermutedFGVCAircraft(CLPermutedDataset):
    r"""Permuted FGVC-Aircraft dataset. The [FGVC-Aircraft dataset](https://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/) is a collection of aircraft images. It consists of 10,200 images, each color image.

    FGVC-Aircraft has 3 different class labels by variant, family and manufacturer, which has 102, 70, 41 classes respectively. We support all of them in Permuted FGVC-Aircraft.
    """

    def __init__(
        self,
        root: str,
        annotation_level: str,
        num_tasks: int,
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
        permutation_mode: str = "first_channel_only",
        permutation_seeds: dict[int, int] | None = None,
    ) -> None:
        r"""
        **Args:**
        - **root** (`str`): the root directory where the original FGVCAircraft data 'FGVCAircraft/' live.
        - **annotation_level** (`str`): The annotation level, supports 'variant', 'family' and 'manufacturer'.
        - **num_tasks** (`int`): the maximum number of tasks supported by the CL dataset. This decides the valid task IDs from 1 to `num_tasks`.
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
        - **permutation_mode** (`str`): the mode of permutation; one of:
            1. 'all': permute all pixels.
            2. 'by_channel': permute channel by channel separately. All channels are applied the same permutation order.
            3. 'first_channel_only': permute only the first channel.
        - **permutation_seeds** (`dict[int, int]` | `None`): the dict of seeds for permutation operations used to construct each task. Keys are task IDs and the values are permutation seeds for each task. Default is `None`, which creates a dict of seeds from 0 to `num_tasks`-1.
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
            num_tasks=num_tasks,
            batch_size=batch_size,
            num_workers=num_workers,
            custom_transforms=custom_transforms,
            repeat_channels=repeat_channels,
            to_tensor=to_tensor,
            resize=resize,
            permutation_mode=permutation_mode,
            permutation_seeds=permutation_seeds,
        )

        self.annotation_level: str = annotation_level
        r"""The annotation level, supports 'variant', 'family' and 'manufacturer'."""

    def prepare_data(self) -> None:
        r"""Download the original FGVC-Aircraft dataset if haven't."""

        if self.task_id != 1:
            return  # download all original datasets only at the beginning of first task

        FGVCAircraft(root=self.root_t, split="train", download=True)
        FGVCAircraft(root=self.root_t, split="val", download=True)
        FGVCAircraft(root=self.root_t, split="test", download=True)

        pylogger.debug(
            "The original FGVC-Aircraft dataset has been downloaded to %s.",
            self.root_t,
        )

    def train_and_val_dataset(self) -> tuple[Dataset, Dataset]:
        """Get the training and validation dataset of task `self.task_id`.

        **Returns:**
        - **train_and_val_dataset** (`tuple[Dataset, Dataset]`): the train and validation dataset of task `self.task_id`.
        """
        dataset_train = FGVCAircraft(
            root=self.root_t,
            split="train",
            annotation_level=self.annotation_level,
            transform=self.train_and_val_transforms(),
            target_transform=self.target_transform(),
            download=False,
        )

        dataset_val = FGVCAircraft(
            root=self.root_t,
            split="val",
            annotation_level=self.annotation_level,
            transform=self.train_and_val_transforms(),
            target_transform=self.target_transform(),
            download=False,
        )

        return dataset_train, dataset_val

    def test_dataset(self) -> Dataset:
        r"""Get the test dataset of task `self.task_id`.

        **Returns:**
        - **test_dataset** (`Dataset`): the test dataset of task `self.task_id`.
        """
        dataset_test = FGVCAircraft(
            root=self.root_t,
            split="test",
            annotation_level=self.annotation_level,
            transform=self.test_transforms(),
            target_transform=self.target_transform(),
            download=False,
        )

        return dataset_test
