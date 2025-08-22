r"""
The submodule in `cl_datasets` for Permuted DTD dataset.
"""

__all__ = ["PermutedDTD"]

import logging
from typing import Callable

from torch.utils.data import Dataset
from torchvision.datasets import DTD
from torchvision.transforms import transforms

from clarena.cl_datasets import CLPermutedDataset
from clarena.utils.transforms import ClassMapping

# always get logger for built-in logging in each module
pylogger = logging.getLogger(__name__)


class PermutedDTD(CLPermutedDataset):
    r"""Permuted DTD dataset. The [DTD dataset](https://www.robots.ox.ac.uk/~vgg/data/dtd/) is a collection of describable texture pictures. It consists of 5,640 images of 47 kinds of textures (classes), each 300x300-640x640 color image."""

    original_dataset_python_class: type[Dataset] = DTD
    r"""The original dataset class."""

    def __init__(
        self,
        root: str,
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
        r"""Initialize the dataset object providing the root where data files live.

        **Args:**
        - **root** (`str`): the root directory where the original DTD data 'DTD/' live.
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
        - **permutation_mode** (`str`): the mode of permutation, should be one of the following:
            1. 'all': permute all pixels.
            2. 'by_channel': permute channel by channel separately. All channels are applied the same permutation order.
            3. 'first_channel_only': permute only the first channel.
        - **permutation_seeds** (`dict[int, int]` | `None`): the dict of seeds for permutation operations used to construct each task. Keys are task IDs and the values are permutation seeds for each task. Default is `None`, which creates a dict of seeds from 0 to `num_tasks`-1.
        """

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

    def prepare_data(self) -> None:
        r"""Download the original DTD dataset if haven't."""

        if self.task_id != 1:
            return  # download all original datasets only at the beginning of first task

        DTD(root=self.root_t, split="train", download=True)
        DTD(root=self.root_t, split="val", download=True)
        DTD(root=self.root_t, split="test", download=True)

        pylogger.debug(
            "The original DTD dataset has been downloaded to %s.", self.root_t
        )

    def train_and_val_dataset(self) -> tuple[Dataset, Dataset]:
        """Get the training and validation dataset of task `self.task_id`.

        **Returns:**
        - **train_and_val_dataset** (`tuple[Dataset, Dataset]`): the train and validation dataset of task `self.task_id`.
        """
        dataset_train = DTD(
            root=self.root_t,
            split="train",
            transform=self.train_and_val_transforms(),
            target_transform=ClassMapping(self.get_cl_class_map(self.task_id)),
            download=False,
        )

        dataset_val = DTD(
            root=self.root_t,
            split="val",
            transform=self.train_and_val_transforms(),
            target_transform=ClassMapping(self.get_cl_class_map(self.task_id)),
            download=False,
        )

        return dataset_train, dataset_val

    def test_dataset(self) -> Dataset:
        r"""Get the test dataset of task `self.task_id`.

        **Returns:**
        - **test_dataset** (`Dataset`): the test dataset of task `self.task_id`.
        """
        dataset_test = DTD(
            root=self.root_t,
            split="test",
            transform=self.test_transforms(),
            target_transform=ClassMapping(self.get_cl_class_map(self.task_id)),
            download=False,
        )

        return dataset_test
