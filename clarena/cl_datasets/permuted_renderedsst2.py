r"""
The submodule in `cl_datasets` for Permuted Rendered SST2 dataset.
"""

__all__ = ["PermutedRenderedSST2"]

import logging
from typing import Callable

from torch.utils.data import Dataset
from torchvision.datasets import RenderedSST2
from torchvision.transforms import transforms

from clarena.cl_datasets import CLClassMapping, CLPermutedDataset

# always get logger for built-in logging in each module
pylogger = logging.getLogger(__name__)


class PermutedRenderedSST2(CLPermutedDataset):
    r"""Permuted Rendered SST2 dataset. The [original Rendered SST2 dataset](https://github.com/openai/CLIP/blob/main/data/rendered-sst2.md) is a collection of 9,613 32x32 RGB images in 2 classes (correspond to positive and negative sentiment)."""

    original_dataset_python_class: type[Dataset] = RenderedSST2

    r"""The original dataset class."""

    def __init__(
        self,
        root: str,
        num_tasks: int,
        batch_size: int | list[int] = 1,
        num_workers: int | list[int] = 0,
        custom_transforms: (
            Callable
            | transforms.Compose
            | None
            | list[Callable | transforms.Compose | None]
        ) = None,
        repeat_channels: int | None | list[int | None] = None,
        to_tensor: bool | list[bool] = True,
        resize: tuple[int, int] | None | list[tuple[int, int] | None] = None,
        permutation_mode: str = "first_channel_only",
        permutation_seeds: list[int] | None = None,
    ) -> None:
        r"""Initialise the Permuted Rendered SST2 dataset object providing the root where data files live.

        **Args:**
        - **root** (`str`): the root directory where the original Rendered SST2 data 'RenderedSST2/' live.
        - **num_tasks** (`int`): the maximum number of tasks supported by the CL dataset.
        - **batch_size** (`int` | `list[int]`): The batch size in train, val, test dataloader. If `list[str]`, it should be a list of integers, each integer is the batch size for each task.
        - **num_workers** (`int` | `list[int]`): the number of workers for dataloaders. If `list[str]`, it should be a list of integers, each integer is the num of workers for each task.
        - **custom_transforms** (`transform` or `transforms.Compose` or `None` or list of them): the custom transforms to apply to ONLY TRAIN dataset. Can be a single transform, composed transforms or no transform. `ToTensor()`, normalise, permute and so on are not included. If it is a list, each item is the custom transforms for each task.
        - **repeat_channels** (`int` | `None` | list of them): the number of channels to repeat for each task. Default is None, which means no repeat. If not None, it should be an integer. If it is a list, each item is the number of channels to repeat for each task.
        - **to_tensor** (`bool` | `list[bool]`): whether to include `ToTensor()` transform. Default is True.
        - **resize** (`tuple[int, int]` | `None` or list of them): the size to resize the images to. Default is None, which means no resize. If not None, it should be a tuple of two integers. If it is a list, each item is the size to resize for each task.
        - **permutation_mode** (`str`): the mode of permutation, should be one of the following:
            1. 'all': permute all pixels.
            2. 'by_channel': permute channel by channel separately. All channels are applied the same permutation order.
            3. 'first_channel_only': permute only the first channel.
        - **permutation_seeds** (`list[int]` or `None`): the seeds for permutation operations used to construct tasks. Make sure it has the same number of seeds as `num_tasks`. Default is None, which creates a list of seeds from 0 to `num_tasks`-1.
        """
        CLPermutedDataset.__init__(
            self,
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
        r"""Download the original Rendered SST2 dataset if haven't."""

        if self.task_id != 1:
            return  # download all original datasets only at the beginning of first task

        RenderedSST2(root=self.root_t, split="train", download=True)
        RenderedSST2(root=self.root_t, split="val", download=True)
        RenderedSST2(root=self.root_t, split="test", download=True)

        pylogger.debug(
            "The original Rendered SST2 dataset has been downloaded to %s.",
            self.root_t,
        )

    def train_and_val_dataset(self) -> tuple[Dataset, Dataset]:
        """Get the training and validation dataset of task `self.task_id`.

        **Returns:**
        - **train_and_val_dataset** (`tuple[Dataset, Dataset]`): the train and validation dataset of task `self.task_id`.
        """
        dataset_train = RenderedSST2(
            root=self.root_t,
            split="train",
            transform=self.train_and_val_transforms(),
            target_transform=CLClassMapping(self.cl_class_map_t),
            download=False,
        )

        dataset_val = RenderedSST2(
            root=self.root_t,
            split="val",
            transform=self.train_and_val_transforms(),
            target_transform=CLClassMapping(self.cl_class_map_t),
            download=False,
        )

        return dataset_train, dataset_val

    def test_dataset(self) -> Dataset:
        r"""Get the test dataset of task `self.task_id`.

        **Returns:**
        - **test_dataset** (`Dataset`): the test dataset of task `self.task_id`.
        """
        dataset_test = RenderedSST2(
            root=self.root_t,
            split="test",
            transform=self.test_transforms(),
            target_transform=CLClassMapping(self.cl_class_map_t),
            download=False,
        )

        return dataset_test
