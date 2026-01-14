r"""
The submodule in `cl_datasets` for Rotated MNIST dataset.
"""

__all__ = ["RotatedMNIST"]

import logging
from typing import Callable

import torch
from omegaconf import DictConfig, ListConfig, OmegaConf
from torch.utils.data import Dataset, random_split
from torchvision.datasets import MNIST
from torchvision.transforms import transforms

from clarena.cl_datasets import CLDataset
from clarena.stl_datasets.raw.constants import (
    DATASET_CONSTANTS_MAPPING,
    DatasetConstants,
)

# always get logger for built-in logging in each module
pylogger = logging.getLogger(__name__)


class RotatedMNIST(CLDataset):
    r"""Rotated MNIST dataset. The [MNIST dataset](http://yann.lecun.com/exdb/mnist/) is a collection of handwritten digits. It consists of 60,000 training and 10,000 test images of handwritten digit images (10 classes), each 28x28 grayscale image. Rotated MNIST creates multiple tasks by rotating all images in each task by a fixed angle."""

    original_dataset_python_class: type[Dataset] = MNIST
    r"""The original dataset class."""

    def __init__(
        self,
        root: str,
        num_tasks: int,
        validation_percentage: float,
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
        rotation_degrees: dict[int, float] | list[float] | None = None,
    ) -> None:
        r"""
        **Args:**
        - **root** (`str`): the root directory where the original MNIST data 'MNIST/' live.
        - **num_tasks** (`int`): the maximum number of tasks supported by the CL dataset. This decides the valid task IDs from 1 to `num_tasks`.
        - **validation_percentage** (`float`): the percentage to randomly split some training data into validation data.
        - **batch_size** (`int` | `dict[int, int]`): the batch size for train, val, and test dataloaders.
        If it is a dict, the keys are task IDs and the values are the batch sizes for each task. If it is an `int`, it is the same batch size for all tasks.
        - **num_workers** (`int` | `dict[int, int]`): the number of workers for dataloaders.
        If it is a dict, the keys are task IDs and the values are the number of workers for each task. If it is an `int`, it is the same number of workers for all tasks.
        - **custom_transforms** (`transform` or `transforms.Compose` or `None` or dict of them): the custom transforms to apply ONLY to the TRAIN dataset. Can be a single transform, composed transforms, or no transform. `ToTensor()`, normalization, and rotation are not included.
        If it is a dict, the keys are task IDs and the values are the custom transforms for each task. If it is a single transform or composed transforms, it is applied to all tasks. If it is `None`, no custom transforms are applied.
        - **repeat_channels** (`int` | `None` | dict of them): the number of channels to repeat for each task. Default is `None`, which means no repeat.
        If it is a dict, the keys are task IDs and the values are the number of channels to repeat for each task. If it is an `int`, it is the same number of channels to repeat for all tasks. If it is `None`, no repeat is applied.
        - **to_tensor** (`bool` | `dict[int, bool]`): whether to include the `ToTensor()` transform. Default is `True`.
        If it is a dict, the keys are task IDs and the values are whether to include the `ToTensor()` transform for each task. If it is a single boolean value, it is applied to all tasks.
        - **resize** (`tuple[int, int]` | `None` or dict of them): the size to resize the images to. Default is `None`, which means no resize.
        If it is a dict, the keys are task IDs and the values are the sizes to resize for each task. If it is a single tuple of two integers, it is applied to all tasks. If it is `None`, no resize is applied.
        - **rotation_degrees** (`dict[int, float]` | `list[float]` | `None`): the rotation degrees for each task. If it is a list, its length must match `num_tasks` and it is mapped to task IDs in order. If it is `None`, angles are evenly spaced in [0, 180) across tasks.
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
        )

        self.original_dataset_constants: type[DatasetConstants] = (
            DATASET_CONSTANTS_MAPPING[self.original_dataset_python_class]
        )
        r"""The original dataset constants class."""

        if isinstance(rotation_degrees, (DictConfig, ListConfig)):
            rotation_degrees = OmegaConf.to_container(rotation_degrees)

        if rotation_degrees is None:
            step = 180.0 / num_tasks
            rotation_degrees = {
                t: (t - 1) * step for t in range(1, num_tasks + 1)
            }
        elif isinstance(rotation_degrees, (list, tuple)):
            if len(rotation_degrees) != num_tasks:
                raise ValueError(
                    "rotation_degrees length must match num_tasks."
                )
            rotation_degrees = {
                t: float(rotation_degrees[t - 1])
                for t in range(1, num_tasks + 1)
            }
        elif isinstance(rotation_degrees, dict):
            rotation_degrees = {
                int(task_id): float(angle)
                for task_id, angle in rotation_degrees.items()
            }
        else:
            raise TypeError(
                "rotation_degrees must be a dict, list, or None."
            )

        self.rotation_degrees: dict[int, float] = rotation_degrees
        r"""The rotation degrees for each task."""
        self.rotation_degree_t: float
        r"""The rotation degree for the current task `self.task_id`."""
        self.rotation_transform_t: transforms.RandomRotation
        r"""The rotation transform for the current task `self.task_id`."""

        self.validation_percentage: float = validation_percentage
        r"""The percentage to randomly split some training data into validation data."""

        RotatedMNIST.sanity_check(self)

    def sanity_check(self) -> None:
        r"""Sanity check."""

        expected_keys = set(range(1, self.num_tasks + 1))
        if set(self.rotation_degrees.keys()) != expected_keys:
            raise ValueError(
                "rotation_degrees dict keys must be consecutive integers from 1 to num_tasks."
            )

    def get_cl_class_map(self, task_id: int) -> dict[str | int, int]:
        r"""Get the mapping of classes of task `task_id` to fit continual learning settings `self.cl_paradigm`.

        **Args:**
        - **task_id** (`int`): the task ID to query the CL class map.

        **Returns:**
        - **cl_class_map** (`dict[str | int, int]`): the CL class map of the task. Keys are the original class labels and values are the integer class label for continual learning.
            - If `self.cl_paradigm` is 'TIL' or 'DIL', the mapped class labels of a task should be continuous integers from 0 to the number of classes.
            - If `self.cl_paradigm` is 'CIL', the mapped class labels of a task should be continuous integers from the number of classes of previous tasks to the number of classes of the current task.
        """
        num_classes_t = self.original_dataset_constants.NUM_CLASSES
        class_map_t = self.original_dataset_constants.CLASS_MAP

        if self.cl_paradigm in ["TIL", "DIL"]:
            return {class_map_t[i]: i for i in range(num_classes_t)}
        if self.cl_paradigm == "CIL":
            return {
                class_map_t[i]: i + (task_id - 1) * num_classes_t
                for i in range(num_classes_t)
            }

        raise ValueError(f"Unsupported cl_paradigm: {self.cl_paradigm}")

    def setup_task_id(self, task_id: int) -> None:
        r"""Set up which task's dataset the CL experiment is on. This must be done before `setup()` method is called.

        **Args:**
        - **task_id** (`int`): the target task ID.
        """
        super().setup_task_id(task_id)

        self.mean_t = self.original_dataset_constants.MEAN
        self.std_t = self.original_dataset_constants.STD

        self.rotation_degree_t = self.rotation_degrees[task_id]
        self.rotation_transform_t = transforms.RandomRotation(
            degrees=(self.rotation_degree_t, self.rotation_degree_t),
            fill=0,
        )

    def prepare_data(self) -> None:
        r"""Download the original MNIST dataset if haven't."""

        if self.task_id != 1:
            return  # download all original datasets only at the beginning of first task

        MNIST(root=self.root_t, train=True, download=True)
        MNIST(root=self.root_t, train=False, download=True)

        pylogger.debug(
            "The original MNIST dataset has been downloaded to %s.", self.root_t
        )

    def train_and_val_transforms(self) -> transforms.Compose:
        r"""Transforms for training and validation datasets. Rotation is applied before `ToTensor()` to keep PIL-based rotation.

        **Returns:**
        - **train_and_val_transforms** (`transforms.Compose`): the composed train/val transforms.
        """
        repeat_channels_transform = (
            transforms.Grayscale(num_output_channels=self.repeat_channels_t)
            if self.repeat_channels_t is not None
            else None
        )
        to_tensor_transform = transforms.ToTensor() if self.to_tensor_t else None
        resize_transform = (
            transforms.Resize(self.resize_t) if self.resize_t is not None else None
        )
        normalization_transform = transforms.Normalize(self.mean_t, self.std_t)

        return transforms.Compose(
            list(
                filter(
                    None,
                    [
                        repeat_channels_transform,
                        self.rotation_transform_t,
                        to_tensor_transform,
                        resize_transform,
                        self.custom_transforms_t,
                        normalization_transform,
                    ],
                )
            )
        )  # the order of transforms matters

    def test_transforms(self) -> transforms.Compose:
        r"""Transforms for the test dataset. Rotation is applied before `ToTensor()` to keep PIL-based rotation.

        **Returns:**
        - **test_transforms** (`transforms.Compose`): the composed test transforms.
        """
        repeat_channels_transform = (
            transforms.Grayscale(num_output_channels=self.repeat_channels_t)
            if self.repeat_channels_t is not None
            else None
        )
        to_tensor_transform = transforms.ToTensor() if self.to_tensor_t else None
        resize_transform = (
            transforms.Resize(self.resize_t) if self.resize_t is not None else None
        )
        normalization_transform = transforms.Normalize(self.mean_t, self.std_t)

        return transforms.Compose(
            list(
                filter(
                    None,
                    [
                        repeat_channels_transform,
                        self.rotation_transform_t,
                        to_tensor_transform,
                        resize_transform,
                        normalization_transform,
                    ],
                )
            )
        )  # the order of transforms matters. No custom transforms for test

    def train_and_val_dataset(self) -> tuple[Dataset, Dataset]:
        """Get the training and validation dataset of task `self.task_id`.

        **Returns:**
        - **train_and_val_dataset** (`tuple[Dataset, Dataset]`): the train and validation dataset of task `self.task_id`.
        """
        dataset_train_and_val = MNIST(
            root=self.root_t,
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
        r"""Get the test dataset of task `self.task_id`.

        **Returns:**
        - **test_dataset** (`Dataset`): the test dataset of task `self.task_id`.
        """
        dataset_test = MNIST(
            root=self.root_t,
            train=False,
            transform=self.test_transforms(),
            target_transform=self.target_transform(),
            download=False,
        )

        return dataset_test
