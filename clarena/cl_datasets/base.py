r"""
The submodule in `cl_datasets` for CL dataset bases.
"""

__all__ = [
    "CLDataset",
    "CLPermutedDataset",
    "CLClassMapping",
    "Permute",
    "CLSplitDataset",
    "CLCombinedDataset",
    "JointDataset",
]

import ast
import logging
import types
from abc import abstractmethod
from copy import deepcopy
from typing import Any, Callable

import torch
from lightning import LightningDataModule
from omegaconf import ListConfig
from torch.utils.data import ConcatDataset, DataLoader, Dataset
from torchvision import transforms

from clarena.cl_datasets.original.constants import (
    DATASET_CONSTANTS_MAPPING,
    DatasetConstants,
)
from clarena.utils import str_to_class

# always get logger for built-in logging in each module
pylogger = logging.getLogger(__name__)


class CLDataset(LightningDataModule):
    r"""The base class of continual learning datasets, inherited from `LightningDataModule`."""

    def __init__(
        self,
        root: str | list[str],
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
    ) -> None:
        r"""Initialise the CL dataset object providing the root where data files live.

        **Args:**
        - **root** (`str` | `list[str]`): the root directory where the original data files for constructing the CL dataset physically live. If `list[str]`, it should be a list of strings, each string is the root directory for each task.
        - **num_tasks** (`int`): the maximum number of tasks supported by the CL dataset.
        - **batch_size** (`int` | `list[int]`): The batch size in train, val, test dataloader. If `list[str]`, it should be a list of integers, each integer is the batch size for each task.
        - **num_workers** (`int` | `list[int]`): the number of workers for dataloaders. If `list[str]`, it should be a list of integers, each integer is the num of workers for each task.
        - **custom_transforms** (`transform` or `transforms.Compose` or `None` or list of them): the custom transforms to apply to ONLY TRAIN dataset. Can be a single transform, composed transforms or no transform. `ToTensor()`, normalise, permute and so on are not included. If it is a list, each item is the custom transforms for each task.
        - **repeat_channels** (`int` | `None` | list of them): the number of channels to repeat for each task. Default is None, which means no repeat. If not None, it should be an integer. If it is a list, each item is the number of channels to repeat for each task.
        - **to_tensor** (`bool` | `list[bool]`): whether to include `ToTensor()` transform. Default is True.
        - **resize** (`tuple[int, int]` | `None` or list of them): the size to resize the images to. Default is None, which means no resize. If not None, it should be a tuple of two integers. If it is a list, each item is the size to resize for each task.
        """
        LightningDataModule.__init__(self)

        self.root: str | list[str] = (
            root if isinstance(root, ListConfig) else [root] * num_tasks
        )
        r"""Store the list of root directory of the original data files for each task. """
        self.root_t: str
        r"""Store the root directory of the original data files for the current task `self.task_id`. Used when constructing the dataset."""
        self.num_tasks: int = num_tasks
        r"""Store the maximum number of tasks supported by the dataset."""
        self.batch_size: int | list[int] = (
            batch_size
            if isinstance(batch_size, ListConfig)
            else [batch_size] * num_tasks
        )
        r"""Store the list of batch size for each task. """
        self.batch_size_t: int
        r"""Store the batch size for the current task `self.task_id`. Used when constructing train, val, test dataloader."""
        self.num_workers: int | list[int] = (
            num_workers
            if isinstance(num_workers, ListConfig)
            else [num_workers] * num_tasks
        )
        r"""Store the list of number of workers for each task. """
        self.num_workers_t: int
        r"""Store the number of workers for the current task `self.task_id`. Used when constructing train, val, test dataloader."""
        self.custom_transforms: (
            Callable
            | transforms.Compose
            | None
            | list[Callable | transforms.Compose | None]
        ) = (
            custom_transforms
            if isinstance(custom_transforms, ListConfig)
            else [custom_transforms] * num_tasks
        )
        r"""Store the list of custom transforms for each task. """
        self.custom_transforms_t: Callable | transforms.Compose | None
        r"""Store the custom transforms for the current task `self.task_id`. Used when constructing the dataset."""
        self.repeat_channels: int | None | list[int | None] = (
            repeat_channels
            if isinstance(repeat_channels, ListConfig)
            else [repeat_channels] * num_tasks
        )
        r"""Store the list of number of channels to repeat for each task. """
        self.repeat_channels_t: int | None
        r"""Store the number of channels to repeat for the current task `self.task_id`. Used when constructing the transforms."""
        self.to_tensor: bool | list[bool] = (
            to_tensor if isinstance(to_tensor, ListConfig) else [to_tensor] * num_tasks
        )
        r"""Store the list of to_tensor flag for each task. """
        self.to_tensor_t: bool
        r"""Store the to_tensor flag for the current task `self.task_id`. Used when constructing the transforms."""
        self.resize: tuple[int, int] | None | list[tuple[int, int] | None] = (
            [ast.literal_eval(rs) if rs else None for rs in resize]
            if isinstance(resize, ListConfig)
            else [ast.literal_eval(resize) if resize else None] * num_tasks
        )
        r"""Store the list of size to resize for each task. """
        self.resize_t: tuple[int, int] | None
        r"""Store the size to resize for the current task `self.task_id`. Used when constructing the transforms."""
        self.mean_t: float
        r"""Store the mean values for normalisation for the current task `self.task_id`. Used when constructing the transforms."""
        self.std_t: float
        r"""Store the standard deviation values for normalisation for the current task `self.task_id`. Used when constructing the transforms."""

        self.task_id: int
        r"""Task ID counter indicating which task is being processed. Self updated during the task loop. Starting from 1. """
        self.seen_task_ids: list[int] = []
        r"""The list of task IDs that have been seen in the experiment."""

        self.cl_paradigm: str
        r"""Store the continual learning paradigm, either 'TIL' (Task-Incremental Learning) or 'CIL' (Class-Incremental Learning). Gotten from `set_cl_paradigm` and used to define the CL class map."""

        self.num_classes_t: int
        r"""The number of classes in each task. """
        self.class_map_t: dict[int, str | int]
        r"""Store the class map for the current task `self.task_id`. The key is the integer class label, and the value is the original class label. It is used to get the original class label from the integer class label."""
        self.cl_class_map_t: dict[str | int, int]
        r"""Store the CL class map for the current task `self.task_id`. """

        self.dataset_train_t: Any
        r"""The training dataset object. Can be a PyTorch Dataset object or any other dataset object."""
        self.dataset_val_t: Any
        r"""The validation dataset object. Can be a PyTorch Dataset object or any other dataset object."""
        self.dataset_test: dict[str, Any] = {}
        r"""The dictionary to store test dataset object of each task. Keys are task IDs (string type) and values are the dataset objects. Can be PyTorch Dataset objects or any other dataset objects."""

        CLDataset.sanity_check(self)

    def sanity_check(self) -> None:
        r"""Check the sanity of the arguments."""
        pass

    @abstractmethod
    def cl_class_map(self, task_id: int) -> dict[str | int, int]:
        r"""The mapping of classes of task `task_id` to fit continual learning settings `self.cl_paradigm`. It must be implemented by subclasses.

        **Args:**
        - **task_id** (`int`): The task ID to query CL class map.

        **Returns:**
        - **cl_class_map**(`dict[str | int, int]`): the CL class map of the task. Key is original class label, value is integer class label for continual learning.
            - If `self.cl_paradigm` is 'TIL', the mapped class labels of a task should be continuous integers from 0 to the number of classes.
            - If `self.cl_paradigm` is 'CIL', the mapped class labels of a task should be continuous integers from the number of classes of previous tasks to the number of classes of the current task.
        """

    @abstractmethod
    def prepare_data(self) -> None:
        r"""Use this to download and prepare data. It must be implemented by subclasses, regulated by `LightningDatamodule`. This method is called at the beginning of each task."""

    def setup(self, stage: str) -> None:
        r"""Set up the dataset for different stages. This method is called at the beginning of each task.

        **Args:**
        - **stage** (`str`): the stage of the experiment. Should be one of the following:
            - 'fit': training and validation dataset of current task `self.task_id` should be assigned to `self.dataset_train_t` and `self.dataset_val_t`.
            - 'test': a list of test dataset of all seen tasks (from task 0 to `self.task_id`) should be assigned to `self.dataset_test`.
        """
        if stage == "fit":
            # these two stages must be done together because a sanity check for validation is conducted before training
            pylogger.debug(
                "Construct train and validation dataset for task %d...", self.task_id
            )

            self.dataset_train_t, self.dataset_val_t = self.train_and_val_dataset()

            pylogger.debug(
                "Train and validation dataset for task %d are ready.", self.task_id
            )

        if stage == "test":

            pylogger.debug("Construct test dataset for task %d...", self.task_id)

            self.dataset_test[f"{self.task_id}"] = self.test_dataset()

            pylogger.debug("Test dataset for task %d are ready.", self.task_id)

    def setup_task_id(self, task_id: int) -> None:
        r"""Set up which task's dataset the CL experiment is on. This must be done before `setup()` method is called.

        **Args:**
        - **task_id** (`int`): the target task ID.
        """

        self.task_id = task_id
        self.seen_task_ids.append(task_id)

        self.root_t = self.root[task_id - 1]
        self.batch_size_t = self.batch_size[task_id - 1]
        self.num_workers_t = self.num_workers[task_id - 1]
        self.custom_transforms_t = self.custom_transforms[task_id - 1]
        self.repeat_channels_t = self.repeat_channels[task_id - 1]
        self.to_tensor_t = self.to_tensor[task_id - 1]
        self.resize_t = self.resize[task_id - 1]

        self.cl_class_map_t = self.cl_class_map(task_id)

    def set_cl_paradigm(self, cl_paradigm: str) -> None:
        r"""Set the continual learning paradigm to `self.cl_paradigm`. It is used to define the CL class map.

        **Args:**
        - **cl_paradigm** (`str`): the continual learning paradigmeither 'TIL' (Task-Incremental Learning) or 'CIL' (Class-Incremental Learning).
        """
        self.cl_paradigm = cl_paradigm

    def train_and_val_transforms(self) -> transforms.Compose:
        r"""Transforms generator for train and validation dataset incorporating the custom transforms with basic transforms like `normalisation` and `ToTensor()`. It is a handy tool to use in subclasses when constructing the dataset.

        **Returns:**
        - **train_and_val_transforms** (`transforms.Compose`): the composed training transforms.
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
        normalisation_transform = transforms.Normalize(self.mean_t, self.std_t)

        return transforms.Compose(
            list(
                filter(
                    None,
                    [
                        repeat_channels_transform,
                        to_tensor_transform,
                        resize_transform,
                        self.custom_transforms_t,
                        normalisation_transform,
                    ],
                )
            )
        )  # the order of transforms matters

    def test_transforms(self) -> transforms.Compose:
        r"""Transforms generator for test dataset. Only basic transforms like `normalisation` and `ToTensor()` are included. It is a handy tool to use in subclasses when constructing the dataset.

        **Returns:**
        - **test_transforms** (`transforms.Compose`): the composed training transforms.
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
        normalisation_transform = transforms.Normalize(self.mean_t, self.std_t)

        return transforms.Compose(
            list(
                filter(
                    None,
                    [
                        repeat_channels_transform,
                        to_tensor_transform,
                        resize_transform,
                        normalisation_transform,
                    ],
                )
            )
        )  # the order of transforms matters

    @abstractmethod
    def train_and_val_dataset(self) -> Any:
        r"""Get the training and validation dataset of task `self.task_id`. It must be implemented by subclasses.

        **Returns:**
        - **train_and_val_dataset** (`Any`): the train and validation dataset of task `self.task_id`.
        """

    @abstractmethod
    def test_dataset(self) -> Any:
        """Get the test dataset of task `self.task_id`. It must be implemented by subclasses.

        **Returns:**
        - **test_dataset** (`Any`): the test dataset of task `self.task_id`.
        """

    def train_dataloader(self) -> DataLoader:
        r"""DataLoader generator for stage train of task `self.task_id`. It is automatically called before training.

        **Returns:**
        - **train_dataloader** (`Dataloader`): the train DataLoader of task `self.task_id`.
        """

        pylogger.debug("Construct train dataloader for task %d...", self.task_id)

        return DataLoader(
            dataset=self.dataset_train_t,
            batch_size=self.batch_size_t,
            shuffle=True,  # shuffle train batch to prevent overfitting
            num_workers=self.num_workers_t,
        )

    def val_dataloader(self) -> DataLoader:
        r"""DataLoader generator for stage validate. It is automatically called before validation.

        **Returns:**
        - **val_dataloader** (`Dataloader`): the validation DataLoader of task `self.task_id`.
        """

        pylogger.debug("Construct validation dataloader for task %d...", self.task_id)

        return DataLoader(
            dataset=self.dataset_val_t,
            batch_size=self.batch_size_t,
            shuffle=False,  # don't have to shuffle val or test batch
            num_workers=self.num_workers_t,
        )

    def test_dataloader(self) -> dict[str, DataLoader]:
        r"""DataLoader generator for stage test. It is automatically called before testing.

        **Returns:**
        - **test_dataloader** (`dict[str, DataLoader]`): the test DataLoader dict of `self.task_id` and all tasks before (as the test is conducted on all seen tasks). Keys are task IDs (string type) and values are the DataLoaders.
        """

        pylogger.debug("Construct test dataloader for task %d...", self.task_id)

        return {
            f"{task_id}": DataLoader(
                dataset=dataset_test_t,
                batch_size=self.batch_size_t,
                shuffle=False,  # don't have to shuffle val or test batch
                num_workers=self.num_workers_t,
            )
            for task_id, dataset_test_t in self.dataset_test.items()
        }


class CLPermutedDataset(CLDataset):
    r"""The base class of continual learning datasets which are constructed as permutations from an original dataset, inherited from `CLDataset`."""

    original_dataset_python_class: type[Dataset]
    r"""The original dataset class. It must be provided in subclasses."""

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
        r"""Initialise the CL dataset object providing the root where data files live.

        **Args:**
        - **root** (`str`): the root directory where the original data files for constructing the CL dataset physically live.
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
        - **permutation_seeds** (`list[int]` or `None`): the seeds for permutation operations used to construct tasks. Make sure it has the same number of seeds as `num_tasks`. Default is None, which creates a list of seeds from 1 to `num_tasks`.
        """
        CLDataset.__init__(
            self,
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
        r"""The original dataset constants class. """

        self.permutation_mode: str = permutation_mode
        r"""Store the mode of permutation. Used when permutation operations used to construct tasks. """

        self.permutation_seeds: list[int] = (
            permutation_seeds if permutation_seeds else list(range(num_tasks))
        )
        r"""Store the permutation seeds for all tasks. Use when permutation operations used to construct tasks. """

        self.permutation_seed_t: int
        r"""Store the permutation seed for the current task `self.task_id`."""
        self.permute_transform_t: Permute
        r"""Store the permutation transform for the current task `self.task_id`. """

        CLPermutedDataset.sanity_check(self)

    def sanity_check(self) -> None:
        r"""Check the sanity of the arguments.

        **Raises:**
        - **ValueError**: when the `permutation_seeds` is not equal to `num_tasks`, or the `permutation_mode` is not one of the valid options.
        """
        if self.permutation_seeds and self.num_tasks != len(self.permutation_seeds):
            raise ValueError(
                "The number of permutation seeds is not equal to number of tasks!"
            )
        if self.permutation_mode not in ["all", "by_channel", "first_channel_only"]:
            raise ValueError(
                "The permutation_mode should be one of 'all', 'by_channel', 'first_channel_only'."
            )

    def cl_class_map(self, task_id: int) -> dict[str | int, int]:
        r"""The mapping of classes of task `task_id` to fit continual learning settings `self.cl_paradigm`.

        **Args:**
        - **task_id** (`int`): The task ID to query CL class map.

        **Returns:**
        - **cl_class_map**(`dict[str | int, int]`): the CL class map of the task. Key is original class label, value is integer class label for continual learning.
            - If `self.cl_paradigm` is 'TIL', the mapped class labels of a task should be continuous integers from 0 to the number of classes.
            - If `self.cl_paradigm` is 'CIL', the mapped class labels of a task should be continuous integers from the number of classes of previous tasks to the number of classes of the current task.
        """
        if self.cl_paradigm == "TIL":
            return {self.class_map_t[i]: i for i in range(self.num_classes_t)}
        if self.cl_paradigm == "CIL":
            return {
                self.class_map_t[i]: i + (task_id - 1) * self.num_classes_t
                for i in range(self.num_classes_t)
            }

    def setup_task_id(self, task_id: int) -> None:
        r"""Set up which task's dataset the CL experiment is on. This must be done before `setup()` method is called.

        **Args:**
        - **task_id** (`int`): the target task ID.
        """
        self.num_classes_t = (
            self.original_dataset_constants.NUM_CLASSES
        )  # the same with the original dataset
        self.class_map_t = (
            self.original_dataset_constants.CLASS_MAP
        )  # the same with the original dataset

        CLDataset.setup_task_id(self, task_id)

        self.mean_t = (
            self.original_dataset_constants.MEAN
        )  # the same with the original dataset
        self.std_t = (
            self.original_dataset_constants.STD
        )  # the same with the original dataset

        num_channels = (
            self.original_dataset_constants.NUM_CHANNELS
            if self.repeat_channels_t is None
            else self.repeat_channels_t
        )

        if (
            hasattr(self.original_dataset_constants, "IMG_SIZE")
            or self.resize_t is not None
        ):
            img_size = (
                self.original_dataset_constants.IMG_SIZE
                if self.resize_t is None
                else torch.Size(self.resize_t)
            )
        else:
            raise AttributeError(
                "The original dataset has different image sizes. Please resize the images to a fixed size by specifying hyperparameter: resize."
            )

        # set up the permutation transform
        self.permutation_seed_t = self.permutation_seeds[task_id - 1]
        self.permute_transform_t = Permute(
            num_channels=num_channels,
            img_size=img_size,
            mode=self.permutation_mode,
            seed=self.permutation_seed_t,
        )

    def train_and_val_transforms(self) -> transforms.Compose:
        r"""Transforms generator for train and validation dataset incorporating the custom transforms with basic transforms like `normalisation` and `ToTensor()`. In permuted CL datasets, permute transform also applies. It is a handy tool to use in subclasses when constructing the dataset.

        **Returns:**
        - **train_and_val_transforms** (`transforms.Compose`): the composed training transforms.
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
        normalisation_transform = transforms.Normalize(self.mean_t, self.std_t)

        return transforms.Compose(
            list(
                filter(
                    None,
                    [
                        repeat_channels_transform,
                        to_tensor_transform,
                        resize_transform,
                        self.permute_transform_t,
                        self.custom_transforms_t,
                        normalisation_transform,
                    ],
                )
            )
        )  # the order of transforms matters

    def test_transforms(self) -> transforms.Compose:
        r"""Transforms generator for test dataset. Only basic transforms like `normalisation` and `ToTensor()` are included. It is a handy tool to use in subclasses when constructing the dataset.

        **Returns:**
        - **test_transforms** (`transforms.Compose`): the composed training transforms.
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
        normalisation_transform = transforms.Normalize(self.mean_t, self.std_t)

        return transforms.Compose(
            list(
                filter(
                    None,
                    [
                        repeat_channels_transform,
                        to_tensor_transform,
                        resize_transform,
                        self.permute_transform_t,
                        normalisation_transform,
                    ],
                )
            )
        )  # the order of transforms matters


class CLSplitDataset(CLDataset):
    r"""The base class of continual learning datasets, which are constructed as permutations from an original dataset, inherited from `CLDataset`."""

    original_dataset_python_class: type[Dataset]
    r"""The original dataset class. It must be provided in subclasses."""

    def __init__(
        self,
        root: str,
        class_split: list[list[int]],
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
    ) -> None:
        r"""Initialise the CL dataset object providing the root where data files live.

        **Args:**
        - **root** (`str`): the root directory where the original data files for constructing the CL dataset physically live.
        - **class_split** (`list[list[int]]`): the class split for each task. Each element in the list is a list of class labels (integers starting from 0) to split for a task.
        - **batch_size** (`int` | `list[int]`): The batch size in train, val, test dataloader. If `list[str]`, it should be a list of integers, each integer is the batch size for each task.
        - **num_workers** (`int` | `list[int]`): the number of workers for dataloaders. If `list[str]`, it should be a list of integers, each integer is the num of workers for each task.
        - **custom_transforms** (`transform` or `transforms.Compose` or `None` or list of them): the custom transforms to apply to ONLY TRAIN dataset. Can be a single transform, composed transforms or no transform. `ToTensor()`, normalise, permute and so on are not included. If it is a list, each item is the custom transforms for each task.
        - **repeat_channels** (`int` | `None` | list of them): the number of channels to repeat for each task. Default is None, which means no repeat. If not None, it should be an integer. If it is a list, each item is the number of channels to repeat for each task.
        - **to_tensor** (`bool` | `list[bool]`): whether to include `ToTensor()` transform. Default is True.
        - **resize** (`tuple[int, int]` | `None` or list of them): the size to resize the images to. Default is None, which means no resize. If not None, it should be a tuple of two integers. If it is a list, each item is the size to resize for each task.
        """
        CLDataset.__init__(
            self,
            root=root,
            num_tasks=len(class_split),
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
        r"""The original dataset constants class. """

        self.class_split: list[list[int]] = class_split
        r"""Store the class split for each task. Used when constructing the split dataset."""

        CLSplitDataset.sanity_check(self)

    def sanity_check(self) -> None:
        r"""Check the sanity of the arguments.

        **Raises:**
        - **ValueError**: when the length of `class_split` is not equal to `num_tasks`.
        - **ValueError**: when any of the lists in `class_split` has less than 2 elements. A classification task must have less than 2 classes.
        """
        if len(self.class_split) != self.num_tasks:
            raise ValueError(
                "The length of class split is not equal to number of tasks!"
            )
        if any(len(split) < 2 for split in self.class_split):
            raise ValueError("Each class split must contain at least 2 elements!")

    def cl_class_map(self, task_id: int) -> dict[str | int, int]:
        r"""The mapping of classes of task `task_id` to fit continual learning settings `self.cl_paradigm`.

        **Args:**
        - **task_id** (`int`): The task ID to query CL class map.

        **Returns:**
        - **cl_class_map**(`dict[str | int, int]`): the CL class map of the task. Key is original class label, value is integer class label for continual learning.
            - If `self.cl_paradigm` is 'TIL', the mapped class labels of a task should be continuous integers from 0 to the number of classes.
            - If `self.cl_paradigm` is 'CIL', the mapped class labels of a task should be continuous integers from the number of classes of previous tasks to the number of classes of the current task.
        """
        if self.cl_paradigm == "TIL":
            return {
                self.class_map_t[self.class_split[task_id - 1][i]]: i
                for i in range(self.num_classes_t)
            }
        if self.cl_paradigm == "CIL":
            num_classes_previous = sum(
                [len(self.class_split[i]) for i in range(self.task_id - 1)]
            )
            return {
                self.class_map_t[self.class_split[task_id - 1][i]]: num_classes_previous
                + i
                for i in range(self.num_classes_t)
            }

    def setup_task_id(self, task_id: int) -> None:
        r"""Set up which task's dataset the CL experiment is on. This must be done before `setup()` method is called.

        **Args:**
        - **task_id** (`int`): the target task ID.
        """
        self.num_classes_t = len(
            self.class_split[task_id - 1]
        )  # the number of classes in the current task, i.e. the length of the class split
        self.class_map_t = (
            self.original_dataset_constants.CLASS_MAP
        )  # the same with the original dataset

        CLDataset.setup_task_id(self, task_id)

        self.mean_t = (
            self.original_dataset_constants.MEAN
        )  # the same with the original dataset
        self.std_t = (
            self.original_dataset_constants.STD
        )  # the same with the original dataset

    @abstractmethod
    def get_subset_of_classes(self, dataset: Dataset) -> Dataset:
        r"""Get a subset of classes from the dataset of current classes of `self.task_id`. It is used when constructing the split. It must be implemented by subclasses.

        **Args:**
        - **dataset** (`Dataset`): the dataset to retrieve subset from.

        **Returns:**
        - **subset** (`Dataset`): the subset of classes from the dataset.
        """


class CLCombinedDataset(CLDataset):
    r"""The base class of continual learning datasets which are constructed as combinations of several original datasets (one dataset for one task), inherited from `CLDataset`."""

    def __init__(
        self,
        datasets: list[str],
        root: list[str],
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
    ) -> None:
        r"""Initialise the CL dataset object providing the root where data files live.

        **Args:**
        - **datasets** (`list[str]`): the list of dataset class paths for each task. Each element in the list must be a string referring to a valid PyTorch Dataset class.
        - **root** (`list[str]`): the list of root directory where the original data files for constructing the CL dataset physically live.
        - **batch_size** (`int` | `list[int]`): The batch size in train, val, test dataloader. If `list[str]`, it should be a list of integers, each integer is the batch size for each task.
        - **num_workers** (`int` | `list[int]`): the number of workers for dataloaders. If `list[str]`, it should be a list of integers, each integer is the num of workers for each task.
        - **custom_transforms** (`transform` or `transforms.Compose` or `None` or list of them): the custom transforms to apply to ONLY TRAIN dataset. Can be a single transform, composed transforms or no transform. `ToTensor()`, normalise, permute and so on are not included. If it is a list, each item is the custom transforms for each task.
        - **repeat_channels** (`int` | `None` | list of them): the number of channels to repeat for each task. Default is None, which means no repeat. If not None, it should be an integer. If it is a list, each item is the number of channels to repeat for each task.
        - **to_tensor** (`bool` | `list[bool]`): whether to include `ToTensor()` transform. Default is True.
        - **resize** (`tuple[int, int]` | `None` or list of them): the size to resize the images to. Default is None, which means no resize. If not None, it should be a tuple of two integers. If it is a list, each item is the size to resize for each task.
        """
        CLDataset.__init__(
            self,
            root=root,
            num_tasks=len(datasets),
            batch_size=batch_size,
            num_workers=num_workers,
            custom_transforms=custom_transforms,
            repeat_channels=repeat_channels,
            to_tensor=to_tensor,
            resize=resize,
        )

        self.original_dataset_python_classes: list[Dataset] = [
            str_to_class(dataset_class_path) for dataset_class_path in datasets
        ]
        r"""Store the list of dataset classes for each task."""
        self.original_dataset_python_class_t: Dataset
        r"""Store the dataset class for the current task `self.task_id`."""
        self.original_dataset_constants_t: type[DatasetConstants]
        r"""The original dataset constants class for the current task `self.task_id`. """

    def cl_class_map(self, task_id: int) -> dict[str | int, int]:
        r"""The mapping of classes of task `task_id` to fit continual learning settings `self.cl_paradigm`.

        **Args:**
        - **task_id** (`int`): The task ID to query CL class map.

        **Returns:**
        - **cl_class_map**(`dict[str | int, int]`): the CL class map of the task. Key is original class label, value is integer class label for continual learning.
            - If `self.cl_paradigm` is 'TIL', the mapped class labels of a task should be continuous integers from 0 to the number of classes.
            - If `self.cl_paradigm` is 'CIL', the mapped class labels of a task should be continuous integers from the number of classes of previous tasks to the number of classes of the current task.
        """

        if self.cl_paradigm == "TIL":
            return {self.class_map_t[i]: i for i in range(self.num_classes_t)}
        if self.cl_paradigm == "CIL":
            num_classes_previous = sum(
                [
                    DATASET_CONSTANTS_MAPPING[
                        self.original_dataset_python_classes[i]
                    ].NUM_CLASSES
                    for i in range(self.task_id - 1)
                ]
            )
            return {
                self.class_map_t[i]: num_classes_previous + i
                for i in range(self.num_classes_t)
            }

    def setup_task_id(self, task_id: int) -> None:
        r"""Set up which task's dataset the CL experiment is on. This must be done before `setup()` method is called.

        **Args:**
        - **task_id** (`int`): the target task ID.
        """

        self.original_dataset_python_class_t = self.original_dataset_python_classes[
            task_id - 1
        ]

        self.original_dataset_constants_t: type[DatasetConstants] = (
            DATASET_CONSTANTS_MAPPING[self.original_dataset_python_class_t]
        )
        self.num_classes_t = (
            self.original_dataset_constants_t.NUM_CLASSES
        )  # the same with the task's original dataset
        self.class_map_t = (
            self.original_dataset_constants_t.CLASS_MAP
        )  # the same with the task's original dataset

        CLDataset.setup_task_id(self, task_id)

        self.mean_t = self.original_dataset_constants_t.MEAN
        self.std_t = self.original_dataset_constants_t.STD


class CLClassMapping:
    r"""CL Class mapping to dataset labels. Used as a PyTorch target Transform."""

    def __init__(self, cl_class_map: dict[str | int, int]) -> None:
        r"""Initialise the CL class mapping transform object from the CL class map of a task.

        **Args:**
        - **cl_class_map** (`dict[str | int, int]`): the CL class map for a task.
        """
        self.cl_class_map = cl_class_map

    def __call__(self, target: torch.Tensor) -> torch.Tensor:
        r"""The CL class mapping transform to dataset labels. It is defined as a callable object like a PyTorch Transform.

        **Args:**
        - **target** (`Tensor`): the target tensor.

        **Returns:**
        - **transformed_target** (`Tensor`): the transformed target tensor.
        """

        target = int(
            target
        )  # convert to int if it is a tensor to avoid keyerror in map
        return self.cl_class_map[target]


class Permute:
    r"""Permutation operation to image. Used to construct permuted CL dataset.

    Used as a PyTorch Dataset Transform.
    """

    def __init__(
        self,
        num_channels: int,
        img_size: torch.Size,
        mode: str = "first_channel_only",
        seed: int | None = None,
    ) -> None:
        r"""Initialise the Permute transform object. The permutation order is constructed in the initialisation to save runtime.

        **Args:**
        - **num_channels** (`int`): the number of channels in the image.
        - **img_size** (`torch.Size`): the size of the image to be permuted.
        - **mode** (`str`): the mode of permutation, shouble be one of the following:
            - 'all': permute all pixels.
            - 'by_channel': permute channel by channel separately. All channels are applied the same permutation order.
            - 'first_channel_only': permute only the first channel.
        - **seed** (`int` or `None`): seed for permutation operation. If None, the permutation will use a default seed from PyTorch generator.
        """
        self.mode = mode
        r"""Store the mode of permutation."""

        # get generator for permutation
        torch_generator = torch.Generator()
        if seed:
            torch_generator.manual_seed(seed)

        # calculate the number of pixels from the image size
        if self.mode == "all":
            num_pixels = num_channels * img_size[0] * img_size[1]
        elif self.mode == "by_channel" or "first_channel_only":
            num_pixels = img_size[0] * img_size[1]

        self.permute: torch.Tensor = torch.randperm(
            num_pixels, generator=torch_generator
        )
        r"""The permutation order, a `Tensor` permuted from [1,2, ..., `num_pixels`] with the given seed. It is the core element of permutation operation."""

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        r"""The permutation operation to image is defined as a callable object like a PyTorch Transform.

        **Args:**
        - **img** (`Tensor`): image to be permuted. Must match the size of `img_size` in the initialisation.

        **Returns:**
        - **img_permuted** (`Tensor`): the permuted image.
        """

        if self.mode == "all":

            img_flat = img.view(
                -1
            )  # flatten the whole image to 1d so that it can be applied 1d permuted order
            img_flat_permuted = img_flat[self.permute]  # conduct permutation operation
            img_permuted = img_flat_permuted.view(
                img.size()
            )  # return to the original image shape
            return img_permuted

        if self.mode == "by_channel":

            permuted_channels = []
            for i in range(img.size(0)):
                # act on every channel
                channel_flat = img[i].view(
                    -1
                )  # flatten the channel to 1d so that it can be applied 1d permuted order
                channel_flat_permuted = channel_flat[
                    self.permute
                ]  # conduct permutation operation
                channel_permuted = channel_flat_permuted.view(
                    img[0].size()
                )  # return to the original channel shape
                permuted_channels.append(channel_permuted)
            img_permuted = torch.stack(
                permuted_channels
            )  # stack the permuted channels to restore the image
            return img_permuted

        if self.mode == "first_channel_only":

            first_channel_flat = img[0].view(
                -1
            )  # flatten the first channel to 1d so that it can be applied 1d permuted order
            first_channel_flat_permuted = first_channel_flat[
                self.permute
            ]  # conduct permutation operation
            first_channel_permuted = first_channel_flat_permuted.view(
                img[0].size()
            )  # return to the original channel shape

            img_permuted = img.clone()
            img_permuted[0] = first_channel_permuted

            return img_permuted


class JointDataset(LightningDataModule):
    r"""The class of joint datasets for the joint learning (JL) experiment, inherited from `LightningDataModule`. The joint dataset is a combination of all tasks' datasets. It is used to train the model on all tasks at once.

    This class is dynamically constructed from the CL dataset class.
    """

    def __init__(
        self, cl_dataset: CLDataset, batch_size: int, num_workers: int
    ) -> None:
        r"""Initialise the joint dataset object.

        **Args:**
        - **cl_dataset** (`CLDataset`): the CL dataset object to be used for constructing the joint dataset.
        - **batch_size** (`int`): the batch size in train, val, test dataloader.
        - **num_workers** (`int`): the number of workers for dataloaders.
        """

        LightningDataModule.__init__(self)

        self.cl_dataset: CLDataset = cl_dataset
        r"""Store the CL dataset object."""

        self.batch_size: int = batch_size
        r"""Store the batch size for dataloaders."""

        self.num_workers: int = num_workers
        r"""Store the number of workers for dataloaders."""

        self.dataset_train: ConcatDataset
        r"""The dictionary to store training dataset object of each task. Keys are task IDs (string type) and values are the dataset objects. Can be PyTorch Dataset objects or any other dataset objects."""
        self.dataset_val: dict[str, Any] = {}
        r"""The dictionary to store validation dataset object of each task. Keys are task IDs (string type) and values are the dataset objects. Can be PyTorch Dataset objects or any other dataset objects."""
        self.dataset_test: dict[str, Any] = {}
        r"""The dictionary to store test dataset object of each task. Keys are task IDs (string type) and values are the dataset objects. Can be PyTorch Dataset objects or any other dataset objects."""

    def prepare_data(self) -> None:
        r"""Download and prepare data."""
        self.cl_dataset.prepare_data()

    def setup(self, stage: str) -> None:
        r"""Set up the dataset for different stages.

        **Args:**
        - **stage** (`str`): the stage of the experiment. Should be one of the following:
            - 'fit': training and validation dataset should be assigned to `self.dataset_train` and `self.dataset_val`.
            - 'test': test dataset should be assigned to `self.dataset_test`.
        """
        if stage == "fit":
            pylogger.debug("Construct joint training dataset...")

            dataset_train_list = []
            for task_id in range(1, self.cl_dataset.num_tasks + 1):
                self.cl_dataset.setup_task_id(task_id)
                self.cl_dataset.setup(stage)
                task_labelled_dataset = label_dataset_task(
                    self.cl_dataset.dataset_train_t, task_id
                )
                dataset_train_list.append(task_labelled_dataset)

            self.dataset_train = ConcatDataset(dataset_train_list)
            pylogger.debug("Joint training dataset are ready.")

            pylogger.debug("Construct joint validation dataset...")

            for task_id in range(1, self.cl_dataset.num_tasks + 1):
                self.cl_dataset.setup_task_id(task_id)
                self.cl_dataset.setup(stage)
                self.dataset_val[f"{task_id}"] = self.cl_dataset.dataset_val_t
            pylogger.debug("Joint validation dataset are ready.")

        if stage == "test":

            pylogger.debug("Construct joint test dataset...")

            for task_id in range(1, self.cl_dataset.num_tasks + 1):
                self.cl_dataset.setup_task_id(task_id)
                self.cl_dataset.setup(stage)
                self.dataset_test[f"{task_id}"] = self.cl_dataset.dataset_test[
                    f"{task_id}"
                ]

            pylogger.debug("Joint test dataset is ready.")

    def train_dataloader(self) -> DataLoader:
        r"""DataLoader generator for stage train. It is automatically called before training.

        **Returns:**
        - **train_dataloader** (`Dataloader`): the train DataLoader.
        """

        pylogger.debug("Construct joint train dataloader...")

        return DataLoader(
            dataset=self.dataset_train,
            batch_size=self.batch_size,
            shuffle=True,  # shuffle train batch to prevent overfitting
            num_workers=self.num_workers,
        )

    def val_dataloader(self) -> DataLoader:
        r"""DataLoader generator for stage validate. It is automatically called before validation.

        **Returns:**
        - **train_dataloader** (`Dataloader`): the validation DataLoader.
        """

        pylogger.debug("Construct joint validation dataloader...")

        return {
            f"{task_id}": DataLoader(
                dataset=dataset_val_t,
                batch_size=self.batch_size,
                shuffle=False,  # don't have to shuffle val or test batch
                num_workers=self.num_workers,
            )
            for task_id, dataset_val_t in self.dataset_val.items()
        }

    def test_dataloader(self) -> dict[str, DataLoader]:
        r"""DataLoader generator for stage test. It is automatically called before testing.

        **Returns:**
        - **test_dataloader** (`dict[str, DataLoader]`): the test DataLoader.
        """

        pylogger.debug("Construct joint test dataloader...")

        return {
            f"{task_id}": DataLoader(
                dataset=dataset_test_t,
                batch_size=self.batch_size,
                shuffle=False,  # don't have to shuffle val or test batch
                num_workers=self.num_workers,
            )
            for task_id, dataset_test_t in self.dataset_test.items()
        }


class TaskLabelledDataset(Dataset):
    r"""The dataset class that labels the dataset with the given task ID. It is used to label the dataset with the task ID for joint learning (JL) experiment."""

    def __init__(self, dataset: Dataset, task_id: int) -> None:
        r"""Initialise the task labelled dataset object.

        **Args:**
        - **dataset** (`Dataset`): the dataset to be labelled.
        - **task_id** (`int`): the task ID to be labelled.
        """
        Dataset.__init__(self)

        self.dataset: Dataset = dataset
        r"""Store the dataset object."""
        self.task_id: int = task_id
        r"""Store the task ID."""

    def __len__(self) -> int:
        r"""The length of the dataset. The same as the length of the original dataset.

        **Returns:**
        - **length** (`int`): the length of the dataset.
        """

        return len(self.dataset)

    def __getitem__(self, idx) -> tuple[Any, Any, int]:
        r"""Get the item from the dataset. Labelled with the task ID.

        **Args:**
        - **idx** (`int`): the index of the item to be retrieved.

        **Returns:**
        - **x** (`Any`): the input data.
        - **y** (`Any`): the target data.
        - **task_id** (`int`): the task ID.
        """
        x, y = self.dataset[idx]
        return x, y, self.task_id


def label_dataset_task(dataset: Dataset, task_id: int) -> Dataset:
    r"""Label the dataset with the given task ID by wrapping it with a dataset that returns (x, y, task_id) tuples.

    **Args:**
    - **dataset** (`Dataset`): the dataset to be labelled.
    - **task_id** (`int`): the task ID to be labelled.

    **Returns:**
    - **task_labelled_dataset** (`Dataset`): the labelled dataset.
    """
    return TaskLabelledDataset(dataset, task_id)
