r"""
The submodule in `cl_datasets` for CL dataset bases.
"""

__all__ = [
    "CLDataset",
    "CLPermutedDataset",
    "CLSplitDataset",
    "CLCombinedDataset",
]

import ast
import logging
from abc import abstractmethod
from typing import Any, Callable

import torch
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from clarena.stl_datasets.raw.constants import (
    DATASET_CONSTANTS_MAPPING,
    DatasetConstants,
)
from clarena.utils.misc import str_to_class
from clarena.utils.transforms import ClassMapping, Permute

# always get logger for built-in logging in each module
pylogger = logging.getLogger(__name__)


class CLDataset(LightningDataModule):
    r"""The base class of continual learning datasets."""

    def __init__(
        self,
        root: str | dict[int, str],
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
    ) -> None:
        r"""
        **Args:**
        - **root** (`str` | `dict[int, str]`): the root directory where the original data files for constructing the CL dataset physically live.
        If it is a dict, the keys are task IDs and the values are the root directories for each task. If it is a string, it is the same root directory for all tasks.
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
        """
        super().__init__()

        self.root: dict[int, str] = (
            root
            if isinstance(root, dict)
            else {t: root for t in range(1, num_tasks + 1)}
        )
        r"""The dict of root directories of the original data files for each task."""
        self.num_tasks: int = num_tasks
        r"""The maximum number of tasks supported by the dataset."""
        self.cl_paradigm: str
        r"""The continual learning paradigm."""
        self.batch_size: dict[int, int] = (
            batch_size
            if isinstance(batch_size, dict)
            else {t: batch_size for t in range(1, num_tasks + 1)}
        )
        r"""The dict of batch sizes for each task."""
        self.num_workers: dict[int, int] = (
            num_workers
            if isinstance(num_workers, dict)
            else {t: num_workers for t in range(1, num_tasks + 1)}
        )
        r"""The dict of numbers of workers for each task."""
        self.custom_transforms: dict[int, Callable | transforms.Compose | None] = (
            custom_transforms
            if isinstance(custom_transforms, dict)
            else {t: custom_transforms for t in range(1, num_tasks + 1)}
        )
        r"""The dict of custom transforms for each task."""
        self.repeat_channels: dict[int, int | None] = (
            repeat_channels
            if isinstance(repeat_channels, dict)
            else {t: repeat_channels for t in range(1, num_tasks + 1)}
        )
        r"""The dict of number of channels to repeat for each task."""
        self.to_tensor: dict[int, bool] = (
            to_tensor
            if isinstance(to_tensor, dict)
            else {t: to_tensor for t in range(1, num_tasks + 1)}
        )
        r"""The dict of to_tensor flag for each task. """
        self.resize: dict[int, tuple[int, int] | None] = (
            [ast.literal_eval(rs) if rs else None for rs in resize]
            if isinstance(resize, dict)
            else {
                t: (ast.literal_eval(resize) if resize else None)
                for t in range(1, num_tasks + 1)
            }
        )
        r"""The dict of sizes to resize to for each task."""

        # task-specific attributes
        self.root_t: str
        r"""The root directory of the original data files for the current task `self.task_id`."""
        self.batch_size_t: int
        r"""The batch size for the current task `self.task_id`."""
        self.num_workers_t: int
        r"""The number of workers for the current task `self.task_id`."""
        self.custom_transforms_t: Callable | transforms.Compose | None
        r"""The custom transforms for the current task `self.task_id`."""
        self.repeat_channels_t: int | None
        r"""The number of channels to repeat for the current task `self.task_id`."""
        self.to_tensor_t: bool
        r"""The to_tensor flag for the current task `self.task_id`."""
        self.resize_t: tuple[int, int] | None
        r"""The size to resize for the current task `self.task_id`."""
        self.mean_t: float
        r"""The mean values for normalization for the current task `self.task_id`."""
        self.std_t: float
        r"""The standard deviation values for normalization for the current task `self.task_id`."""

        # dataset containers
        self.dataset_train_t: Any
        r"""The training dataset object. Can be a PyTorch Dataset object or any other dataset object."""
        self.dataset_val_t: Any
        r"""The validation dataset object. Can be a PyTorch Dataset object or any other dataset object."""
        self.dataset_test: dict[int, Any] = {}
        r"""The dictionary to store test dataset object of each task. Keys are task IDs and values are the dataset objects. Can be PyTorch Dataset objects or any other dataset objects."""

        # task ID control
        self.task_id: int
        r"""Task ID counter indicating which task is being processed. Self updated during the task loop. Valid from 1 to the number of tasks in the CL dataset."""
        self.processed_task_ids: list[int] = []
        r"""Task IDs that have been processed."""

        CLDataset.sanity_check(self)

    def sanity_check(self) -> None:
        r"""Sanity check."""

        # check if each task has been provided with necessary arguments
        for attr in [
            "root",
            "batch_size",
            "num_workers",
            "custom_transforms",
            "repeat_channels",
            "to_tensor",
            "resize",
        ]:
            value = getattr(self, attr)
            expected_keys = set(range(1, self.num_tasks + 1))
            if set(value.keys()) != expected_keys:
                raise ValueError(
                    f"{attr} dict keys must be consecutive integers from 1 to num_tasks."
                )

    @abstractmethod
    def get_cl_class_map(self, task_id: int) -> dict[str | int, int]:
        r"""Get the mapping of classes of task `task_id` to fit continual learning settings `self.cl_paradigm`. It must be implemented by subclasses.

        **Args:**
        - **task_id** (`int`): the task ID to query the CL class map.

        **Returns:**
        - **cl_class_map** (`dict[str | int, int]`): the CL class map of the task. Keys are the original class labels and values are the integer class label for continual learning.
            - If `self.cl_paradigm` is 'TIL', the mapped class labels of each task should be continuous integers from 0 to the number of classes.
            - If `self.cl_paradigm` is 'CIL', the mapped class labels of each task should be continuous integers from the number of classes of previous tasks to the number of classes of the current task.
        """

    @abstractmethod
    def prepare_data(self) -> None:
        r"""Use this to download and prepare data. It must be implemented by subclasses, as required by `LightningDataModule`. This method is called at the beginning of each task."""

    def setup(self, stage: str) -> None:
        r"""Set up the dataset for different stages. This method is called at the beginning of each task.

        **Args:**
        - **stage** (`str`): the stage of the experiment; one of:
            - 'fit': training and validation datasets of the current task `self.task_id` are assigned to `self.dataset_train_t` and `self.dataset_val_t`.
            - 'test': a dict of test datasets of all seen tasks should be assigned to `self.dataset_test`.
        """
        if stage == "fit":
            # these two stages must be done together because a sanity check for validation is conducted before training
            pylogger.debug(
                "Construct train and validation dataset for task %d...", self.task_id
            )

            self.dataset_train_t, self.dataset_val_t = self.train_and_val_dataset()

            pylogger.info(
                "Train and validation dataset for task %d are ready.", self.task_id
            )
            pylogger.info(
                "Train dataset for task %d size: %d",
                self.task_id,
                len(self.dataset_train_t),
            )
            pylogger.info(
                "Validation dataset for task %d size: %d",
                self.task_id,
                len(self.dataset_val_t),
            )

        elif stage == "test":

            pylogger.debug("Construct test dataset for task %d...", self.task_id)

            self.dataset_test[self.task_id] = self.test_dataset()

            pylogger.info("Test dataset for task %d are ready.", self.task_id)
            pylogger.info(
                "Test dataset for task %d size: %d",
                self.task_id,
                len(self.dataset_test[self.task_id]),
            )

    def setup_task_id(self, task_id: int) -> None:
        r"""Set up which task's dataset the CL experiment is on. This must be done before `setup()` method is called.

        **Args:**
        - **task_id** (`int`): the target task ID.
        """

        self.task_id = task_id

        self.root_t = self.root[task_id]
        self.batch_size_t = self.batch_size[task_id]
        self.num_workers_t = self.num_workers[task_id]
        self.custom_transforms_t = self.custom_transforms[task_id]
        self.repeat_channels_t = self.repeat_channels[task_id]
        self.to_tensor_t = self.to_tensor[task_id]
        self.resize_t = self.resize[task_id]

        self.processed_task_ids.append(task_id)

    def setup_tasks_eval(self, eval_tasks: list[int]) -> None:
        r"""Set up tasks for continual learning main evaluation.

        **Args:**
        - **eval_tasks** (`list[int]`): the list of task IDs to evaluate.
        """
        for task_id in eval_tasks:
            self.setup_task_id(task_id=task_id)
            self.setup(stage="test")

    def set_cl_paradigm(self, cl_paradigm: str) -> None:
        r"""Set `cl_paradigm` to `self.cl_paradigm`. It is used to define the CL class map.

        **Args:**
        - **cl_paradigm** (`str`): the continual learning paradigm, either 'TIL' (Task-Incremental Learning) or 'CIL' (Class-Incremental Learning).
        """
        self.cl_paradigm = cl_paradigm

    def train_and_val_transforms(self) -> transforms.Compose:
        r"""Transforms for training and validation datasets, incorporating the custom transforms with basic transforms like normalization and `ToTensor()`. It can be used in subclasses when constructing the dataset.

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
                        to_tensor_transform,
                        resize_transform,
                        self.custom_transforms_t,
                        normalization_transform,
                    ],
                )
            )
        )  # the order of transforms matters

    def test_transforms(self) -> transforms.Compose:
        r"""Transforms for the test dataset. Only basic transforms like normalization and `ToTensor()` are included. It is used in subclasses when constructing the dataset.

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
                        to_tensor_transform,
                        resize_transform,
                        normalization_transform,
                    ],
                )
            )
        )  # the order of transforms matters. No custom transforms for test

    def target_transform(self) -> ClassMapping:
        r"""Target transform to map the original class labels to CL class labels according to `self.cl_paradigm`. It can be used in subclasses when constructing the dataset.

        **Returns:**
        - **target_transform** (`Callable`): the target transform function.
        """

        cl_class_map = self.get_cl_class_map(task_id=self.task_id)

        target_transform = ClassMapping(class_map=cl_class_map)

        return target_transform

    @abstractmethod
    def train_and_val_dataset(self) -> tuple[Any, Any]:
        r"""Get the training and validation datasets of the current task `self.task_id`. It must be implemented by subclasses.

        **Returns:**
        - **train_and_val_dataset** (`tuple[Any, Any]`): the train and validation datasets of the current task `self.task_id`.
        """

    @abstractmethod
    def test_dataset(self) -> Any:
        r"""Get the test dataset of the current task `self.task_id`. It must be implemented by subclasses.

        **Returns:**
        - **test_dataset** (`Any`): the test dataset of the current task `self.task_id`.
        """

    def train_dataloader(self) -> DataLoader:
        r"""DataLoader generator for the train stage of the current task `self.task_id`. It is automatically called before training the task.

        **Returns:**
        - **train_dataloader** (`DataLoader`): the train DataLoader of task `self.task_id`.
        """

        pylogger.debug("Construct train dataloader for task %d...", self.task_id)

        return DataLoader(
            dataset=self.dataset_train_t,
            batch_size=self.batch_size_t,
            shuffle=True,  # shuffle train batch to prevent overfitting
            num_workers=self.num_workers_t,
            drop_last=True, # to avoid batchnorm error (when batch_size is 1)
        )

    def val_dataloader(self) -> DataLoader:
        r"""DataLoader generator for the validation stage of the current task `self.task_id`. It is automatically called before the task's validation.

        **Returns:**
        - **val_dataloader** (`DataLoader`): the validation DataLoader of task `self.task_id`.
        """

        pylogger.debug("Construct validation dataloader for task %d...", self.task_id)

        return DataLoader(
            dataset=self.dataset_val_t,
            batch_size=self.batch_size_t,
            shuffle=False,  # don't have to shuffle val or test batch
            num_workers=self.num_workers_t,
        )

    def test_dataloader(self) -> dict[int, DataLoader]:
        r"""DataLoader generator for the test stage of the current task `self.task_id`. It is automatically called before testing the task.

        **Returns:**
        - **test_dataloader** (`dict[int, DataLoader]`): the test DataLoader dict of `self.task_id` and all tasks before (as the test is conducted on all seen tasks). Keys are task IDs and values are the DataLoaders.
        """

        pylogger.debug("Construct test dataloader for task %d...", self.task_id)

        return {
            task_id: DataLoader(
                dataset=dataset_test_t,
                batch_size=self.batch_size_t,
                shuffle=False,  # don't have to shuffle val or test batch
                num_workers=self.num_workers_t,
            )
            for task_id, dataset_test_t in self.dataset_test.items()
        }

    def __len__(self) -> int:
        r"""Get the number of tasks in the dataset.

        **Returns:**
        - **num_tasks** (`int`): the number of tasks in the dataset.
        """
        return self.num_tasks


class CLPermutedDataset(CLDataset):
    r"""The base class of continual learning datasets constructed as permutations of an original dataset."""

    original_dataset_python_class: type[Dataset]
    r"""The original dataset class. **It must be provided in subclasses.** """

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
        r"""
        **Args:**
        - **root** (`str`): the root directory where the original dataset live.
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

        self.permutation_mode: str = permutation_mode
        r"""The mode of permutation."""
        self.permutation_seeds: dict[int, int] = (
            permutation_seeds
            if permutation_seeds
            else {t: t - 1 for t in range(1, num_tasks + 1)}
        )
        r"""The dict of permutation seeds for each task."""

        self.permutation_seed_t: int
        r"""The permutation seed for the current task `self.task_id`."""
        self.permute_transform_t: Permute
        r"""The permutation transform for the current task `self.task_id`."""

        CLPermutedDataset.sanity_check(self)

    def sanity_check(self) -> None:
        r"""Sanity check."""

        # check the permutation mode
        if self.permutation_mode not in ["all", "by_channel", "first_channel_only"]:
            raise ValueError(
                "The permutation_mode should be one of 'all', 'by_channel', 'first_channel_only'."
            )

        # check the permutation seeds
        expected_keys = set(range(1, self.num_tasks + 1))
        if set(self.permutation_seeds.keys()) != expected_keys:
            raise ValueError(
                f"{self.permutation_seeds} dict keys must be consecutive integers from 1 to num_tasks."
            )

    def get_cl_class_map(self, task_id: int) -> dict[str | int, int]:
        r"""Get the mapping of classes of task `task_id` to fit continual learning settings `self.cl_paradigm`.

        **Args:**
        - **task_id** (`int`): the task ID to query the CL class map.

        **Returns:**
        - **cl_class_map** (`dict[str | int, int]`): the CL class map of the task. Keys are the original class labels and values are the integer class label for continual learning.
            - If `self.cl_paradigm` is 'TIL', the mapped class labels of a task should be continuous integers from 0 to the number of classes.
            - If `self.cl_paradigm` is 'CIL', the mapped class labels of a task should be continuous integers from the number of classes of previous tasks to the number of classes of the current task.
        """

        num_classes_t = (
            self.original_dataset_constants.NUM_CLASSES
        )  # the same with the original dataset
        class_map_t = (
            self.original_dataset_constants.CLASS_MAP
        )  # the same with the original dataset

        if self.cl_paradigm == "TIL":
            return {class_map_t[i]: i for i in range(num_classes_t)}
        if self.cl_paradigm == "CIL":
            return {
                class_map_t[i]: i + (task_id - 1) * num_classes_t
                for i in range(num_classes_t)
            }

    def setup_task_id(self, task_id: int) -> None:
        r"""Set up which task's dataset the CL experiment is on. This must be done before `setup()` method is called.

        **Args:**
        - **task_id** (`int`): the target task ID.
        """

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
        self.permutation_seed_t = self.permutation_seeds[task_id]
        self.permute_transform_t = Permute(
            num_channels=num_channels,
            img_size=img_size,
            mode=self.permutation_mode,
            seed=self.permutation_seed_t,
        )

    def train_and_val_transforms(self) -> transforms.Compose:
        r"""Transforms for training and validation datasets, incorporating the custom transforms with basic transforms like normalization and `ToTensor()`. In permuted CL datasets, a permute transform also applies.

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
                        to_tensor_transform,
                        resize_transform,
                        self.permute_transform_t,  # permutation is included here
                        self.custom_transforms_t,
                        normalization_transform,
                    ],
                )
            )
        )  # the order of transforms matters

    def test_transforms(self) -> transforms.Compose:
        r"""Transforms for the test dataset. Only basic transforms like normalization and `ToTensor()` are included. In permuted CL datasets, a permute transform also applies.

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
                        to_tensor_transform,
                        resize_transform,
                        self.permute_transform_t,  # permutation is included here
                        normalization_transform,
                    ],
                )
            )
        )  # the order of transforms matters. No custom transforms for test


class CLSplitDataset(CLDataset):
    r"""The base class of continual learning datasets constructed as splits of an original dataset."""

    original_dataset_python_class: type[Dataset]
    r"""The original dataset class. **It must be provided in subclasses.** """

    def __init__(
        self,
        root: str,
        class_split: dict[int, list[int]],
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
    ) -> None:
        r"""
        **Args:**
        - **root** (`str`): the root directory where the original dataset live.
        - **class_split** (`dict[int, list[int]]`): the dict of classes for each task. The keys are task IDs ane the values are lists of class labels (integers starting from 0) to split for each task.
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
        """
        super().__init__(
            root=root,
            num_tasks=len(
                class_split
            ),  # num_tasks is not explicitly provided, but derived from the class_split length
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

        self.class_split: dict[int, list[int]] = class_split
        r"""The dict of class splits for each task."""

        CLSplitDataset.sanity_check(self)

    def sanity_check(self) -> None:
        r"""Sanity check."""

        # check the class split
        expected_keys = set(range(1, self.num_tasks + 1))
        if set(self.class_split.keys()) != expected_keys:
            raise ValueError(
                f"{self.class_split} dict keys must be consecutive integers from 1 to num_tasks."
            )
        if any(len(split) < 2 for split in self.class_split.values()):
            raise ValueError("Each class split must contain at least 2 elements!")

    def get_cl_class_map(self, task_id: int) -> dict[str | int, int]:
        r"""Get the mapping of classes of task `task_id` to fit continual learning settings `self.cl_paradigm`.

        **Args:**
        - **task_id** (`int`): the task ID to query the CL class map.

        **Returns:**
        - **cl_class_map** (`dict[str | int, int]`): the CL class map of the task. Keys are the original class labels and values are the integer class label for continual learning.
            - If `self.cl_paradigm` is 'TIL', the mapped class labels of a task should be continuous integers from 0 to the number of classes.
            - If `self.cl_paradigm` is 'CIL', the mapped class labels of a task should be continuous integers from the number of classes of previous tasks to the number of classes of the current task.
        """
        num_classes_t = len(
            self.class_split[task_id]
        )  # the number of classes in the current task, i.e. the length of the class split
        class_map_t = (
            self.original_dataset_constants.CLASS_MAP
        )  # the same with the original dataset

        if self.cl_paradigm == "TIL":
            return {
                class_map_t[self.class_split[task_id][i]]: i
                for i in range(num_classes_t)
            }
        if self.cl_paradigm == "CIL":
            num_classes_previous = sum(
                len(self.class_split[i]) for i in range(1, task_id)
            )
            return {
                class_map_t[self.class_split[task_id][i]]: num_classes_previous + i
                for i in range(num_classes_t)
            }

    def setup_task_id(self, task_id: int) -> None:
        r"""Set up which task's dataset the CL experiment is on. This must be done before `setup()` method is called.

        **Args:**
        - **task_id** (`int`): the target task ID.
        """
        super().setup_task_id(task_id)

        self.mean_t = (
            self.original_dataset_constants.MEAN
        )  # the same with the original dataset
        self.std_t = (
            self.original_dataset_constants.STD
        )  # the same with the original dataset

    @abstractmethod
    def get_subset_of_classes(self, dataset: Dataset) -> Dataset:
        r"""Get a subset of classes from the dataset for the current task `self.task_id`. It is used when constructing the split. **It must be implemented by subclasses.**

        **Args:**
        - **dataset** (`Dataset`): the dataset to retrieve the subset from.

        **Returns:**
        - **subset** (`Dataset`): the subset of classes from the dataset.
        """


class CLCombinedDataset(CLDataset):
    r"""The base class of continual learning datasets constructed as combinations of several single-task datasets (one dataset per task)."""

    def __init__(
        self,
        datasets: dict[int, str],
        root: str | dict[int, str],
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
    ) -> None:
        r"""
        **Args:**
        - **datasets** (`dict[int, str]`): the dict of dataset class paths for each task. The keys are task IDs and the values are the dataset class paths (as strings) to use for each task.
        - **root** (`str` | `dict[int, str]`): the root directory where the original data files for constructing the CL dataset physically live.
        If it is a dict, the keys are task IDs and the values are the root directories for each task. If it is a string, it is the same root directory for all tasks.
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
        """
        super().__init__(
            root=root,
            num_tasks=len(
                datasets
            ),  # num_tasks is not explicitly provided, but derived from the datasets length
            batch_size=batch_size,
            num_workers=num_workers,
            custom_transforms=custom_transforms,
            repeat_channels=repeat_channels,
            to_tensor=to_tensor,
            resize=resize,
        )

        self.original_dataset_python_classes: dict[int, Dataset] = {
            t: str_to_class(dataset_class_path) for t, dataset_class_path in datasets
        }
        r"""The dict of dataset classes for each task."""
        self.original_dataset_python_class_t: Dataset
        r"""The dataset class for the current task `self.task_id`."""
        self.original_dataset_constants_t: type[DatasetConstants]
        r"""The original dataset constants class for the current task `self.task_id`."""

    def get_cl_class_map(self, task_id: int) -> dict[str | int, int]:
        r"""Get the mapping of classes of task `task_id` to fit continual learning settings `self.cl_paradigm`.

        **Args:**
        - **task_id** (`int`): the task ID to query the CL class map.

        **Returns:**
        - **cl_class_map** (`dict[str | int, int]`): the CL class map of the task. Keys are the original class labels and values are the integer class label for continual learning.
            - If `self.cl_paradigm` is 'TIL', the mapped class labels of a task should be continuous integers from 0 to the number of classes.
            - If `self.cl_paradigm` is 'CIL', the mapped class labels of a task should be continuous integers from the number of classes of previous tasks to the number of classes of the current task.
        """
        original_dataset_python_class_t = self.original_dataset_python_classes[task_id]
        original_dataset_constants_t = DATASET_CONSTANTS_MAPPING[
            original_dataset_python_class_t
        ]
        num_classes_t = original_dataset_constants_t.NUM_CLASSES
        class_map_t = original_dataset_constants_t.CLASS_MAP

        if self.cl_paradigm == "TIL":
            return {class_map_t[i]: i for i in range(num_classes_t)}
        if self.cl_paradigm == "CIL":
            num_classes_previous = sum(
                [
                    DATASET_CONSTANTS_MAPPING[
                        self.original_dataset_python_classes[i]
                    ].NUM_CLASSES
                    for i in range(1, task_id)
                ]
            )
            return {
                class_map_t[i]: num_classes_previous + i for i in range(num_classes_t)
            }

    def setup_task_id(self, task_id: int) -> None:
        r"""Set up which task's dataset the CL experiment is on. This must be done before `setup()` method is called.

        **Args:**
        - **task_id** (`int`): the target task ID.
        """

        self.original_dataset_python_class_t = self.original_dataset_python_classes[
            task_id
        ]

        self.original_dataset_constants_t: type[DatasetConstants] = (
            DATASET_CONSTANTS_MAPPING[self.original_dataset_python_class_t]
        )

        super().setup_task_id(task_id)

        self.mean_t = self.original_dataset_constants_t.MEAN
        self.std_t = self.original_dataset_constants_t.STD
