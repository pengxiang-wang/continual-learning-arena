r"""
The submodule in `mtl_datasets` for MTL dataset bases.
"""

__all__ = [
    "MTLDataset",
    "MTLDatasetFromRaw",
    "MTLDatasetFromCL",
    "TaskLabelledDataset",
    "label_dataset_task",
]
import ast
import logging
from abc import abstractmethod
from typing import Any, Callable

from lightning import LightningDataModule
from torch.utils.data import ConcatDataset, DataLoader, Dataset
from torchvision import transforms

from clarena.cl_datasets import CLDataset
from clarena.stl_datasets.raw.constants import (
    DATASET_CONSTANTS_MAPPING,
    DatasetConstants,
)
from clarena.utils.misc import str_to_class

# always get logger for built-in logging in each module
pylogger = logging.getLogger(__name__)


class MTLDataset(LightningDataModule):
    r"""The base class of multi-task learning datasets, inherited from `LightningDataModule`."""

    def __init__(
        self,
        root: str | dict[int, str],
        num_tasks: int,
        sampling_strategy: str = "mixed",
        batch_size: int = 1,
        num_workers: int = 0,
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
        r"""Initialize the MTL dataset object providing the root where data files live.

        **Args:**
        - **root** (`str` | `list[str]`): the root directory where the original data files for constructing the MTL dataset physically live. If `list[str]`, it should be a list of strings, each string is the root directory for each task.
        - **num_tasks** (`int`): the maximum number of tasks supported by the MTL dataset.
        - **sampling_strategy** (`str`): the sampling strategy that construct training batch from each task's dataset. Should be one of the following:
            - 'mixed': mixed sampling strategy, which samples from all tasks' datasets.
        - **batch_size** (`int`): The batch size in train, val, test dataloader.
        - **num_workers** (`int`): the number of workers for dataloaders.
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
        self.sampling_strategy: str = sampling_strategy
        r"""The sampling strategy for constructing training batch from each task's dataset."""
        self.batch_size: int = batch_size
        r"""The batch size for dataloaders."""
        self.num_workers: int = num_workers
        r"""The number of workers for dataloaders."""

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

        # dataset containers
        self.dataset_train: dict[int, Any] = {}
        r"""The dictionary to store training dataset object of each task. Keys are task IDs and values are the dataset objects. Can be PyTorch Dataset objects or any other dataset objects."""
        self.dataset_val: dict[int, Any] = {}
        r"""The dictionary to store validation dataset object of each task. Keys are task IDs and values are the dataset objects. Can be PyTorch Dataset objects or any other dataset objects."""
        self.dataset_test: dict[int, Any] = {}
        r"""The dictionary to store test dataset object of each task. Keys are task IDs and values are the dataset objects. Can be PyTorch Dataset objects or any other dataset objects."""

        self.mean: dict[int, float] = {}
        r"""Tthe list of mean values for normalization for all tasks. Used when constructing the transforms."""
        self.std: dict[int, float] = {}
        r"""The list of standard deviation values for normalization for all tasks. Used when constructing the transforms."""

        # task ID controls
        self.train_tasks: list[int]
        r""""The list of task IDs to be trained. It should be a list of integers, each integer is the task ID."""
        self.eval_tasks: list[int]
        r"""The list of task IDs to be evaluated. It should be a list of integers, each integer is the task ID."""

        MTLDataset.sanity_check(self)

    def sanity_check(self) -> None:
        r"""Check the sanity of the arguments."""
        for attr in [
            "root",
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
    def get_class_map(self, task_id: int) -> dict[str | int, int]:
        r"""Get the mapping of classes of task `task_id`. It must be implemented by subclasses.

        **Args:**
        - **task_id** (`int`): The task ID to query class map.

        **Returns:**
        - **class_map**(`dict[str | int, int]`): the class map of the task. Key is original class label, value is integer class label for multi-task learning. The mapped class labels of each task should be continuous integers from 0 to the number of classes.
        """

    @abstractmethod
    def prepare_data(self) -> None:
        r"""Use this to download and prepare data. It must be implemented by subclasses, regulated by `LightningDatamodule`."""

    def setup(self, stage: str) -> None:
        r"""Set up the dataset for different stages.

        **Args:**
        - **stage** (`str`): the stage of the experiment. Should be one of the following:
            - 'fit': training and validation dataset should be assigned to `self.dataset_train` and `self.dataset_val`.
            - 'test': test dataset should be assigned to `self.dataset_test`.
        """
        if stage == "fit":
            # these two stages must be done together because a sanity check for validation is conducted before training
            pylogger.debug("Construct train and validation dataset ...")

            for task_id in range(1, self.num_tasks + 1):

                self.dataset_train[task_id], self.dataset_val[task_id] = (
                    self.train_and_val_dataset(task_id)
                )

                pylogger.debug(
                    "Train and validation dataset for task %d are ready.", task_id
                )
                pylogger.info(
                    "Train dataset for task %d size: %d",
                    task_id,
                    len(self.dataset_train[task_id]),
                )
                pylogger.info(
                    "Validation dataset for task %d size: %d",
                    task_id,
                    len(self.dataset_val[task_id]),
                )

        elif stage == "test":

            pylogger.debug("Construct test dataset ...")

            for task_id in range(1, self.num_tasks + 1):

                self.dataset_test[task_id] = self.test_dataset(task_id)

                pylogger.debug("Test dataset for task %d are ready.", task_id)
                pylogger.info(
                    "Test dataset for task %d size: %d",
                    task_id,
                    len(self.dataset_test[task_id]),
                )

    def setup_tasks(self, train_tasks: list[int], eval_tasks: list[int]) -> None:
        r"""Set up the tasks for the dataset.

        **Args:**
        - **train_tasks** (`list[int]`): the list of task IDs to be trained. It should be a list of integers, each integer is the task ID. This is used when constructing the dataloader.
        - **eval_tasks** (`list[int]`): the list of task IDs to be evaluated. It should be a list of integers, each integer is the task ID. This is used when constructing the dataloader.
        """
        self.train_tasks = train_tasks
        self.eval_tasks = eval_tasks

    def train_and_val_transforms(self, task_id: int) -> transforms.Compose:
        r"""Transforms generator for train and validation dataset of task `task_id` incorporating the custom transforms with basic transforms like `normalization` and `ToTensor()`. It is a handy tool to use in subclasses when constructing the dataset.

        **Args:**
        - **task_id** (`int`): the task ID of training and validation dataset to get the transforms for.

        **Returns:**
        - **train_and_val_transforms** (`transforms.Compose`): the composed training transforms.
        """
        repeat_channels_transform = (
            transforms.Grayscale(num_output_channels=self.repeat_channels[task_id])
            if self.repeat_channels[task_id] is not None
            else None
        )
        to_tensor_transform = transforms.ToTensor() if self.to_tensor[task_id] else None
        resize_transform = (
            transforms.Resize(self.resize[task_id])
            if self.resize[task_id] is not None
            else None
        )
        normalization_transform = transforms.Normalize(
            self.mean[task_id], self.std[task_id]
        )

        return transforms.Compose(
            list(
                filter(
                    None,
                    [
                        repeat_channels_transform,
                        to_tensor_transform,
                        resize_transform,
                        self.custom_transforms[task_id],
                        normalization_transform,
                    ],
                )
            )
        )  # the order of transforms matters

    def test_transforms(self, task_id: int) -> transforms.Compose:
        r"""Transforms generator for test dataset of task `task_id`. Only basic transforms like `normalization` and `ToTensor()` are included. It is a handy tool to use in subclasses when constructing the dataset.

        **Args:**
        - **task_id** (`int`): the task ID of test dataset to get the transforms for.

        **Returns:**
        - **test_transforms** (`transforms.Compose`): the composed training transforms.
        """

        repeat_channels_transform = (
            transforms.Grayscale(num_output_channels=self.repeat_channels[task_id])
            if self.repeat_channels[task_id] is not None
            else None
        )
        to_tensor_transform = transforms.ToTensor() if self.to_tensor[task_id] else None
        resize_transform = (
            transforms.Resize(self.resize[task_id])
            if self.resize[task_id] is not None
            else None
        )
        normalization_transform = transforms.Normalize(
            self.mean[task_id], self.std[task_id]
        )

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
        )  # the order of transforms matters

    @abstractmethod
    def train_and_val_dataset(self, task_id: int) -> Any:
        r"""Get the training and validation dataset of task `task_id`. It must be implemented by subclasses.

        **Args:**
        - **task_id** (`int`): the task ID to get the training and validation dataset for.

        **Returns:**
        - **train_and_val_dataset** (`Any`): the train and validation dataset of task `task_id`.
        """

    @abstractmethod
    def test_dataset(self, task_id: int) -> Any:
        """Get the test dataset of task `task_id`. It must be implemented by subclasses.

        **Args:**
        - **task_id** (`int`): the task ID to get the test dataset for.

        **Returns:**
        - **test_dataset** (`Any`): the test dataset of task `task_id`.
        """

    def train_dataloader(self) -> DataLoader:
        r"""DataLoader generator for stage train. It is automatically called before training.

        **Returns:**
        - **train_dataloader** (`Dataloader`): the train DataLoader of task.
        """

        pylogger.debug(
            "Construct train dataloader ... sampling_strategy method: %s",
            self.sampling_strategy,
        )

        if self.sampling_strategy == "mixed":
            # mixed sampling strategy, which samples from all tasks' datasets

            concatenated_dataset = ConcatDataset(
                [self.dataset_train[task_id] for task_id in self.train_tasks]
            )

            return DataLoader(
                dataset=concatenated_dataset,
                batch_size=self.batch_size,
                shuffle=True,  # shuffle train batch to prevent overfitting
                num_workers=self.num_workers,
            )

    def val_dataloader(self) -> DataLoader:
        r"""DataLoader generator for the validation stage. It is automatically called before validation.

        **Returns:**
        - **val_dataloader** (`dict[int, DataLoader]`): the validation DataLoader.
        """

        pylogger.debug("Construct validation dataloader...")

        return {
            task_id: DataLoader(
                dataset=dataset_val_t,
                batch_size=self.batch_size,
                shuffle=False,  # don't have to shuffle val or test batch
                num_workers=self.num_workers,
            )
            for task_id, dataset_val_t in self.dataset_val.items()
        }

    def test_dataloader(self) -> dict[int, DataLoader]:
        r"""DataLoader generator for stage test. It is automatically called before testing.

        **Returns:**
        - **test_dataloader** (`dict[int, DataLoader]`): the test DataLoader.
        """

        pylogger.debug("Construct test dataloader...")

        return {
            task_id: DataLoader(
                dataset=dataset_test_t,
                batch_size=self.batch_size,
                shuffle=False,  # don't have to shuffle val or test batch
                num_workers=self.num_workers,
            )
            for task_id, dataset_test_t in self.dataset_test.items()
        }


class MTLDatasetFromRaw(MTLDataset):
    r"""Multi-task learning datasets constructed from raw data files. Inherits from `MTLDataset`.

    This is usually for constructing the reference joint learning experiment for multi-task learning.
    """

    def __init__(
        self,
        datasets: dict[int, str],
        root: str | dict[int, str],
        sampling_strategy: str = "mixed",
        batch_size: int = 1,
        num_workers: int = 0,
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
        r"""Initialize the `MTLDatasetFromRaw` object.

        **Args:**
        - **datasets** (`dict[int, str]`): the dict of dataset class paths for each task. The keys are task IDs and the values are the dataset class paths (as strings) to use for each task.
        - **root** (`str` | `dict[int, str]`): the root directory where the original data files for constructing the MTL dataset physically live. If `dict[int, str]`, it should be a dict of task IDs and their corresponding root directories.
        - **sampling_strategy** (`str`): the sampling strategy that construct training batch from each task's dataset. Should be one of the following:
            - 'mixed': mixed sampling strategy, which samples from all tasks' datasets.
        - **batch_size** (`int`): The batch size in train, val, test dataloader.
        - **num_workers** (`int`): the number of workers for dataloaders.
        - **custom_transforms** (`transform` or `transforms.Compose` or `None` or dict of them): the custom transforms to apply ONLY to the TRAIN dataset. Can be a single transform, composed transforms, or no transform. `ToTensor()`, normalization, permute, and so on are not included.
        If it is a dict, the keys are task IDs and the values are the custom transforms for each task. If it is a single transform or composed transforms, it is applied to all tasks. If it is `None`, no custom transforms are applied.
        - **repeat_channels** (`int` | `None` | dict of them): the number of channels to repeat for each task. Default is `None`, which means no repeat.
        If it is a dict, the keys are task IDs and the values are the number of channels to repeat for each task. If it is an `int`, it is the same number of channels to repeat for all tasks. If it is `None`, no repeat is applied.
        - **to_tensor** (`bool` | `dict[int, bool]`): whether to include the `ToTensor()` transform. Default is `True`.
        If it is a dict, the keys are task IDs and the values are whether to include the `ToTensor()` transform for each task. If it is a single boolean value, it is applied to all tasks.
        - **resize** (`tuple[int, int]` | `None` or dict of them): the size to resize the images to. Default is `None`, which means no resize. If it is a dict, the keys are task IDs and the values are the sizes to resize for each task. If it is a single tuple of two integers, it is applied to all tasks. If it is `None`, no resize is applied.
        """
        super().__init__(
            root=root,
            num_tasks=len(
                datasets
            ),  # num_tasks is not explicitly provided, but derived from the datasets length
            sampling_strategy=sampling_strategy,
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

    def get_class_map(self, task_id: int) -> dict[str | int, int]:
        r"""Get the mapping of classes of task `task_id` to fit continual learning settings `self.cl_paradigm`.

        **Args:**
        - **task_id** (`int`): the task ID to query the CL class map.

        **Returns:**
        - **cl_class_map** (`dict[str | int, int]`): the CL class map of the task. Key is the original class label, value is the integer class label for continual learning.
        """
        original_dataset_python_class_t = self.original_dataset_python_classes[task_id]
        original_dataset_constants_t = DATASET_CONSTANTS_MAPPING[
            original_dataset_python_class_t
        ]
        num_classes_t = original_dataset_constants_t.NUM_CLASSES
        class_map_t = original_dataset_constants_t.CLASS_MAP

        return {class_map_t[i]: i for i in range(num_classes_t)}

    def setup_tasks(self, train_tasks: list[int], eval_tasks: list[int]) -> None:
        r"""Set up the tasks for the dataset.

        **Args:**
        - **train_tasks** (`list[int]`): the list of task IDs to be trained. It should be a list of integers, each integer is the task ID. This is used when constructing the dataloader.
        - **eval_tasks** (`list[int]`): the list of task IDs to be evaluated. It should be a list of integers, each integer is the task ID. This is used when constructing the dataloader.
        """
        super().setup_tasks(train_tasks, eval_tasks)

        for task_id in train_tasks + eval_tasks:
            original_dataset_python_class_t = self.original_dataset_python_classes[
                task_id
            ]
            original_dataset_constants_t = DATASET_CONSTANTS_MAPPING[
                original_dataset_python_class_t
            ]
            self.mean[task_id] = original_dataset_constants_t.MEAN
            self.std[task_id] = original_dataset_constants_t.STD


class MTLDatasetFromCL(MTLDataset):
    r"""Multi-task learning datasets constructed from the CL datasets. Inherits from `MTLDataset`.

    This is usually for constructing the reference joint learning experiment for continual learning.
    """

    def __init__(
        self,
        cl_dataset: CLDataset,
        sampling_strategy: str = "mixed",
    ) -> None:
        r"""Initialize the `MTLDatasetFromCL` object.

        **Args:**
        - **cl_dataset** (`CLDataset`): the CL dataset object to be used for constructing the MTL dataset.
        - **sampling_strategy** (`str`): the sampling strategy that construct training batch from each task's dataset. Should be one of the following:
            - 'mixed': mixed sampling strategy, which samples from all tasks' datasets.
        """

        self.cl_dataset: CLDataset = cl_dataset
        r"""The CL dataset for constructing the MTL dataset."""

        batch_size = (
            cl_dataset.batch_size
            if isinstance(cl_dataset.batch_size, int)
            else cl_dataset.batch_size[1]
        )

        num_workers = (
            cl_dataset.num_workers
            if isinstance(cl_dataset.num_workers, int)
            else cl_dataset.num_workers[1]
        )

        super().__init__(
            root=None,
            num_tasks=cl_dataset.num_tasks,
            sampling_strategy=sampling_strategy,
            batch_size=batch_size,
            num_workers=num_workers,
            custom_transforms=None,
            repeat_channels=None,
            to_tensor=None,
            resize=None,
        )

    def prepare_data(self) -> None:
        r"""Download and prepare data."""
        self.cl_dataset.prepare_data()  # Prepare the CL dataset

    def setup(self, stage: str) -> None:
        r"""Set up the dataset for different stages.

        **Args:**
        - **stage** (`str`): the stage of the experiment. Should be one of the following:
            - 'fit': training and validation dataset should be assigned to `self.dataset_train` and `self.dataset_val`.
            - 'test': test dataset should be assigned to `self.dataset_test`.
        """
        if stage == "fit":
            pylogger.debug("Construct train and validation dataset ...")

            # go through each task of continual learning to get the training dataset of each task
            for task_id in range(1, self.num_tasks + 1):
                self.cl_dataset.setup_task_id(task_id)
                self.cl_dataset.setup(stage)

                # label the training dataset with the task ID
                task_labelled_dataset_train_t = label_dataset_task(
                    self.cl_dataset.dataset_train_t, task_id
                )
                self.dataset_train[task_id] = task_labelled_dataset_train_t

                # label the validation dataset with the task ID
                task_labelled_dataset_val_t = label_dataset_task(
                    self.cl_dataset.dataset_val_t, task_id
                )
                self.dataset_val[task_id] = task_labelled_dataset_val_t

                pylogger.debug(
                    "Train and validation dataset for task %d are ready.", task_id
                )
                pylogger.info(
                    "Train dataset for task %d size: %d",
                    task_id,
                    len(self.dataset_train[task_id]),
                )
                pylogger.info(
                    "Validation dataset for task %d size: %d",
                    task_id,
                    len(self.dataset_val[task_id]),
                )

        elif stage == "test":

            pylogger.debug("Construct test dataset ...")

            for task_id in self.eval_tasks:

                self.cl_dataset.setup_task_id(task_id)
                self.cl_dataset.setup(stage)

                task_labelled_dataset_test_t = label_dataset_task(
                    self.cl_dataset.dataset_test[task_id], task_id
                )

                self.dataset_test[task_id] = task_labelled_dataset_test_t

                pylogger.debug("Test dataset for task %d are ready.", task_id)
                pylogger.info(
                    "Test dataset for task %d size: %d",
                    task_id,
                    len(self.dataset_test[task_id]),
                )

    def get_class_map(self, task_id: int) -> dict[str | int, int]:
        r"""Get the mapping of classes of task `task_id`. It is inherited from `CLDataset` and returns the class map of the task.

        **Args:**
        - **task_id** (`int`): The task ID to query class map.

        **Returns:**
        - **class_map**(`dict[str | int, int]`): the class map of the task. Key is original class label, value is integer class label for multi-task learning. The mapped class labels of each task should be continuous integers from 0 to the number of classes.
        """
        return self.cl_dataset.get_cl_class_map(task_id)


class TaskLabelledDataset(Dataset):
    r"""The dataset class that labels the a task's dataset with the given task ID. It is used to label the dataset with the task ID for MTL experiment."""

    def __init__(self, dataset: Dataset, task_id: int) -> None:
        r"""Initialize the task labelled dataset object.

        **Args:**
        - **dataset** (`Dataset`): the dataset to be labelled.
        - **task_id** (`int`): the task ID to be labelled.
        """
        super().__init__()

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
