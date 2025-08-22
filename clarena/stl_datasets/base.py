r"""
The submodule in `cl_datasets` for STL dataset bases.
"""

__all__ = [
    "STLDataset",
    "STLDatasetFromRaw",
]


import logging
from abc import abstractmethod
from typing import Any, Callable

from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from clarena.stl_datasets.raw.constants import (
    DATASET_CONSTANTS_MAPPING,
    DatasetConstants,
)

# always get logger for built-in logging in each module
pylogger = logging.getLogger(__name__)


class STLDataset(LightningDataModule):
    r"""The base class of single-task learning datasets, inherited from `LightningDataModule`."""

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
        r"""Initialize the STL dataset object providing the root where data files live.

        **Args:**
        - **root** (`str`): the root directory where the original data files for constructing the STL dataset physically live.
        - **batch_size** (`int`): The batch size in train, val, test dataloader.
        - **num_workers** (`int`): the number of workers for dataloaders.
        - **custom_transforms** (`transform` or `transforms.Compose` or `None`): the custom transforms to apply to ONLY TRAIN dataset. Can be a single transform, composed transforms or no transform. `ToTensor()`, normalize, permute and so on are not included.
        - **repeat_channels** (`int` | `None`): the number of channels to repeat for each task. Default is None, which means no repeat. If not None, it should be an integer.
        - **to_tensor** (`bool`): whether to include `ToTensor()` transform. Default is True.
        - **resize** (`tuple[int, int]` | `None` or list of them): the size to resize the images to. Default is None, which means no resize. If not None, it should be a tuple of two integers.
        """
        super().__init__()

        self.root: str = root
        r"""The root directory of the original data files."""
        self.batch_size: int = batch_size
        r"""The batch size for dataloaders."""
        self.num_workers: int = num_workers
        r"""The number of workers for dataloaders."""

        self.custom_transforms: Callable | transforms.Compose = custom_transforms
        r"""The custom transforms."""
        self.repeat_channels: int | None = repeat_channels
        r"""The number of channels to repeat."""
        self.to_tensor: bool = to_tensor
        r"""The to_tensor flag."""
        self.resize: tuple[int, int] | None = resize
        r"""The size to resize."""

        # classes information
        self.num_classes: int
        r"""The number of classes in each task."""
        self.class_map: dict[int, str | int]
        r"""The class map for the current task `self.task_id`. The key is the integer class label, and the value is the original class label. It is used to get the original class label from the integer class label."""

        self.dataset_train: Any
        r"""Training dataset object. Can be PyTorch Dataset objects or any other dataset objects."""
        self.dataset_val: Any
        r"""Validation dataset object. Can be PyTorch Dataset objects or any other dataset objects."""
        self.dataset_test: Any
        r"""Test dataset object. Can be PyTorch Dataset objects or any other dataset objects."""
        self.mean: float
        r"""Mean value for normalization. Used when constructing the transforms."""
        self.std: float
        r"""Standard deviation value for normalization. Used when constructing the transforms."""

        STLDataset.sanity_check(self)

    def sanity_check(self) -> None:
        r"""Check the sanity of the arguments."""

    @abstractmethod
    def get_class_map(self) -> dict[str | int, int]:
        r"""Get the mapping of classes of task `task_id`. It must be implemented by subclasses.

        **Returns:**
        - **class_map**(`dict[str | int, int]`): the class map of the task. Key is original class label, value is integer class label for single-task learning.
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

            self.dataset_train, self.dataset_val = self.train_and_val_dataset()

            pylogger.debug("Train and validation dataset are ready.")
            pylogger.info(
                "Train dataset size: %d",
                len(self.dataset_train),
            )
            pylogger.info(
                "Validation dataset size: %d",
                len(self.dataset_val),
            )

        elif stage == "test":

            pylogger.debug("Construct test dataset ...")

            self.dataset_test = self.test_dataset()

            pylogger.debug("Test dataset are ready.")
            pylogger.info(
                "Test dataset for size: %d",
                len(self.dataset_test),
            )

    def setup_task(self) -> None:
        r"""Set up the tasks for the dataset.

        **Args:**
        - **train_tasks** (`list[int]`): the list of task IDs to be trained. It should be a list of integers, each integer is the task ID. This is used when constructing the dataloader.
        - **eval_tasks** (`list[int]`): the list of task IDs to be evaluated. It should be a list of integers, each integer is the task ID. This is used when constructing the dataloader.
        """
        pass

    def train_and_val_transforms(self) -> transforms.Compose:
        r"""Transforms generator for train and validation dataset incorporating the custom transforms with basic transforms like `normalization` and `ToTensor()`. It is a handy tool to use in subclasses when constructing the dataset.

        **Returns:**
        - **train_and_val_transforms** (`transforms.Compose`): the composed training transforms.
        """
        repeat_channels_transform = (
            transforms.Grayscale(num_output_channels=self.repeat_channels)
            if self.repeat_channels is not None
            else None
        )
        to_tensor_transform = transforms.ToTensor() if self.to_tensor else None
        resize_transform = (
            transforms.Resize(self.resize) if self.resize is not None else None
        )
        normalization_transform = transforms.Normalize(self.mean, self.std)

        return transforms.Compose(
            list(
                filter(
                    None,
                    [
                        repeat_channels_transform,
                        to_tensor_transform,
                        resize_transform,
                        self.custom_transforms,
                        normalization_transform,
                    ],
                )
            )
        )  # the order of transforms matters

    def test_transforms(self) -> transforms.Compose:
        r"""Transforms generator for test dataset. Only basic transforms like `normalization` and `ToTensor()` are included. It is a handy tool to use in subclasses when constructing the dataset.

        **Returns:**
        - **test_transforms** (`transforms.Compose`): the composed training transforms.
        """

        repeat_channels_transform = (
            transforms.Grayscale(num_output_channels=self.repeat_channels)
            if self.repeat_channels is not None
            else None
        )
        to_tensor_transform = transforms.ToTensor() if self.to_tensor else None
        resize_transform = (
            transforms.Resize(self.resize) if self.resize is not None else None
        )
        normalization_transform = transforms.Normalize(self.mean, self.std)

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
    def train_and_val_dataset(self) -> Any:
        r"""Get the training and validation dataset. It must be implemented by subclasses.

        **Returns:**
        - **train_and_val_dataset** (`Any`): the train and validation dataset.
        """

    @abstractmethod
    def test_dataset(self) -> Any:
        """Get the test dataset. It must be implemented by subclasses.

        **Args:**
        - **task_id** (`int`): the task ID to get the test dataset.

        **Returns:**
        - **test_dataset** (`Any`): the test dataset.
        """

    def train_dataloader(self) -> DataLoader:
        r"""DataLoader generator for stage train. It is automatically called before training.

        **Returns:**
        - **train_dataloader** (`Dataloader`): the train DataLoader.
        """

        pylogger.debug("Construct train dataloader ...")

        return DataLoader(
            dataset=self.dataset_train,
            batch_size=self.batch_size,
            shuffle=True,  # shuffle train batch to prevent overfitting
            num_workers=self.num_workers,
        )

    def val_dataloader(self) -> DataLoader:
        r"""DataLoader generator for the validation stage. It is automatically called before validation.

        **Returns:**
        - **val_dataloader** (`DataLoader`): the validation DataLoader.
        """

        pylogger.debug("Construct validation dataloader...")

        return DataLoader(
            dataset=self.dataset_val,
            batch_size=self.batch_size,
            shuffle=False,  # don't have to shuffle val or test batch
            num_workers=self.num_workers,
        )

    def test_dataloader(self) -> dict[str, DataLoader]:
        r"""DataLoader generator for stage test. It is automatically called before testing.

        **Returns:**
        - **test_dataloader** (`Dataloader`): the test DataLoader.
        """

        pylogger.debug("Construct test dataloader...")

        return DataLoader(
            dataset=self.dataset_test,
            batch_size=self.batch_size,
            shuffle=False,  # don't have to shuffle val or test batch
            num_workers=self.num_workers,
        )


class STLDatasetFromRaw(STLDataset):
    r"""The base class of single-task learning datasets from raw data, inherited from `STLDataset`.

    It is used to construct the STL dataset from raw data files.
    """

    original_dataset_python_class: type[Dataset]
    r"""The original dataset class. It must be provided in subclasses."""

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
        r"""Initialize the STL dataset object providing the root where data files live.

        **Args:**
        - **root** (`str`): the root directory where the original data files for constructing the STL dataset physically live.
        - **batch_size** (`int`): The batch size in train, val, test dataloader.
        - **num_workers** (`int`): the number of workers for dataloaders.
        - **custom_transforms** (`transform` or `transforms.Compose` or `None`): the custom transforms to apply to ONLY TRAIN dataset. Can be a single transform, composed transforms or no transform. `ToTensor()`, normalize, permute and so on are not included.
        - **repeat_channels** (`int` | `None`): the number of channels to repeat for each task. Default is None, which means no repeat. If not None, it should be an integer.
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

        self.original_dataset_constants: type[DatasetConstants] = (
            DATASET_CONSTANTS_MAPPING[self.original_dataset_python_class]
        )
        r"""The original dataset constants class."""

    def get_class_map(self) -> dict[str | int, int]:
        r"""Get the mapping of classes of task `task_id`. It must be implemented by subclasses.

        **Returns:**
        - **class_map**(`dict[str | int, int]`): the class map of the task. Key is original class label, value is integer class label for single-task learning.
        """
        return self.original_dataset_constants.CLASS_MAP

    def setup_task(self) -> None:
        r"""Set up the tasks for the dataset.

        **Args:**
        - **train_tasks** (`list[int]`): the list of task IDs to be trained. It should be a list of integers, each integer is the task ID. This is used when constructing the dataloader.
        - **eval_tasks** (`list[int]`): the list of task IDs to be evaluated. It should be a list of integers, each integer is the task ID. This is used when constructing the dataloader.
        """
        super().setup_task()

        self.mean = (
            self.original_dataset_constants.MEAN
        )  # the same with the original dataset
        self.std = (
            self.original_dataset_constants.STD
        )  # the same with the original dataset
