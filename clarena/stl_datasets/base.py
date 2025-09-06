r"""
The submodule in `stl_datasets` for STL dataset bases.
"""

__all__ = [
    "STLDataset",
    "STLDatasetFromRaw",
    "TaskLabelledDataset",
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
from clarena.utils.transforms import ClassMapping

# always get logger for built-in logging in each module
pylogger = logging.getLogger(__name__)


class STLDataset(LightningDataModule):
    r"""The base class of single-task learning datasets."""

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
        - **root** (`str`): the root directory where the original data files for constructing the STL dataset physically live.
        - **batch_size** (`int`): The batch size in train, val, test dataloader.
        - **num_workers** (`int`): the number of workers for dataloaders.
        - **custom_transforms** (`transform` or `transforms.Compose` or `None`): the custom transforms to apply to ONLY TRAIN dataset. Can be a single transform, composed transforms or no transform. `ToTensor()`, normalize and so on are not included.
        - **repeat_channels** (`int` | `None`): the number of channels to repeat. Default is None, which means no repeat. If not None, it should be an integer.
        - **to_tensor** (`bool`): whether to include `ToTensor()` transform. Default is True.
        - **resize** (`tuple[int, int]` | `None`): the size to resize the images to. Default is None, which means no resize. If not None, it should be a tuple of two integers.
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

        self.dataset_train: Any
        r"""Training dataset object. Can be PyTorch Dataset objects or any other dataset objects."""
        self.dataset_val: Any
        r"""Validation dataset object. Can be PyTorch Dataset objects or any other dataset objects."""
        self.dataset_test: Any
        r"""Test dataset object. Can be PyTorch Dataset objects or any other dataset objects."""
        self.mean: float
        r"""Mean value for normalization."""
        self.std: float
        r"""Standard deviation value for normalization."""

        STLDataset.sanity_check(self)

    def sanity_check(self) -> None:
        r"""Sanity check."""

    @abstractmethod
    def get_class_map(self) -> dict[str | int, int]:
        r"""Get the mapping of classes. It must be implemented by subclasses.

        **Returns:**
        - **class_map**(`dict[str | int, int]`): the class map. Keys are original class labels and values are integer class labels for single-task learning.
        """

    @abstractmethod
    def prepare_data(self) -> None:
        r"""Use this to download and prepare data. It must be implemented by subclasses, as required by `LightningDatamodule`."""

    def setup(self, stage: str) -> None:
        r"""Set up the dataset for different stages.

        **Args:**
        - **stage** (`str`): the stage of the experiment; one of:
            - 'fit': training and validation dataset should be assigned to `self.dataset_train` and `self.dataset_val`.
            - 'test': test dataset should be assigned to `self.dataset_test`.
        """
        if stage == "fit":
            # these two stages must be done together because a sanity check for validation is conducted before training
            pylogger.debug("Construct train and validation dataset ...")

            self.dataset_train, self.dataset_val = self.train_and_val_dataset()

            pylogger.info("Train and validation dataset are ready.")
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

            pylogger.info("Test dataset are ready.")
            pylogger.info(
                "Test dataset for size: %d",
                len(self.dataset_test),
            )

    def setup_task(self) -> None:
        r"""Set up the task for the dataset."""
        pass

    def train_and_val_transforms(self) -> transforms.Compose:
        r"""Transforms for training and validation dataset, incorporating the custom transforms with basic transforms like normalization and `ToTensor()`. It can be used in subclasses when constructing the dataset.

        **Returns:**
        - **train_and_val_transforms** (`transforms.Compose`): the composed train/val transforms.
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
        r"""Transforms for test dataset. Only basic transforms like normalization and `ToTensor()` are included. It can be used in subclasses when constructing the dataset.

        **Returns:**
        - **test_transforms** (`transforms.Compose`): the composed test transforms.
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
        )  # the order of transforms matters. No custom transforms for test

    def target_transform(self) -> Callable:
        r"""Target transform to map the original class labels to CL class labels. It can be used in subclasses when constructing the dataset.

        **Returns:**
        - **target_transform** (`Callable`): the target transform.
        """
        class_map = self.get_class_map()

        target_transform = ClassMapping(class_map=class_map)

        return target_transform

    @abstractmethod
    def train_and_val_dataset(self) -> tuple[Any, Any]:
        r"""Get the training and validation dataset. It must be implemented by subclasses.

        **Returns:**
        - **train_and_val_dataset** (`tuple[Any, Any]`): the train and validation dataset.
        """

    @abstractmethod
    def test_dataset(self) -> Any:
        """Get the test dataset. It must be implemented by subclasses.

        **Returns:**
        - **test_dataset** (`Any`): the test dataset.
        """

    def train_dataloader(self) -> DataLoader:
        r"""DataLoader generator for the stage train. It is automatically called before training.

        **Returns:**
        - **train_dataloader** (`DataLoader`): the train DataLoader.
        """

        pylogger.debug("Construct train dataloader ...")

        return DataLoader(
            dataset=self.dataset_train,
            batch_size=self.batch_size,
            shuffle=True,  # shuffle train batch to prevent overfitting
            num_workers=self.num_workers,
            drop_last=True,  # to avoid batchnorm error (when batch_size is 1)
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

    def test_dataloader(self) -> dict[int, DataLoader]:
        r"""DataLoader generator for stage test. It is automatically called before testing.

        **Returns:**
        - **test_dataloader** (`DataLoader`): the test DataLoader.
        """

        pylogger.debug("Construct test dataloader...")

        return DataLoader(
            dataset=self.dataset_test,
            batch_size=self.batch_size,
            shuffle=False,  # don't have to shuffle val or test batch
            num_workers=self.num_workers,
        )


class STLDatasetFromRaw(STLDataset):
    r"""The base class of single-task learning datasets from raw PyTorch Dataset."""

    original_dataset_python_class: type[Dataset]
    r"""The original dataset class. **It must be provided in subclasses.**"""

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
        - **root** (`str`): the root directory where the original data files for constructing the STL dataset physically live.
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

        self.original_dataset_constants: type[DatasetConstants] = (
            DATASET_CONSTANTS_MAPPING[self.original_dataset_python_class]
        )
        r"""The original dataset constants class."""

    def get_class_map(self) -> dict[str | int, int]:
        r"""Get the mapping of classes.

        **Returns:**
        - **class_map**(`dict[str | int, int]`): the class map. Key is original class label, value is integer class label for single-task learning.
        """
        return self.original_dataset_constants.CLASS_MAP

    def setup_task(self) -> None:
        r"""Set up the task for the dataset."""
        super().setup_task()

        self.mean = (
            self.original_dataset_constants.MEAN
        )  # the same with the original dataset
        self.std = (
            self.original_dataset_constants.STD
        )  # the same with the original dataset


class TaskLabelledDataset(Dataset):
    r"""The dataset class that labels the a task's dataset with the given task ID. It is used to label the dataset with the task ID for MTL experiment."""

    def __init__(self, dataset: Dataset, task_id: int) -> None:
        r"""
        **Args:**
        - **dataset** (`Dataset`): the dataset to be labelled.
        - **task_id** (`int`): the task ID to be labelled.
        """
        super().__init__()

        self.dataset: Dataset = dataset
        r"""The original dataset object."""
        self.task_id: int = task_id
        r"""The task ID."""

    def __len__(self) -> int:
        r"""The length of the dataset.

        **Returns:**
        - **length** (`int`): the length of the dataset.
        """

        return len(self.dataset)  # the same as the length of the original dataset.

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
