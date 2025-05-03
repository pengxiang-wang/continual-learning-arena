# r"""
# The submodule in `cl_datasets` for fake dataset for continual learning.
# """

# __all__ = ["CLFakeData"]

# import logging
# from typing import Callable

# import torch
# from torch.utils.data import Dataset, random_split
# from torchvision.datasets import FakeData
# from torchvision.transforms import transforms

# from clarena.cl_datasets import CLDataset

# # always get logger for built-in logging in each module
# pylogger = logging.getLogger(__name__)


# class CLFakeData(CLDataset):
#     r"""Permuted fake dataset. A fake dataset returns randomly generated images."""

#     def __init__(
#         self,
#         num_tasks: int,
#         num_samples: int | list[int],
#         img_sizes: tuple | list[tuple],
#         num_classes: int | list[int],
#         validation_percentage: float,
#         test_percentage: float,
#         batch_size: int | list[int] = 1,
#         num_workers: int | list[int] = 0,
#         custom_transforms: (
#             Callable
#             | transforms.Compose
#             | None
#             | list[Callable | transforms.Compose | None]
#         ) = None,
#         custom_target_transforms: (
#             Callable
#             | transforms.Compose
#             | None
#             | list[Callable | transforms.Compose | None]
#         ) = None,
#         random_offsets: list[int] | None = None,
#     ) -> None:
#         r"""Initialise the Permuted Caltech dataset object providing the root where data files live.

#         **Args:**
#         - **num_tasks** (`int`): the maximum number of tasks supported by the CL dataset.
#         - **num_samples** (`int` | `list[int]`): list of number of samples for each task. If it is a single integer, it will be used for all tasks.
#         - **img_sizes** (`tuple` | `list[tuple]`): list of image sizes for each task. If it is a single tuple, it will be used for all tasks.
#         - **num_classes** (`int` | `list[int]`): list of number of classes for each task. If it is a single integer, it will be used for all tasks.
#         - **validation_percentage** (`float`): the percentage to randomly split some of the training data into validation data.
#         - **test_percentage** (`float`): the percentage to randomly split some of the entire data into test data.
#         - **batch_size** (`int` | `list[int]`): The batch size in train, val, test dataloader. If `list[str]`, it should be a list of integers, each integer is the batch size for each task.
#         - **num_workers** (`int` | `list[int]`): the number of workers for dataloaders. If `list[str]`, it should be a list of integers, each integer is the num of workers for each task.
#         - **custom_transforms** (`transform` or `transforms.Compose` or `None` or list of them): the custom transforms to apply to ONLY TRAIN dataset. Can be a single transform, composed transforms or no transform. `ToTensor()`, normalise, permute and so on are not included. If it is a list, each item is the custom transforms for each task.
#         - **custom_target_transforms** (`transform` or `transforms.Compose` or `None` or list of them): the custom target transforms to apply to dataset labels. Can be a single transform, composed transforms or no transform. CL class mapping is not included. If it is a list, each item is the custom transforms for each task.
#         - **random_offsets** (`list[int]` | `None`): list of random offsets (fake data generation seeds) for each task. Make sure it has the same number of seeds as `num_tasks`. Default is None, which creates a list of seeds from 1 to `num_tasks`.
#         """
#         CLDataset.__init__(
#             self,
#             root=None,
#             num_tasks=num_tasks,
#             batch_size=batch_size,
#             num_workers=num_workers,
#             custom_transforms=custom_transforms,
#             repeat_channels=None,
#             to_tensor=True,  # fake data are PIL images
#             resize=None,
#
#         )

#         self.validation_percentage: float = validation_percentage
#         """Store the percentage to randomly split some of the training data into validation data."""
#         self.test_percentage: float = test_percentage
#         """Store the percentage to randomly split some of the entire data into test data."""

#     def prepare_data(self) -> None:
#         r"""Download the original fake dataset if haven't."""
#         # just download the original dataset once
# if self.task_id != 1:
# return  # download all original datasets only at the beginning of first task
#         FakeData(root=self.root_t, download=True)

#         pylogger.debug(
#             "The original fake dataset has been downloaded to %s.", self.root_t
#         )

#     def train_and_val_dataset(self) -> tuple[Dataset, Dataset]:
#         """Get the training and validation dataset of task `self.task_id`.

#         **Returns:**
#         - **train_and_val_dataset** (`tuple[Dataset, Dataset]`): the train and validation dataset of task `self.task_id`.
#         """
#         dataset_all = FakeData(
#             root=self.root_t,
#             transform=self.train_and_val_transforms(),
#             download=False,
#         )
#         dataset_all.target_transform = (
#             self.target_transforms()
#         )  # must be set here before random split, otherwise the target transform will not be applied to the dataset

#         dataset_train_and_val, _ = random_split(
#             dataset_all,
#             lengths=[
#                 1 - self.test_percentage,
#                 self.test_percentage,
#             ],
#             generator=torch.Generator().manual_seed(
#                 42
#             ),  # this must be set fixed to make sure the datasets across experiments are the same. Don't handle it to global seed as it might vary across experiments
#         )

#         return random_split(
#             dataset_train_and_val,
#             lengths=[1 - self.validation_percentage, self.validation_percentage],
#             generator=torch.Generator().manual_seed(
#                 42
#             ),  # this must be set fixed to make sure the datasets across experiments are the same. Don't handle it to global seed as it might vary across experiments
#         )

#     def test_dataset(self) -> Dataset:
#         r"""Get the test dataset of task `self.task_id`.

#         **Returns:**
#         - **test_dataset** (`Dataset`): the test dataset of task `self.task_id`.
#         """
#         dataset_all = FakeData(
#             root=self.root_t,
#             transform=self.train_and_val_transforms(),
#             download=False,
#         )

#         dataset_all.target_transform = (
#             self.target_transforms()
#         )  # must be set here before random split, otherwise the target transform will not be applied to the dataset

#         _, dataset_test = random_split(
#             dataset_all,
#             lengths=[1 - self.test_percentage, self.test_percentage],
#             generator=torch.Generator().manual_seed(42),
#         )

#         return dataset_test
