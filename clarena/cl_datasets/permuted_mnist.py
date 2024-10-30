"""
The submodule in `cl_datasets` for Permuted MNIST dataset.
"""

__all__ = ["PermutedMNIST"]

import logging
from typing import Callable

import torch
from torch.utils.data import Dataset, random_split
from torchvision.datasets import MNIST
from torchvision.transforms import transforms

from clarena.cl_datasets import CLPermutedDataset

# always get logger for built-in logging in each module
pylogger = logging.getLogger(__name__)


class PermutedMNIST(CLPermutedDataset):
    """Permuted MNIST dataset."""

    img_size: torch.Size = torch.Size([1, 28, 28])
    """The size of MNIST images."""

    mean_original: tuple[float] = (0.1307,)
    """The mean values for normalisation."""

    std_original: tuple[float] = (0.3081,)
    """The standard deviation values for normalisation."""

    def __init__(
        self,
        root: str,
        num_tasks: int,
        validation_percentage: float,
        batch_size: int = 1,
        num_workers: int = 10,
        custom_transforms: Callable | transforms.Compose | None = None,
        custom_target_transforms: Callable | transforms.Compose | None = None,
        permutation_mode: str = "first_channel_only",
        permutation_seeds: list[int] | None = None,
    ) -> None:
        """Initialise the Permuted MNIST dataset.

        **Args:**
        - **root** (`str`): the root directory where the original MNIST data 'MNIST/raw/train-images-idx3-ubyte' and 'MNIST/raw/t10k-images-idx3-ubyte' live.
        - **num_tasks** (`int`): the maximum number of tasks supported by the CL dataset.
        - **validation_percentage** (`float`): the percentage to randomly split some of the training data into validation data.
        - **batch_size** (`int`): The batch size in train, val, test dataloader.
        - **num_workers** (`int`): the number of workers for dataloaders.
        - **custom_transforms** (`transform` or `transforms.Compose` or `None`): the custom transforms to apply to ONLY TRAIN dataset. Can be a single transform, composed transforms or no transform.
        `ToTensor()`, normalise, permute and so on are not included.
        - **custom_target_transforms** (`transform` or `transforms.Compose` or `None`): the custom target transforms to apply to dataset labels. Can be a single transform, composed transforms or no transform. CL class mapping is not included.
        - **permutation_mode** (`str`): the mode of permutation, should be one of the following:
            1. 'all': permute all pixels.
            2. 'by_channel': permute channel by channel separately. All channels are applied the same permutation order.
            3. 'first_channel_only': permute only the first channel.
        - **permutation_seeds** (`list[int]` or `None`): the seeds for permutation operations used to construct tasks. Make sure it has the same number of seeds as `num_tasks`. Default is None, which creates a list of seeds from 1 to `num_tasks`.
        """
        super().__init__(
            root=root,
            num_tasks=num_tasks,
            validation_percentage=validation_percentage,
            batch_size=batch_size,
            num_workers=num_workers,
            custom_transforms=custom_transforms,
            custom_target_transforms=custom_target_transforms,
            permutation_mode=permutation_mode,
            permutation_seeds=permutation_seeds,
        )

    def cl_class_map(self, task_id: int) -> dict[str | int, int]:
        """The mapping of classes of task `task_id` to fit continual learning settings `self.cl_paradigm`.

        **Args:**
        - **task_id** (`int`): The task ID to query CL class map.

        **Returns:**
        - The CL class map of the task. Key is original class label, value is integer class label for continual learning.
            - If `self.cl_paradigm` is 'TIL', the mapped class labels of a task should be continuous integers from 0 to the number of classes.
            - If `self.cl_paradigm` is 'CIL', the mapped class labels of a task should be continuous integers from the number of classes of previous tasks to the number of classes of the current task.
        """
        if self.cl_paradigm == "TIL":
            return {i: i for i in range(10)}
        if self.cl_paradigm == "CIL":
            return {i: i + (task_id - 1) * 10 for i in range(10)}

    def prepare_data(self) -> None:
        """Download the original MNIST dataset if haven't."""
        # just download
        MNIST(root=self.root, train=True, download=True)
        MNIST(root=self.root, train=False, download=True)

        pylogger.debug(
            "The original MNIST dataset has been downloaded to %s.", self.root
        )

    def train_and_val_dataset(self) -> tuple[Dataset, Dataset]:
        """Get the training and validation dataset of task `self.task_id`.

        **Returns:**
        - The train and validation dataset of task `self.task_id`."""
        dataset_train_and_val = MNIST(
            root=self.root,
            train=True,
            transform=self.train_and_val_transforms(to_tensor=True),
            target_transform=self.target_transforms(),
            download=False,
        )
        return random_split(
            dataset_train_and_val,
            lengths=[1 - self.validation_percentage, self.validation_percentage],
        )

    def test_dataset(self) -> Dataset:
        """Get the test dataset of task `self.task_id`.

        **Returns:**
        - The test dataset of task `self.task_id`.
        """
        return MNIST(
            root=self.root,
            train=False,
            transform=self.test_transforms(to_tensor=True),
            target_transform=self.target_transforms(),
            download=False,
        )
