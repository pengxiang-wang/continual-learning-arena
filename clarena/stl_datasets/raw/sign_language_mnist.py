"""
The submodule in `cl_datasets.original` for the original Sign Language MNIST dataset.
"""

__all__ = ["SignLanguageMNIST"]

import logging
import os
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
from PIL import Image
from torchvision.datasets import VisionDataset

# always get logger for built-in logging in each module
pylogger = logging.getLogger(__name__)


class SignLanguageMNIST(VisionDataset):
    r"""Sign Language MNIST dataset. The [Sign Language MNIST dataset](https://www.kaggle.com/datamunge/sign-language-mnist) is a collection of hand gesture images representing ASL letters (A-Y, excluding J). It consists of 34,627 28x28 grayscale images in 24 classes."""

    base_folder: str = "archive"
    r"""The folder name where the main dataset (images) are stored."""

    train_csv: str = "sign_mnist_train.csv"
    r"""The CSV file name for the training dataset."""

    test_csv: str = "sign_mnist_test.csv"
    r"""The CSV file name for the test dataset."""

    def __init__(
        self,
        root: str | Path,
        train: bool = True,
        transform: Callable | None = None,
        target_transform: Callable | None = None,
        download: bool = False,
    ) -> None:
        r"""Initialize the Sign Language MNIST dataset.

        **Args:**
        - **root** (`str` | `Path`): Root directory of the dataset.
        - **train** (`bool`): If True, creates dataset from training set, otherwise from test set.
        - **transform** (`callable` | `None`): A function/transform that takes in a PIL image and returns a transformed version.
        - **target_transform** (`callable` | `None`): A function/transform that takes in the target and transforms it.
        - **download** (`bool`): Placeholder flag. Download is not supported. Dataset must be manually placed in the root directory.
        """
        super().__init__(root, transform=transform, target_transform=target_transform)

        self.train: bool = train
        r"""Store the flag to indicate whether to load training or test dataset."""
        self.csv_file: str = self.train_csv if self.train else self.test_csv
        r"""Store the filename of the CSV data."""
        self.class_names: list[str] = list("ABCDEFGHIKLMNOPQRSTUVWXY")
        r"""Store the class names of the dataset (A-Y, excluding J)."""
        self.data: pd.DataFrame
        r"""Store the full dataset loaded from CSV (image pixels and labels)."""

        if download:
            raise NotImplementedError(
                "Automatic download is not supported. Please manually download the entire data folder 'archive' from https://www.kaggle.com/datamunge/sign-language-mnist and extract it in the correct folder."
            )

        csv_path = os.path.join(self.root, self.base_folder, self.csv_file)
        if not os.path.isfile(csv_path):
            raise RuntimeError(
                f"{self.csv_file} not found in {self.root}/{self.base_folder}. Please manually download the entire data folder 'archive' from https://www.kaggle.com/datamunge/sign-language-mnist and extract it in the correct folder."
            )

        self.data = pd.read_csv(csv_path)

    def __getitem__(self, index: int) -> tuple[Any, Any]:
        r"""Get a data sample from the dataset.

        **Args:**
        - **index** (`int`): Index of the item to retrieve.

        **Returns:**
        - **img** (`PIL.Image`): The image.
        - **target** (`int`): The label.
        """
        row = self.data.iloc[index]
        label = int(row.iloc[0])
        img_array = np.array(row[1:], dtype=np.uint8).reshape(28, 28)
        img = Image.fromarray(img_array, mode="L")

        if self.transform:
            img = self.transform(img)
        if self.target_transform:
            label = self.target_transform(label)

        return img, label

    def __len__(self) -> int:
        r"""Get the number of samples in the dataset.

        **Returns:**
        - **int**: Total number of samples.
        """
        return len(self.data)
