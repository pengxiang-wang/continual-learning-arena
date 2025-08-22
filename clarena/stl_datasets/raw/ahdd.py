"""
The submodule in `cl_datasets.original` for the original Arabic Handwritten Digits dataset.
"""

__all__ = ["ArabicHandwrittenDigits"]


import logging
import os
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
from PIL import Image
from torchvision.datasets.vision import VisionDataset

# always get logger for built-in logging in each module
pylogger = logging.getLogger(__name__)


class ArabicHandwrittenDigits(VisionDataset):
    r"""Arabic Handwritten Digits Dataset (CSV version). The [Arabic Handwritten Digits dataset](https://www.kaggle.com/datasets/mloey1/ahdd1) is a collection of handwritten Arabic digits (0-9). The dataset contains 60,000 training and 10,000 testing samples of handwritten Arabic digits (0-9), each 28x28 grayscale image."""

    base_folder: str = "Arabic Handwritten Digits Dataset CSV"
    train_images_file: str = "csvTrainImages 60k x 784.csv"
    train_labels_file: str = "csvTrainLabel 60k x 1.csv"
    test_images_file: str = "csvTestImages 10k x 784.csv"
    test_labels_file: str = "csvTestLabel 10k x 1.csv"

    def __init__(
        self,
        root: str | Path,
        train: bool = True,
        transform: Callable | None = None,
        target_transform: Callable | None = None,
        download: bool = False,
    ) -> None:
        r"""Initialize the Arabic Handwritten Digits CSV dataset.

        **Args:**
        - **root** (`str` | `Path`): Root directory containing extracted CSV files.
        - **train** (`bool`): If True, use training set; else use test set.
        - **transform** (`callable` | `None`): Image transformation function.
        - **target_transform** (`callable` | `None`): Target transformation function.
        - **download** (`bool`): Not implemented. Must place files manually.
        """
        super().__init__(root, transform=transform, target_transform=target_transform)

        if download:
            raise NotImplementedError(
                "Automatic download is not supported. Please manually download the dataset folder containing 4 csv files 'Arabic Handwritten Digits Dataset CSV' from https://www.kaggle.com/datasets/mloey1/ahdd1 and extract it in the correct folder."
            )

        self.train: bool = train
        r"""Flag indicating whether to use training or testing dataset."""

        if self.train:
            images_path = os.path.join(
                self.root, self.base_folder, self.train_images_file
            )
            labels_path = os.path.join(
                self.root, self.base_folder, self.train_labels_file
            )
        else:
            images_path = os.path.join(
                self.root, self.base_folder, self.test_images_file
            )
            labels_path = os.path.join(
                self.root, self.base_folder, self.test_labels_file
            )

        if not os.path.isfile(images_path) or not os.path.isfile(labels_path):
            raise RuntimeError(f"Missing CSV files in {self.root}/{self.base_folder}.")

        self.images = pd.read_csv(images_path, header=None).values.astype(np.uint8)
        self.labels = pd.read_csv(labels_path, header=None).values.flatten().astype(int)

        assert len(self.images) == len(
            self.labels
        ), "Mismatched number of images and labels."

        self.class_names: list[str] = [str(i) for i in range(10)]
        r"""Digit classes from 0 to 9."""

    def __getitem__(self, index: int) -> tuple[Any, int]:
        r"""Get a sample from the dataset.

        **Args:**
        - **index** (`int`): Index of the sample.

        **Returns:**
        - **img** (`PIL.Image`): The 28x28 grayscale image.
        - **target** (`int`): Corresponding digit label.
        """
        img_array = self.images[index].reshape(28, 28)
        label = self.labels[index]
        img = Image.fromarray(img_array, mode="L")

        if self.transform:
            img = self.transform(img)
        if self.target_transform:
            label = self.target_transform(label)

        return img, label

    def __len__(self) -> int:
        r"""Return the total number of samples."""
        return len(self.labels)
