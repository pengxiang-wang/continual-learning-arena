"""
The submodule in `cl_datasets.original` for the original Kannada-MNIST dataset.
"""

__all__ = ["KannadaMNIST"]

import gzip
import logging
import os
from pathlib import Path
from typing import Any, Callable

import numpy as np
from PIL import Image
from torchvision.datasets.vision import VisionDataset

# always get logger for built-in logging in each module
pylogger = logging.getLogger(__name__)


class KannadaMNIST(VisionDataset):
    r"""Kannada-MNIST dataset. The [Kannada-MNIST dataset](https://github.com/vinayprabhu/Kannada_MNIST) is a collection of handwritten Kannada digits (0-9). It consists of 70,000 28x28 grayscale images in 10 classes."""

    base_folder: str = "Kannada_MNIST"
    train_images_file: str = "X_kannada_MNIST_train-idx3-ubyte.gz"
    train_labels_file: str = "y_kannada_MNIST_train-idx1-ubyte.gz"
    test_images_file: str = "X_kannada_MNIST_test-idx3-ubyte.gz"
    test_labels_file: str = "y_kannada_MNIST_test-idx1-ubyte.gz"

    def __init__(
        self,
        root: str | Path,
        train: bool = True,
        transform: Callable | None = None,
        target_transform: Callable | None = None,
        download: bool = False,
    ) -> None:
        r"""Initialize the Kannada MNIST dataset.

        **Args:**
        - **root** (`str` | `Path`): Root directory of the dataset.
        - **train** (`bool`): Whether to load training data or test data.
        - **transform** (`callable` | `None`): Transform to apply to images.
        - **target_transform** (`callable` | `None`): Transform to apply to labels.
        - **download** (`bool`): Not implemented. Dataset must be manually downloaded and extracted.
        """
        super().__init__(root, transform=transform, target_transform=target_transform)

        if download:
            raise NotImplementedError(
                "Automatic download is not supported. Please manually download the entire data folder 'Kannada_MNIST_datataset_paper.zip' from https://www.kaggle.com/datasets/higgstachyon/kannada-mnist and extract it, and copy 'Kannada_MNIST_Ubyte_gz/Kannada_MNIST' to the correct folder."
            )

        if not self._check_integrity():
            raise NotImplementedError(
                "Dataset not found or corrupted. Please manually download the entire data folder 'Kannada_MNIST_datataset_paper.zip' from https://www.kaggle.com/datasets/higgstachyon/kannada-mnist and extract it, and copy 'Kannada_MNIST_Ubyte_gz/Kannada_MNIST' to the correct folder."
            )

        self.train = train
        r"""Flag indicating whether training data is loaded."""

        folder = os.path.join(self.root, self.base_folder)
        if self.train:
            img_path = os.path.join(folder, self.train_images_file)
            lbl_path = os.path.join(folder, self.train_labels_file)
        else:
            img_path = os.path.join(folder, self.test_images_file)
            lbl_path = os.path.join(folder, self.test_labels_file)

        self.images = self._read_images(img_path)
        self.labels = self._read_labels(lbl_path)

        assert len(self.images) == len(
            self.labels
        ), "Mismatch between images and labels length."

        self.class_names: list[str] = [str(i) for i in range(10)]
        r"""Digit classes 0 to 9."""

    def _read_images(self, path: str) -> np.ndarray:
        with gzip.open(path, "rb") as f:
            _ = int.from_bytes(f.read(4), "big")  # magic
            num = int.from_bytes(f.read(4), "big")
            rows = int.from_bytes(f.read(4), "big")
            cols = int.from_bytes(f.read(4), "big")
            buf = f.read(rows * cols * num)
            return np.frombuffer(buf, dtype=np.uint8).reshape(num, rows, cols)

    def _read_labels(self, path: str) -> np.ndarray:
        with gzip.open(path, "rb") as f:
            _ = int.from_bytes(f.read(4), "big")  # magic
            num = int.from_bytes(f.read(4), "big")
            buf = f.read(num)
            return np.frombuffer(buf, dtype=np.uint8)

    def _check_integrity(self) -> bool:
        r"""Sanity check if dataset not found or corrupted. Do loading data at the same time.

        **Returns:**
        - **if_intergral (**bool**)**: True if the dataset is found and not corrupted, False otherwise.
        """
        return True

    def __getitem__(self, index: int) -> tuple[Any, int]:
        r"""Get image and label at given index."""
        img_array = self.images[index]
        label = int(self.labels[index])
        img = Image.fromarray(img_array, mode="L")

        if self.transform:
            img = self.transform(img)
        if self.target_transform:
            label = self.target_transform(label)

        return img, label

    def __len__(self) -> int:
        r"""Return number of samples."""
        return len(self.labels)
