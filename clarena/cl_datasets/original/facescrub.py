r"""
The submodule in `cl_datasets.original` for the original FaceScrub subset dataset.
"""

__all__ = ["FaceScrub10", "FaceScrub20", "FaceScrub50", "FaceScrub100"]

import logging
import os
import pickle
import urllib.request
import zipfile
from pathlib import Path
from typing import Any, Callable

import numpy as np
from PIL import Image
from torchvision.datasets import VisionDataset

pylogger = logging.getLogger(__name__)


class FaceScrub10(VisionDataset):
    """FaceScrub-10 dataset. [FaceScrub-10](https://github.com/nkundiushuti/facescrub_subset/) is a 10-class subset of the official [Megaface Facescrub challenge](http://megaface.cs.washington.edu/participate/challenge.html), cropped and resized to 32x32 pixels."""

    base_folder: str = "facescrub_10"
    url: str = (
        "https://github.com/nkundiushuti/facescrub_subset/blob/master/data/facescrub_10.zip?raw=true"
    )
    filename: str = "facescrub_10.zip"
    train_file: str = "facescrub_train_10.pkl"
    test_file: str = "facescrub_test_10.pkl"

    def __init__(
        self,
        root: str | Path,
        train: bool = True,
        transform: Callable | None = None,
        target_transform: Callable | None = None,
        download: bool = False,
    ) -> None:
        r"""Initialize the FaceScrub10 dataset.

        **Args:**
        - **root** (`str` | `Path`): Root directory of the dataset.
        - **train** (`bool`): If True, load the training set. Otherwise, load the test set.
        - **transform** (`Callable`, optional): Transform to apply to each image.
        - **target_transform** (`Callable`, optional): Transform to apply to each label.
        - **download** (`bool`): If True, download the dataset if it's not already available.
        """
        super().__init__(root, transform=transform, target_transform=target_transform)

        self.root = os.path.expanduser(root)
        self.train = train

        if download:
            self._download()

        if not self._check_integrity():
            raise RuntimeError("Dataset not found. Use download=True to fetch it.")

        self._load_data()

    def _check_integrity(self) -> bool:
        """Check if dataset files exist."""
        file_path = os.path.join(
            self.root, self.train_file if self.train else self.test_file
        )
        return os.path.exists(file_path)

    def _download(self) -> None:
        """Download and extract the dataset."""
        os.makedirs(self.root, exist_ok=True)
        archive_path = os.path.join(self.root, self.filename)

        if not os.path.exists(archive_path):
            pylogger.debug(f"Downloading from {self.url}")
            urllib.request.urlretrieve(self.url, archive_path)

        with zipfile.ZipFile(archive_path, "r") as zip_ref:
            zip_ref.extractall(self.root)
        pylogger.debug("Downloaded and extracted dataset.")

    def _load_data(self) -> None:
        """Load features and labels from pickle."""
        file_path = os.path.join(
            self.root, self.train_file if self.train else self.test_file
        )

        with open(file_path, "rb") as f:
            data_dict = pickle.load(f)

        self.data = data_dict["features"].astype(np.uint8)
        self.labels = data_dict["labels"].astype(np.uint8)

    def __getitem__(self, index: int) -> tuple[Any, Any]:
        """Get image and label at index."""
        img_array, target = self.data[index], self.labels[index]
        img = Image.fromarray(np.transpose(img_array, (1, 2, 0)))

        if self.transform:
            img = self.transform(img)
        if self.target_transform:
            target = self.target_transform(target)

        return img, target

    def __len__(self) -> int:
        """Return number of items."""
        return len(self.data)


class FaceScrub20(VisionDataset):
    """FaceScrub-20 dataset. [FaceScrub-20](https://github.com/nkundiushuti/facescrub_subset/) is a 20-class subset of the official [Megaface Facescrub challenge](http://megaface.cs.washington.edu/participate/challenge.html), cropped and resized to 32x32 pixels."""

    base_folder: str = "facescrub_20"
    url: str = (
        "https://github.com/nkundiushuti/facescrub_subset/blob/master/data/facescrub_20.zip?raw=true"
    )
    filename: str = "facescrub_20.zip"
    train_file: str = "facescrub_train_20.pkl"
    test_file: str = "facescrub_test_20.pkl"

    def __init__(
        self,
        root: str | Path,
        train: bool = True,
        transform: Callable | None = None,
        target_transform: Callable | None = None,
        download: bool = False,
    ) -> None:
        r"""Initialize the FaceScrub20 dataset.

        **Args:**
        - **root** (`str` | `Path`): Root directory of the dataset.
        - **train** (`bool`): If True, load the training set. Otherwise, load the test set.
        - **transform** (`Callable`, optional): Transform to apply to each image.
        - **target_transform** (`Callable`, optional): Transform to apply to each label.
        - **download** (`bool`): If True, download the dataset if it's not already available.
        """
        super().__init__(root, transform=transform, target_transform=target_transform)

        self.root = os.path.expanduser(root)
        self.train = train

        if download:
            self._download()

        if not self._check_integrity():
            raise RuntimeError("Dataset not found. Use download=True to fetch it.")

        self._load_data()

    def _check_integrity(self) -> bool:
        """Check if dataset files exist."""
        file_path = os.path.join(
            self.root, self.train_file if self.train else self.test_file
        )
        return os.path.exists(file_path)

    def _download(self) -> None:
        """Download and extract the dataset."""
        os.makedirs(self.root, exist_ok=True)
        archive_path = os.path.join(self.root, self.filename)

        if not os.path.exists(archive_path):
            pylogger.debug(f"Downloading from {self.url}")
            urllib.request.urlretrieve(self.url, archive_path)

        with zipfile.ZipFile(archive_path, "r") as zip_ref:
            zip_ref.extractall(self.root)
        pylogger.debug("Downloaded and extracted dataset.")

    def _load_data(self) -> None:
        """Load features and labels from pickle."""
        file_path = os.path.join(
            self.root, self.train_file if self.train else self.test_file
        )

        with open(file_path, "rb") as f:
            data_dict = pickle.load(f)

        self.data = data_dict["features"].astype(np.uint8)
        self.labels = data_dict["labels"].astype(np.uint8)

    def __getitem__(self, index: int) -> tuple[Any, Any]:
        """Get image and label at index."""
        img_array, target = self.data[index], self.labels[index]
        img = Image.fromarray(np.transpose(img_array, (1, 2, 0)))

        if self.transform:
            img = self.transform(img)
        if self.target_transform:
            target = self.target_transform(target)

        return img, target

    def __len__(self) -> int:
        """Return number of items."""
        return len(self.data)


class FaceScrub50(VisionDataset):
    """FaceScrub-50 dataset. [FaceScrub-50](https://github.com/nkundiushuti/facescrub_subset/) is a 50-class subset of the official [Megaface Facescrub challenge](http://megaface.cs.washington.edu/participate/challenge.html), cropped and resized to 32x32 pixels."""

    base_folder: str = "facescrub_50"
    url: str = (
        "https://github.com/nkundiushuti/facescrub_subset/blob/master/data/facescrub_50.zip?raw=true"
    )
    filename: str = "facescrub_50.zip"
    train_file: str = "facescrub_train_50.pkl"
    test_file: str = "facescrub_test_50.pkl"

    def __init__(
        self,
        root: str | Path,
        train: bool = True,
        transform: Callable | None = None,
        target_transform: Callable | None = None,
        download: bool = False,
    ) -> None:
        r"""Initialize the FaceScrub50 dataset.

        **Args:**
        - **root** (`str` | `Path`): Root directory of the dataset.
        - **train** (`bool`): If True, load the training set. Otherwise, load the test set.
        - **transform** (`Callable`, optional): Transform to apply to each image.
        - **target_transform** (`Callable`, optional): Transform to apply to each label.
        - **download** (`bool`): If True, download the dataset if it's not already available.
        """
        super().__init__(root, transform=transform, target_transform=target_transform)

        self.root = os.path.expanduser(root)
        self.train = train

        if download:
            self._download()

        if not self._check_integrity():
            raise RuntimeError("Dataset not found. Use download=True to fetch it.")

        self._load_data()

    def _check_integrity(self) -> bool:
        """Check if dataset files exist."""
        file_path = os.path.join(
            self.root, self.train_file if self.train else self.test_file
        )
        return os.path.exists(file_path)

    def _download(self) -> None:
        """Download and extract the dataset."""
        os.makedirs(self.root, exist_ok=True)
        archive_path = os.path.join(self.root, self.filename)

        if not os.path.exists(archive_path):
            pylogger.debug(f"Downloading from {self.url}")
            urllib.request.urlretrieve(self.url, archive_path)

        with zipfile.ZipFile(archive_path, "r") as zip_ref:
            zip_ref.extractall(self.root)
        pylogger.debug("Downloaded and extracted dataset.")

    def _load_data(self) -> None:
        """Load features and labels from pickle."""
        file_path = os.path.join(
            self.root, self.train_file if self.train else self.test_file
        )

        with open(file_path, "rb") as f:
            data_dict = pickle.load(f)

        self.data = data_dict["features"].astype(np.uint8)
        self.labels = data_dict["labels"].astype(np.uint8)

    def __getitem__(self, index: int) -> tuple[Any, Any]:
        """Get image and label at index."""
        img_array, target = self.data[index], self.labels[index]
        img = Image.fromarray(np.transpose(img_array, (1, 2, 0)))

        if self.transform:
            img = self.transform(img)
        if self.target_transform:
            target = self.target_transform(target)

        return img, target

    def __len__(self) -> int:
        """Return number of items."""
        return len(self.data)


class FaceScrub100(VisionDataset):
    """FaceScrub-100 dataset. [FaceScrub-100](https://github.com/nkundiushuti/facescrub_subset/) is a 100-class subset of the official [Megaface Facescrub challenge](http://megaface.cs.washington.edu/participate/challenge.html), cropped and resized to 32x32 pixels.

    The dataset contains images of 100 different individuals with separate train/test splits.
    """

    base_folder: str = "facescrub_100"
    url: str = (
        "https://github.com/nkundiushuti/facescrub_subset/blob/master/data/facescrub_100.zip?raw=true"
    )
    filename: str = "facescrub_100.zip"
    train_file: str = "facescrub_train_100.pkl"
    test_file: str = "facescrub_test_100.pkl"

    def __init__(
        self,
        root: str | Path,
        train: bool = True,
        transform: Callable | None = None,
        target_transform: Callable | None = None,
        download: bool = False,
    ) -> None:
        r"""Initialize the FaceScrub100 dataset.

        **Args:**
        - **root** (`str` | `Path`): Root directory of the dataset.
        - **train** (`bool`): If True, load the training split; otherwise, load the test split.
        - **transform** (`Callable`, optional): Function to apply to each image.
        - **target_transform** (`Callable`, optional): Function to apply to each label.
        - **download** (`bool`): If True, download the dataset if it's not found in root.
        """
        super().__init__(root, transform=transform, target_transform=target_transform)

        self.root = os.path.expanduser(root)
        self.train = train

        if download:
            self._download()

        if not self._check_integrity():
            raise RuntimeError("Dataset not found. Use download=True to download it.")

        self._load_data()

    def _check_integrity(self) -> bool:
        """Check if the dataset files are present."""
        expected_file = os.path.join(
            self.root, self.train_file if self.train else self.test_file
        )
        return os.path.exists(expected_file)

    def _download(self) -> None:
        """Download the dataset archive and extract."""
        os.makedirs(self.root, exist_ok=True)
        archive_path = os.path.join(self.root, self.filename)

        if not os.path.exists(archive_path):
            pylogger.debug(f"Downloading from {self.url} to {archive_path}")
            urllib.request.urlretrieve(self.url, archive_path)

        with zipfile.ZipFile(archive_path, "r") as zip_ref:
            zip_ref.extractall(self.root)
        pylogger.debug("Extraction completed.")

    def _load_data(self) -> None:
        """Load the dataset from the pickle files."""
        file_path = os.path.join(
            self.root, self.train_file if self.train else self.test_file
        )

        with open(file_path, "rb") as f:
            data_dict = pickle.load(f)

        self.data = data_dict["features"].astype(np.uint8)  # Shape: (N, C, H, W)
        self.labels = data_dict["labels"].astype(np.uint8)

    def __getitem__(self, index: int) -> tuple[Any, Any]:
        """Get item by index.

        **Args:**
        - **index** (`int`): Index of the item to fetch.

        **Returns:**
        - (`PIL.Image`, `int`): Tuple of the image and its label.
        """
        img_array, target = self.data[index], self.labels[index]
        img = Image.fromarray(np.transpose(img_array, (1, 2, 0)))

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.data)
