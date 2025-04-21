r"""
The submodule in `cl_datasets.original` for the original NotMNIST dataset.
"""

__all__ = ["NotMNIST"]


import logging
import os
import tarfile
from pathlib import Path
from typing import Any, Callable

import pandas as pd
from PIL import Image
from torchvision.datasets import VisionDataset
from torchvision.datasets.folder import default_loader
from torchvision.datasets.utils import download_url

# always get logger for built-in logging in each module
pylogger = logging.getLogger(__name__)


class NotMNIST(VisionDataset):
    r"""NotMNIST dataset. The [original NotMNIST dataset](https://yaroslavvb.blogspot.com/2011/09/notmnist-dataset.html) is a collection of letters (including A-J). It consists 28x28 grayscale images in 10 classes. The larger dataset contains 500,000 images, while the smaller dataset contains around 19,000 images. This class loads the larger dataset as training and the smaller dataset as test set."""

    small_base_folder: str = "notMNIST_small"
    r"""The folder name where the main small dataset (images) are stored."""

    large_base_folder: str = "notMNIST_large"
    r"""The folder name where the main large dataset (images) are stored."""

    small_url: str = "http://yaroslavvb.com/upload/notMNIST/notMNIST_small.tar.gz"
    r"""The URL to download the smaller NotMNIST dataset."""

    large_url: str = "http://yaroslavvb.com/upload/notMNIST/notMNIST_large.tar.gz"
    r"""The URL to download the larger NotMNIST dataset."""

    small_filename: str = "notMNIST_small.tar.gz"
    r"""The filename of the smaller dataset tar file."""

    large_filename: str = "notMNIST_large.tar.gz"
    r"""The filename of the larger dataset tar file."""

    small_tgz_md5: str = "c9890a473a9769fda4bdf314aaf500dd"
    r"""The md5 hash of the dataset smaller tar file."""

    large_tgz_md5: str = "70a95b805ecfb6592c48e196df7c1499"
    r"""The md5 hash of the dataset larger tar file."""

    def __init__(
        self,
        root: str | Path,
        train: bool = True,
        transform: Callable | None = None,
        target_transform: Callable | None = None,
        download: bool = False,
    ) -> None:
        r"""Initialise the NotMNIST dataset.

        **Args:**
        - **root** (`str` | `Path`): Root directory of the dataset.
        - **train** (`bool` | `None`): If True, creates dataset from training set, otherwise creates from test set.
        - **transform** (`callable` | `None`): A function/transform that  takes in an PIL image and returns a transformed version. E.g, `transforms.RandomCrop`
        - **target_transform** (`callable` | `None`): A function/transform that takes in the target and transforms it.
        - **download** (`bool`): If true, downloads the dataset from the internet and puts it in root directory. If dataset is already downloaded, it is not downloaded again.
        """
        VisionDataset.__init__(
            self, root=root, transform=transform, target_transform=target_transform
        )

        self.data: pd.DataFrame
        r"""Store the metadata of the dataset. The metadata includes the image file paths and the class labels."""
        self.class_names: list[str] = list("ABCDEFGHIJ")
        r"""Store the class names of the dataset."""

        self.loader: str = default_loader
        r"""Store the loader function to load the images."""

        self.train: bool = train
        r"""Store the flag to indicate whether to load training or test dataset."""
        self.url: str = self.large_url if self.train else self.small_url
        r"""Store the URL to download the dataset from."""
        self.base_folder: str = (
            self.large_base_folder if self.train else self.small_base_folder
        )
        r"""Store the folder name where the main data (images) are stored."""
        self.filename: str = self.large_filename if self.train else self.small_filename
        r"""Store the filename of the dataset."""
        self.tgz_md5: str = self.large_tgz_md5 if self.train else self.small_tgz_md5
        r"""Store the md5 hash of the dataset tar file."""

        if download:
            self.download()  # download the dataset if not already downloaded

        if not self._check_integrity():
            raise RuntimeError(
                "Dataset not found or corrupted. Use download=True to get it."
            )

    def _load_data(self) -> None:
        r"""Load the dataset."""

        data_dir = os.path.join(self.root, self.base_folder)
        samples = []

        for label_index, folder in enumerate(sorted(os.listdir(data_dir))):
            folder_path = os.path.join(data_dir, folder)
            if not os.path.isdir(folder_path):
                continue

            for fname in os.listdir(folder_path):
                img_path = os.path.join(folder_path, fname)
                try:
                    with Image.open(img_path) as img:
                        img.verify()
                    samples.append((img_path, label_index))
                except Exception:
                    pylogger.warning(f"Corrupted file skipped: {img_path}")
                    continue

        self.data = pd.DataFrame(samples, columns=["filepath", "target"])

    def _check_integrity(self) -> bool:
        r"""Sanity check if dataset not found or corrupted. Do loading data at the same time.

        **Returns:**
        - **if_intergral (**bool**)**: True if the dataset is found and not corrupted, False otherwise.
        """
        try:
            self._load_data()
        except Exception:
            return False

        for path in self.data["filepath"]:
            if not os.path.isfile(path):
                pylogger.warning(f"Missing file: {path}")
                return False
        return True

    def download(self) -> None:
        r"""Download the NotMNIST data if it doesn't exist already."""

        if self._check_integrity():
            pylogger.info("Files already downloaded and verified.")
            return

        # Choose the appropriate URL and filename
        filename = os.path.basename(self.url)
        archive_path = os.path.join(self.root, filename)

        # Download and verify the archive
        download_url(self.url, self.root, filename=filename, md5=self.tgz_md5)

        # Extract the contents of the archive
        with tarfile.open(archive_path, "r:gz") as tar:
            tar.extractall(path=self.root)

    def __getitem__(self, index: int) -> tuple[Any, Any]:
        r"""Get a data sample from the dataset.

        **Args:**
        - **index** (`int`): Index of the item to retrieve.

        **Returns:**
        - **img** (`PIL.Image`): The image.
        - **target** (`int`): The label.
        """
        sample = self.data.iloc[index]
        img = Image.open(sample.filepath).convert("L")
        target = sample.target

        if self.transform:
            img = self.transform(img)
        if self.target_transform:
            target = self.target_transform(target)

        return img, target

    def __len__(self) -> int:
        r"""Get the number of samples in the dataset.

        **Returns:**
        - **int**: Total number of samples.
        """
        return len(self.data)
