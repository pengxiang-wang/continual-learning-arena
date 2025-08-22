r"""
The submodule in `cl_datasets.original` for the original NotMNIST dataset.
"""

__all__ = [
    "NotMNIST",
    "NotMNISTFromHAT",
]

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

pylogger = logging.getLogger(__name__)


class NotMNIST(VisionDataset):
    r"""NotMNIST dataset. The [original NotMNIST dataset](https://yaroslavvb.blogspot.com/2011/09/notmnist-dataset.html) is a collection of letters (including A-J). It consists 28x28 grayscale images in 10 classes. The larger dataset contains 500,000 images, while the smaller dataset contains around 19,000 images. This class loads the larger dataset as training and the smaller dataset as test set."""

    small_base_folder: str = "notMNIST_small"
    large_base_folder: str = "notMNIST_large"

    small_url: str = "http://yaroslavvb.com/upload/notMNIST/notMNIST_small.tar.gz"
    large_url: str = "http://yaroslavvb.com/upload/notMNIST/notMNIST_large.tar.gz"

    small_filename: str = "notMNIST_small.tar.gz"
    large_filename: str = "notMNIST_large.tar.gz"

    small_tgz_md5: str = "c9890a473a9769fda4bdf314aaf500dd"
    large_tgz_md5: str = "70a95b805ecfb6592c48e196df7c1499"

    def __init__(
        self,
        root: str | Path,
        train: bool = True,
        transform: Callable | None = None,
        target_transform: Callable | None = None,
        download: bool = False,
    ) -> None:
        r"""Initialize the NotMNIST dataset.

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
        self.class_names: list[str] = list("ABCDEFGHIJ")
        self.loader: str = default_loader
        self.train: bool = train
        self.url: str = self.large_url if self.train else self.small_url
        self.base_folder: str = (
            self.large_base_folder if self.train else self.small_base_folder
        )
        self.filename: str = self.large_filename if self.train else self.small_filename
        self.tgz_md5: str = self.large_tgz_md5 if self.train else self.small_tgz_md5

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError(
                "Dataset not found or corrupted. Use download=True to get it."
            )

    def _load_data(self) -> None:
        r"""Load the dataset."""
        cache_file = os.path.join(self.root, f"{self.base_folder}_metadata.csv")
        if os.path.exists(cache_file):
            self.data = pd.read_csv(cache_file)
            return

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
        self.data.to_csv(cache_file, index=False)

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
            pylogger.debug("Files already downloaded and verified.")
            return

        filename = os.path.basename(self.url)
        archive_path = os.path.join(self.root, filename)

        download_url(self.url, self.root, filename=filename, md5=self.tgz_md5)

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


# The following implementation is copied from HAT


import pickle
import urllib

import numpy as np
import torch


class NotMNISTFromHAT(torch.utils.data.Dataset):
    """The notMNIST dataset is a image recognition dataset of font glypyhs for the letters A through J useful with simple neural networks. It is quite similar to the classic MNIST dataset of handwritten digits 0 through 9.

    Args:
        root (string): Root directory of dataset where directory ``Traffic signs`` exists.
        split (string): One of {'train', 'test'}.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and puts it in root directory.
            If dataset is already downloaded, it is not downloaded again.

    """

    def __init__(
        self, root, train=True, transform=None, target_transform=None, download=False
    ):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.filename = "notmnist.zip"
        self.url = "https://github.com/nkundiushuti/notmnist_convert/blob/master/notmnist.zip?raw=true"

        fpath = os.path.join(root, self.filename)
        if not os.path.isfile(fpath):
            if not download:
                raise RuntimeError(
                    "Dataset not found. You can use download=True to download it"
                )
            else:
                print("Downloading from " + self.url)
                self.download()

        training_file = "notmnist_train.pkl"
        testing_file = "notmnist_test.pkl"
        if train:
            with open(os.path.join(root, training_file), "rb") as f:
                # u = pickle._Unpickler(f)
                # u.encoding = 'latin1'
                # train  = u.load()
                train = pickle.load(f)
            self.data = train["features"].astype(np.uint8)
            self.labels = train["labels"].astype(np.uint8)
        else:
            with open(os.path.join(root, testing_file), "rb") as f:
                # u = pickle._Unpickler(f)
                # u.encoding = 'latin1'
                # test  = u.load()
                test = pickle.load(f)

            self.data = test["features"].astype(np.uint8)
            self.labels = test["labels"].astype(np.uint8)

    def __getitem__(self, index):
        """
        Args: index (int): Index
        Returns: tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.labels[index]
        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img[0])

        if self.transform is not None:
            img = self.transform(img)

        return img, target

    def __len__(self):
        return len(self.data)

    def download(self):
        import errno

        root = os.path.expanduser(self.root)

        fpath = os.path.join(root, self.filename)

        try:
            os.makedirs(root)
        except OSError as e:
            if e.errno == errno.EEXIST:
                pass
            else:
                raise
        urllib.request.urlretrieve(self.url, fpath)

        import zipfile

        zip_ref = zipfile.ZipFile(fpath, "r")
        zip_ref.extractall(root)
        zip_ref.close()
