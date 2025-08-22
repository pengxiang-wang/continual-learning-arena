r"""
The submodule in `cl_datasets.original` for the original CUB-200-2011 dataset.
"""

__all__ = ["CUB2002011"]

import logging
import os
import tarfile
from pathlib import Path
from typing import Any, Callable

import pandas as pd
from torchvision.datasets import VisionDataset
from torchvision.datasets.folder import default_loader
from torchvision.datasets.utils import download_file_from_google_drive

# always get logger for built-in logging in each module
pylogger = logging.getLogger(__name__)


class CUB2002011(VisionDataset):
    """CUB-200-2011 dataset. [CUB(Caltech-UCSD Birds)-200-2011)](https://www.vision.caltech.edu/datasets/cub_200_2011/) is a bird image dataset. It consists of 120,000 64x64 colour images in 200 classes, with 500 training, 50 validation and 50 test examples per class."""

    base_folder: str = "CUB_200_2011/images"
    r"""The folder name where the main data (images) are stored."""

    url = "http://www.vision.caltech.edu/visipedia-data/CUB-200-2011/CUB_200_2011.tgz"
    r"""The url to download the dataset from."""

    file_id = "1hbzc_P1FuxMkcabkgn9ZKinBwW683j45"
    r"""The file id to download the dataset from Google Drive."""

    filename = "CUB_200_2011.tgz"
    r"""The filename of the dataset."""

    tgz_md5 = "97eceeb196236b17998738112f37df78"
    r"""The md5 hash of the dataset tar file."""

    def __init__(
        self,
        root: str | Path,
        train: bool = True,
        transform: Callable | None = None,
        target_transform: Callable | None = None,
        download: bool = False,
    ) -> None:
        r"""Initialize the CUB-200-2011 dataset.

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
        self.class_names: list[str]
        r"""Store the class names of the dataset."""

        self.loader: str = default_loader
        r"""Store the loader function to load the images."""

        self.train: bool = train
        r"""Store the flag to indicate whether to load training or test dataset."""

        if download:
            self.download()  # download the dataset if not already downloaded

        if not self._check_integrity():
            raise RuntimeError(
                "Dataset not found or corrupted. You can use download=True to download it"
            )

    def _load_data(self) -> None:
        r"""Load the dataset."""

        images = pd.read_csv(
            os.path.join(self.root, "CUB_200_2011", "images.txt"),
            sep=" ",
            names=["img_id", "filepath"],
        )
        image_class_labels = pd.read_csv(
            os.path.join(self.root, "CUB_200_2011", "image_class_labels.txt"),
            sep=" ",
            names=["img_id", "target"],
        )
        train_test_split = pd.read_csv(
            os.path.join(self.root, "CUB_200_2011", "train_test_split.txt"),
            sep=" ",
            names=["img_id", "is_training_img"],
        )

        data = images.merge(image_class_labels, on="img_id")
        self.data = data.merge(train_test_split, on="img_id")

        class_names = pd.read_csv(
            os.path.join(self.root, "CUB_200_2011", "classes.txt"),
            sep=" ",
            names=["class_name"],
            usecols=[1],
        )
        self.class_names = class_names["class_name"].to_list()
        if self.train:
            self.data = self.data[self.data.is_training_img == 1]
        else:
            self.data = self.data[self.data.is_training_img == 0]

    def _check_integrity(self) -> bool:
        r"""Sanity check if dataset not found or corrupted. Do loading data at the same time.

        **Returns:**
        - **if_intergral (**bool**)**: True if the dataset is found and not corrupted, False otherwise.
        """
        try:
            self._load_data()
        except Exception:
            return False

        for index, row in self.data.iterrows():
            filepath = os.path.join(self.root, self.base_folder, row.filepath)
            if not os.path.isfile(filepath):
                print(filepath)
                return False
        return True

    def download(self) -> None:
        """Download the MNIST data if it doesn't exist already."""

        if self._check_integrity():
            print("Files already downloaded and verified")
            return

        download_file_from_google_drive(
            self.file_id, self.root, self.filename, self.tgz_md5
        )

        with tarfile.open(os.path.join(self.root, self.filename), "r:gz") as tar:
            tar.extractall(path=self.root)

    def __getitem__(self, index: int) -> tuple[Any, Any]:
        r"""Get the data and target by index.

        **Args:**
        - **idx** (`int`): The index of the item to get.

        **Returns:**
        - **img** (`PIL.Image`): The image at the given index.
        - **target** (`int`): The target label of the image at the given index.
        """

        sample = self.data.iloc[index]
        path = os.path.join(self.root, self.base_folder, sample.filepath)
        target = sample.target - 1  # Targets start at 1 by default, so shift to 0
        img = self.loader(path)

        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self) -> int:
        r"""Get the number of data in the dataset.

        **Returns:**
        - **len** (`int`): The number of data of the dataset.
        """
        return len(self.data)
