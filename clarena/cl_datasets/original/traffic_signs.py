r"""
The submodule in `cl_datasets.original` for the original CUB-200-2011 dataset.
"""

__all__ = ["TrafficSigns"]

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


class TrafficSigns(VisionDataset):
    r"""German Traffic Signs dataset. [German Traffic Signs Recognition Benchmark (GTSRB)](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset) is a traffic sign dataset. It consists of more than 50,000 32×32 color images in 43 classes, with 39,209 training and 12,630 test examples."""

    def __init__(
        self,
        root: str | Path,
        train: bool = True,
        transform: Callable | None = None,
        target_transform: Callable | None = None,
        download: bool = False,
    ) -> None:
        r"""Initialise the German Traffic Signs dataset.

        **Args:**
        - **root** (`str`): Root directory of the dataset.
        - **train** (`bool`): If True, creates dataset from training set, otherwise creates from test set.
        - **transform** (`callable` | `None`): A function/transform that  takes in an PIL image and returns a transformed version. E.g, ``transforms.RandomCrop``
        - **target_transform** (`callable` | `None`): A function/transform that takes in the target and transforms it.
        - **download** (`bool`): If true, downloads the dataset from the internet and puts it in root directory. If dataset is already downloaded, it is not downloaded again.
        """
        VisionDataset.__init__(
            self, root=root, transform=transform, target_transform=target_transform
        )

        self.root = os.path.expanduser(root)
        self.transform = transform
        self.filename = "traffic_signs_dataset.zip"
        self.url = "https://d17h27t6h515a5.cloudfront.net/topher/2016/October/580d53ce_traffic-sign-data/traffic-sign-data.zip"
        # Other options for the same 32x32 pickled dataset
        # url="https://d17h27t6h515a5.cloudfront.net/topher/2016/November/581faac4_traffic-signs-data/traffic-signs-data.zip"
        # url_train="https://drive.google.com/open?id=0B5WIzrIVeL0WR1dsTC1FdWEtWFE"
        # url_test="https://drive.google.com/open?id=0B5WIzrIVeL0WLTlPNlR2RG95S3c"

        fpath = os.path.join(root, self.filename)
        if not os.path.isfile(fpath):
            if not download:
                raise RuntimeError(
                    "Dataset not found. You can use download=True to download it"
                )
            else:
                print("Downloading from " + self.url)
                self.download()

        training_file = "lab 2 data/train.p"
        testing_file = "lab 2 data/test.p"
        if train:
            with open(os.path.join(root, training_file), mode="rb") as f:
                train = pickle.load(f)
            self.data = train["features"]
            self.labels = train["labels"]
        else:
            with open(os.path.join(root, testing_file), mode="rb") as f:
                test = pickle.load(f)
            self.data = test["features"]
            self.labels = test["labels"]

        self.data = np.transpose(self.data, (0, 3, 1, 2))
        # print(self.data.shape); sys.exit()

    def __getitem__(self, index):
        """
        Args: index (int): Index
        Returns: tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.labels[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(np.transpose(img, (1, 2, 0)))

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
