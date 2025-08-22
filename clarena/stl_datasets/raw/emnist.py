r"""
The submodule in `cl_datasets.original` for the original EMNIST dataset.
"""

__all__ = [
    "EMNISTByClass",
    "EMNISTByMerge",
    "EMNISTBalanced",
    "EMNISTLetters",
    "EMNISTDigits",
]

import logging
from pathlib import Path
from typing import Callable

from torchvision.datasets import EMNIST

# always get logger for built-in logging in each module
pylogger = logging.getLogger(__name__)


class EMNISTByClass(EMNIST):
    r"""EMNIST ByClass dataset. The [original EMNIST dataset](https://www.nist.gov/itl/products-and-services/emnist-dataset/) is a collection of handwritten letters and digits (including A-Z, a-z, 0-9). The ByClass split consists of 814,255 28x28 grayscale images in 62 unbalanced classes."""

    def __init__(
        self,
        root: str | Path,
        train: bool = True,
        transform: Callable | None = None,
        target_transform: Callable | None = None,
        download: bool = False,
    ) -> None:
        r"""Initialize the EMNIST ByClass dataset.

        **Args:**
        - **root** (`str`): Root directory of the dataset where `EMNIST/raw/train-images-idx3-ubyte` and `EMNIST/raw/t10k-images-idx3-ubyte` exist.
        - **train** (`bool` | `None`): If True, creates dataset from training set, otherwise creates from test set.
        - **transform** (`callable` | `None`): A function/transform that  takes in an PIL image and returns a transformed version. E.g, ``transforms.RandomCrop``
        - **target_transform** (`callable` | `None`): A function/transform that takes in the target and transforms it.
        - **download** (`bool`): If true, downloads the dataset from the internet and puts it in root directory. If dataset is already downloaded, it is not downloaded again.
        """
        super().__init__(
            root=root,
            split="byclass",
            train=train,
            transform=transform,
            target_transform=target_transform,
            download=download,
        )


class EMNISTByMerge(EMNIST):
    r"""EMNIST ByMerge dataset. The [original EMNIST dataset](https://www.nist.gov/itl/products-and-services/emnist-dataset/) is a collection of handwritten letters and digits (including A-Z, a-z, 0-9). The ByMerge split consists of 814,255 28x28 grayscale images in 47 unbalanced classes."""

    def __init__(
        self,
        root: str | Path,
        train: bool = True,
        transform: Callable | None = None,
        target_transform: Callable | None = None,
        download: bool = False,
    ) -> None:
        r"""Initialize the EMNIST ByMerge dataset.

        **Args:**
        - **root** (`str`): Root directory of the dataset where `EMNIST/raw/train-images-idx3-ubyte` and `EMNIST/raw/t10k-images-idx3-ubyte` exist.
        - **train** (`bool` | `None`): If True, creates dataset from training set, otherwise creates from test set.
        - **transform** (`callable` | `None`): A function/transform that  takes in an PIL image and returns a transformed version. E.g, ``transforms.RandomCrop``
        - **target_transform** (`callable` | `None`): A function/transform that takes in the target and transforms it.
        - **download** (`bool`): If true, downloads the dataset from the internet and puts it in root directory. If dataset is already downloaded, it is not downloaded again.
        """
        super().__init__(
            root=root,
            split="bymerge",
            train=train,
            transform=transform,
            target_transform=target_transform,
            download=download,
        )


class EMNISTBalanced(EMNIST):
    r"""EMNIST Balanced dataset. The [original EMNIST dataset](https://www.nist.gov/itl/products-and-services/emnist-dataset/) is a collection of handwritten letters and digits (including A-Z, a-z, 0-9). The Balanced split consists of 131,600 28x28 grayscale images in 47 balanced classes."""

    def __init__(
        self,
        root: str | Path,
        train: bool = True,
        transform: Callable | None = None,
        target_transform: Callable | None = None,
        download: bool = False,
    ) -> None:
        r"""Initialize the EMNIST Balanced dataset.

        **Args:**
        - **root** (`str`): Root directory of the dataset where `EMNIST/raw/train-images-idx3-ubyte` and `EMNIST/raw/t10k-images-idx3-ubyte` exist.
        - **train** (`bool` | `None`): If True, creates dataset from training set, otherwise creates from test set.
        - **transform** (`callable` | `None`): A function/transform that  takes in an PIL image and returns a transformed version. E.g, ``transforms.RandomCrop``
        - **target_transform** (`callable` | `None`): A function/transform that takes in the target and transforms it.
        - **download** (`bool`): If true, downloads the dataset from the internet and puts it in root directory. If dataset is already downloaded, it is not downloaded again.
        """
        super().__init__(
            root=root,
            split="balanced",
            train=train,
            transform=transform,
            target_transform=target_transform,
            download=download,
        )


class EMNISTLetters(EMNIST):
    r"""EMNIST Letters dataset. The [original EMNIST dataset](https://www.nist.gov/itl/products-and-services/emnist-dataset/) is a collection of handwritten letters and digits (including A-Z, a-z, 0-9). The Letters split consists of 145,600 28x28 grayscale letters images in 26 balanced classes."""

    def __init__(
        self,
        root: str | Path,
        train: bool = True,
        transform: Callable | None = None,
        target_transform: Callable | None = None,
        download: bool = False,
    ) -> None:
        r"""Initialize the EMNIST Letters dataset.

        **Args:**
        - **root** (`str`): Root directory of the dataset where `EMNIST/raw/train-images-idx3-ubyte` and `EMNIST/raw/t10k-images-idx3-ubyte` exist.
        - **train** (`bool` | `None`): If True, creates dataset from training set, otherwise creates from test set.
        - **transform** (`callable` | `None`): A function/transform that  takes in an PIL image and returns a transformed version. E.g, ``transforms.RandomCrop``
        - **target_transform** (`callable` | `None`): A function/transform that takes in the target and transforms it.
        - **download** (`bool`): If true, downloads the dataset from the internet and puts it in root directory. If dataset is already downloaded, it is not downloaded again.
        """
        super().__init__(
            root=root,
            split="letters",
            train=train,
            transform=transform,
            target_transform=target_transform,
            download=download,
        )


class EMNISTDigits(EMNIST):
    r"""EMNIST Digits dataset. The [original EMNIST dataset](https://www.nist.gov/itl/products-and-services/emnist-dataset/) is a collection of handwritten letters and digits (including A-Z, a-z, 0-9). The Digits split consists of 280,000 28x28 grayscale digits images in 10 balanced classes."""

    def __init__(
        self,
        root: str | Path,
        train: bool = True,
        transform: Callable | None = None,
        target_transform: Callable | None = None,
        download: bool = False,
    ) -> None:
        r"""Initialize the EMNIST Digits dataset.

        **Args:**
        - **root** (`str`): Root directory of the dataset where `EMNIST/raw/train-images-idx3-ubyte` and `EMNIST/raw/t10k-images-idx3-ubyte` exist.
        - **train** (`bool` | `None`): If True, creates dataset from training set, otherwise creates from test set.
        - **transform** (`callable` | `None`): A function/transform that  takes in an PIL image and returns a transformed version. E.g, ``transforms.RandomCrop``
        - **target_transform** (`callable` | `None`): A function/transform that takes in the target and transforms it.
        - **download** (`bool`): If true, downloads the dataset from the internet and puts it in root directory. If dataset is already downloaded, it is not downloaded again.
        """
        super().__init__(
            root=root,
            split="digits",
            train=train,
            transform=transform,
            target_transform=target_transform,
            download=download,
        )
