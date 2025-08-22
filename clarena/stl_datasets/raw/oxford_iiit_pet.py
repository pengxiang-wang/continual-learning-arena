r"""
The submodule in `cl_datasets.original` for the original Oxford-IIIT Pet dataset.
"""

__all__ = [
    "OxfordIIITPet37",
    "OxfordIIITPet2",
]

import logging
from pathlib import Path
from typing import Callable

from torchvision.datasets import OxfordIIITPet

# always get logger for built-in logging in each module
pylogger = logging.getLogger(__name__)


class OxfordIIITPet37(OxfordIIITPet):
    r"""Oxford-IIIT Pet dataset with 37 breed classes. The [original Oxford IIIT Pet dataset](http://www.robots.ox.ac.uk/~vgg/data/pets/) is a collection of pet images. It consists of 37 breeds of pets with 200 images per breed."""

    def __init__(
        self,
        root: str | Path,
        split: str = "trainval",
        transform: Callable | None = None,
        target_transform: Callable | None = None,
        download: bool = False,
    ) -> None:
        r"""Initialize the Oxford-IIIT Pet dataset with 37 breed classes.

        **Args:**
        - **root** (`str`): Root directory of the dataset.
        - **split** (`str`): The dataset split to use. Can be 'trainval', or 'test'. Default is 'trainval'.
        - **transform** (`callable` | `None`): A function/transform that takes in an PIL image and returns a transformed version. E.g, ``transforms.RandomCrop``
        - **target_transform** (`callable` | `None`): A function/transform that takes in the target and transforms it.
        - **download** (`bool`): If true, downloads the dataset from the internet and puts it in root directory. If dataset is already downloaded, it is not downloaded again.
        """
        OxfordIIITPet.__init__(
            self,
            root=root,
            split=split,
            target_types="category",
            transform=transform,
            target_transform=target_transform,
            download=download,
        )


class OxfordIIITPet2(OxfordIIITPet):
    r"""Oxford-IIIT Pet dataset with 2 classes (cat, dog). The [original Oxford IIIT Pet dataset](http://www.robots.ox.ac.uk/~vgg/data/pets/) is a collection of pet images. It consists of 37 breeds of pets with 200 images per breed."""

    def __init__(
        self,
        root: str | Path,
        split: str = "trainval",
        transform: Callable | None = None,
        target_transform: Callable | None = None,
        download: bool = False,
    ) -> None:
        r"""Initialize the Oxford-IIIT Pet dataset with 37 breed classes.

        **Args:**
        - **root** (`str`): Root directory of the dataset.
        - **split** (`str`): The dataset split to use. Can be 'trainval', or 'test'. Default is 'trainval'.
        - **transform** (`callable` | `None`): A function/transform that takes in an PIL image and returns a transformed version. E.g, ``transforms.RandomCrop``
        - **target_transform** (`callable` | `None`): A function/transform that takes in the target and transforms it.
        - **download** (`bool`): If true, downloads the dataset from the internet and puts it in root directory. If dataset is already downloaded, it is not downloaded again.
        """
        OxfordIIITPet.__init__(
            self,
            root=root,
            split=split,
            target_types="binary-category",
            transform=transform,
            target_transform=target_transform,
            download=download,
        )
