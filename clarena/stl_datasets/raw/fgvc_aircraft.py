r"""
The submodule in `cl_datasets.original` for the original FGVC-Aircraft dataset.
"""

__all__ = [
    "FGVCAircraftVariant",
    "FGVCAircraftFamily",
    "FGVCAircraftManufacturer",
]

import logging
from pathlib import Path
from typing import Callable

from torchvision.datasets import FGVCAircraft

# always get logger for built-in logging in each module
pylogger = logging.getLogger(__name__)


class FGVCAircraftVariant(FGVCAircraft):
    r"""FGVC-Aircraft dataset annotated by variant. The [original FGVC-Aircraft dataset](https://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/) is a collection of aircraft images. It consists of 10,000 images."""

    def __init__(
        self,
        root: str | Path,
        split: str = "trainval",
        transform: Callable | None = None,
        target_transform: Callable | None = None,
        download: bool = False,
    ) -> None:
        r"""Initialize the FGVC-Aircraft dataset annotated by variant.

        **Args:**
        - **root** (`str`): Root directory of the FGVC-Aircraft dataset.
        - **split** (`str`): The dataset split, supports 'train', 'val', 'trainval', 'test'.
        - **transform** (`callable` | `None`): A function/transform that takes in an PIL image and returns a transformed version. E.g, ``transforms.RandomCrop``
        - **target_transform** (`callable` | `None`): A function/transform that takes in the target and transforms it.
        - **download** (`bool`): If true, downloads the dataset from the internet and puts it in root directory. If dataset is already downloaded, it is not downloaded again.
        """
        super().__init__(
            root=root,
            split=split,
            annotation_level="variant",
            transform=transform,
            target_transform=target_transform,
            download=download,
        )


class FGVCAircraftFamily(FGVCAircraft):
    r"""FGVC-Aircraft dataset annotated by family. The [original FGVC-Aircraft dataset](https://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/) is a collection of aircraft images. It consists of 10,000 images."""

    def __init__(
        self,
        root: str | Path,
        split: str = "trainval",
        transform: Callable | None = None,
        target_transform: Callable | None = None,
        download: bool = False,
    ) -> None:
        r"""Initialize the FGVC-Aircraft dataset annotated by family.

        **Args:**
        - **root** (`str`): Root directory of the FGVC-Aircraft dataset.
        - **split** (`str`): The dataset split, supports 'train', 'val', 'trainval', 'test'.
        - **transform** (`callable` | `None`): A function/transform that takes in an PIL image and returns a transformed version. E.g, ``transforms.RandomCrop``
        - **target_transform** (`callable` | `None`): A function/transform that takes in the target and transforms it.
        - **download** (`bool`): If true, downloads the dataset from the internet and puts it in root directory. If dataset is already downloaded, it is not downloaded again.
        """
        super().__init__(
            root=root,
            split=split,
            annotation_level="family",
            transform=transform,
            target_transform=target_transform,
            download=download,
        )


class FGVCAircraftManufacturer(FGVCAircraft):
    r"""FGVC-Aircraft dataset annotated by manufacturer. The [original FGVC-Aircraft dataset](https://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/) is a collection of aircraft images. It consists of 10,000 images."""

    def __init__(
        self,
        root: str | Path,
        split: str = "trainval",
        transform: Callable | None = None,
        target_transform: Callable | None = None,
        download: bool = False,
    ) -> None:
        r"""Initialize the FGVC-Aircraft dataset annotated by manufacturer.

        **Args:**
        - **root** (`str`): Root directory of the FGVC-Aircraft dataset.
        - **split** (`str`): The dataset split, supports 'train', 'val', 'trainval', 'test'.
        - **transform** (`callable` | `None`): A function/transform that takes in an PIL image and returns a transformed version. E.g, ``transforms.RandomCrop``
        - **target_transform** (`callable` | `None`): A function/transform that takes in the target and transforms it.
        - **download** (`bool`): If true, downloads the dataset from the internet and puts it in root directory. If dataset is already downloaded, it is not downloaded again.
        """
        super().__init__(
            root=root,
            split=split,
            annotation_level="manufacturer",
            transform=transform,
            target_transform=target_transform,
            download=download,
        )
