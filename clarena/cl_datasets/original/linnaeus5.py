r"""
The submodule in `cl_datasets.original` for the original Linnaeus 5 dataset.
"""

__all__ = [
    "Linnaeus5",
    "Linnaeus5_32",
    "Linnaeus5_64",
    "Linnaeus5_128",
    "Linnaeus5_256",
]


import logging
import os
from pathlib import Path
from typing import Any, Callable

import pandas as pd
import rarfile
from PIL import Image
from torchvision.datasets import VisionDataset
from torchvision.datasets.folder import default_loader
from torchvision.datasets.utils import download_url

# always get logger for built-in logging in each module
pylogger = logging.getLogger(__name__)


class Linnaeus5(VisionDataset):
    r"""Linnaeus 5 dataset. The [original Linnaeus 5 dataset](https://chaladze.com/l5/) contains flower images in 5 classes at various resolutions. It has built-in `train/` and `test/` splits. This class handles loading and using those splits."""

    base_url: str = "https://chaladze.com/l5/img/"
    r"""Base URL where the dataset archives are hosted."""

    versions: dict[str, str] = {
        "256": "Linnaeus%205%2032X32.rar",
        "128": "Linnaeus%205%20128X128.rar",
        "64": "Linnaeus%205%2064X64.rar",
        "32": "Linnaeus%205%2032X32.rar",
    }
    r"""Dictionary mapping resolution keys to their .rar archive filenames."""

    def __init__(
        self,
        root: str | Path,
        resolution: str = "256",
        train: bool = True,
        transform: Callable | None = None,
        target_transform: Callable | None = None,
        download: bool = False,
    ) -> None:
        r"""Initialise the Linnaeus 5 dataset.

        **Args:**
        - **root** (`str` | `Path`): Root directory of the dataset.
        - **resolution** (`str`): Image resolution to use: "256", "128", "64", or "32".
        - **train** (`bool`): If True, uses the train set. If False, uses the test set.
        - **transform** (`callable` | `None`): Image transform pipeline.
        - **target_transform** (`callable` | `None`): Label transform.
        - **download** (`bool`): If True, downloads the dataset if not found.
        """
        super().__init__(
            root=root, transform=transform, target_transform=target_transform
        )

        if resolution not in self.versions:
            raise ValueError(
                f"Invalid resolution '{resolution}', choose from {list(self.versions.keys())}"
            )

        self.resolution = resolution
        self.train = train

        self.class_names: list[str] = [
            "buttercup",
            "colts_foot",
            "daisy",
            "dandelion",
            "fritillary",
        ]
        r"""List of class names in the dataset."""

        self.filename: str = self.versions[resolution]
        self.base_folder: str = self.filename.replace(".rar", "").replace("%20", " ")
        self.url: str = self.base_url + self.filename
        self.split_folder: str = "train" if self.train else "test"
        self.data: pd.DataFrame

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError(
                "Dataset not found or corrupted. Use download=True to get it."
            )

        self._load_data()
        self.loader = default_loader

    def _load_data(self) -> None:
        r"""Load file paths and labels from `train/` or `test/` folders."""
        split_dir = os.path.join(self.root, self.base_folder, self.split_folder)
        samples = []

        for label_index, class_folder in enumerate(sorted(os.listdir(split_dir))):
            class_path = os.path.join(split_dir, class_folder)
            if not os.path.isdir(class_path):
                continue

            for fname in os.listdir(class_path):
                img_path = os.path.join(class_path, fname)
                try:
                    with Image.open(img_path) as img:
                        img.verify()
                    samples.append((img_path, label_index))
                except Exception:
                    pylogger.warning(f"Corrupted file skipped: {img_path}")
                    continue

        self.data = pd.DataFrame(samples, columns=["filepath", "target"])

    def _check_integrity(self) -> bool:
        r"""Check that data is extracted and usable."""
        split_dir = os.path.join(self.root, self.base_folder, self.split_folder)
        return os.path.isdir(split_dir)

    def download(self) -> None:
        r"""Download and extract the Linnaeus 5 dataset .rar file."""
        archive_path = os.path.join(self.root, self.filename)

        if self._check_integrity():
            pylogger.info("Files already downloaded and verified.")
            return

        download_url(self.url, self.root, filename=self.filename)

        # Extract the .rar file
        try:
            with rarfile.RarFile(archive_path) as rf:
                rf.extractall(self.root)
        except rarfile.Error as e:
            raise RuntimeError(
                f"Failed to extract the file {archive_path}. Please extract it manually. Error: {e}"
            )

    def __getitem__(self, index: int) -> tuple[Any, Any]:
        r"""Get an image and its label by index."""
        sample = self.data.iloc[index]
        img = Image.open(sample.filepath).convert("RGB")
        target = sample.target

        if self.transform:
            img = self.transform(img)
        if self.target_transform:
            target = self.target_transform(target)

        return img, target

    def __len__(self) -> int:
        return len(self.data)


class Linnaeus5_32(Linnaeus5):
    r"""Linnaeus 5 dataset with 32x32 resolution. This is a subclass of Linnaeus5 that sets the resolution to 32x32."""

    def __init__(
        self,
        root: str | Path,
        train: bool = True,
        transform: Callable | None = None,
        target_transform: Callable | None = None,
        download: bool = False,
    ) -> None:
        r"""Initialise the Linnaeus 5 dataset with 32x32 resolution.

        **Args:**
        - **root** (`str` | `Path`): Root directory of the dataset.
        - **train** (`bool`): If True, uses the train set. If False, uses the test set.
        - **transform** (`callable` | `None`): Image transform pipeline.
        - **target_transform** (`callable` | `None`): Label transform.
        - **download** (`bool`): If True, downloads the dataset if not found.
        """
        super().__init__(
            root=root,
            resolution="32",
            train=train,
            transform=transform,
            target_transform=target_transform,
            download=download,
        )


class Linnaeus5_64(Linnaeus5):
    r"""Linnaeus 5 dataset with 64x64 resolution. This is a subclass of Linnaeus5 that sets the resolution to 64x64."""

    def __init__(
        self,
        root: str | Path,
        train: bool = True,
        transform: Callable | None = None,
        target_transform: Callable | None = None,
        download: bool = False,
    ) -> None:
        r"""Initialise the Linnaeus 5 dataset with 64x64 resolution.

        **Args:**
        - **root** (`str` | `Path`): Root directory of the dataset.
        - **train** (`bool`): If True, uses the train set. If False, uses the test set.
        - **transform** (`callable` | `None`): Image transform pipeline.
        - **target_transform** (`callable` | `None`): Label transform.
        - **download** (`bool`): If True, downloads the dataset if not found.
        """
        super().__init__(
            root=root,
            resolution="64",
            train=train,
            transform=transform,
            target_transform=target_transform,
            download=download,
        )


class Linnaeus5_128(Linnaeus5):
    r"""Linnaeus 5 dataset with 128x128 resolution. This is a subclass of Linnaeus5 that sets the resolution to 128x128."""

    def __init__(
        self,
        root: str | Path,
        train: bool = True,
        transform: Callable | None = None,
        target_transform: Callable | None = None,
        download: bool = False,
    ) -> None:
        r"""Initialise the Linnaeus 5 dataset with 128x128 resolution.

        **Args:**
        - **root** (`str` | `Path`): Root directory of the dataset.
        - **train** (`bool`): If True, uses the train set. If False, uses the test set.
        - **transform** (`callable` | `None`): Image transform pipeline.
        - **target_transform** (`callable` | `None`): Label transform.
        - **download** (`bool`): If True, downloads the dataset if not found.
        """
        super().__init__(
            root=root,
            resolution="128",
            train=train,
            transform=transform,
            target_transform=target_transform,
            download=download,
        )


class Linnaeus5_256(Linnaeus5):
    r"""Linnaeus 5 dataset with 256x256 resolution. This is a subclass of Linnaeus5 that sets the resolution to 256x256."""

    def __init__(
        self,
        root: str | Path,
        train: bool = True,
        transform: Callable | None = None,
        target_transform: Callable | None = None,
        download: bool = False,
    ) -> None:
        r"""Initialise the Linnaeus 5 dataset with 256x256 resolution.

        **Args:**
        - **root** (`str` | `Path`): Root directory of the dataset.
        - **train** (`bool`): If True, uses the train set. If False, uses the test set.
        - **transform** (`callable` | `None`): Image transform pipeline.
        - **target_transform** (`callable` | `None`): Label transform.
        - **download** (`bool`): If True, downloads the dataset if not found.
        """
        super().__init__(
            root=root,
            resolution="256",
            train=train,
            transform=transform,
            target_transform=target_transform,
            download=download,
        )
