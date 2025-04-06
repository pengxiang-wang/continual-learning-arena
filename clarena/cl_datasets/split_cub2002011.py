r"""
The submodule in `cl_datasets` for Split CUB-200-2011 dataset.
"""

__all__ = ["SplitCUB2002011"]

import logging
import os
from pathlib import Path
from typing import Callable

import pandas as pd
import torch
from torch.utils.data import Dataset, random_split
from torchvision import transforms
from torchvision.datasets import VisionDataset
from torchvision.datasets.folder import default_loader
from torchvision.datasets.utils import download_file_from_google_drive

from clarena.cl_datasets import CLSplitDataset

# always get logger for built-in logging in each module
pylogger = logging.getLogger(__name__)


class SplitCUB2002011(CLSplitDataset):
    r"""Split CUB-200-2011 dataset. [CUB(Caltech-UCSD Birds)-200-2011)](https://www.vision.caltech.edu/datasets/cub_200_2011/) is a bird image dataset. It consists of 120,000 64x64 colour images in 200 classes, with 500 training, 50 validation and 50 test examples per class."""

    num_classes: int = 200
    r"""The number of classes in TinyImageNet dataset."""

    mean_original: tuple[float] = (0.4802, 0.4481, 0.3975)
    r"""The mean values for normalisation."""

    std_original: tuple[float] = (0.2302, 0.2265, 0.2262)
    r"""The standard deviation values for normalisation."""

    def __init__(
        self,
        root: str,
        num_tasks: int,
        class_split: list[list[int]],
        validation_percentage: float,
        batch_size: int = 1,
        num_workers: int = 0,
        custom_transforms: Callable | transforms.Compose | None = None,
        to_tensor: bool = True,
        resize: tuple[int, int] = (224, 224),
        custom_target_transforms: Callable | transforms.Compose | None = None,
    ) -> None:
        r"""Initialise the Split TinyImageNet dataset.

        **Args:**
        - **root** (`str`): the root directory where the original TinyImageNet data 'tiny-imagenet-200/' live.
        - **num_tasks** (`int`): the maximum number of tasks supported by the CL dataset.
        - **class_split** (`list[list[int]]`): the class split for each task. Each element in the list is a list of class labels (integers starting from 0) to split for a task.
        - **validation_percentage** (`float`): the percentage to randomly split some of the training data into validation data.
        - **batch_size** (`int`): The batch size in train, val, test dataloader.
        - **num_workers** (`int`): the number of workers for dataloaders.
        - **custom_transforms** (`transform` or `transforms.Compose` or `None`): the custom transforms to apply to ONLY TRAIN dataset. Can be a single transform, composed transforms or no transform.
        `ToTensor()`, normalise, permute and so on are not included.
        - **to_tensor** (`bool`): whether to include `ToTensor()` transform. Default is True.
        - **resize** (`tuple[int, int]`): the size to resize the images to, which should be a tuple of two integers. Resizing is mandatory for CUB-200-2011 dataset because the original images are not all the same size. Default is (224, 224).
        - **custom_target_transforms** (`transform` or `transforms.Compose` or `None`): the custom target transforms to apply to dataset labels. Can be a single transform, composed transforms or no transform. CL class mapping is not included.
        - **permutation_mode** (`str`): the mode of permutation, should be one of the following:
            1. 'all': permute all pixels.
            2. 'by_channel': permute channel by channel separately. All channels are applied the same permutation order.
            3. 'first_channel_only': permute only the first channel.
        """
        CLSplitDataset.__init__(
            self,
            root=root,
            num_tasks=num_tasks,
            class_split=class_split,
            batch_size=batch_size,
            num_workers=num_workers,
            custom_transforms=custom_transforms,
            to_tensor=to_tensor,
            resize=resize,
            custom_target_transforms=custom_target_transforms,
        )

        self.validation_percentage: float = validation_percentage
        """Store the percentage to randomly split some of the training data into validation data."""

    def prepare_data(self) -> None:
        r"""Download the original TinyImagenet dataset if haven't."""
        CUB2002011(root=self.root, train=True, download=True)
        CUB2002011(root=self.root, train=False, download=True)

    def get_class_subset(self, dataset: Dataset) -> Dataset:
        r"""Provide a util method here to retrieve a subset from PyTorch Dataset of current classes of `self.task_id`. It could be useful when you constructing the split CL dataset.

        **Args:**
        - **dataset** (`Dataset`): the original dataset to retrieve subset from.

        **Returns:**
        - **subset** (`Dataset`): subset of original dataset in classes.
        """
        classes = self.class_split[self.task_id - 1]

        # get the indices of the dataset that belong to the classes
        idx = [i for i, (_, target) in enumerate(dataset) if target in classes]

        # subset the dataset by the indices, in-place operation
        dataset.data = dataset.data.iloc[idx]  # data is a Pandas DataFrame

        return dataset

    def train_and_val_dataset(self) -> tuple[Dataset, Dataset]:
        r"""Get the training and validation dataset of task `self.task_id`.

        **Returns:**
        - **train_and_val_dataset** (`tuple[Dataset, Dataset]`): the train and validation dataset of task `self.task_id`.
        """
        dataset_train_and_val = self.get_class_subset(
            CUB2002011(
                root=self.root,
                train=True,
                transform=self.train_and_val_transforms(),
                download=False,
            )
        )
        dataset_train_and_val.target_transform = self.target_transforms()

        return random_split(
            dataset_train_and_val,
            lengths=[1 - self.validation_percentage, self.validation_percentage],
            generator=torch.Generator().manual_seed(
                42
            ),  # this must be set fixed to make sure the datasets across experiments are the same. Don't handle it to global seed as it might vary across experiments
        )

    def test_dataset(self) -> Dataset:
        r"""Get the test dataset of task `self.task_id`.

        **Returns:**
        - **test_dataset** (`Dataset`): the test dataset of task `self.task_id`.
        """
        dataset_test = self.get_class_subset(
            CUB2002011(
                root=self.root,
                train=False,
                transform=self.test_transforms(),
                download=False,
            )
        )
        dataset_test.target_transform = self.target_transforms()

        return dataset_test


class CUB2002011(VisionDataset):
    """`CUB-200-2011 dataset. [CUB(Caltech-UCSD Birds)-200-2011)](https://www.vision.caltech.edu/datasets/cub_200_2011/) is a bird image dataset. It consists of 120,000 64x64 colour images in 200 classes, with 500 training, 50 validation and 50 test examples per class."""

    base_folder = "CUB_200_2011/images"
    url = "http://www.vision.caltech.edu/visipedia-data/CUB-200-2011/CUB_200_2011.tgz"
    file_id = "1hbzc_P1FuxMkcabkgn9ZKinBwW683j45"
    filename = "CUB_200_2011.tgz"
    tgz_md5 = "97eceeb196236b17998738112f37df78"

    def __init__(
        self,
        root: str | Path,
        train: bool = True,
        transform: Callable | None = None,
        target_transform: Callable | None = None,
        download: bool = False,
    ) -> None:
        r"""Initialise the CUB-200-2011 dataset.

        **Args:**
        - **root** (`str`): Root directory of the dataset.
        - train (`bool` | `None`): If True, creates dataset from training set, otherwise
        creates from test set.
        - **transform** (`callable` | `None`): A function/transform that  takes in an PIL image
        and returns a transformed version. E.g, ``transforms.RandomCrop``
        - **target_transform** (`callable` | `None`): A function/transform that takes in the
        target and transforms it.
        - **download** (`bool`): If true, downloads the dataset from the internet and
        puts it in root directory. If dataset is already downloaded, it is not
        downloaded again.
        """
        VisionDataset.__init__(
            self, root=root, transform=transform, target_transform=target_transform
        )

        self.loader = default_loader
        self.train = train
        if download:
            self._download()

        if not self._check_integrity():
            raise RuntimeError(
                "Dataset not found or corrupted. You can use download=True to download it"
            )

    def _load_metadata(self) -> None:
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
        try:
            self._load_metadata()
        except Exception:
            return False

        for index, row in self.data.iterrows():
            filepath = os.path.join(self.root, self.base_folder, row.filepath)
            if not os.path.isfile(filepath):
                print(filepath)
                return False
        return True

    def _download(self) -> None:
        import tarfile

        if self._check_integrity():
            print("Files already downloaded and verified")
            return

        download_file_from_google_drive(
            self.file_id, self.root, self.filename, self.tgz_md5
        )

        with tarfile.open(os.path.join(self.root, self.filename), "r:gz") as tar:
            tar.extractall(path=self.root)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data.iloc[idx]
        path = os.path.join(self.root, self.base_folder, sample.filepath)
        target = sample.target - 1  # Targets start at 1 by default, so shift to 0
        img = self.loader(path)

        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target
