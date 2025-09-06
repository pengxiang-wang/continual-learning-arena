r"""
The submodule in `cl_datasets` for combined datasets.
"""

__all__ = ["Combined"]

import logging
from typing import Callable

import torch
from tinyimagenet import TinyImageNet
from torch.utils.data import Dataset, random_split
from torchvision import transforms
from torchvision.datasets import (
    CIFAR10,
    CIFAR100,
    DTD,
    FER2013,
    GTSRB,
    KMNIST,
    MNIST,
    PCAM,
    SEMEION,
    SUN397,
    SVHN,
    USPS,
    Caltech101,
    Caltech256,
    CelebA,
    Country211,
    EuroSAT,
    FashionMNIST,
    Flowers102,
    Food101,
    RenderedSST2,
    StanfordCars,
)
from torchvision.datasets.vision import VisionDataset

from clarena.cl_datasets import CLCombinedDataset
from clarena.stl_datasets.raw import (
    CUB2002011,
    ArabicHandwrittenDigits,
    EMNISTBalanced,
    EMNISTByClass,
    EMNISTByMerge,
    EMNISTDigits,
    EMNISTLetters,
    FaceScrub10,
    FaceScrub20,
    FaceScrub50,
    FaceScrub100,
    FaceScrubFromHAT,
    FGVCAircraftFamily,
    FGVCAircraftManufacturer,
    FGVCAircraftVariant,
    KannadaMNIST,
    Linnaeus5_32,
    Linnaeus5_64,
    Linnaeus5_128,
    Linnaeus5_256,
    NotMNIST,
    NotMNISTFromHAT,
    OxfordIIITPet2,
    OxfordIIITPet37,
    SignLanguageMNIST,
    TrafficSignsFromHAT,
)

# always get logger for built-in logging in each module
pylogger = logging.getLogger(__name__)


class Combined(CLCombinedDataset):
    r"""Combined CL dataset from available datasets."""

    AVAILABLE_DATASETS: list[VisionDataset] = [
        ArabicHandwrittenDigits,
        CIFAR10,
        CIFAR100,
        CUB2002011,
        Caltech101,
        Caltech256,
        CelebA,
        Country211,
        DTD,
        EMNISTBalanced,
        EMNISTByClass,
        EMNISTByMerge,
        EMNISTDigits,
        EMNISTLetters,
        EuroSAT,
        FER2013,
        FGVCAircraftFamily,
        FGVCAircraftManufacturer,
        FGVCAircraftVariant,
        FaceScrub10,
        FaceScrub100,
        FaceScrubFromHAT,
        FaceScrub20,
        FaceScrub50,
        FashionMNIST,
        Flowers102,
        Food101,
        GTSRB,
        KMNIST,
        KannadaMNIST,
        Linnaeus5_128,
        Linnaeus5_256,
        Linnaeus5_32,
        Linnaeus5_64,
        MNIST,
        NotMNIST,
        NotMNISTFromHAT,
        OxfordIIITPet2,
        OxfordIIITPet37,
        PCAM,
        RenderedSST2,
        SEMEION,
        SUN397,
        SVHN,
        SignLanguageMNIST,
        StanfordCars,
        TrafficSignsFromHAT,
        TinyImageNet,
        USPS,
    ]
    r"""The list of available datasets."""

    def __init__(
        self,
        datasets: list[str],
        root: list[str],
        validation_percentage: float,
        test_percentage: float,
        batch_size: int | list[int] = 1,
        num_workers: int | list[int] = 0,
        custom_transforms: (
            Callable
            | transforms.Compose
            | None
            | list[Callable | transforms.Compose | None]
        ) = None,
        repeat_channels: int | None | list[int | None] = None,
        to_tensor: bool | list[bool] = True,
        resize: tuple[int, int] | None | list[tuple[int, int] | None] = None,
    ) -> None:
        r"""Initialize the Combined Torchvision dataset object providing the root where data files live.

        **Args:**
        - **datasets** (`list[str]`): the list of dataset class paths for each task. Each element in the list must be a string referring to a valid PyTorch Dataset class. It needs to be one in `self.AVAILABLE_DATASETS`.
        - **root** (`list[str]`): the list of root directory where the original data files for constructing the CL dataset physically live.
        - **validation_percentage** (`float`): the percentage to randomly split some of the training data into validation data (only if validation set is not provided in the dataset).
        - **test_percentage** (`float`): the percentage to randomly split some of the entire data into test data (only if test set is not provided in the dataset).
        - **batch_size** (`int` | `list[int]`): The batch size in train, val, test dataloader. If `list[str]`, it should be a list of integers, each integer is the batch size for each task.
        - **num_workers** (`int` | `list[int]`): the number of workers for dataloaders. If `list[str]`, it should be a list of integers, each integer is the num of workers for each task.
        - **custom_transforms** (`transform` or `transforms.Compose` or `None` or list of them): the custom transforms to apply to ONLY TRAIN dataset. Can be a single transform, composed transforms or no transform. `ToTensor()`, normalize, permute and so on are not included. If it is a list, each item is the custom transforms for each task.
        - **repeat_channels** (`int` | `None` | list of them): the number of channels to repeat for each task. Default is None, which means no repeat. If not None, it should be an integer. If it is a list, each item is the number of channels to repeat for each task.
        - **to_tensor** (`bool` | `list[bool]`): whether to include `ToTensor()` transform. Default is True.
        - **resize** (`tuple[int, int]` | `None` or list of them): the size to resize the images to. Default is None, which means no resize. If not None, it should be a tuple of two integers. If it is a list, each item is the size to resize for each task.
        """
        super().__init__(
            datasets=datasets,
            root=root,
            batch_size=batch_size,
            num_workers=num_workers,
            custom_transforms=custom_transforms,
            repeat_channels=repeat_channels,
            to_tensor=to_tensor,
            resize=resize,
        )

        self.test_percentage: float = test_percentage
        """The percentage to randomly split some data into test data."""
        self.validation_percentage: float = validation_percentage
        """The percentage to randomly split some training data into validation data."""

    def prepare_data(self) -> None:
        r"""Download the original datasets if haven't."""

        if self.task_id != 1:
            return  # download all original datasets only at the beginning of first task

        failed_dataset_classes = []
        for task_id in range(1, self.num_tasks + 1):
            root = self.root[task_id]
            dataset_class = self.original_dataset_python_classes[task_id]
            # torchvision datasets might have different APIs
            try:
                # collect the error and raise it at the end to avoid stopping the whole download process

                if dataset_class in [
                    ArabicHandwrittenDigits,
                    KannadaMNIST,
                    SignLanguageMNIST,
                ]:
                    # these datasets have no automatic download function, we require users to download them manually
                    # the following code is just to check if the dataset is already downloaded
                    dataset_class(root=root, train=True, download=False)
                    dataset_class(root=root, train=False, download=False)

                elif dataset_class in [
                    Caltech101,
                    Caltech256,
                    EuroSAT,
                    SEMEION,
                    SUN397,
                ]:
                    # dataset classes that don't have any train, val, test split
                    dataset_class(root=root, download=True)

                elif dataset_class in [
                    ArabicHandwrittenDigits,
                    CIFAR10,
                    CIFAR100,
                    CUB2002011,
                    EMNISTByClass,
                    EMNISTByMerge,
                    EMNISTBalanced,
                    EMNISTLetters,
                    EMNISTDigits,
                    FaceScrub10,
                    FaceScrub20,
                    FaceScrub50,
                    FaceScrub100,
                    FaceScrubFromHAT,
                    FashionMNIST,
                    KannadaMNIST,
                    KMNIST,
                    Linnaeus5_32,
                    Linnaeus5_64,
                    Linnaeus5_128,
                    Linnaeus5_256,
                    MNIST,
                    NotMNIST,
                    NotMNISTFromHAT,
                    SignLanguageMNIST,
                    TrafficSignsFromHAT,
                    USPS,
                ]:
                    # dataset classes that have `train` bool argument
                    dataset_class(root=root, train=True, download=True)
                    dataset_class(root=root, train=False, download=True)

                elif dataset_class in [
                    Food101,
                    GTSRB,
                    StanfordCars,
                    SVHN,
                ]:
                    # dataset classes that have `split` argument with 'train', 'test'
                    dataset_class(
                        root=root,
                        split="train",
                        download=True,
                    )
                    dataset_class(
                        root=root,
                        split="test",
                        download=True,
                    )
                elif dataset_class in [Country211]:
                    # dataset classes that have `split` argument with 'train', 'valid', 'test'
                    dataset_class(
                        root=root,
                        split="train",
                        download=True,
                    )
                    dataset_class(
                        root=root,
                        split="valid",
                        download=True,
                    )
                    dataset_class(
                        root=root,
                        split="test",
                        download=True,
                    )

                elif dataset_class in [
                    DTD,
                    FGVCAircraftVariant,
                    FGVCAircraftFamily,
                    FGVCAircraftManufacturer,
                    Flowers102,
                    PCAM,
                    RenderedSST2,
                ]:
                    # dataset classes that have `split` argument with 'train', 'val', 'test'
                    dataset_class(
                        root=root,
                        split="train",
                        download=True,
                    )
                    dataset_class(
                        root=root,
                        split="val",
                        download=True,
                    )
                    dataset_class(
                        root=root,
                        split="test",
                        download=True,
                    )
                elif dataset_class in [OxfordIIITPet2, OxfordIIITPet37]:
                    # dataset classes that have `split` argument with 'trainval', 'test'
                    dataset_class(
                        root=root,
                        split="trainval",
                        download=True,
                    )
                    dataset_class(
                        root=root,
                        split="test",
                        download=True,
                    )
                elif dataset_class == CelebA:
                    # special case
                    dataset_class(
                        root=root,
                        split="train",
                        target_type="identity",
                        download=True,
                    )
                    dataset_class(
                        root=root,
                        split="valid",
                        target_type="identity",
                        download=True,
                    )
                    dataset_class(
                        root=root,
                        split="test",
                        target_type="identity",
                        download=True,
                    )
                elif dataset_class == FER2013:
                    # special case
                    dataset_class(
                        root=root,
                        split="train",
                    )
                    dataset_class(
                        root=root,
                        split="test",
                    )
                elif dataset_class == TinyImageNet:
                    # special case
                    dataset_class(root=root)

            except RuntimeError:
                failed_dataset_classes.append(dataset_class)  # save for later prompt
            else:
                pylogger.debug(
                    "The original %s dataset for task %s has been downloaded to %s.",
                    dataset_class,
                    task_id,
                    root,
                )

        if failed_dataset_classes:
            raise RuntimeError(
                f"The following datasets failed to download: {failed_dataset_classes}. Please try downloading them again or manually."
            )

    def train_and_val_dataset(self) -> tuple[Dataset, Dataset]:
        """Get the training and validation dataset of task `self.task_id`.

        **Returns:**
        - **train_and_val_dataset** (`tuple[Dataset, Dataset]`): the train and validation dataset of task `self.task_id`.
        """
        # torchvision datasets might have different APIs
        if self.original_dataset_python_class_t in [
            ArabicHandwrittenDigits,
            CIFAR10,
            CIFAR100,
            CUB2002011,
            EMNISTByClass,
            EMNISTByMerge,
            EMNISTBalanced,
            EMNISTLetters,
            EMNISTDigits,
            FaceScrub10,
            FaceScrub20,
            FaceScrub50,
            FaceScrub100,
            FaceScrubFromHAT,
            FashionMNIST,
            KannadaMNIST,
            KMNIST,
            Linnaeus5_32,
            Linnaeus5_64,
            Linnaeus5_128,
            Linnaeus5_256,
            MNIST,
            NotMNIST,
            NotMNISTFromHAT,
            SignLanguageMNIST,
            TrafficSignsFromHAT,
            USPS,
        ]:
            # dataset classes that have `train` bool argument
            dataset_train_and_val = self.original_dataset_python_class_t(
                root=self.root_t,
                train=True,
                transform=self.train_and_val_transforms(),
                target_transform=self.target_transform(),
            )

            return random_split(
                dataset_train_and_val,
                lengths=[1 - self.validation_percentage, self.validation_percentage],
                generator=torch.Generator().manual_seed(
                    42
                ),  # this must be set fixed to make sure the datasets across experiments are the same. Don't handle it to global seed as it might vary across experiments
            )
        elif self.original_dataset_python_class_t in [
            Caltech101,
            Caltech256,
            EuroSAT,
            SEMEION,
            SUN397,
        ]:
            # dataset classes that don't have train and test splt
            dataset_all = self.original_dataset_python_class_t(
                root=self.root_t,
                transform=self.train_and_val_transforms(),
                target_transform=self.target_transform(),
            )

            dataset_train_and_val, _ = random_split(
                dataset_all,
                lengths=[
                    1 - self.test_percentage,
                    self.test_percentage,
                ],
                generator=torch.Generator().manual_seed(
                    42
                ),  # this must be set fixed to make sure the datasets across experiments are the same. Don't handle it to global seed as it might vary across experiments
            )

            return random_split(
                dataset_train_and_val,
                lengths=[1 - self.validation_percentage, self.validation_percentage],
                generator=torch.Generator().manual_seed(
                    42
                ),  # this must be set fixed to make sure the datasets across experiments are the same. Don't handle it to global seed as it might vary across experiments
            )
        elif self.original_dataset_python_class_t in [Country211]:
            # dataset classes that have `split` argument with 'train', 'valid', 'test'
            dataset_train = self.original_dataset_python_class_t(
                root=self.root_t,
                split="train",
                transform=self.train_and_val_transforms(),
                target_transform=self.target_transform(),
            )
            dataset_val = self.original_dataset_python_class_t(
                root=self.root_t,
                split="valid",
                transform=self.train_and_val_transforms(),
                target_transform=self.target_transform(),
            )

            return dataset_train, dataset_val

        elif self.original_dataset_python_class_t in [
            DTD,
            FGVCAircraftVariant,
            FGVCAircraftFamily,
            FGVCAircraftManufacturer,
            Flowers102,
            PCAM,
            RenderedSST2,
        ]:
            # dataset classes that have `split` argument with 'train', 'val', 'test'
            dataset_train = self.original_dataset_python_class_t(
                root=self.root_t,
                split="train",
                transform=self.train_and_val_transforms(),
                target_transform=self.target_transform(),
            )

            dataset_val = self.original_dataset_python_class_t(
                root=self.root_t,
                split="val",
                transform=self.train_and_val_transforms(),
                target_transform=self.target_transform(),
            )

            return dataset_train, dataset_val

        elif self.original_dataset_python_class_t in [
            FER2013,
            Food101,
            GTSRB,
            StanfordCars,
            SVHN,
            TinyImageNet,
        ]:
            # dataset classes that have `split` argument with 'train', 'test'

            dataset_train_and_val = self.original_dataset_python_class_t(
                root=self.root_t,
                split="train",
                transform=self.train_and_val_transforms(),
                target_transform=self.target_transform(),
            )

            return random_split(
                dataset_train_and_val,
                lengths=[1 - self.validation_percentage, self.validation_percentage],
                generator=torch.Generator().manual_seed(
                    42
                ),  # this must be set fixed to make sure the datasets across experiments are the same. Don't handle it to global seed as it might vary across experiments
            )

        elif self.original_dataset_python_class_t in [OxfordIIITPet2, OxfordIIITPet37]:
            # dataset classes that have `split` argument with 'trainval', 'test'

            dataset_train_and_val = self.original_dataset_python_class_t(
                root=self.root_t,
                split="trainval",
                transform=self.train_and_val_transforms(),
                target_transform=self.target_transform(),
            )

            return random_split(
                dataset_train_and_val,
                lengths=[1 - self.validation_percentage, self.validation_percentage],
                generator=torch.Generator().manual_seed(
                    42
                ),  # this must be set fixed to make sure the datasets across experiments are the same. Don't handle it to global seed as it might vary across experiments
            )

        elif self.original_dataset_python_class_t in [CelebA]:
            # special case
            dataset_train = self.original_dataset_python_class_t(
                root=self.root_t,
                split="train",
                target_type="identity",
                transform=self.train_and_val_transforms(),
                target_transform=self.target_transform(),
            )

            dataset_val = self.original_dataset_python_class_t(
                root=self.root_t,
                split="valid",
                target_type="identity",
                transform=self.train_and_val_transforms(),
                target_transform=self.target_transform(),
            )

            return dataset_train, dataset_val

    def test_dataset(self) -> Dataset:
        r"""Get the test dataset of task `self.task_id`.

        **Returns:**
        - **test_dataset** (`Dataset`): the test dataset of task `self.task_id`.
        """
        # torchvision datasets might have different APIs

        if self.original_dataset_python_class_t in [
            ArabicHandwrittenDigits,
            CIFAR10,
            CIFAR100,
            CUB2002011,
            EMNISTByClass,
            EMNISTByMerge,
            EMNISTBalanced,
            EMNISTLetters,
            EMNISTDigits,
            FaceScrub10,
            FaceScrub20,
            FaceScrub50,
            FaceScrub100,
            FaceScrubFromHAT,
            FashionMNIST,
            KannadaMNIST,
            KMNIST,
            Linnaeus5_32,
            Linnaeus5_64,
            Linnaeus5_128,
            Linnaeus5_256,
            MNIST,
            NotMNIST,
            NotMNISTFromHAT,
            SignLanguageMNIST,
            TrafficSignsFromHAT,
            USPS,
        ]:
            # dataset classes that have `train` bool argument
            dataset_test = self.original_dataset_python_class_t(
                root=self.root_t,
                train=False,
                transform=self.test_transforms(),
                target_transform=self.target_transform(),
            )

            return dataset_test

        elif self.original_dataset_python_class_t in [
            Country211,
            DTD,
            FER2013,
            FGVCAircraftVariant,
            FGVCAircraftFamily,
            FGVCAircraftManufacturer,
            Flowers102,
            Food101,
            GTSRB,
            OxfordIIITPet2,
            OxfordIIITPet37,
            PCAM,
            RenderedSST2,
            StanfordCars,
            SVHN,
        ]:
            # dataset classes that have `split` argument with 'test'

            dataset_test = self.original_dataset_python_class_t(
                root=self.root_t,
                split="test",
                transform=self.test_transforms(),
                target_transform=self.target_transform(),
            )

            return dataset_test

        elif self.original_dataset_python_class_t in [
            Caltech101,
            Caltech256,
            EuroSAT,
            SEMEION,
            SUN397,
        ]:
            # dataset classes that don't have train and test splt

            dataset_all = self.original_dataset_python_class_t(
                root=self.root_t,
                transform=self.train_and_val_transforms(),
                target_transform=self.target_transform(),
            )

            _, dataset_test = random_split(
                dataset_all,
                lengths=[1 - self.test_percentage, self.test_percentage],
                generator=torch.Generator().manual_seed(42),
            )

            return dataset_test

        elif self.original_dataset_python_class_t in [CelebA]:
            # special case
            dataset_test = self.original_dataset_python_class_t(
                root=self.root_t,
                split="test",
                target_type="identity",
                transform=self.test_transforms(),
                target_transform=self.target_transform(),
            )

            return dataset_test

        elif self.original_dataset_python_class_t in [TinyImageNet]:
            # special case
            dataset_test = self.original_dataset_python_class_t(
                root=self.root_t,
                split="val",
                transform=self.test_transforms(),
                target_transform=self.target_transform(),
            )

            return dataset_test
