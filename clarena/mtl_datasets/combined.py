r"""
The submodule in `mtl_datasets` for CelebA dataset used for multi-task learning.
"""

__all__ = ["Combined"]

import logging
from typing import Callable

import torch
from tinyimagenet import TinyImageNet
from torch.utils.data import Dataset, random_split
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
from torchvision.transforms import transforms

from clarena.mtl_datasets import MTLCombinedDataset
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


class Combined(MTLCombinedDataset):
    r"""Combined MTL dataset from available datasets."""

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
        datasets: dict[int, str],
        root: str | dict[int, str],
        validation_percentage: float,
        test_percentage: float,
        sampling_strategy: str = "mixed",
        batch_size: int = 1,
        num_workers: int = 0,
        custom_transforms: (
            Callable
            | transforms.Compose
            | None
            | dict[int, Callable | transforms.Compose | None]
        ) = None,
        repeat_channels: int | None | dict[int, int | None] = None,
        to_tensor: bool | dict[int, bool] = True,
        resize: tuple[int, int] | None | dict[int, tuple[int, int] | None] = None,
    ) -> None:
        r"""
        **Args:**
        - **datasets** (`dict[int, str]`): the dict of dataset class paths for each task. The keys are task IDs and the values are the dataset class paths (as strings) to use for each task.
        - **root** (`str` | `dict[int, str]`): the root directory where the original data files for constructing the MTL dataset physically live. If `dict[int, str]`, it should be a dict of task IDs and their corresponding root directories.
        - **validation_percentage** (`float`): the percentage to randomly split some of the training data into validation data (only if validation set is not provided in the dataset).
        - **test_percentage** (`float`): the percentage to randomly split some of the entire data into test data (only if test set is not provided in the dataset).
        - **sampling_strategy** (`str`): the sampling strategy that construct training batch from each task's dataset; one of:
            - 'mixed': mixed sampling strategy, which samples from all tasks' datasets.
        - **batch_size** (`int`): The batch size in train, val, test dataloader.
        - **num_workers** (`int`): the number of workers for dataloaders.
        - **custom_transforms** (`transform` or `transforms.Compose` or `None` or dict of them): the custom transforms to apply ONLY to the TRAIN dataset. Can be a single transform, composed transforms, or no transform. `ToTensor()`, normalization and so on are not included.
        If it is a dict, the keys are task IDs and the values are the custom transforms for each task. If it is a single transform or composed transforms, it is applied to all tasks. If it is `None`, no custom transforms are applied.
        - **repeat_channels** (`int` | `None` | dict of them): the number of channels to repeat for each task. Default is `None`, which means no repeat.
        If it is a dict, the keys are task IDs and the values are the number of channels to repeat for each task. If it is an `int`, it is the same number of channels to repeat for all tasks. If it is `None`, no repeat is applied.
        - **to_tensor** (`bool` | `dict[int, bool]`): whether to include the `ToTensor()` transform. Default is `True`.
        If it is a dict, the keys are task IDs and the values are whether to include the `ToTensor()` transform for each task. If it is a single boolean value, it is applied to all tasks.
        - **resize** (`tuple[int, int]` | `None` or dict of them): the size to resize the images to. Default is `None`, which means no resize. If it is a dict, the keys are task IDs and the values are the sizes to resize for each task. If it is a single tuple of two integers, it is applied to all tasks. If it is `None`, no resize is applied.
        """
        super().__init__(
            datasets=datasets,
            root=root,
            sampling_strategy=sampling_strategy,
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

    def train_and_val_dataset(self, task_id: int) -> tuple[Dataset, Dataset]:
        r"""Get the training and validation dataset of task `task_id`.

        **Args:**
        - **task_id** (`int`): the task ID to get the training and validation dataset for.

        **Returns:**
        - **train_and_val_dataset** (`tuple[Dataset, Dataset]`): the train and validation dataset of task `task_id`.
        """
        original_dataset_python_class_t = self.original_dataset_python_classes[task_id]

        # torchvision datasets might have different APIs
        if original_dataset_python_class_t in [
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
            dataset_train_and_val = original_dataset_python_class_t(
                root=self.root[task_id],
                train=True,
                transform=self.train_and_val_transforms(task_id),
                target_transform=self.target_transform(task_id),
            )

            return random_split(
                dataset_train_and_val,
                lengths=[1 - self.validation_percentage, self.validation_percentage],
                generator=torch.Generator().manual_seed(
                    42
                ),  # this must be set fixed to make sure the datasets across experiments are the same. Don't handle it to global seed as it might vary across experiments
            )
        elif original_dataset_python_class_t in [
            Caltech101,
            Caltech256,
            EuroSAT,
            SEMEION,
            SUN397,
        ]:
            # dataset classes that don't have train and test splt
            dataset_all = original_dataset_python_class_t(
                root=self.root[task_id],
                transform=self.train_and_val_transforms(task_id),
                target_transform=self.target_transform(task_id),
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
        elif original_dataset_python_class_t in [Country211]:
            # dataset classes that have `split` argument with 'train', 'valid', 'test'
            dataset_train = original_dataset_python_class_t(
                root=self.root[task_id],
                split="train",
                transform=self.train_and_val_transforms(task_id),
                target_transform=self.target_transform(task_id),
            )
            dataset_val = original_dataset_python_class_t(
                root=self.root[task_id],
                split="valid",
                transform=self.train_and_val_transforms(task_id),
                target_transform=self.target_transform(task_id),
            )

            return dataset_train, dataset_val

        elif original_dataset_python_class_t in [
            DTD,
            FGVCAircraftVariant,
            FGVCAircraftFamily,
            FGVCAircraftManufacturer,
            Flowers102,
            PCAM,
            RenderedSST2,
        ]:
            # dataset classes that have `split` argument with 'train', 'val', 'test'
            dataset_train = original_dataset_python_class_t(
                root=self.root[task_id],
                split="train",
                transform=self.train_and_val_transforms(task_id),
                target_transform=self.target_transform(task_id),
            )

            dataset_val = original_dataset_python_class_t(
                root=self.root[task_id],
                split="val",
                transform=self.train_and_val_transforms(task_id),
                target_transform=self.target_transform(task_id),
            )

            return dataset_train, dataset_val

        elif original_dataset_python_class_t in [
            FER2013,
            Food101,
            GTSRB,
            StanfordCars,
            SVHN,
            TinyImageNet,
        ]:
            # dataset classes that have `split` argument with 'train', 'test'

            dataset_train_and_val = original_dataset_python_class_t(
                root=self.root[task_id],
                split="train",
                transform=self.train_and_val_transforms(task_id),
                target_transform=self.target_transform(task_id),
            )

            return random_split(
                dataset_train_and_val,
                lengths=[1 - self.validation_percentage, self.validation_percentage],
                generator=torch.Generator().manual_seed(
                    42
                ),  # this must be set fixed to make sure the datasets across experiments are the same. Don't handle it to global seed as it might vary across experiments
            )

        elif original_dataset_python_class_t in [OxfordIIITPet2, OxfordIIITPet37]:
            # dataset classes that have `split` argument with 'trainval', 'test'

            dataset_train_and_val = original_dataset_python_class_t(
                root=self.root[task_id],
                split="trainval",
                transform=self.train_and_val_transforms(task_id),
                target_transform=self.target_transform(task_id),
            )

            return random_split(
                dataset_train_and_val,
                lengths=[1 - self.validation_percentage, self.validation_percentage],
                generator=torch.Generator().manual_seed(
                    42
                ),  # this must be set fixed to make sure the datasets across experiments are the same. Don't handle it to global seed as it might vary across experiments
            )

        elif original_dataset_python_class_t in [CelebA]:
            # special case
            dataset_train = original_dataset_python_class_t(
                root=self.root[task_id],
                split="train",
                target_type="identity",
                transform=self.train_and_val_transforms(task_id),
                target_transform=self.target_transform(task_id),
            )

            dataset_val = original_dataset_python_class_t(
                root=self.root[task_id],
                split="valid",
                target_type="identity",
                transform=self.train_and_val_transforms(task_id),
                target_transform=self.target_transform(task_id),
            )

            return dataset_train, dataset_val

    def test_dataset(self, task_id: int) -> Dataset:
        """Get the test dataset of task `task_id`.

        **Args:**
        - **task_id** (`int`): the task ID to get the test dataset for.

        **Returns:**
        - **test_dataset** (`Dataset`): the test dataset of task `task_id`.
        """
        original_dataset_python_class_t = self.original_dataset_python_classes[task_id]

        if original_dataset_python_class_t in [
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
            dataset_test = original_dataset_python_class_t(
                root=self.root[task_id],
                train=False,
                transform=self.test_transforms(task_id),
                target_transform=self.target_transform(task_id),
            )

            return dataset_test

        elif original_dataset_python_class_t in [
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

            dataset_test = original_dataset_python_class_t(
                root=self.root[task_id],
                split="test",
                transform=self.test_transforms(task_id),
                target_transform=self.target_transform(task_id),
            )

            return dataset_test

        elif original_dataset_python_class_t in [
            Caltech101,
            Caltech256,
            EuroSAT,
            SEMEION,
            SUN397,
        ]:
            # dataset classes that don't have train and test splt

            dataset_all = original_dataset_python_class_t(
                root=self.root[task_id],
                transform=self.train_and_val_transforms(task_id),
                target_transform=self.target_transform(task_id),
            )

            _, dataset_test = random_split(
                dataset_all,
                lengths=[1 - self.test_percentage, self.test_percentage],
                generator=torch.Generator().manual_seed(42),
            )

            return dataset_test

        elif original_dataset_python_class_t in [CelebA]:
            # special case
            dataset_test = original_dataset_python_class_t(
                root=self.root[task_id],
                split="test",
                target_type="identity",
                transform=self.test_transforms(task_id),
                target_transform=self.target_transform(task_id),
            )

            return dataset_test

        elif original_dataset_python_class_t in [TinyImageNet]:
            # special case
            dataset_test = original_dataset_python_class_t(
                root=self.root[task_id],
                split="val",
                transform=self.test_transforms(task_id),
                target_transform=self.target_transform(task_id),
            )

            return dataset_test
