r"""
The submodule in `cl_datasets` for constants about datasets.
"""

__all__ = [
    "DatasetConstants",
    "MNISTConstants",
    "CIFAR10Constants",
    "CIFAR100Constants",
    "TinyImageNetConstants",
    "CUB2002011Constants",
    "DATASET_CONSTANTS_MAPPING",
]

import tinyimagenet
import torch
import torchvision
from torch.utils.data import Dataset

from clarena.cl_datasets import original


class DatasetConstants:
    r"""Base class for constants about datasets."""

    NUM_CLASSES: int
    r"""The number of classes in the dataset."""

    IMG_SIZE: torch.Size
    r"""The size of images in the dataset."""

    MEAN: tuple[float]
    r"""The mean values of each channel. """

    STD: tuple[float]
    r"""The standard deviation values of each channel. """

    CLASS_MAP: dict[int | str, int]
    r"""The mapping from class name to class index."""


class MNISTConstants(DatasetConstants):
    r"""Constants about MNIST dataset."""

    NUM_CLASSES: int = 10
    r"""The number of classes in MNIST dataset."""

    IMG_SIZE: torch.Size = torch.Size([1, 28, 28])
    r"""The size of MNIST images."""

    MEAN: tuple[float] = (0.1307,)
    r"""The mean values of each channel. """

    STD: tuple[float] = (0.3081,)
    r"""The standard deviation values of each channel. """

    CLASS_MAP: dict[int | str, int] = {
        0: 0,  # digit 0
        1: 1,  # digit 1
        2: 2,  # digit 2
        3: 3,  # digit 3
        4: 4,  # digit 4
        5: 5,  # digit 5
        6: 6,  # digit 6
        7: 7,  # digit 7
        8: 8,  # digit 8
        9: 9,  # digit 9
    }
    r"""The mapping from class name to class index. They correspond to the digits 0-9."""


class EMNISTByClassConstants(DatasetConstants):
    r"""Constants about EMNIST ByClass dataset."""

    NUM_CLASSES: int = 62
    r"""The number of classes in EMNIST ByClass dataset."""

    IMG_SIZE: torch.Size = torch.Size([1, 28, 28])
    r"""The size of EMNIST ByClass images."""

    MEAN: tuple[float] = (0.1751,)
    r"""The mean values of each channel. """

    STD: tuple[float] = (0.3332,)
    r"""The standard deviation values of each channel. """

    # CLASS_MAP: dict[int | str, int] = {
    #     0: 0, # ???
    #     1: 1, #


class EMNISTByMergeConstants(DatasetConstants):
    r"""Constants about EMNIST ByMerge dataset."""

    NUM_CLASSES: int = 47
    r"""The number of classes in EMNIST ByMerge dataset."""

    IMG_SIZE: torch.Size = torch.Size([1, 28, 28])
    r"""The size of EMNIST ByMerge images."""

    MEAN: tuple[float] = (0.1740,)
    r"""The mean values of each channel. """

    STD: tuple[float] = (0.3310,)
    r"""The standard deviation values of each channel. """


class EMNISTBalancedConstants(DatasetConstants):
    r"""Constants about EMNIST Balanced dataset."""

    NUM_CLASSES: int = 47
    r"""The number of classes in EMNIST Balanced dataset."""

    IMG_SIZE: torch.Size = torch.Size([1, 28, 28])
    r"""The size of EMNIST Balanced images."""

    MEAN: tuple[float] = (0.1754,)
    r"""The mean values of each channel. """

    STD: tuple[float] = (0.3336,)
    r"""The standard deviation values of each channel. """


class EMNISTLettersConstants(DatasetConstants):
    r"""Constants about EMNIST Letters dataset."""

    NUM_CLASSES: int = 26
    r"""The number of classes in EMNIST Letters dataset."""

    IMG_SIZE: torch.Size = torch.Size([1, 28, 28])
    r"""The size of EMNIST Letters images."""

    MEAN: tuple[float] = (0.1722,)
    r"""The mean values of each channel. """

    STD: tuple[float] = (0.3310,)
    r"""The standard deviation values of each channel. """


class EMNISTDigitsConstants(DatasetConstants):
    r"""Constants about EMNIST Digits dataset."""

    NUM_CLASSES: int = 10
    r"""The number of classes in EMNIST Digits dataset."""

    IMG_SIZE: torch.Size = torch.Size([1, 28, 28])
    r"""The size of EMNIST Digits images."""

    MEAN: tuple[float] = (0.1736,)
    r"""The mean values of each channel. """

    STD: tuple[float] = (0.3317,)
    r"""The standard deviation values of each channel. """


class FashionMNISTConstants(DatasetConstants):
    r"""Constants about Fashion-MNIST dataset."""

    NUM_CLASSES: int = 10
    r"""The number of classes in Fashion-MNIST dataset."""

    IMG_SIZE: torch.Size = torch.Size([1, 28, 28])
    r"""The size of Fashion-MNIST images."""

    MEAN: tuple[float] = (0.2860,)
    r"""The mean values of each channel. """

    STD: tuple[float] = (0.3530,)
    r"""The standard deviation values of each channel. """


class KMNISTConstants(DatasetConstants):
    r"""Constants about Kuzushiji-MNIST dataset."""

    NUM_CLASSES: int = 10
    r"""The number of classes in Kuzushiji-MNIST dataset."""

    IMG_SIZE: torch.Size = torch.Size([1, 28, 28])
    r"""The size of Kuzushiji-MNIST images."""

    MEAN: tuple[float] = (0.1904,)
    r"""The mean values of each channel. """

    STD: tuple[float] = (0.3475,)
    r"""The standard deviation values of each channel. """


class NotMNISTConstants(DatasetConstants):
    r"""Constants about NotMNIST dataset."""

    NUM_CLASSES: int = 10
    r"""The number of classes in NotMNIST dataset."""

    IMG_SIZE: torch.Size = torch.Size([1, 28, 28])
    r"""The size of NotMNIST images."""

    MEAN: tuple[float] = (0.4254,)
    r"""The mean values of each channel. """

    STD: tuple[float] = (0.4501,)
    r"""The standard deviation values of each channel. """


class SignLanguageMNISTConstants(DatasetConstants):
    r"""Constants about Sign Language MNIST dataset."""

    NUM_CLASSES: int = 24
    r"""The number of classes in Sign Language MNIST dataset."""

    IMG_SIZE: torch.Size = torch.Size([1, 28, 28])
    r"""The size of Sign Language MNIST images."""

    MEAN: tuple[float] = (0.3079,)
    r"""The mean values of each channel. """

    STD: tuple[float] = (0.2741,)
    r"""The standard deviation values of each channel. """


class SVHNConstants(DatasetConstants):
    r"""Constants about SVHN dataset."""

    NUM_CLASSES: int = 10
    r"""The number of classes in SVHN dataset."""

    IMG_SIZE: torch.Size = torch.Size([3, 32, 32])
    r"""The size of SVHN images."""

    MEAN: tuple[float] = (0.4377, 0.4438, 0.4728)
    r"""The mean values of each channel. """

    STD: tuple[float] = (0.1980, 0.2010, 0.1970)
    r"""The standard deviation values of each channel. """


class CIFAR10Constants(DatasetConstants):
    r"""Constants about CIFAR-10 dataset."""

    NUM_CLASSES: int = 10
    r"""The number of classes in CIFAR-10 dataset."""

    IMG_SIZE: torch.Size = torch.Size([1, 32, 32])
    r"""The size of CIFAR images."""

    MEAN: tuple[float] = (0.5074, 0.4867, 0.4411)
    r"""The mean values of each channel. """

    STD: tuple[float] = (0.2011, 0.1987, 0.2025)
    r"""The standard deviation values of each channel. """


class CIFAR100Constants(DatasetConstants):
    r"""Constants about CIFAR-100 dataset."""

    NUM_CLASSES: int = 100
    r"""The number of classes in CIFAR-100 dataset."""

    IMG_SIZE: torch.Size = torch.Size([1, 32, 32])
    r"""The size of CIFAR images."""

    MEAN: tuple[float] = (0.5074, 0.4867, 0.4411)
    r"""The mean values of each channel. """

    STD: tuple[float] = (0.2011, 0.1987, 0.2025)
    r"""The standard deviation values of each channel. """


class TinyImageNetConstants(DatasetConstants):
    r"""Constants about TinyImageNet dataset."""

    NUM_CLASSES: int = 200
    r"""The number of classes in Tiny ImageNet dataset."""

    IMG_SIZE: torch.Size = torch.Size([3, 64, 64])
    r"""The size of Tiny ImageNet images."""

    MEAN: tuple[float] = (0.4802, 0.4481, 0.3975)
    r"""The mean values of each channel. """

    STD: tuple[float] = (0.2302, 0.2265, 0.2262)
    r"""The standard deviation values of each channel. """


class CUB2002011Constants(DatasetConstants):
    r"""Constants about CUB-200-2011 dataset."""

    NUM_CLASSES: int = 200
    r"""The number of classes in CUB-200-2011 dataset."""

    MEAN: tuple[float] = (0.4853, 0.4994, 0.4324)
    r"""The mean values of each channel. """

    STD: tuple[float] = (0.2290, 0.2242, 0.2605)
    r"""The standard deviation values of each channel. """


class GTSRBConstants(DatasetConstants):
    r"""Constants about GTSRB dataset."""

    NUM_CLASSES: int = 43
    r"""The number of classes in GTSRB dataset."""

    IMG_SIZE: torch.Size = torch.Size([3, 32, 32])
    r"""The size of GTSRB images."""

    MEAN: tuple[float] = (0.4850, 0.4560, 0.4060)
    r"""The mean values of each channel. GTSRB does not have official mean values, so we use the ImageNet mean values."""

    STD: tuple[float] = (0.2290, 0.2240, 0.2250)
    r"""The standard deviation values of each channel. GTSRB does not have official std values, so we use the ImageNet std values."""


DATASET_CONSTANTS_MAPPING: dict[type[Dataset], type[DatasetConstants]] = {
    torchvision.datasets.MNIST: MNISTConstants,
    original.EMNISTByClass: EMNISTByClassConstants,
    original.EMNISTByMerge: EMNISTByMergeConstants,
    original.EMNISTBalanced: EMNISTBalancedConstants,
    original.EMNISTLetters: EMNISTLettersConstants,
    original.EMNISTDigits: EMNISTDigitsConstants,
    original.SignLanguageMNIST: SignLanguageMNISTConstants,
    original.NotMNIST: NotMNISTConstants,
    torchvision.datasets.FashionMNIST: FashionMNISTConstants,
    torchvision.datasets.KMNIST: KMNISTConstants,
    torchvision.datasets.SVHN: SVHNConstants,
    torchvision.datasets.CIFAR10: CIFAR10Constants,
    torchvision.datasets.CIFAR100: CIFAR100Constants,
    tinyimagenet.TinyImageNet: TinyImageNetConstants,
    original.CUB2002011: CUB2002011Constants,
    torchvision.datasets.GTSRB: GTSRBConstants,
}
r"""A dictionary that maps dataset classes to their corresponding constants."""
