r"""
The submodule in `cl_datasets` for constants about datasets.
"""

__all__ = ["MNISTConstants"]

import tinyimagenet
import torch
from torch.utils.data import Dataset
from torchvision import datasets as implemented_by_torchvision

from clarena.cl_datasets import original as implemented_by_clarena


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


DATASET_CONSTANTS: dict[type[Dataset], type[DatasetConstants]] = {
    implemented_by_torchvision.MNIST: MNISTConstants,
    implemented_by_torchvision.CIFAR10: CIFAR10Constants,
    implemented_by_torchvision.CIFAR100: CIFAR100Constants,
    tinyimagenet.TinyImageNet: TinyImageNetConstants,
    implemented_by_clarena.CUB2002011: CUB2002011Constants,
}
