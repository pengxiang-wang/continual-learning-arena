r"""

# Continual Learning Datasets

This submodule provides the **continual learning datasets** that can be used in CLArena.

Here are the base classes for continual learning datasets, which inherit from Lightning `LightningDataModule`:

- `CLDataset`: The base class for all continual learning datasets.
    - `CLPermutedDataset`: The base class for permuted continual learning datasets. A child class of `CLDataset`.
    - `CLSplitDataset`: The base class for split continual learning datasets. A child class of `CLDataset`.
    - `CLCombinedDataset`: The base class for combined continual learning datasets. A child class of `CLDataset`.

Please note that this is an API documantation. Please refer to the main documentation pages for more information about how to configure and implement continual learning datasets:

- [**Configure CL Dataset**](https://pengxiang-wang.com/projects/continual-learning-arena/docs/components/CL-dataset)
- [**Implement Custom CL Dataset**](https://pengxiang-wang.com/projects/continual-learning-arena/docs/custom-implementation/cl_dataset)
- [**A Beginners' Guide to Continual Learning (CL Dataset)**](https://pengxiang-wang.com/posts/continual-learning-beginners-guide#sec-CL-dataset)


"""

from .base import (
    CLDataset,
    CLPermutedDataset,
    CLSplitDataset,
    CLCombinedDataset,
)

from .permuted_mnist import PermutedMNIST
from .permuted_emnist import PermutedEMNIST
from .permuted_fashionmnist import PermutedFashionMNIST
from .permuted_kmnist import PermutedKMNIST
from .permuted_notmnist import PermutedNotMNIST
from .permuted_sign_language_mnist import PermutedSignLanguageMNIST
from .permuted_ahdd import PermutedArabicHandwrittenDigits
from .permuted_kannadamnist import PermutedKannadaMNIST
from .permuted_svhn import PermutedSVHN
from .permuted_country211 import PermutedCountry211
from .permuted_imagenette import PermutedImagenette
from .permuted_dtd import PermutedDTD
from .permuted_cifar10 import PermutedCIFAR10
from .permuted_cifar100 import PermutedCIFAR100
from .permuted_caltech101 import PermutedCaltech101
from .permuted_caltech256 import PermutedCaltech256
from .permuted_eurosat import PermutedEuroSAT
from .permuted_fgvc_aircraft import PermutedFGVCAircraft
from .permuted_flowers102 import PermutedFlowers102
from .permuted_food101 import PermutedFood101
from .permuted_celeba import PermutedCelebA
from .permuted_fer2013 import PermutedFER2013
from .permuted_tinyimagenet import PermutedTinyImageNet
from .permuted_oxford_iiit_pet import PermutedOxfordIIITPet
from .permuted_pcam import PermutedPCAM
from .permuted_renderedsst2 import PermutedRenderedSST2
from .permuted_stanfordcars import PermutedStanfordCars
from .permuted_sun397 import PermutedSUN397
from .permuted_usps import PermutedUSPS
from .permuted_SEMEION import PermutedSEMEION
from .permuted_facescrub import PermutedFaceScrub
from .permuted_cub2002011 import PermutedCUB2002011
from .permuted_gtsrb import PermutedGTSRB
from .permuted_linnaeus5 import PermutedLinnaeus5

from .split_cifar10 import SplitCIFAR10
from .split_mnist import SplitMNIST
from .split_cifar100 import SplitCIFAR100
from .split_tinyimagenet import SplitTinyImageNet
from .split_cub2002011 import SplitCUB2002011

from .combined import Combined


__all__ = [
    "CLDataset",
    "CLPermutedDataset",
    "CLSplitDataset",
    "CLCombinedDataset",
    "combined",
    "permuted_mnist",
    "permuted_emnist",
    "permuted_fashionmnist",
    "permuted_imagenette",
    "permuted_sign_language_mnist",
    "permuted_ahdd",
    "permuted_kannadamnist",
    "permuted_country211",
    "permuted_dtd",
    "permuted_fer2013",
    "permuted_fgvc_aircraft",
    "permuted_flowers102",
    "permuted_food101",
    "permuted_kmnist",
    "permuted_notmnist",
    "permuted_svhn",
    "permuted_cifar10",
    "permuted_cifar100",
    "permuted_caltech101",
    "permuted_caltech256",
    "permuted_oxford_iiit_pet",
    "permuted_celeba",
    "permuted_eurosat",
    "permuted_facescrub",
    "permuted_pcam",
    "permuted_renderedsst2",
    "permuted_stanfordcars",
    "permuted_sun397",
    "permuted_usps",
    "permuted_semeion",
    "permuted_tinyimagenet",
    "permuted_cub2002011",
    "permuted_gtsrb",
    "permuted_linnaeus5",
    "split_mnist",
    "split_cifar10",
    "split_cifar100",
    "split_tinyimagenet",
    "split_cub2002011",
]
