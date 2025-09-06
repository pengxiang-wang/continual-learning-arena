r"""

# Single-Task Learning Datasets

This submodule provides the **single-task learning datasets** that can be used in CLArena.

Here are the base classes for single-task learning datasets, which inherit from Lightning `LightningDataModule`:

- `STLDataset`: The base class for all single-task learning datasets.
    - `STLDatasetFromRaw`: The base class for constructing single-task learning datasets from raw datasets. A child class of `STLDataset`.

Please note that this is an API documantation. Please refer to the main documentation pages for more information about how to configure and implement single-task learning datasets:

- [**Configure STL Dataset**](https://pengxiang-wang.com/projects/continual-learning-arena/docs/components/STL-dataset)
- [**Implement Custom STL Dataset**](https://pengxiang-wang.com/projects/continual-learning-arena/docs/custom-implementation/STL-dataset)

"""

from .base import STLDataset, STLDatasetFromRaw, TaskLabelledDataset

from .ahdd import ArabicHandwrittenDigits
from .caltech101 import Caltech101
from .caltech256 import Caltech256
from .celeba import CelebA
from .cifar10 import CIFAR10
from .cifar100 import CIFAR100
from .country211 import Country211
from .cub2002011 import CUB2002011
from .dtd import DTD
from .facescrub import FaceScrub
from .fashionmnist import FashionMNIST
from .fer2013 import FER2013
from .fgvc_aircraft import FGVCAircraft
from .flowers102 import Flowers102
from .food101 import Food101
from .emnist import EMNIST
from .eurosat import EuroSAT
from .gtsrb import GTSRB
from .imagenette import Imagenette
from .kannadamnist import KannadaMNIST
from .kmnist import KMNIST
from .linnaeus5 import Linnaeus5
from .mnist import MNIST
from .notmnist import NotMNIST
from .oxford_iiit_pet import OxfordIIITPet
from .pcam import PCAM
from .renderedsst2 import RenderedSST2
from .SEMEION import SEMEION
from .sign_language_mnist import SignLanguageMNIST
from .stanfordcars import StanfordCars
from .sun397 import SUN397
from .svhn import SVHN
from .tinyimagenet import TinyImageNet
from .usps import USPS


__all__ = [
    "STLDataset",
    "STLDatasetFromRaw",
    "TaskLabelledDataset",
    "ahdd",
    "caltech101",
    "caltech256",
    "celeba",
    "cifar10",
    "cifar100",
    "country211",
    "cub2002011",
    "dtd",
    "facescrub",
    "fashionmnist",
    "fer2013",
    "fgvc_aircraft",
    "flowers102",
    "food101",
    "emnist",
    "eurosat",
    "gtsrb",
    "imagenette",
    "kannadamnist",
    "kmnist",
    "linnaeus5",
    "mnist",
    "notmnist",
    "oxford_iiit_pet",
    "pcam",
    "renderedsst2",
    "SEMEION",
    "sign_language_mnist",
    "stanfordcars",
    "sun397",
    "svhn",
    "tinyimagenet",
    "usps",
]
