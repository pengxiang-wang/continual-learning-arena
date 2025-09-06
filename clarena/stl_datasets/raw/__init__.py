r"""

# Raw Datasets

This submodule provides the **raw datasets** that are used for constructing continual learning, multi-task learning and single-task learning datasets.

The datasets are implemented as subclasses of PyTorch `Dataset` classes.
"""

from .cub2002011 import CUB2002011
from .emnist import (
    EMNISTByClass,
    EMNISTByMerge,
    EMNISTBalanced,
    EMNISTLetters,
    EMNISTDigits,
)
from .traffic_signs import TrafficSignsFromHAT
from .notmnist import NotMNIST, NotMNISTFromHAT
from .sign_language_mnist import SignLanguageMNIST
from .ahdd import ArabicHandwrittenDigits
from .kannada_mnist import KannadaMNIST
from .linnaeus5 import (
    Linnaeus5,
    Linnaeus5_32,
    Linnaeus5_64,
    Linnaeus5_128,
    Linnaeus5_256,
)
from .fgvc_aircraft import (
    FGVCAircraftVariant,
    FGVCAircraftFamily,
    FGVCAircraftManufacturer,
)
from .oxford_iiit_pet import OxfordIIITPet37, OxfordIIITPet2
from .facescrub import (
    FaceScrub10,
    FaceScrub20,
    FaceScrub50,
    FaceScrub100,
    FaceScrubFromHAT,
)


__all__ = [
    "cub2002011",
    "emnist",
    "notmnist",
    "traffic_signs",
    "sign_language_mnist",
    "ahdd",
    "kannada_mnist",
    "linnaeus5",
    "fgvc_aircraft",
    "oxford_iiit_pet",
    "facescrub",
]
