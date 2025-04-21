r"""

# Original Datasets

This submodule provides the **original datasets** that are used for constructing continual learning datasets.

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
from .notmnist import NotMNIST
from .sign_language_mnist import SignLanguageMNIST

__all__ = [
    "cub2002011",
    "emnist",
    "notmnist",
    "sign_language_mnist",
]
