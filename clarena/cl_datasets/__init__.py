r"""

# Continual Learning Datasets

This submodule provides the **continual learning datasets** that can be used in CLArena.

Please note that this is an API documantation. Please refer to the main documentation pages for more information about the CL datasets and how to configure and implement them:

- [**Configure CL Dataset**](https://pengxiang-wang.com/projects/continual-learning-arena/docs/configure-your-experiment/cl-dataset)
- [**Implement Your CL Dataset Class**](https://pengxiang-wang.com/projects/continual-learning-arena/docs/implement-your-cl-modules/cl-dataset)
- [**A Beginners' Guide to Continual Learning (CL Dataset)**](https://pengxiang-wang.com/posts/continual-learning-beginners-guide#sec-CL-dataset)

The datasets are implemented as subclasses of `CLDataset` classes, which are the base class for all continual learning datasets in CLArena.

- `CLDataset`: The base class for continual learning datasets.
- `CLPermutedDataset`: The base class for permuted continual learning datasets. A child class of `CLDataset`.

"""

from .base import (
    CLClassMapping,
    CLDataset,
    CLPermutedDataset,
    CLSplitDataset,
    CLCombinedDataset,
    Permute,
)
from .combined_torchvision import CombinedTorchvision
from .permuted_mnist import PermutedMNIST
from .permuted_emnist import PermutedEMNIST
from .permuted_fashionmnist import PermutedFashionMNIST
from .permuted_kmnist import PermutedKMNIST
from .permuted_notmnist import PermutedNotMNIST
from .permuted_sign_language_mnist import PermutedSignLanguageMNIST
from .permuted_svhn import PermutedSVHN
from .permuted_cifar10 import PermutedCIFAR10
from .permuted_cifar100 import PermutedCIFAR100
from .permuted_tinyimagenet import PermutedTinyImageNet
from .permuted_cub2002011 import PermutedCUB2002011
from .permuted_gtsrb import PermutedGTSRB
from .split_cifar10 import SplitCIFAR10
from .split_mnist import SplitMNIST
from .split_cifar100 import SplitCIFAR100
from .split_tinyimagenet import SplitTinyImageNet
from .split_cub2002011 import SplitCUB2002011

__all__ = [
    "CLDataset",
    "CLPermutedDataset",
    "CLSplitDataset",
    "CLCombinedDataset",
    "CLClassMapping",
    "Permute",
    "combined_torchvision",
    "permuted_mnist",
    "permuted_emnist",
    "permuted_fashionmnist",
    "permuted_sign_language_mnist",
    "permuted_kmnist",
    "permuted_notmnist",
    "permuted_svhn",
    "permuted_cifar10",
    "permuted_cifar100",
    "permuted_tinyimagenet",
    "permuted_cub2002011",
    "permuted_gtsrb",
    "split_mnist",
    "split_cifar10",
    "split_cifar100",
    "split_tinyimagenet",
    "split_cub2002011",
]
