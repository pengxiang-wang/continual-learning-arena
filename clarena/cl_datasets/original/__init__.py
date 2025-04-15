r"""

# Original Datasets

This submodule provides the **original datasets** that are used for constructing continual learning datasets.

The datasets are implemented as subclasses of PyTorch `Dataset` classes.
"""

from .cub2002011 import CUB2002011

__all__ = [
    "cub2002011",
]
