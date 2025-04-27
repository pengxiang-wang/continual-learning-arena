r"""
The submodule in `cl_algorithms` for [WSN (Winning Subnetworks)](https://proceedings.mlr.press/v162/kang22b/kang22b.pdf) algorithm.
"""

__all__ = ["WSN"]

import logging
from typing import Any

from torch import Tensor
from torch.utils.data import DataLoader

from clarena.backbones import CLBackbone
from clarena.cl_algorithms import CLAlgorithm
from clarena.cl_heads import HeadsCIL, HeadsTIL

# always get logger for built-in logging in each module
pylogger = logging.getLogger(__name__)


class WSN(CLAlgorithm):
    r"""WSN (Winning Subnetworks) algorithm.

    [WSN (Winning Subnetworks, 2022)](https://proceedings.mlr.press/v162/kang22b/kang22b.pdf) is an architecture-based continual learning algorithm. It trains learnable parameter-wise importance and select the most important $c\%$ of the network parameters to be used for each task.
    """

    def __init__(
        self,
        backbone: CLBackbone,
        heads: HeadsTIL | HeadsCIL,
    ) -> None:
        r"""Initialise the WSN algorithm with the network.

        **Args:**
        - **backbone** (`CLBackbone`): backbone network.
        - **heads** (`HeadsTIL` | `HeadsCIL`): output heads.
        """
        CLAlgorithm.__init__(self, backbone=backbone, heads=heads)
