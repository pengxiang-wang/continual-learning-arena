"""
The submodule in `cl_algorithms` for [AdaHAT (Adaptive Hard Attention to the Task) algorithm](https://link.springer.com/chapter/10.1007/978-3-031-70352-2_9).
"""

__all__ = ["AdaHAT"]

import logging
from typing import Any

import torch
from lightning import Trainer
from torch import Tensor, nn
from torch.utils.data import DataLoader

from clarena.backbones.base import HATMaskBackbone
from clarena.cl_algorithms import CLAlgorithm
from clarena.cl_algorithms.regularisers import HATMaskSparsityReg

# always get logger for built-in logging in each module
pylogger = logging.getLogger(__name__)


class AdaHAT(CLAlgorithm):
    """AdaHAT (Adaptive Hard Attention to the Task) algorithm.

    [Adaptive HAT (Hard Attention to the Task, 2024)](http://proceedings.mlr.press/v80/serra18a) is
    """