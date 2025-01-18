"""
The submodule in `backbones` for ResNet backbone network.
"""

__all__ = ["ResNet", "HATMaskResNet"]

from typing import Callable

from torch import Tensor, nn
from torchvision.ops import MLP as TorchvisionMLP

from clarena.backbones import CLBackbone, HATMaskBackbone


class ResNet(CLBackbone):
    """Residual network (ReseNet)."""


class HATMaskResNet(HATMaskBackbone):
    """HAT masked residual network (ReseNet).
    
    [HAT (Hard Attention to the Task, 2018)](http://proceedings.mlr.press/v80/serra18a) is an architecture-based continual learning approach that uses learnable hard attention masks to select the task-specific parameters. 
    """
