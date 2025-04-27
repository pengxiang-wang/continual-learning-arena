r"""
The submodule in `backbones` for [WSN (Winning Subnetworks, 2022)](https://proceedings.mlr.press/v162/kang22b/kang22b.pdf) masked MLP backbone network.
"""

__all__ = ["WSNMaskMLP"]

from torch import Tensor, nn

from clarena.backbones import MLP, HATMaskBackbone


class WSNMaskMLP(MLP, HATMaskBackbone):
    r"""[WSN (Winning Subnetworks, 2022)](https://proceedings.mlr.press/v162/kang22b/kang22b.pdf) masked multi-Layer perceptron (MLP).

    [WSN (Winning Subnetworks, 2022)](https://proceedings.mlr.press/v162/kang22b/kang22b.pdf) is an architecture-based continual learning algorithm. It trains learnable parameter-wise importance and select the most important $c\%$ of the network parameters to be used for each task.

    MLP is a dense network architecture, which has several fully-connected layers, each followed by an activation function. The last layer connects to the CL output heads.

    Mask...
    """
