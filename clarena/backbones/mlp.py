"""
The submodule in `backbones` for MLP backbone network.
"""

__all__ = ["MLP"]

from typing import Callable

from torch import Tensor, nn
from torchvision.ops import MLP as TorchvisionMLP

from clarena.backbones import CLBackbone


class MLP(CLBackbone):
    """Multi-Layer Perceptron a.k.a. Fully-Connected Network.

    Modified from `torchvision.ops.MLP` in accordance with this framework.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int],
        output_dim: int,
        activation_layer: Callable[..., nn.Module] | None = nn.ReLU,
        bias: bool = True,
        dropout: float = 0.0,
    ):
        """Initialise the MLP backbone network.

        **Args:**
        - **input_dim** (`int`): the input dimension. Any data need to be flattened before going in MLP. Note that it is not required in convolutional networks.
        - **hidden_dims** (`list[int]`): list of hidden layer dimensions. It can be empty list which means single-layer MLP, and it can be as many layers as you want. Note that it doesn't include the last dimension which we take as output dimension.
        - **output_dim** (`int`): the output dimension which connects to CL output heads.
        - **activation_layer** (`Callable[..., torch.nn.Module]` | `None`): activation function of each layer (if not `None`), if `None` this layer won't be used.
        - **bias** (`bool`): whether to use bias in the linear layer. Default `True`.
        - **dropout** (`float`): The probability for the dropout layer. Default: 0.0.
        """
        super().__init__(output_dim=output_dim)

        self.mlp = TorchvisionMLP(
            in_channels=input_dim,
            hidden_channels=hidden_dims + [output_dim],
            activation_layer=activation_layer,
            bias=bias,
            dropout=dropout,
        )

    def forward(self, input: Tensor, task_id: int | None = None) -> Tensor:
        """The forward pass for data. It is the same for all tasks.

        **Args:**
        - **input** (`Tensor`): The input tensor from data.
        - **task_id** (`int`): the task ID where the data are from. It is just a placeholder for API consistence but never used.

        **Returns:**
        - The output feature tensor to be passed into heads.
        """
        input_flat = input.view(input.size(0), -1)  # flatten before going through MLP
        feature = self.mlp(input_flat)

        return feature
