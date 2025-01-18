"""
The submodule in `backbones` for MLP backbone network.
"""

__all__ = ["MLP", "HATMaskMLP"]


from typing import Callable

from torch import Tensor, nn
from torchvision.ops import MLP as TorchvisionMLP

from clarena.backbones import CLBackbone, HATMaskBackbone


class MLP(CLBackbone):
    """Multi-layer perceptron (MLP) a.k.a. fully-connected network.

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
    ) -> None:
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

    def forward(self, input: Tensor, stage: str, task_id: int | None = None) -> Tensor:
        """The forward pass for data. It is the same for all tasks.

        **Args:**
        - **input** (`Tensor`): the input tensor from data.
        - **task_id** (`int`): the task ID where the data are from. It is just a placeholder for API consistence but never used.

        **Returns:**
        - **feature** (`Tensor`): the output feature tensor to be passed into heads.
        """
        input_flat = input.view(input.size(0), -1)  # flatten before going through MLP
        feature = self.mlp(input_flat)

        return feature


class HATMaskMLP(HATMaskBackbone):
    """HAT masked multi-Layer perceptron (MLP).

    [HAT (Hard Attention to the Task, 2018)](http://proceedings.mlr.press/v80/serra18a) is an architecture-based continual learning approach that uses learnable hard attention masks to select the task-specific parameters. 
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int],
        output_dim: int,
        gate: str,
        activation_layer: Callable[..., nn.Module] | None = nn.ReLU,
        bias: bool = True,
        dropout: float = 0.0,
    ) -> None:
        """Initialise the HAT masked MLP backbone network.

        **Args:**
        - **input_dim** (`int`): the input dimension. Any data need to be flattened before going in MLP. Note that it is not required in convolutional networks.
        - **hidden_dims** (`list[int]`): list of hidden layer dimensions. It can be empty list which means single-layer MLP, and it can be as many layers as you want. Note that it doesn't include the last dimension which we take as output dimension.
        - **output_dim** (`int`): the output dimension which connects to CL output heads.
        - **gate** (`str`): the type of gate function turning the real value task embeddings into attention masks, should be one of the following:
            - `sigmoid`: the sigmoid function.
        - **activation_layer** (`Callable[..., torch.nn.Module]` | `None`): activation function of each layer (if not `None`), if `None` this layer won't be used.
        - **bias** (`bool`): whether to use bias in the linear layer. Default `True`.
        - **dropout** (`float`): The probability for the dropout layer. Default: 0.0."""
        super().__init__(output_dim=output_dim, gate=gate)

        self.mlp = TorchvisionMLP(
            in_channels=input_dim,
            hidden_channels=hidden_dims + [output_dim],
            activation_layer=activation_layer,
            bias=bias,
            dropout=dropout,
        )

        self.masked_layer_order: list[str] = []

        # Construct task embedding over each fully connected layer
        for fc_index, fc in enumerate(
            [layer for layer in self.mlp.layers if isinstance(layer, nn.Linear)]
        ):
            layer_name = f"fc{fc_index}"
            self.masked_layers[layer_name] = fc
            self.task_embedding_t[layer_name] = nn.Embedding(1, fc.output_dim)
            self.masked_layer_order.append(layer_name)

    def forward(
        self,
        input: Tensor,
        stage: str,
        s_max: float | None = None,
        batch_idx: int | None = None,
        num_batches: int | None = None,
        task_id: int | None = None,
    ) -> Tensor:
        """The forward pass for data from task `task_id`. Task-specific mask for `task_id` are applied to the neurons in each layer.

        **Args:**
        - **input** (`Tensor`): The input tensor from data.
        - **stage** (`str`): the stage of the forward pass, should be one of the following:
            1. 'train': training stage.
            2. 'validate': validation stage.
            3. 'test': testing stage.
        - **s_max** (`float`): the maximum scaling factor in the gate function. See chapter 2.4. Hard Attention Training in HAT paper: <http://proceedings.mlr.press/v80/serra18a>.
        - **batch_idx** (`int` | `None`): the current batch index. Applies only to training stage. For other stages, it is default `None`.
        - **num_batches** (`int` | `None`): the total number of batches. Applies only to training stage. For other stages, it is default `None`.
        - **task_id** (`int`): the task ID where the data are from. Applies only to testing stage. For other stages, it is automatically the current task `self.task_id`. If stage is 'test', it could be from any seen task. In TIL, the task IDs of test data are provided thus this argument can be used. HAT algorithm works only for TIL.

        **Returns:**
        - **feature** (`Tensor`): the output feature tensor to be passed.
        - **mask** (`dict[str, Tensor]`): the mask for the current task. Key (`str`) is layer name, value (`Tensor`) is the mask tensor.
        """
        # get the mask for the current task from the task embedding in this stage
        mask = self.get_mask(
            stage,
            s_max=s_max,
            batch_idx=batch_idx,
            num_batches=num_batches,
            task_id=task_id,
        )
        batch_size = input.size(0)

        # flatten: (batch, channels, width, height) -> (batch, channels * width * height)
        x = input.view(batch_size, -1)

        fc_index = 0
        for layer in self.masked_layers:
            x = layer(x)
            if isinstance(layer, nn.Linear):
                # apply the mask to the weights
                x *= mask[f"fc{fc_index}"]
                fc_index += 1

        feature = x

        return feature, mask
