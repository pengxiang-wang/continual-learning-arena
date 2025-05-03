r"""
The submodule in `backbones` for [HAT (Hard Attention to the Task, 2018)](http://proceedings.mlr.press/v80/serra18a) masked MLP backbone network.
"""

__all__ = ["HATMaskMLP"]

from torch import Tensor, nn

from clarena.backbones import MLP, HATMaskBackbone


class HATMaskMLP(MLP, HATMaskBackbone):
    r"""HAT masked multi-Layer perceptron (MLP).

    [HAT (Hard Attention to the Task, 2018)](http://proceedings.mlr.press/v80/serra18a) is an architecture-based continual learning approach that uses learnable hard attention masks to select the task-specific parameters.

    MLP is a dense network architecture, which has several fully-connected layers, each followed by an activation function. The last layer connects to the CL output heads.

    Mask is applied to the units which are neurons in each fully-connected layer. The mask is generated from the unit-wise task embedding and gate function.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int],
        output_dim: int,
        gate: str,
        activation_layer: nn.Module | None = nn.ReLU,
        bias: bool = True,
        dropout: float | None = None,
    ) -> None:
        r"""Construct and initialise the HAT masked MLP backbone network with task embedding. Note that batch normalisation is incompatible with HAT mechanism.

        **Args:**
        - **input_dim** (`int`): the input dimension. Any data need to be flattened before going in MLP.
        - **hidden_dims** (`list[int]`): list of hidden layer dimensions. It can be empty list which means single-layer MLP, and it can be as many layers as you want. Note that it doesn't include the last dimension which we take as output dimension.
        - **output_dim** (`int`): the output dimension which connects to CL output heads.
        - **gate** (`str`): the type of gate function turning the real value task embeddings into attention masks, should be one of the following:
            - `sigmoid`: the sigmoid function.
        - **activation_layer** (`nn.Module` | `None`): activation function of each layer (if not `None`), if `None` this layer won't be used. Default `nn.ReLU`.
        - **bias** (`bool`): whether to use bias in the linear layer. Default `True`.
        - **dropout** (`float` | `None`): the probability for the dropout layer, if `None` this layer won't be used. Default `None`.
        """
        # init from both inherited classes
        HATMaskBackbone.__init__(self, output_dim=output_dim, gate=gate)
        MLP.__init__(
            self,
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            output_dim=output_dim,
            activation_layer=activation_layer,
            batch_normalisation=False,  # batch normalisation is incompatible with HAT mechanism
            bias=bias,
            dropout=dropout,
        )
        self.register_hat_mask_module_explicitly(
            gate=gate
        )  # register all `nn.Module`s for HATMaskBackbone explicitly because the second `__init__()` wipes out them inited by the first `__init__()`

        # construct the task embedding for each weighted layer
        for layer_idx in range(self.num_fc_layers):
            full_layer_name = f"fc/{layer_idx}"
            layer_output_dim = (
                hidden_dims[layer_idx] if layer_idx != len(hidden_dims) else output_dim
            )
            self.task_embedding_t[full_layer_name] = nn.Embedding(
                num_embeddings=1, embedding_dim=layer_output_dim
            )

    def forward(
        self,
        input: Tensor,
        stage: str,
        s_max: float | None = None,
        batch_idx: int | None = None,
        num_batches: int | None = None,
        test_mask: dict[str, Tensor] | None = None,
    ) -> tuple[Tensor, dict[str, Tensor], dict[str, Tensor]]:
        r"""The forward pass for data from task `task_id`. Task-specific mask for `task_id` are applied to the units which are neurons in each fully-connected layer.

        **Args:**
        - **input** (`Tensor`): the input tensor from data.
        - **stage** (`str`): the stage of the forward pass, should be one of the following:
            1. 'train': training stage.
            2. 'validation': validation stage.
            3. 'test': testing stage.
        - **s_max** (`float` | `None`): the maximum scaling factor in the gate function. Doesn't apply to testing stage. See chapter 2.4 "Hard Attention Training" in [HAT paper](http://proceedings.mlr.press/v80/serra18a).
        - **batch_idx** (`int` | `None`): the current batch index. Applies only to training stage. For other stages, it is default `None`.
        - **num_batches** (`int` | `None`): the total number of batches. Applies only to training stage. For other stages, it is default `None`.
        - **test_mask** (`dict[str, Tensor]` | `None`): the binary mask used for test. Applies only to testing stage. For other stages, it is default `None`.

        **Returns:**
        - **output_feature** (`Tensor`): the output feature tensor to be passed into heads. This is the main target of backpropagation.
        - **mask** (`dict[str, Tensor]`): the mask for the current task. Key (`str`) is layer name, value (`Tensor`) is the mask tensor. The mask tensor has size (number of units, ).
        - **activations** (`dict[str, Tensor]`): the hidden features (after activation) in each weighted layer. Key (`str`) is the weighted layer name, value (`Tensor`) is the hidden feature tensor. This is used for the continual learning algorithms that need to use the hidden features for various purposes. Although HAT algorithm does not need this, it is still provided for API consistence for other HAT-based algorithms inherited this `forward()` method of `HAT` class.
        """
        batch_size = input.size(0)
        activations = {}

        # get the mask for the current task from the task embedding in this stage
        mask = self.get_mask(
            stage=stage,
            s_max=s_max,
            batch_idx=batch_idx,
            num_batches=num_batches,
            test_mask=test_mask,
        )

        x = input.view(batch_size, -1)  # flatten before going through MLP

        for layer_idx, layer_name in enumerate(self.weighted_layer_names):
            x = self.fc[layer_idx](x)  # fully-connected layer first
            x = x * mask[f"fc/{layer_idx}"]  # apply the mask to the parameters second
            if self.activation:
                x = self.fc_activation[layer_idx](x)  # activation function third
            activations[layer_name] = x  # store the hidden feature
            if self.dropout:
                x = self.fc_dropout[layer_idx](x)  # dropout last

        output_feature = x

        return output_feature, mask, activations
