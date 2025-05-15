r"""
The submodule in `backbones` for [NISPA (Neuro-Inspired Stability-Plasticity Adaptation)](https://proceedings.mlr.press/v162/gurbuz22a/gurbuz22a.pdf) masked MLP backbone network.
"""

__all__ = ["NISPAMaskMLP"]

from torch import Tensor, nn

from clarena.backbones import MLP, NISPAMaskBackbone


class NISPAMaskMLP(MLP, NISPAMaskBackbone):
    r"""[NISPA (Neuro-Inspired Stability-Plasticity Adaptation)](https://proceedings.mlr.press/v162/gurbuz22a/gurbuz22a.pdf) masked multi-Layer perceptron (MLP).

    [NISPA (Neuro-Inspired Stability-Plasticity Adaptation)](https://proceedings.mlr.press/v162/gurbuz22a/gurbuz22a.pdf) is an architecture-based continual learning algorithm. It

    MLP is a dense network architecture, which has several fully-connected layers, each followed by an activation function. The last layer connects to the CL output heads.

    Mask is applied to the weights and biases in each fully-connected layer. The mask is generated from the parameter-wise score and gate function.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int],
        output_dim: int,
        activation_layer: nn.Module | None = nn.ReLU,
        bias: bool = True,
        dropout: float | None = None,
    ) -> None:
        r"""Construct and initialise the NISPA masked MLP backbone network with task embedding.
        **Args:**
        - **input_dim** (`int`): the input dimension. Any data need to be flattened before going in MLP.
        - **hidden_dims** (`list[int]`): list of hidden layer dimensions. It can be empty list which means single-layer MLP, and it can be as many layers as you want. Note that it doesn't include the last dimension which we take as output dimension.
        - **output_dim** (`int`): the output dimension which connects to CL output heads.
        - **activation_layer** (`nn.Module` | `None`): activation function of each layer (if not `None`), if `None` this layer won't be used. Default `nn.ReLU`.
        - **bias** (`bool`): whether to use bias in the linear layer. Default `True`.
        - **dropout** (`float` | `None`): the probability for the dropout layer, if `None` this layer won't be used. Default `None`.
        """
        # init from both inherited classes
        NISPAMaskBackbone.__init__(self, output_dim=output_dim)
        MLP.__init__(
            self,
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            output_dim=output_dim,
            activation_layer=activation_layer,
            batch_normalisation=False,
            bias=bias,
            dropout=dropout,
        )
        self.register_nispa_mask_module_explicitly()  # register all `nn.Module`s for NISPAMaskBackbone explicitly because the second `__init__()` wipes out them inited by the first `__init__()`

    def forward(
        self,
        input: Tensor,
        stage: str,
    ) -> tuple[Tensor, dict[str, Tensor], dict[str, Tensor], dict[str, Tensor]]:
        r"""The forward pass for data from task `task_id`. Parameter mask is applied to the parameters in each layer.

        **Args:**
        - **input** (`Tensor`): The input tensor from data.
        - **stage** (`str`): the stage of the forward pass, should be one of the following:
            1. 'train': training stage.
            2. 'validation': validation stage.
            3. 'test': testing stage.

        **Returns:**
        - **output_feature** (`Tensor`): the output feature tensor to be passed into heads. This is the main target of backpropagation.
        - **weight_mask** (`dict[str, Tensor]`): the weight mask. Key (`str`) is layer name, value (`Tensor`) is the mask tensor. The mask tensor has same (output features, input features) as weight.
        - **bias_mask** (`dict[str, Tensor]`): the bias mask. Key (`str`) is layer name, value (`Tensor`) is the mask tensor. The mask tensor has same (output features, ) as bias. If the layer doesn't have bias, it is `None`.
        - **activations** (`dict[str, Tensor]`): the hidden features (after activation) in each weighted layer. Key (`str`) is the weighted layer name, value (`Tensor`) is the hidden feature tensor. This is used for the continual learning algorithms that need to use the hidden features for various purposes.
        """
        batch_size = input.size(0)
        activations = {}

        weight_mask, bias_mask = (
            self.weight_mask_t,
            self.bias_mask_t,
        )  # get the stored parameter mask

        x = input.view(batch_size, -1)  # flatten before going through MLP

        for layer_idx, layer_name in enumerate(self.weighted_layer_names):
            weighted_layer = self.fc[layer_idx]

            # mask the weight and bias
            weighted_layer.weight.data *= weight_mask[f"fc.{layer_idx}"]
            if (
                weighted_layer.bias is not None
                and bias_mask[f"fc.{layer_idx}"] is not None
            ):
                weighted_layer.bias.data *= bias_mask[f"fc.{layer_idx}"]

            # forward pass
            x = weighted_layer(x)

            if self.activation:
                x = self.fc_activation[layer_idx](x)  # activation function third
            activations[layer_name] = x  # store the hidden feature
            if self.dropout:
                x = self.fc_dropout[layer_idx](x)  # dropout last

        output_feature = x

        return output_feature, weight_mask, bias_mask, activations
