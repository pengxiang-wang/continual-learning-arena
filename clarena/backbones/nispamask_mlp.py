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
        r"""Construct and initialise the WSN masked MLP backbone network with task embedding. Note that batch normalisation is incompatible with WSN mechanism.

        **Args:**
        - **input_dim** (`int`): the input dimension. Any data need to be flattened before going in MLP.
        - **hidden_dims** (`list[int]`): list of hidden layer dimensions. It can be empty list which means single-layer MLP, and it can be as many layers as you want. Note that it doesn't include the last dimension which we take as output dimension.
        - **output_dim** (`int`): the output dimension which connects to CL output heads.
        - **activation_layer** (`nn.Module` | `None`): activation function of each layer (if not `None`), if `None` this layer won't be used. Default `nn.ReLU`.
        - **bias** (`bool`): whether to use bias in the linear layer. Default `True`.
        - **dropout** (`float` | `None`): the probability for the dropout layer, if `None` this layer won't be used. Default `None`.
        """
        # init from both inherited classes
        WSNMaskBackbone.__init__(self, output_dim=output_dim)
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
        self.register_wsn_mask_module_explicitly()  # register all `nn.Module`s for WSNMaskBackbone explicitly because the second `__init__()` wipes out them inited by the first `__init__()`

        # construct the parameter score for each weighted layer
        for layer_idx in range(self.num_fc_layers):
            full_layer_name = f"fc/{layer_idx}"
            layer_input_dim = (
                input_dim if layer_idx == 0 else hidden_dims[layer_idx - 1]
            )
            layer_output_dim = (
                hidden_dims[layer_idx] if layer_idx != len(hidden_dims) else output_dim
            )
            self.weight_score_t[full_layer_name] = nn.Embedding(
                num_embeddings=layer_output_dim,
                embedding_dim=layer_input_dim,
            )
            self.bias_score_t[full_layer_name] = nn.Embedding(
                num_embeddings=1,
                embedding_dim=layer_output_dim,
            )

    def forward(
        self,
        input: Tensor,
        stage: str,
        mask_percentage: float,
        test_mask: tuple[dict[str, Tensor], dict[str, Tensor]] | None = None,
    ) -> tuple[Tensor, dict[str, Tensor], dict[str, Tensor], dict[str, Tensor]]:
        r"""The forward pass for data from task `task_id`. Task-specific mask for `task_id` are applied to the units which are neurons in each fully-connected layer.

        **Args:**
        - **input** (`Tensor`): The input tensor from data.
        - **stage** (`str`): the stage of the forward pass, should be one of the following:
            1. 'train': training stage.
            2. 'validation': validation stage.
            3. 'test': testing stage.
        - **mask_percentage** (`float`): the percentage of parameters to be masked. The value should be between 0 and 1.
        - **test_mask** (`tuple[dict[str, Tensor], dict[str, Tensor]]` | `None`): the binary weight and bias mask used for test. Applies only to testing stage. For other stages, it is default `None`.

        **Returns:**
        - **output_feature** (`Tensor`): the output feature tensor to be passed into heads. This is the main target of backpropagation.
        - **weight_mask** (`dict[str, Tensor]`): the weight mask for the current task. Key (`str`) is layer name, value (`Tensor`) is the mask tensor. The mask tensor has same (output features, input features) as weight.
        - **bias_mask** (`dict[str, Tensor]`): the bias mask for the current task. Key (`str`) is layer name, value (`Tensor`) is the mask tensor. The mask tensor has same (output features, ) as bias. If the layer doesn't have bias, it is `None`.
        - **activations** (`dict[str, Tensor]`): the hidden features (after activation) in each weighted layer. Key (`str`) is the weighted layer name, value (`Tensor`) is the hidden feature tensor. This is used for the continual learning algorithms that need to use the hidden features for various purposes.
        """
        batch_size = input.size(0)
        activations = {}

        weight_mask, bias_mask = self.get_mask(
            stage,
            mask_percentage=mask_percentage,
            test_mask=test_mask,
        )

        x = input.view(batch_size, -1)  # flatten before going through MLP

        for layer_idx, layer_name in enumerate(self.weighted_layer_names):
            weighted_layer = self.fc[layer_idx]
            weight = weighted_layer.weight
            bias = weighted_layer.bias

            # mask the weight and bias
            masked_weight = weight * weight_mask[f"fc/{layer_idx}"]
            if bias is not None and bias_mask[f"fc/{layer_idx}"] is not None:
                masked_bias = bias * bias_mask[f"fc/{layer_idx}"]
            else:
                masked_bias = None

            # do the forward pass using the masked weight and bias. Do not modify the weight and bias data in the original layer object or it will lose the computation graph.
            x = nn.functional.linear(x, masked_weight, masked_bias)

            if self.activation:
                x = self.fc_activation[layer_idx](x)  # activation function third
            activations[layer_name] = x  # store the hidden feature
            if self.dropout:
                x = self.fc_dropout[layer_idx](x)  # dropout last

        output_feature = x

        return output_feature, weight_mask, bias_mask, activations
