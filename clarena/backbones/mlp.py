r"""
The submodule in `backbones` for MLP backbone network.
"""

__all__ = ["MLP", "HATMaskMLP"]

from torch import Tensor, nn

from clarena.backbones import CLBackbone, HATMaskBackbone


class MLP(CLBackbone):
    """**Multi-layer perceptron (MLP)** a.k.a. fully-connected network.

    MLP is an dense network architecture, which has several fully-connected layers, each followed by an activation function. The last layer connects to the CL output heads.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int],
        output_dim: int,
        activation_layer: nn.Module | None = nn.ReLU,
        batch_normalisation: bool = False,
        bias: bool = True,
        dropout: float | None = None,
    ) -> None:
        r"""Construct and initialise the MLP backbone network.

        **Args:**
        - **input_dim** (`int`): the input dimension. Any data need to be flattened before going in MLP. Note that it is not required in convolutional networks.
        - **hidden_dims** (`list[int]`): list of hidden layer dimensions. It can be empty list which means single-layer MLP, and it can be as many layers as you want. Note that it doesn't include the last dimension which we take as output dimension.
        - **output_dim** (`int`): the output dimension which connects to CL output heads.
        - **activation_layer** (`nn.Module` | `None`): activation function of each layer (if not `None`), if `None` this layer won't be used. Default `nn.ReLU`.
        - **batch_normalisation** (`bool`): whether to use batch normalisation after the fully-connected layers. Default `False`.
        - **bias** (`bool`): whether to use bias in the linear layer. Default `True`.
        - **dropout** (`float` | `None`): the probability for the dropout layer, if `None` this layer won't be used. Default `None`.
        """
        CLBackbone.__init__(self, output_dim=output_dim)

        self.num_fc_layers: int = len(hidden_dims) + 1
        r"""Store the number of fully-connected layers in the MLP backbone network, which helps form the loops in constructing layers and forward pass."""
        self.batch_normalisation: bool = batch_normalisation
        r"""Store whether to use batch normalisation after the fully-connected layers."""
        self.activation: bool = activation_layer is not None
        r"""Store whether to use activation function after the fully-connected layers."""
        self.dropout: bool = dropout is not None
        r"""Store whether to use dropout after the fully-connected layers."""

        self.fc: nn.ModuleList = nn.ModuleList()
        r"""The list of fully-connected (`nn.Linear`) layers. """
        if self.batch_normalisation:
            self.fc_bn: nn.ModuleList = nn.ModuleList()
            r"""The list of batch normalisation (`nn.BatchNorm1d`) layers after the fully-connected layers."""
        if self.activation:
            self.fc_activation: nn.ModuleList = nn.ModuleList()
            r"""The list of activation layers after the fully-connected layers. """
        if self.dropout:
            self.fc_dropout: nn.ModuleList = nn.ModuleList()
            r"""The list of dropout layers after the fully-connected layers. """

        # construct the weighted fully-connected layers and attached layers (batchnorm, activation, dropout, etc) in a loop
        for layer_idx in range(self.num_fc_layers):

            # the input and output dim of the current weighted layer
            layer_input_dim = (
                input_dim if layer_idx == 0 else hidden_dims[layer_idx - 1]
            )
            layer_output_dim = (
                hidden_dims[layer_idx] if layer_idx != len(hidden_dims) else output_dim
            )

            # construct the fully connected layer
            self.fc.append(
                nn.Linear(
                    in_features=layer_input_dim,
                    out_features=layer_output_dim,
                    bias=bias,
                )
            )

            # update the weighted layer names
            full_layer_name = f"fc/{layer_idx}"
            self.weighted_layer_names.append(full_layer_name)

            # construct the batch normalisation layer
            if self.batch_normalisation:
                self.fc_bn.append(nn.BatchNorm1d(num_features=(layer_output_dim)))

            # construct the activation layer
            if self.activation:
                self.fc_activation.append(activation_layer())

            # construct the dropout layer
            if self.dropout:
                self.fc_dropout.append(nn.Dropout(dropout))

    def forward(
        self, input: Tensor, stage: str = None, task_id: int | None = None
    ) -> tuple[Tensor, dict[str, Tensor]]:
        r"""The forward pass for data. It is the same for all tasks.

        **Args:**
        - **input** (`Tensor`): the input tensor from data.

        **Returns:**
        - **output_feature** (`Tensor`): the output feature tensor to be passed into heads. This is the main target of backpropagation.
        - **hidden_features** (`dict[str, Tensor]`): the hidden features (after activation) in each weighted layer. Key (`str`) is the weighted layer name, value (`Tensor`) is the hidden feature tensor. This is used for the continual learning algorithms that need to use the hidden features for various purposes.
        """
        batch_size = input.size(0)
        hidden_features = {}

        x = input.view(batch_size, -1)  # flatten before going through MLP

        for layer_idx, layer_name in enumerate(self.weighted_layer_names):
            x = self.fc[layer_idx](x)  # fully-connected layer first
            if self.batch_normalisation:
                x = self.fc_bn[layer_idx](x)  # batch normalisation second
            if self.activation:
                x = self.fc_activation[layer_idx](x)  # activation function third
            hidden_features[layer_name] = x  # store the hidden feature
            if self.dropout:
                x = self.fc_dropout[layer_idx](x)  # dropout last

        output_feature = x

        return output_feature, hidden_features


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
        - **mask** (`dict[str, Tensor]`): the mask for the current task. Key (`str`) is layer name, value (`Tensor`) is the mask tensor. The mask tensor has size (number of units).
        - **hidden_features** (`dict[str, Tensor]`): the hidden features (after activation) in each weighted layer. Key (`str`) is the weighted layer name, value (`Tensor`) is the hidden feature tensor. This is used for the continual learning algorithms that need to use the hidden features for various purposes. Although HAT algorithm does not need this, it is still provided for API consistence for other HAT-based algorithms inherited this `forward()` method of `HAT` class.
        """
        batch_size = input.size(0)
        hidden_features = {}

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
            hidden_features[layer_name] = x  # store the hidden feature
            if self.dropout:
                x = self.fc_dropout[layer_idx](x)  # dropout last

        output_feature = x

        return output_feature, mask, hidden_features
