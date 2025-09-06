r"""
The submodule in `backbones` for the MLP backbone network. It includes multiple versions of MLP, including the basic MLP, the continual learning MLP, the [HAT](http://proceedings.mlr.press/v80/serra18a) masked MLP, and the [WSN](https://proceedings.mlr.press/v162/kang22b/kang22b.pdf) masked MLP.
"""

__all__ = ["MLP", "CLMLP", "HATMaskMLP", "WSNMaskMLP"]

import logging
from copy import deepcopy

from torch import Tensor, nn

from clarena.backbones import Backbone, CLBackbone, HATMaskBackbone, WSNMaskBackbone

# always get logger for built-in logging in each module
pylogger = logging.getLogger(__name__)


class MLP(Backbone):
    """Multi-layer perceptron (MLP), a.k.a. fully connected network.

    MLP is a dense network architecture with several fully connected layers, each followed by an activation function. The last layer connects to the output heads.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int],
        output_dim: int,
        activation_layer: nn.Module | None = nn.ReLU,
        batch_normalization: bool = False,
        bias: bool = True,
        dropout: float | None = None,
        **kwargs,
    ) -> None:
        r"""Construct and initialize the MLP backbone network.

        **Args:**
        - **input_dim** (`int`): The input dimension. Any data need to be flattened before entering the MLP. Note that it is not required in convolutional networks.
        - **hidden_dims** (`list[int]`): List of hidden layer dimensions. It can be an empty list, which means a single-layer MLP, and it can be as many layers as you want. Note that it doesn't include the last dimension, which we take as the output dimension.
        - **output_dim** (`int`): The output dimension that connects to output heads.
        - **activation_layer** (`nn.Module` | `None`): Activation function of each layer (if not `None`). If `None`, this layer won't be used. Default `nn.ReLU`.
        - **batch_normalization** (`bool`): Whether to use batch normalization after the fully connected layers. Default `False`.
        - **bias** (`bool`): Whether to use bias in the linear layer. Default `True`.
        - **dropout** (`float` | `None`): The probability for the dropout layer. If `None`, this layer won't be used. Default `None`.
        - **kwargs**: Reserved for multiple inheritance.
        """
        super().__init__(output_dim=output_dim, **kwargs)

        self.input_dim: int = input_dim
        r"""The input dimension of the MLP backbone network."""
        self.hidden_dims: list[int] = hidden_dims
        r"""The hidden dimensions of the MLP backbone network."""
        self.output_dim: int = output_dim
        r"""The output dimension of the MLP backbone network."""

        self.num_fc_layers: int = len(hidden_dims) + 1
        r"""The number of fully-connected layers in the MLP backbone network, which helps form the loops in constructing layers and forward pass."""
        self.batch_normalization: bool = batch_normalization
        r"""Whether to use batch normalization after the fully-connected layers."""
        self.activation: bool = activation_layer is not None
        r"""Whether to use activation function after the fully-connected layers."""
        self.dropout: bool = dropout is not None
        r"""Whether to use dropout after the fully-connected layers."""

        self.fc: nn.ModuleList = nn.ModuleList()
        r"""The list of fully connected (`nn.Linear`) layers."""
        if self.batch_normalization:
            self.fc_bn: nn.ModuleList = nn.ModuleList()
            r"""The list of batch normalization (`nn.BatchNorm1d`) layers after the fully connected layers."""
        if self.activation:
            self.fc_activation: nn.ModuleList = nn.ModuleList()
            r"""The list of activation layers after the fully connected layers."""
        if self.dropout:
            self.fc_dropout: nn.ModuleList = nn.ModuleList()
            r"""The list of dropout layers after the fully connected layers."""

        # construct the weighted fully connected layers and attached layers (batch norm, activation, dropout, etc.) in a loop
        for layer_idx in range(self.num_fc_layers):

            # the input and output dim of the current weighted layer
            layer_input_dim = (
                self.input_dim if layer_idx == 0 else self.hidden_dims[layer_idx - 1]
            )
            layer_output_dim = (
                self.hidden_dims[layer_idx]
                if layer_idx != len(self.hidden_dims)
                else self.output_dim
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

            # construct the batch normalization layer
            if self.batch_normalization:
                self.fc_bn.append(nn.BatchNorm1d(num_features=(layer_output_dim)))

            # construct the activation layer
            if self.activation:
                self.fc_activation.append(activation_layer())

            # construct the dropout layer
            if self.dropout:
                self.fc_dropout.append(nn.Dropout(dropout))

    def forward(
        self, input: Tensor, stage: str = None
    ) -> tuple[Tensor, dict[str, Tensor]]:
        r"""The forward pass for data. It is the same for all tasks.

        **Args:**
        - **input** (`Tensor`): The input tensor from data.

        **Returns:**
        - **output_feature** (`Tensor`): The output feature tensor to be passed into heads. This is the main target of backpropagation.
        - **activations** (`dict[str, Tensor]`): The hidden features (after activation) in each weighted layer. Keys (`str`) are the weighted layer names and values (`Tensor`) are the hidden feature tensors. This is used for certain algorithms that need to use hidden features for various purposes.
        """
        batch_size = input.size(0)
        activations = {}

        x = input.view(batch_size, -1)  # flatten before going through MLP

        for layer_idx, layer_name in enumerate(self.weighted_layer_names):
            x = self.fc[layer_idx](x)  # fully-connected layer first
            if self.batch_normalization:
                x = self.fc_bn[layer_idx](
                    x
                )  # batch normalization can be before or after activation. We put it before activation here
            if self.activation:
                x = self.fc_activation[layer_idx](x)  # activation function third
            activations[layer_name] = x  # store the hidden feature
            if self.dropout:
                x = self.fc_dropout[layer_idx](x)  # dropout last

        output_feature = x

        return output_feature, activations


class CLMLP(CLBackbone, MLP):
    """Multi-layer perceptron (MLP), a.k.a. fully connected network. Used as a continual learning backbone.

    MLP is a dense network architecture with several fully connected layers, each followed by an activation function. The last layer connects to the CL output heads.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int],
        output_dim: int,
        activation_layer: nn.Module | None = nn.ReLU,
        batch_normalization: bool = False,
        bias: bool = True,
        dropout: float | None = None,
        **kwargs,
    ) -> None:
        r"""Construct and initialize the CLMLP backbone network.

        **Args:**
        - **input_dim** (`int`): the input dimension. Any data need to be flattened before going in MLP. Note that it is not required in convolutional networks.
        - **hidden_dims** (`list[int]`): list of hidden layer dimensions. It can be empty list which means single-layer MLP, and it can be as many layers as you want. Note that it doesn't include the last dimension which we take as output dimension.
        - **output_dim** (`int`): the output dimension which connects to CL output heads.
        - **activation_layer** (`nn.Module` | `None`): activation function of each layer (if not `None`), if `None` this layer won't be used. Default `nn.ReLU`.
        - **batch_normalization** (`bool`): whether to use batch normalization after the fully-connected layers. Default `False`.
        - **bias** (`bool`): whether to use bias in the linear layer. Default `True`.
        - **dropout** (`float` | `None`): the probability for the dropout layer, if `None` this layer won't be used. Default `None`.        - **kwargs**: Reserved for multiple inheritance.
        """
        super().__init__(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            output_dim=output_dim,
            activation_layer=activation_layer,
            batch_normalization=batch_normalization,
            bias=bias,
            dropout=dropout,
            **kwargs,
        )

    def forward(
        self, input: Tensor, stage: str = None, task_id: int | None = None
    ) -> tuple[Tensor, dict[str, Tensor]]:
        r"""The forward pass for data. It is the same for all tasks.

        **Args:**
        - **input** (`Tensor`): The input tensor from data.

        **Returns:**
        - **output_feature** (`Tensor`): The output feature tensor to be passed into heads. This is the main target of backpropagation.
        - **activations** (`dict[str, Tensor]`): The hidden features (after activation) in each weighted layer. Key (`str`) is the weighted layer name; value (`Tensor`) is the hidden feature tensor. This is used for continual learning algorithms that need hidden features for various purposes.
        """
        return MLP.forward(self, input, stage)  # call the MLP forward method


class HATMaskMLP(HATMaskBackbone, MLP):
    r"""HAT-masked multi-layer perceptron (MLP).

    [HAT (Hard Attention to the Task, 2018)](http://proceedings.mlr.press/v80/serra18a) is an architecture-based continual learning approach that uses learnable hard attention masks to select task-specific parameters.

    MLP is a dense network architecture with several fully connected layers, each followed by an activation function. The last layer connects to the CL output heads.

    The mask is applied to units (neurons) in each fully connected layer. The mask is generated from the neuron-wise task embedding and the gate function.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int],
        output_dim: int,
        gate: str,
        activation_layer: nn.Module | None = nn.ReLU,
        batch_normalization: str | None = None,
        bias: bool = True,
        dropout: float | None = None,
    ) -> None:
        r"""Construct and initialize the HAT-masked MLP backbone network with task embeddings. Note that batch normalization is incompatible with the HAT mechanism.

        **Args:**
        - **input_dim** (`int`): The input dimension. Any data need to be flattened before entering the MLP.
        - **hidden_dims** (`list[int]`): List of hidden layer dimensions. It can be an empty list, which means a single-layer MLP, and it can be as many layers as you want. Note that it doesn't include the last dimension, which we take as the output dimension.
        - **output_dim** (`int`): The output dimension that connects to CL output heads.
        - **gate** (`str`): The type of gate function turning real-valued task embeddings into attention masks; one of:
            - `sigmoid`: the sigmoid function.
        - **activation_layer** (`nn.Module` | `None`): Activation function of each layer (if not `None`). If `None`, this layer won't be used. Default `nn.ReLU`.
        - **batch_normalization** (`str` | `None`): How to use batch normalization after the fully connected layers; one of:
            - `None`: no batch normalization layers.
            - `shared`: use a single batch normalization layer for all tasks. Note that this can cause catastrophic forgetting.
            - `independent`: use independent batch normalization layers for each task.
        - **bias** (`bool`): Whether to use bias in the linear layer. Default `True`.
        - **dropout** (`float` | `None`): The probability for the dropout layer. If `None`, this layer won't be used. Default `None`.        - **kwargs**: Reserved for multiple inheritance.
        """
        super().__init__(
            output_dim=output_dim,
            gate=gate,
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            activation_layer=activation_layer,
            batch_normalization=(
                True if batch_normalization == "shared" or "independent" else False
            ),
            bias=bias,
            dropout=dropout,
        )

        # construct the task embedding for each weighted layer
        for layer_idx in range(self.num_fc_layers):
            full_layer_name = f"fc/{layer_idx}"
            layer_output_dim = (
                hidden_dims[layer_idx] if layer_idx != len(hidden_dims) else output_dim
            )
            self.task_embedding_t[full_layer_name] = nn.Embedding(
                num_embeddings=1, embedding_dim=layer_output_dim
            )

        self.batch_normalization: str | None = batch_normalization
        r"""The way to use batch normalization after the fully-connected layers. This overrides the `batch_normalization` argument in `MLP` class. """

        # construct the batch normalization layers if needed
        if self.batch_normalization == "independent":
            self.fc_bns: nn.ModuleDict = nn.ModuleDict()  # initially empty
            r"""Independent batch normalization layers are stored in a `ModuleDict`. Keys are task IDs and values are the corresponding batch normalization layers for the `nn.Linear`. We use `ModuleDict` rather than `dict` to ensure `LightningModule` can track these model parameters for purposes such as automatic device transfer and model summaries.
            
            Note that the task IDs must be string type in order to let `LightningModule` identify this part of the model."""
            self.original_fc_bn_state_dict: dict = deepcopy(self.fc_bn.state_dict())
            r"""The original batch normalization state dict as the source for creating new independent batch normalization layers. """

    def setup_task_id(self, task_id: int) -> None:
        r"""Set up task `task_id`. This must be done before the `forward()` method is called.

        **Args:**
        - **task_id** (`int`): The target task ID.
        """
        HATMaskBackbone.setup_task_id(self, task_id=task_id)

        if self.batch_normalization == "independent":
            if self.task_id not in self.fc_bns.keys():
                self.fc_bns[f"{self.task_id}"] = deepcopy(self.fc_bn)

    def get_bn(self, stage: str, test_task_id: int | None) -> nn.Module | None:
        r"""Get the batch normalization layer used in the `forward()` method for different stages.

        **Args:**
        - **stage** (`str`): The stage of the forward pass; one of:
            1. 'train': training stage.
            2. 'validation': validation stage.
            3. 'test': testing stage.
        - **test_task_id** (`int` | `None`): The test task ID. Applies only to the testing stage. For other stages, it is `None`.

        **Returns:**
        - **fc_bn** (`nn.Module` | `None`): The batch normalization module.
        """
        if self.batch_normalization == "independent" and stage == "test":
            return self.fc_bns[f"{test_task_id}"]
        else:
            return self.fc_bn

    def initialize_independent_bn(self) -> None:
        r"""Initialize the independent batch normalization layer for the current task. This is called when a new task is created. Applies only when `batch_normalization` is 'independent'."""

        if self.batch_normalization == "independent":
            self.fc_bn.load_state_dict(self.original_fc_bn_state_dict)

    def store_bn(self) -> None:
        r"""Store the batch normalization layer for the current task `self.task_id`. Applies only when `batch_normalization` is 'independent'."""

        if self.batch_normalization == "independent":
            self.fc_bns[f"{self.task_id}"] = deepcopy(self.fc_bn)

    def forward(
        self,
        input: Tensor,
        stage: str,
        s_max: float | None = None,
        batch_idx: int | None = None,
        num_batches: int | None = None,
        test_task_id: int | None = None,
    ) -> tuple[Tensor, dict[str, Tensor], dict[str, Tensor]]:
        r"""The forward pass for data from task `self.task_id`. Task-specific masks for `self.task_id` are applied to units (neurons) in each fully connected layer.

        **Args:**
        - **input** (`Tensor`): The input tensor from data.
        - **stage** (`str`): The stage of the forward pass; one of:
            1. 'train': training stage.
            2. 'validation': validation stage.
            3. 'test': testing stage.
        - **s_max** (`float`): The maximum scaling factor in the gate function. Doesn't apply to the testing stage. See Sec. 2.4 in the [HAT paper](http://proceedings.mlr.press/v80/serra18a).
        - **batch_idx** (`int` | `None`): The current batch index. Applies only to the training stage. For other stages, it is `None`.
        - **num_batches** (`int` | `None`): The total number of batches. Applies only to the training stage. For other stages, it is `None`.
        - **test_task_id** (`int` | `None`): The test task ID. Applies only to the testing stage. For other stages, it is `None`.

        **Returns:**
        - **output_feature** (`Tensor`): The output feature tensor to be passed into heads. This is the main target of backpropagation.
        - **mask** (`dict[str, Tensor]`): The mask for the current task. Keys (`str`) are the layer names and values (`Tensor`) are the mask tensors. The mask tensor has size (number of units, ).
        - **activations** (`dict[str, Tensor]`): The hidden features (after activation) in each weighted layer. Keys (`str`) are the weighted layer names and values (`Tensor`) are the hidden feature tensors. This is used for continual learning algorithms that need hidden features. Although the HAT algorithm does not need this, it is still provided for API consistency for other HAT-based algorithms that inherit this `forward()` method of the `HAT` class.
        """
        batch_size = input.size(0)
        activations = {}

        mask = self.get_mask(
            stage=stage,
            s_max=s_max,
            batch_idx=batch_idx,
            num_batches=num_batches,
            test_task_id=test_task_id,
        )
        if self.batch_normalization:
            fc_bn = self.get_bn(stage=stage, test_task_id=test_task_id)
        x = input.view(batch_size, -1)  # flatten before going through MLP

        for layer_idx, layer_name in enumerate(self.weighted_layer_names):

            x = self.fc[layer_idx](x)  # fully-connected layer first
            if self.batch_normalization:
                x = fc_bn[layer_idx](x)  # batch normalization second
            x = x * mask[f"fc/{layer_idx}"]  # apply the mask to the parameters second

            if self.activation:
                x = self.fc_activation[layer_idx](x)  # activation function third
            activations[layer_name] = x  # store the hidden feature
            if self.dropout:
                x = self.fc_dropout[layer_idx](x)  # dropout last

        output_feature = x

        return output_feature, mask, activations


class WSNMaskMLP(MLP, WSNMaskBackbone):
    r"""[WSN (Winning Subnetworks, 2022)](https://proceedings.mlr.press/v162/kang22b/kang22b.pdf) masked multi-layer perceptron (MLP).

    [WSN (Winning Subnetworks, 2022)](https://proceedings.mlr.press/v162/kang22b/kang22b.pdf) is an architecture-based continual learning algorithm. It trains learnable parameter-wise importance and selects the most important $c\%$ of the network parameters to be used for each task.

    MLP is a dense network architecture with several fully connected layers, each followed by an activation function. The last layer connects to the CL output heads.

    The mask is applied to the weights and biases in each fully connected layer. The mask is generated from the parameter-wise score and the gate function.
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
        r"""Construct and initialize the WSN-masked MLP backbone network with task embeddings.

        **Args:**
        - **input_dim** (`int`): The input dimension. Any data need to be flattened before entering the MLP.
        - **hidden_dims** (`list[int]`): List of hidden layer dimensions. It can be an empty list, which means a single-layer MLP, and it can be as many layers as you want. Note that it doesn't include the last dimension, which we take as the output dimension.
        - **output_dim** (`int`): The output dimension that connects to CL output heads.
        - **activation_layer** (`nn.Module` | `None`): Activation function of each layer (if not `None`). If `None`, this layer won't be used. Default `nn.ReLU`.
        - **bias** (`bool`): Whether to use bias in the linear layer. Default `True`.
        - **dropout** (`float` | `None`): The probability for the dropout layer. If `None`, this layer won't be used. Default `None`.        - **kwargs**: Reserved for multiple inheritance.
        """
        # init from both inherited classes
        super().__init__(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            output_dim=output_dim,
            activation_layer=activation_layer,
            batch_normalization=False,
            bias=bias,
            dropout=dropout,
        )

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
        r"""The forward pass for data from task `self.task_id`. Task-specific masks for `self.task_id` are applied to units (neurons) in each fully connected layer.

        **Args:**
        - **input** (`Tensor`): The input tensor from data.
        - **stage** (`str`): The stage of the forward pass; one of:
            1. 'train': training stage.
            2. 'validation': validation stage.
            3. 'test': testing stage.
        - **mask_percentage** (`float`): The percentage of parameters to be masked. The value should be between 0 and 1.
        - **test_mask** (`tuple[dict[str, Tensor], dict[str, Tensor]]` | `None`): The binary weight and bias masks used for testing. Applies only to the testing stage. For other stages, it is `None`.

        **Returns:**
        - **output_feature** (`Tensor`): The output feature tensor to be passed into heads. This is the main target of backpropagation.
        - **weight_mask** (`dict[str, Tensor]`): The weight mask for the current task. Keys (`str`) are the layer names and values (`Tensor`) are the mask tensors. The mask tensor has the same size (output features, input features) as the weight.
        - **bias_mask** (`dict[str, Tensor]`): The bias mask for the current task. Keys (`str`) are the layer names and values (`Tensor`) are the mask tensors. The mask tensor has the same size (output features, ) as the bias. If the layer doesn't have a bias, it is `None`.
        - **activations** (`dict[str, Tensor]`): The hidden features (after activation) in each weighted layer. Keys (`str`) are the weighted layer names and values (`Tensor`) are the hidden feature tensors. This is used for continual learning algorithms that need hidden features for various purposes.
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


# r"""
# The submodule in `backbones` for [NISPA (Neuro-Inspired Stability-Plasticity Adaptation)](https://proceedings.mlr.press/v162/gurbuz22a/gurbuz22a.pdf) masked MLP backbone network.
# """

# __all__ = ["NISPAMaskMLP"]

# from torch import Tensor, nn

# from clarena.backbones import MLP, NISPAMaskBackbone


# class NISPAMaskMLP(MLP, NISPAMaskBackbone):
#     r"""[NISPA (Neuro-Inspired Stability-Plasticity Adaptation)](https://proceedings.mlr.press/v162/gurbuz22a/gurbuz22a.pdf) masked multi-Layer perceptron (MLP).

#     [NISPA (Neuro-Inspired Stability-Plasticity Adaptation)](https://proceedings.mlr.press/v162/gurbuz22a/gurbuz22a.pdf) is an architecture-based continual learning algorithm. It

#     MLP is a dense network architecture, which has several fully-connected layers, each followed by an activation function. The last layer connects to the CL output heads.

#     Mask is applied to the weights and biases in each fully-connected layer. The mask is generated from the parameter-wise score and gate function.
#     """

#     def __init__(
#         self,
#         input_dim: int,
#         hidden_dims: list[int],
#         output_dim: int,
#         activation_layer: nn.Module | None = nn.ReLU,
#         bias: bool = True,
#         dropout: float | None = None,
#     ) -> None:
#         r"""Construct and initialize the WSN masked MLP backbone network with task embedding. Note that batch normalization is incompatible with WSN mechanism.

#         **Args:**
#         - **input_dim** (`int`): the input dimension. Any data need to be flattened before going in MLP.
#         - **hidden_dims** (`list[int]`): list of hidden layer dimensions. It can be empty list which means single-layer MLP, and it can be as many layers as you want. Note that it doesn't include the last dimension which we take as output dimension.
#         - **output_dim** (`int`): the output dimension which connects to CL output heads.
#         - **activation_layer** (`nn.Module` | `None`): activation function of each layer (if not `None`), if `None` this layer won't be used. Default `nn.ReLU`.
#         - **bias** (`bool`): whether to use bias in the linear layer. Default `True`.
#         - **dropout** (`float` | `None`): the probability for the dropout layer, if `None` this layer won't be used. Default `None`.
#         """
#         # init from both inherited classes
#         super().__init__(
#             input_dim=input_dim,
#             hidden_dims=hidden_dims,
#             output_dim=output_dim,
#             activation_layer=activation_layer,
#             batch_normalization=False,  # batch normalization is incompatible with HAT mechanism
#             bias=bias,
#             dropout=dropout,
#         )
#         self.register_wsn_mask_module_explicitly()  # register all `nn.Module`s for WSNMaskBackbone explicitly because the second `__init__()` wipes out them inited by the first `__init__()`

#         # construct the parameter score for each weighted layer
#         for layer_idx in range(self.num_fc_layers):
#             full_layer_name = f"fc/{layer_idx}"
#             layer_input_dim = (
#                 input_dim if layer_idx == 0 else hidden_dims[layer_idx - 1]
#             )
#             layer_output_dim = (
#                 hidden_dims[layer_idx] if layer_idx != len(hidden_dims) else output_dim
#             )
#             self.weight_score_t[full_layer_name] = nn.Embedding(
#                 num_embeddings=layer_output_dim,
#                 embedding_dim=layer_input_dim,
#             )
#             self.bias_score_t[full_layer_name] = nn.Embedding(
#                 num_embeddings=1,
#                 embedding_dim=layer_output_dim,
#             )

#     def forward(
#         self,
#         input: Tensor,
#         stage: str,
#         mask_percentage: float,
#         test_mask: tuple[dict[str, Tensor], dict[str, Tensor]] | None = None,
#     ) -> tuple[Tensor, dict[str, Tensor], dict[str, Tensor], dict[str, Tensor]]:
#         r"""The forward pass for data from task `self.task_id`. Task-specific mask for `self.task_id` are applied to the units which are neurons in each fully-connected layer.

#         **Args:**
#         - **input** (`Tensor`): The input tensor from data.
#         - **stage** (`str`): the stage of the forward pass; one of:
#             1. 'train': training stage.
#             2. 'validation': validation stage.
#             3. 'test': testing stage.
#         - **mask_percentage** (`float`): the percentage of parameters to be masked. The value should be between 0 and 1.
#         - **test_mask** (`tuple[dict[str, Tensor], dict[str, Tensor]]` | `None`): the binary weight and bias mask used for test. Applies only to testing stage. For other stages, it is default `None`.

#         **Returns:**
#         - **output_feature** (`Tensor`): the output feature tensor to be passed into heads. This is the main target of backpropagation.
#         - **weight_mask** (`dict[str, Tensor]`): the weight mask for the current task. Key (`str`) is layer name, value (`Tensor`) is the mask tensor. The mask tensor has same (output features, input features) as weight.
#         - **bias_mask** (`dict[str, Tensor]`): the bias mask for the current task. Key (`str`) is layer name, value (`Tensor`) is the mask tensor. The mask tensor has same (output features, ) as bias. If the layer doesn't have bias, it is `None`.
#         - **activations** (`dict[str, Tensor]`): the hidden features (after activation) in each weighted layer. Key (`str`) is the weighted layer name, value (`Tensor`) is the hidden feature tensor. This is used for the continual learning algorithms that need to use the hidden features for various purposes.
#         """
#         batch_size = input.size(0)
#         activations = {}

#         weight_mask, bias_mask = self.get_mask(
#             stage,
#             mask_percentage=mask_percentage,
#             test_mask=test_mask,
#         )

#         x = input.view(batch_size, -1)  # flatten before going through MLP

#         for layer_idx, layer_name in enumerate(self.weighted_layer_names):
#             weighted_layer = self.fc[layer_idx]
#             weight = weighted_layer.weight
#             bias = weighted_layer.bias

#             # mask the weight and bias
#             masked_weight = weight * weight_mask[f"fc/{layer_idx}"]
#             if bias is not None and bias_mask[f"fc/{layer_idx}"] is not None:
#                 masked_bias = bias * bias_mask[f"fc/{layer_idx}"]
#             else:
#                 masked_bias = None

#             # do the forward pass using the masked weight and bias. Do not modify the weight and bias data in the original layer object or it will lose the computation graph.
#             x = nn.functional.linear(x, masked_weight, masked_bias)

#             if self.activation:
#                 x = self.fc_activation[layer_idx](x)  # activation function third
#             activations[layer_name] = x  # store the hidden feature
#             if self.dropout:
#                 x = self.fc_dropout[layer_idx](x)  # dropout last

#         output_feature = x

#         return output_feature, weight_mask, bias_mask, activations
#         return output_feature, weight_mask, bias_mask, activations
