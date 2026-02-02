r"""
The submodule in `backbones` for the HAT masked MLP backbone network.
"""

__all__ = ["HATMaskMLP"]

import logging
from copy import deepcopy

from torch import Tensor, nn

from clarena.backbones import MLP, HATMaskBackbone

# always get logger for built-in logging in each module
pylogger = logging.getLogger(__name__)


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
        **kwargs,
    ) -> None:
        r"""Construct and initialize the HAT-masked MLP backbone network with task embeddings.

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
        - **dropout** (`float` | `None`): The probability for the dropout layer. If `None`, this layer won't be used. Default `None`.
        - **kwargs**: Reserved for multiple inheritance.
        """

        super().__init__(
            output_dim=output_dim,
            gate=gate,
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            activation_layer=activation_layer,
            batch_normalization=(
                True
                if batch_normalization == "shared"
                or batch_normalization == "independent"
                else False
            ),
            bias=bias,
            dropout=dropout,
            **kwargs,
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
            4. 'unlearning_test': unlearning testing stage.
        - **test_task_id** (`int` | `None`): The test task ID. Applies only to the testing stage. For other stages, it is `None`.

        **Returns:**
        - **fc_bn** (`nn.Module` | `None`): The batch normalization module.
        """
        if self.batch_normalization == "independent" and stage in (
            "test",
            "unlearning_test",
        ):
            test_task_id_for_bn = self.task_id if test_task_id is None else test_task_id
            return self.fc_bns[f"{test_task_id_for_bn}"]
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
            4. 'unlearning_test': unlearning testing stage.
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
