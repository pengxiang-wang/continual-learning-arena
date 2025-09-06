r"""
The submodule in `backbones` for backbone network base classes.
"""

__all__ = [
    "Backbone",
    "CLBackbone",
    "HATMaskBackbone",
    "WSNMaskBackbone",
]

import logging
import math
from typing import Callable

import torch
from torch import Tensor, nn
from typing_extensions import override

# always get logger for built-in logging in each module
pylogger = logging.getLogger(__name__)


class Backbone(nn.Module):
    r"""The base class for backbone networks."""

    def __init__(self, output_dim: int | None, **kwargs) -> None:
        r"""
        **Args:**
        - **output_dim** (`int` | `None`): The output dimension that connects to output heads. The `input_dim` of output heads is expected to be the same as this `output_dim`. In some cases, this class is used as a block in the backbone network that doesn't have an output dimension. In this case, it can be `None`.
        - **kwargs**: Reserved for multiple inheritance.
        """
        super().__init__()

        self.output_dim: int = output_dim
        r"""The output dimension of the backbone network."""

        self.weighted_layer_names: list[str] = []
        r"""The list of the weighted layer names in order (from input to output). A weighted layer has weights connecting to other weighted layers. They are the main part of neural networks. **It must be provided in subclasses.**
        
        The layer names must match the names of weighted layers defined in the backbone and include all of them. The names follow the `nn.Module` internal naming mechanism with `.` replaced with `/`. For example: 
        - If a layer is assigned to `self.conv1`, the name becomes `conv1`. 
        - If `nn.Sequential` is used, the name becomes the index of the layer in the sequence, such as `0`, `1`, etc. 
        - If a hierarchical structure is used, for example, a `nn.Module` is assigned to `self.block` which has `self.conv1`, the name becomes `block/conv1`. Note that it should have been `block.conv1` according to `nn.Module`'s rules, but we use '/' instead of '.' to avoid errors when using '.' as keys in a `ModuleDict`.
        """

    def get_layer_by_name(self, layer_name: str | None) -> nn.Module | None:
        r"""Get the layer by its name.

        **Args:**
        - **layer_name** (`str` | `None`): The layer name following the `nn.Module` internal naming mechanism with `.` replaced with `/`. If `None`, return `None`.

        **Returns:**
        - **layer** (`nn.Module` | `None`): The layer. If `layer_name` is `None`, return `None`.
        """
        if layer_name is None:
            return None

        for name, layer in self.named_modules():
            if name == layer_name.replace("/", "."):
                return layer

    def preceding_layer_name(self, layer_name: str | None) -> str | None:
        r"""Get the name of the preceding layer of the given layer from the stored `weighted_layer_names`.

        **Args:**
        - **layer_name** (`str` | `None`): The layer name following the `nn.Module` internal naming mechanism with `.` replaced with `/`. If `None`, return `None`.

        **Returns:**
        - **preceding_layer_name** (`str`): The name of the preceding layer. If the given layer is the first layer, return `None`.
        """
        if layer_name is None:
            return None

        if layer_name not in self.weighted_layer_names:
            raise ValueError(
                f"The layer name {layer_name} doesn't exist in weighted layer names."
            )

        weighted_layer_idx = self.weighted_layer_names.index(layer_name)
        if weighted_layer_idx == 0:
            return None
        preceding_layer_name = self.weighted_layer_names[weighted_layer_idx - 1]
        return preceding_layer_name

    def next_layer_name(self, layer_name: str) -> str:
        r"""Get the name of the next layer of the given layer from the stored `self.masked_layer_order`. If the given layer is the last layer of the BACKBONE, return `None`.

        **Args:**
        - **layer_name** (`str`): The name of the layer.

        **Returns:**
        - **next_layer_name** (`str`): The name of the next layer.

        **Raises:**
        - **ValueError**: If `layer_name` is not in the weighted layer order.
        """

        if layer_name not in self.weighted_layer_names:
            raise ValueError(f"The layer name {layer_name} doesn't exist.")

        weighted_layer_idx = self.weighted_layer_names.index(layer_name)
        if weighted_layer_idx == len(self.weighted_layer_names) - 1:
            return None
        next_layer_name = self.weighted_layer_names[weighted_layer_idx + 1]
        return next_layer_name

    @override  # since `nn.Module` uses it
    def forward(
        self,
        input: Tensor,
        stage: str,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        r"""The forward pass. **It must be implemented by subclasses.**

        **Args:**
        - **input** (`Tensor`): The input tensor from data.
        - **stage** (`str`): The stage of the forward pass; one of:
            1. 'train': training stage.
            2. 'validation': validation stage.
            3. 'test': testing stage.

        **Returns:**
        - **output_feature** (`Tensor`): The output feature tensor to be passed into heads. This is the main target of backpropagation.
        - **activations** (`dict[str, Tensor]`): The hidden features (after activation) in each weighted layer. Key (`str`) is the weighted layer name, value (`Tensor`) is the hidden feature tensor. This is used for certain algorithms that need to use the hidden features for various purposes.
        """


class CLBackbone(Backbone):
    r"""The base class of continual learning backbone networks."""

    def __init__(self, output_dim: int | None, **kwargs) -> None:
        r"""
        **Args:**
        - **output_dim** (`int` | `None`): The output dimension that connects to CL output heads. The `input_dim` of output heads is expected to be the same as this `output_dim`. In some cases, this class is used as a block in the backbone network that doesn't have an output dimension. In this case, it can be `None`.
        - **kwargs**: Reserved for multiple inheritance.
        """
        super().__init__(output_dim=output_dim, **kwargs)

        # task ID control
        self.task_id: int
        r"""Task ID counter indicating which task is being processed. Self updated during the task loop. Valid from 1 to the number of tasks in the CL dataset."""
        self.processed_task_ids: list[int] = []
        r"""Task IDs that have been processed."""

    def setup_task_id(self, task_id: int) -> None:
        r"""Set up task `task_id`. This must be done before the `forward()` method is called."""
        self.task_id = task_id
        self.processed_task_ids.append(task_id)

    @override  # since `nn.Module` uses it
    def forward(
        self,
        input: Tensor,
        stage: str,
        task_id: int | None = None,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        r"""The forward pass for data from task `task_id`. In some backbones, the forward pass might be different for different tasks. **It must be implemented by subclasses.**

        **Args:**
        - **input** (`Tensor`): The input tensor from data.
        - **stage** (`str`): The stage of the forward pass; one of:
            1. 'train': training stage.
            2. 'validation': validation stage.
            3. 'test': testing stage.
        - **task_id** (`int` | `None`): The task ID where the data are from. If the stage is 'train' or 'validation', it is usually the current task `self.task_id`. If the stage is 'test', it could be from any seen task. In TIL, the task IDs of test data are provided; thus this argument can be used. In CIL, they are not provided, so it is just a placeholder for API consistency and is not used. Best practice is not to provide this argument and leave it as the default value.

        **Returns:**
        - **output_feature** (`Tensor`): The output feature tensor to be passed into heads. This is the main target of backpropagation.
        - **activations** (`dict[str, Tensor]`): The hidden features (after activation) in each weighted layer. Key (`str`) is the weighted layer name, value (`Tensor`) is the hidden feature tensor. This is used for continual learning algorithms that need hidden features for various purposes.
        """


class HATMaskBackbone(CLBackbone):
    r"""The backbone network for HAT-based algorithms with learnable hard attention masks.

    HAT-based algorithms include:

    - [**HAT (Hard Attention to the Task, 2018)**](http://proceedings.mlr.press/v80/serra18a) is an architecture-based continual learning approach that uses learnable hard attention masks to select task-specific parameters.
    - [**AdaHAT (Adaptive Hard Attention to the Task, 2024)**](https://link.springer.com/chapter/10.1007/978-3-031-70352-2_9) is an architecture-based continual learning approach that improves HAT by introducing adaptive soft gradient clipping based on parameter importance and network sparsity.
    - **FG-AdaHAT** is an architecture-based continual learning approach that improves HAT by introducing fine-grained neuron-wise importance measures guiding the adaptive adjustment mechanism in AdaHAT.
    """

    def __init__(self, output_dim: int | None, gate: str, **kwargs) -> None:
        r"""
        **Args:**
        - **output_dim** (`int`): The output dimension that connects to CL output heads. The `input_dim` of output heads is expected to be the same as this `output_dim`. In some cases, this class is used as a block in the backbone network that doesn't have an output dimension. In this case, it can be `None`.
        - **gate** (`str`): The type of gate function turning the real value task embeddings into attention masks; one of:
            - `sigmoid`: the sigmoid function.
            - `tanh`: the hyperbolic tangent function.
        - **kwargs**: Reserved for multiple inheritance.
        """
        super().__init__(output_dim=output_dim, **kwargs)

        self.gate: str = gate
        r"""The type of gate function."""
        self.gate_fn: Callable
        r"""The gate function mapping the real value task embeddings into attention masks."""

        if gate == "sigmoid":
            self.gate_fn = nn.Sigmoid()

        self.task_embedding_t: nn.ModuleDict = nn.ModuleDict()
        r"""The task embedding for the current task `self.task_id`. Keys are layer names and values are the task embedding `nn.Embedding` for the layer. Each task embedding has size (1, number of units).
        
        We use `ModuleDict` rather than `dict` to ensure `LightningModule` properly registers these model parameters for purposes such as automatic device transfer and model summaries.
        
        We use `nn.Embedding` rather than `nn.Parameter` to store the task embedding for each layer, which is a type of `nn.Module` and can be accepted by `nn.ModuleDict`. (`nn.Parameter` cannot be accepted by `nn.ModuleDict`.)
        
        **This must be defined to cover each weighted layer (as listed in `weighted_layer_names`) in the backbone network.** Otherwise, the uncovered parts will keep being updated for all tasks and become the source of catastrophic forgetting.
        """

        self.masks: dict[int, dict[str, Tensor]] = {}
        r"""The binary attention mask of each previous task gated from the task embedding. Keys are task IDs and values are the corresponding mask. Each mask is a dict where keys are layer names and values are the binary mask tensor for the layer. The mask tensor has size (number of units, ). """

        HATMaskBackbone.sanity_check(self)

    def initialize_task_embedding(self, mode: str) -> None:
        r"""Initialize the task embedding for the current task `self.task_id`.

        **Args:**
        - **mode** (`str`): The initialization mode for task embeddings; one of:
            1. 'N01' (default): standard normal distribution $N(0, 1)$.
            2. 'U-11': uniform distribution $U(-1, 1)$.
            3. 'U01': uniform distribution $U(0, 1)$.
            4. 'U-10': uniform distribution $U(-1, 0)$.
            5. 'last': inherit task embeddings from the last task.
        """
        for te in self.task_embedding_t.values():
            if mode == "N01":
                nn.init.normal_(te.weight, 0, 1)
            elif mode == "U-11":
                nn.init.uniform_(te.weight, -1, 1)
            elif mode == "U01":
                nn.init.uniform_(te.weight, 0, 1)
            elif mode == "U-10":
                nn.init.uniform_(te.weight, -1, 0)
            elif mode == "last":
                pass

    def sanity_check(self) -> None:
        r"""Sanity check."""

        if self.gate not in ["sigmoid"]:
            raise ValueError("The gate should be one of: 'sigmoid'.")

    def get_mask(
        self,
        stage: str,
        s_max: float | None = None,
        batch_idx: int | None = None,
        num_batches: int | None = None,
        test_task_id: int | None = None,
    ) -> dict[str, Tensor]:
        r"""Get the hard attention mask used in the `forward()` method for different stages.

        **Args:**
        - **stage** (`str`): The stage when applying the conversion; one of:
            1. 'train': training stage. Get the mask from the current task embedding through the gate function, scaled by an annealed scalar. See Sec. 2.4 in the [HAT paper](http://proceedings.mlr.press/v80/serra18a).
            2. 'validation': validation stage. Get the mask from the current task embedding through the gate function, scaled by `s_max`, where large scaling makes masks nearly binary. (Note that in this stage, the binary mask hasn't been stored yet, as training is not over.)
            3. 'test': testing stage. Apply the test mask directly from the stored masks using `test_task_id`.
        - **s_max** (`float`): The maximum scaling factor in the gate function. Doesn't apply to the testing stage. See Sec. 2.4 in the [HAT paper](http://proceedings.mlr.press/v80/serra18a).
        - **batch_idx** (`int` | `None`): The current batch index. Applies only to the training stage. For other stages, it is `None`.
        - **num_batches** (`int` | `None`): The total number of batches. Applies only to the training stage. For other stages, it is `None`.
        - **test_task_id** (`int` | `None`): The test task ID. Applies only to the testing stage. For other stages, it is `None`.

        **Returns:**
        - **mask** (`dict[str, Tensor]`): The hard attention (with values 0 or 1) mask. Keys (`str`) are the layer names and values (`Tensor`) are the mask tensors. The mask tensor has size (number of units, ).
        """

        # sanity check
        if stage == "train" and (
            s_max is None or batch_idx is None or num_batches is None
        ):
            raise ValueError(
                "The `s_max`, `batch_idx` and `batch_num` should be provided at training stage, instead of the default value `None`."
            )
        if stage == "validation" and (s_max is None):
            raise ValueError(
                "The `s_max` should be provided at validation stage, instead of the default value `None`."
            )
        if stage == "test" and (test_task_id is None):
            raise ValueError(
                "The `task_mask` should be provided at testing stage, instead of the default value `None`."
            )

        mask = {}
        if stage == "train":
            for layer_name in self.weighted_layer_names:
                anneal_scalar = 1 / s_max + (s_max - 1 / s_max) * (batch_idx - 1) / (
                    num_batches - 1
                )  # see Eq. (3) in Sec. 2.4 "Hard Attention Training" in the [HAT paper](http://proceedings.mlr.press/v80/serra18a).
                mask[layer_name] = self.gate_fn(
                    self.task_embedding_t[layer_name].weight * anneal_scalar
                ).squeeze()
        elif stage == "validation":
            for layer_name in self.weighted_layer_names:
                mask[layer_name] = self.gate_fn(
                    self.task_embedding_t[layer_name].weight * s_max
                ).squeeze()
        elif stage == "test":
            mask = self.masks[test_task_id]

        return mask

    def te_to_binary_mask(self) -> dict[str, Tensor]:
        r"""Convert the current task embedding into a binary mask.

        This method is used before the testing stage to convert the task embedding into a binary mask for each layer. The binary mask is used to select parameters for the current task.

        **Returns:**
        - **mask_t** (`dict[str, Tensor]`): The binary mask for the current task. Keys (`str`) are the layer names and values (`Tensor`) are the mask tensors. The mask tensor has size (number of units, ).
        """
        # get the mask for the current task
        mask_t = {
            layer_name: (self.task_embedding_t[layer_name].weight > 0)
            .float()
            .squeeze()
            .detach()
            for layer_name in self.weighted_layer_names
        }

        return mask_t

    def store_mask(self) -> None:
        r"""Store the mask for the current task `self.task_id`."""
        mask_t = self.te_to_binary_mask()
        self.masks[self.task_id] = mask_t

        return mask_t

    def get_layer_measure_parameter_wise(
        self,
        neuron_wise_measure: dict[str, Tensor],
        layer_name: str,
        aggregation_mode: str,
    ) -> Tensor:
        r"""Get the parameter-wise measure on the parameters right before the given layer.

        It is calculated from the given neuron-wise measure. It aggregates two feature-sized vectors (corresponding to the given layer and the preceding layer) into a weight-wise matrix (corresponding to the weights in between) and a bias-wise vector (corresponding to the bias of the given layer), using the given aggregation method. For example, given two feature-sized measures $m_{l,i}$ and $m_{l-1,j}$ and 'min' aggregation, the parameter-wise measure is $\min \left(a_{l,i}, a_{l-1,j}\right)$, a matrix with respect to $i, j$.

        Note that if the given layer is the first layer with no preceding layer, we will get the parameter-wise measure directly by broadcasting from the neuron-wise measure of the given layer.

        This method is used to calculate parameter-wise measures in various HAT-based algorithms:

        - **HAT**: the parameter-wise measure is the binary mask for previous tasks from the neuron-wise cumulative mask of previous tasks `cumulative_mask_for_previous_tasks`, which is $\text{Agg} \left(a_{l,i}^{<t}, a_{l-1,j}^{<t}\right)$ in Eq. (2) in the [HAT paper](http://proceedings.mlr.press/v80/serra18a).
        - **AdaHAT**: the parameter-wise measure is the parameter importance for previous tasks from the neuron-wise summative mask of previous tasks `summative_mask_for_previous_tasks`, which is $\text{Agg} \left(m_{l,i}^{<t,\text{sum}}, m_{l-1,j}^{<t,\text{sum}}\right)$ in Eq. (9) in the [AdaHAT paper](https://link.springer.com/chapter/10.1007/978-3-031-70352-2_9).
        - **FG-AdaHAT**: the parameter-wise measure is the parameter importance for previous tasks from the neuron-wise importance of previous tasks `summative_importance_for_previous_tasks`, which is $\text{Agg} \left(I_{l,i}^{<t}, I_{l-1,j}^{<t}\right)$ in Eq. (2) in the FG-AdaHAT paper.

        **Args:**
        - **neuron_wise_measure** (`dict[str, Tensor]`): The neuron-wise measure. Keys are layer names and values are the neuron-wise measure tensor. The tensor has size (number of units, ).
        - **layer_name** (`str`): The name of the given layer.
        - **aggregation_mode** (`str`): The aggregation mode mapping two feature-wise measures into a weight-wise matrix; one of:
            - 'min': takes the minimum of the two connected unit measures.
            - 'max': takes the maximum of the two connected unit measures.
            - 'mean': takes the mean of the two connected unit measures.

        **Returns:**
        - **weight_measure** (`Tensor`): The weight measure matrix, the same size as the corresponding weights.
        - **bias_measure** (`Tensor`): The bias measure vector, the same size as the corresponding bias.
        """

        # initialize the aggregation function
        if aggregation_mode == "min":
            aggregation_func = torch.min
        elif aggregation_mode == "max":
            aggregation_func = torch.max
        elif aggregation_mode == "mean":
            aggregation_func = torch.mean
        else:
            raise ValueError(
                f"The aggregation method {aggregation_mode} is not supported."
            )

        # get the preceding layer
        preceding_layer_name = self.preceding_layer_name(layer_name)

        # get weight size for expanding the measures
        layer = self.get_layer_by_name(layer_name)
        weight_size = layer.weight.size()

        # construct the weight-wise measure
        layer_measure = neuron_wise_measure[layer_name]
        layer_measure_broadcast_size = (-1, 1) + tuple(
            1 for _ in range(len(weight_size) - 2)
        )  # since the size of mask tensor is (number of units, ), we extend it to (number of units, 1) and expand it to the weight size. The weight size has 2 dimensions in fully connected layers and 4 dimensions in convolutional layers

        layer_measure_broadcasted = layer_measure.view(
            *layer_measure_broadcast_size
        ).expand(
            weight_size,
        )  # expand the given layer mask to the weight size and broadcast

        if (
            preceding_layer_name
        ):  # if the layer is not the first layer, where the preceding layer exists

            preceding_layer_measure_broadcast_size = (1, -1) + tuple(
                1 for _ in range(len(weight_size) - 2)
            )  # since the size of mask tensor is (number of units, ), we extend it to (1, number of units) and expand it to the weight size. The weight size has 2 dimensions in fully connected layers and 4 dimensions in convolutional layers
            preceding_layer_measure = neuron_wise_measure[preceding_layer_name]
            preceding_layer_measure_broadcasted = preceding_layer_measure.view(
                *preceding_layer_measure_broadcast_size
            ).expand(
                weight_size
            )  # expand the preceding layer mask to the weight size and broadcast
            weight_measure = aggregation_func(
                layer_measure_broadcasted, preceding_layer_measure_broadcasted
            )  # get the minimum of the two mask vectors, from expanded
        else:  # if the layer is the first layer
            weight_measure = layer_measure_broadcasted

        # construct the bias-wise measure
        bias_measure = layer_measure

        return weight_measure, bias_measure

    @override
    def forward(
        self,
        input: Tensor,
        stage: str,
        s_max: float | None = None,
        batch_idx: int | None = None,
        num_batches: int | None = None,
        test_task_id: int | None = None,
    ) -> tuple[Tensor, dict[str, Tensor], dict[str, Tensor]]:
        r"""The forward pass for data from task `self.task_id`. Task-specific masks for `self.task_id` are applied to the units in each layer.

        **Args:**
        - **input** (`Tensor`): The input tensor from data.
        - **stage** (`str`): The stage of the forward pass; one of:
            1. 'train': training stage.
            2. 'validation': validation stage.
            3. 'test': testing stage.
        - **s_max** (`float`): The maximum scaling factor in the gate function. See Sec. 2.4 in the [HAT paper](http://proceedings.mlr.press/v80/serra18a).
        - **batch_idx** (`int` | `None`): The current batch index. Applies only to the training stage. For other stages, it is `None`.
        - **num_batches** (`int` | `None`): The total number of batches. Applies only to the training stage. For other stages, it is `None`.
        - **test_task_id** (`int` | `None`): The test task ID. Applies only to the testing stage. For other stages, it is `None`.

        **Returns:**
        - **output_feature** (`Tensor`): The output feature tensor to be passed into heads. This is the main target of backpropagation.
        - **mask** (`dict[str, Tensor]`): The mask for the current task. Keys (`str`) are layer names and values (`Tensor`) are the mask tensors. The mask tensor has size (number of units, ).
        - **activations** (`dict[str, Tensor]`): The hidden features (after activation) in each weighted layer. Keys (`str`) are the weighted layer names and values (`Tensor`) are the hidden feature tensors. This is used for continual learning algorithms that need hidden features. Although the HAT algorithm does not need this, it is still provided for API consistency for other HAT-based algorithms that inherit this `forward()` method of the `HAT` class.
        """


class WSNMaskBackbone(CLBackbone):
    r"""The backbone network for the WSN algorithm with learnable parameter masks.

    [WSN (Winning Subnetworks, 2022)](https://proceedings.mlr.press/v162/kang22b/kang22b.pdf) is an architecture-based continual learning algorithm. It trains learnable parameter-wise scores and selects the most scored $c\%$ of the network parameters to be used for each task.
    """

    def __init__(self, output_dim: int | None, **kwargs) -> None:
        r"""
        **Args:**
        - **output_dim** (`int`): The output dimension that connects to CL output heads. The `input_dim` of output heads is expected to be the same as this `output_dim`. In some cases, this class is used as a block in the backbone network that doesn't have an output dimension. In this case, it can be `None`.
        - **kwargs**: Reserved for multiple inheritance.
        """
        super().__init__(output_dim=output_dim, **kwargs)

        self.gate_fn: torch.autograd.Function = PercentileLayerParameterMaskingByScore
        r"""The gate function mapping the real-value parameter score into binary parameter masks. It is a custom autograd function that applies percentile parameter masking by score."""

        self.weight_score_t: nn.ModuleDict = nn.ModuleDict()
        r"""The weight score for the current task `self.task_id`. Keys are the layer names and values are the task embedding `nn.Embedding` for the layer. Each task embedding has the same size (output features, input features) as the weight.
        
        We use `ModuleDict` rather than `dict` to ensure `LightningModule` properly registers these model parameters for purposes such as automatic device transfer and model summaries.
        
        We use `nn.Embedding` rather than `nn.Parameter` to store the task embedding for each layer, which is a type of `nn.Module` and can be accepted by `nn.ModuleDict`. (`nn.Parameter` cannot be accepted by `nn.ModuleDict`.)
        
        **This must be defined to cover each weighted layer (as listed in `weighted_layer_names`) in the backbone network.** Otherwise, the uncovered parts will keep being updated for all tasks and become the source of catastrophic forgetting.
        """

        self.bias_score_t: nn.ModuleDict = nn.ModuleDict()
        r"""The bias score for the current task `self.task_id`. Keys are the layer names and values are the task embedding `nn.Embedding` for the layer. Each task embedding has the same size (1, output features) as the bias. If the layer doesn't have a bias, it is `None`.
        
        We use `ModuleDict` rather than `dict` to ensure `LightningModule` properly registers these model parameters for purposes such as automatic device transfer and model summaries.
        
        We use `nn.Embedding` rather than `nn.Parameter` to store the task embedding for each layer, which is a type of `nn.Module` and can be accepted by `nn.ModuleDict`. (`nn.Parameter` cannot be accepted by `nn.ModuleDict`.)
        
        **This must be defined to cover each weighted layer (as listed in `weighted_layer_names`) in the backbone network.** Otherwise, the uncovered parts will keep being updated for all tasks and become the source of catastrophic forgetting.
        """

        WSNMaskBackbone.sanity_check(self)

    def sanity_check(self) -> None:
        r"""Sanity check."""

    def initialize_parameter_score(self, mode: str) -> None:
        r"""Initialize the parameter score for the current task.

        **Args:**
        - **mode** (`str`): The initialization mode for parameter scores; one of:
            1. 'default': the default initialization mode in the original WSN code.
            2. 'N01': standard normal distribution $N(0, 1)$.
            3. 'U01': uniform distribution $U(0, 1)$.
        """

        for layer_name, weight_score in self.weight_score_t.items():
            if mode == "default":
                # Kaiming Uniform Initialization for weight score
                nn.init.kaiming_uniform_(weight_score.weight, a=math.sqrt(5))

                for layer_name, bias_score in self.bias_score_t.items():
                    if bias_score is not None:
                        # For bias, follow the standard bias initialization using fan_in
                        weight_score = self.weight_score_t[layer_name]
                        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(
                            weight_score.weight
                        )
                        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                        nn.init.uniform_(bias_score.weight, -bound, bound)
            elif mode == "N01":
                nn.init.normal_(weight_score.weight, 0, 1)
                for layer_name, bias_score in self.bias_score_t.items():
                    if bias_score is not None:
                        nn.init.normal_(bias_score.weight, 0, 1)
            elif mode == "U01":
                nn.init.uniform_(weight_score.weight, 0, 1)
                for layer_name, bias_score in self.bias_score_t.items():
                    if bias_score is not None:
                        nn.init.uniform_(bias_score.weight, 0, 1)

    def get_mask(
        self,
        stage: str,
        mask_percentage: float,
        test_mask: tuple[dict[str, Tensor], dict[str, Tensor]] | None = None,
    ) -> dict[str, Tensor]:
        r"""Get the binary parameter mask used in the `forward()` method for different stages.

        **Args:**
        - **stage** (`str`): The stage when applying the conversion; one of:
            1. 'train': training stage. Get the mask from the parameter score of the current task through the gate function that masks the top $c\%$ largest scored parameters. See Sec. 3.1 "Winning Subnetworks" in the [WSN paper](https://proceedings.mlr.press/v162/kang22b/kang22b.pdf).
            2. 'validation': validation stage. Same as 'train'. (Note that in this stage, the binary mask hasn't been stored yet, as training is not over.)
            3. 'test': testing stage. Apply the test mask directly from the argument `test_mask`.
        - **test_mask** (`tuple[dict[str, Tensor], dict[str, Tensor]]` | `None`): The binary weight and bias masks used for testing. Applies only to the testing stage. For other stages, it is `None`.

        **Returns:**
        - **weight_mask** (`dict[str, Tensor]`): The binary mask on weights. Key (`str`) are the layer names and values (`Tensor`) are the mask tensors. The mask tensor has the same size (output features, input features) as the weight.
        - **bias_mask** (`dict[str, Tensor]`): The binary mask on biases. Keys (`str`) are the layer names and values (`Tensor`) are the mask tensors. The mask tensor has the same size (output features, ) as the bias. If the layer doesn't have a bias, it is `None`.
        """
        weight_mask = {}
        bias_mask = {}
        if stage == "train" or stage == "validation":
            for layer_name in self.weighted_layer_names:
                weight_mask[layer_name] = self.gate_fn.apply(
                    self.weight_score_t[layer_name].weight, mask_percentage
                )
                if self.bias_score_t[layer_name] is not None:
                    bias_mask[layer_name] = self.gate_fn.apply(
                        self.bias_score_t[layer_name].weight.squeeze(
                            0
                        ),  # from (1, output_dim) to (output_dim, )
                        mask_percentage,
                    )
                else:
                    bias_mask[layer_name] = None
        elif stage == "test":
            weight_mask, bias_mask = test_mask

        return weight_mask, bias_mask

    @override
    def forward(
        self,
        input: Tensor,
        stage: str,
        mask_percentage: float,
        test_mask: tuple[dict[str, Tensor], dict[str, Tensor]] | None = None,
    ) -> tuple[Tensor, dict[str, Tensor], dict[str, Tensor], dict[str, Tensor]]:
        r"""The forward pass for data from task `self.task_id`. Task-specific mask for `self.task_id` are applied to the units in each layer.

        **Args:**
        - **input** (`Tensor`): The input tensor from data.
        - **stage** (`str`): The stage of the forward pass; one of:
            1. 'train': training stage.
            2. 'validation': validation stage.
            3. 'test': testing stage.
        - **mask_percentage** (`float`): The percentage of parameters to be masked. The value should be between 0 and 1.
        - **test_mask** (`tuple[dict[str, Tensor], dict[str, Tensor]]` | `None`): The binary weight and bias mask used for test. Applies only to the testing stage. For other stages, it is `None`.

        **Returns:**
        - **output_feature** (`Tensor`): The output feature tensor to be passed into heads. This is the main target of backpropagation.
        - **weight_mask** (`dict[str, Tensor]`): The weight mask for the current task. Key (`str`) are layer names and values (`Tensor`) are the mask tensors. The mask tensor has same (output features, input features) as the weight.
        - **bias_mask** (`dict[str, Tensor]`): The bias mask for the current task. Keys (`str`) are layer names and values (`Tensor`) are the mask tensors. The mask tensor has same (output features, ) as the bias. If the layer doesn't have a bias, it is `None`.
        - **activations** (`dict[str, Tensor]`): The hidden features (after activation) in each weighted layer. Key (`str`) are the weighted layer names and values (`Tensor`) are the hidden feature tensors. This is used for the continual learning algorithms that need to use the hidden features for various purposes.
        """


class PercentileLayerParameterMaskingByScore(torch.autograd.Function):
    r"""The custom autograd function that gets the parameter masks of a layer where the top $c\%$ largest scored parameters are masked. This is used in the [WSN (Winning Subnetworks, 2022)](https://proceedings.mlr.press/v162/kang22b/kang22b.pdf) algorithm."""

    @staticmethod
    def forward(ctx, score: Tensor, percentage: float) -> Tensor:
        r"""The forward pass of the custom autograd function.

        **Args:**
        - **ctx**: The context object to save the input for the backward pass. This must be included in the forward pass.
        - **score** (`Tensor`): The parameter score of the layer. It has the same size as the parameter.
        - **percentage** (`float`): The percentage of parameters to be masked. The value should be between 0 and 1.

        **Returns:**
        - **parameter_mask** (`Tensor`): The binary mask. The size is the same as the parameter. The value is 1 for masked parameters and 0 for unmasked parameters.
        """
        # percentile of the scores as the threshold
        threshold = torch.quantile(
            score, 1 - percentage
        )  # keep top c% largest scored parameters

        parameter_size = score.size()
        zeros = torch.zeros(parameter_size).to(score.device)
        ones = torch.ones(parameter_size).to(score.device)

        # the mask is 1 for the parameters with score >= threshold and 0 for the parameters with score < threshold
        return torch.where(score < threshold, zeros, ones)

    @staticmethod
    def backward(ctx, grad_output: Tensor) -> tuple[Tensor, None]:
        r"""The backward pass of the custom autograd function. It applies STE (Straight-through Estimator) to address the problem that this filter layer always has a gradient value of 0; therefore, updating the weight scores with its loss gradient is not possible. See Eq. (5) in Sec. 3.2 "Optimization Procedure for Winning SubNetworks" in the [WSN paper](https://proceedings.mlr.press/v162/kang22b/kang22b.pdf).

        **Args:**
        - **ctx**: The context object to save the input for the backward pass. This must be included in the forward pass.
        - **grad_output** (`Tensor`): The gradient of the output from the forward pass.

        **Returns:**
        - **grad_score_input** (`Tensor`): The gradient of the input (the score).
        - **grad_percentage_input** (`None`): The gradient of the input (percentage). It is `None` because it is not used in the backward pass.
        """


# class NISPAMaskBackbone(CLBackbone):
#     r"""The backbone network for the NISPA algorithm with neuron masks.

#     [NISPA (Neuro-Inspired Stability-Plasticity Adaptation)](https://proceedings.mlr.press/v162/gurbuz22a/gurbuz22a.pdf) is an architecture-based continual learning algorithm.
#     """

#     def __init__(self, output_dim: int | None, **kwargs) -> None:
#         r"""Initialize the NISPA mask backbone network with masks.

#         **Args:**
#         - **output_dim** (`int`): The output dimension that connects to CL output heads. The `input_dim` of output heads is expected to be the same as this `output_dim`. In some cases, this class is used as a block in the backbone network that doesn't have an output dimension. In this case, it can be `None`.
#         """
#         super().__init__(output_dim=output_dim, **kwargs)

#         self.weight_mask_t: dict[str, Tensor] = {}
#         r"""Store the weight mask for each layer. Key (`str`) is layer name, value (`Tensor`) is the mask tensor. The mask tensor has the same size (output features, input features) as weight."""
#         self.frozen_weight_mask_t: dict[str, Tensor] = {}
#         r"""Store the frozen weight mask for each layer. Key (`str`) is layer name, value (`Tensor`) is the mask tensor. The mask tensor has the same size (output features, input features) as weight."""
#         self.bias_mask_t: dict[str, Tensor] = {}
#         r"""Store the bias mask for each layer. Key (`str`) is layer name, value (`Tensor`) is the mask tensor. The mask tensor has the same size (output features, ) as bias. If the layer doesn't have bias, it is `None`."""
#         self.frozen_bias_mask_t: dict[str, Tensor] = {}
#         r"""Store the frozen bias mask for each layer. Key (`str`) is layer name, value (`Tensor`) is the mask tensor. The mask tensor has the same size (output features, ) as bias. If the layer doesn't have bias, it is `None`."""

#         NISPAMaskBackbone.sanity_check(self)

#     def sanity_check(self) -> None:
#         r"""Sanity check."""
#         pass

#     def initialize_parameter_mask(self) -> None:
#         r"""Initialize the parameter masks as zeros."""
#         for layer_name in self.weighted_layer_names:
#             layer = self.get_layer_by_name(layer_name)  # get the layer by its name

#             self.weight_mask_t[layer_name] = torch.zeros_like(layer.weight).to(
#                 self.device
#             )
#             self.frozen_weight_mask_t[layer_name] = torch.zeros_like(layer.weight).to(
#                 self.device
#             )

#             if layer.bias is not None:
#                 self.bias_mask_t[layer_name] = torch.zeros_like(layer.bias).to(
#                     self.device
#                 )
#                 self.frozen_bias_mask_t[layer_name] = torch.zeros_like(layer.bias).to(
#                     self.device
#                 )

#     @override
#     def forward(
#         self,
#         input: Tensor,
#         stage: str,
#     ) -> tuple[Tensor, dict[str, Tensor], dict[str, Tensor], dict[str, Tensor]]:
#         r"""The forward pass for data from task `self.task_id`. The parameter mask is applied to the parameters in each layer.

#         **Args:**
#         - **input** (`Tensor`): The input tensor from data.
#         - **stage** (`str`): The stage of the forward pass; one of:
#             1. 'train': training stage.
#             2. 'validation': validation stage.
#             3. 'test': testing stage.

#         **Returns:**
#         - **output_feature** (`Tensor`): The output feature tensor to be passed into heads. This is the main target of backpropagation.
#         - **weight_mask** (`dict[str, Tensor]`): The weight mask. Key (`str`) is layer name; value (`Tensor`) is the mask tensor. The mask tensor has the same size (output features, input features) as the weight.
#         - **bias_mask** (`dict[str, Tensor]`): The bias mask. Key (`str`) is layer name; value (`Tensor`) is the mask tensor. The mask tensor has the same size (output features, ) as the bias. If the layer doesn't have a bias, it is `None`.
#         - **activations** (`dict[str, Tensor]`): The hidden features (after activation) in each weighted layer. Key (`str`) is the weighted layer name; value (`Tensor`) is the hidden feature tensor. This is used for continual learning algorithms that need hidden features for various purposes.
#         """
