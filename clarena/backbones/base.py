r"""
The submodule in `backbones` for CL backbone network bases.
"""

__all__ = ["CLBackbone", "HATMaskBackbone"]

import logging

import torch
from torch import Tensor, nn
from typing_extensions import override

# always get logger for built-in logging in each module
pylogger = logging.getLogger(__name__)


class CLBackbone(nn.Module):
    r"""The base class of continual learning backbone networks, inherited from `nn.Module`."""

    def __init__(self, output_dim: int | None) -> None:
        r"""Initialise the CL backbone network.

        **Args:**
        - **output_dim** (`int` | `None`): The output dimension which connects to CL output heads. The `input_dim` of output heads are expected to be the same as this `output_dim`. In some cases, this class is used for a block in the backbone network, which doesn't have the output dimension. In this case, it can be `None`.
        """
        nn.Module.__init__(self)

        self.output_dim: int = output_dim
        r"""Store the output dimension of the backbone network."""

        self.weighted_layer_names: list[str] = []
        r"""Maintain a list of the weighted layer names. Weighted layer has weights connecting to other weighted layer. They are the main part of neural networks. **It must be provided in subclasses.**
        
        The names are following the `nn.Module` internal naming mechanism. For example, if the a layer is assigned to `self.conv1`, the name becomes `conv1`. If the `nn.Sequential` is used, the name becomes the index of the layer in the sequence, such as `0`, `1`, etc. If hierarchical structure is used, for example, a `nn.Module` is assigned to `self.block` which has `self.conv1`, the name becomes `block/conv1`. Note that it should be `block.conv1` according to `nn.Module` internal mechanism, but we use '/' instead of '.' to avoid the error of using '.' in the key of `ModuleDict`.
        
        In HAT architecture, it's also the layer names with task embedding masking in the order of forward pass. HAT gives task embedding to every possible weighted layer. 
        """

        self.task_id: int
        r"""Task ID counter indicating which task is being processed. Self updated during the task loop."""

    def setup_task_id(self, task_id: int) -> None:
        r"""Set up which task's dataset the CL experiment is on. This must be done before `forward()` method is called.

        **Args:**
        - **task_id** (`int`): the target task ID.
        """
        self.task_id = task_id

    def get_layer_by_name(self, layer_name: str) -> nn.Module:
        r"""Get the layer by its name.

        **Args:**
        - **layer_name** (`str`): the name of the layer. Note that the name is the name substituting the '.' with '/', like `block/conv1`, rather than `block.conv1`.

        **Returns:**
        - **layer** (`nn.Module`): the layer.
        """
        for name, layer in self.named_modules():
            if name == layer_name.replace("/", "."):
                return layer

    def preceding_layer_name(self, layer_name: str) -> str:
        r"""Get the name of the preceding layer of the given layer from the stored `self.masked_layer_order`. If the given layer is the first layer, return `None`.

        **Args:**
        - **layer_name** (`str`): the name of the layer.

        **Returns:**
        - **preceding_layer_name** (`str`): the name of the preceding layer.

        **Raises:**
        - **ValueError**: if `layer_name` is not in the weighted layer order.
        """

        if layer_name not in self.weighted_layer_names:
            raise ValueError(f"The layer name {layer_name} doesn't exist.")

        weighted_layer_idx = self.weighted_layer_names.index(layer_name)
        if weighted_layer_idx == 0:
            return None
        return self.weighted_layer_names[weighted_layer_idx - 1]

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
        - **stage** (`str`): the stage of the forward pass, should be one of the following:
            1. 'train': training stage.
            2. 'validation': validation stage.
            3. 'test': testing stage.
        - **task_id** (`int` | `None`): the task ID where the data are from. If stage is 'train' or 'validation', it is usually from the current task `self.task_id`. If stage is 'test', it could be from any seen task. In TIL, the task IDs of test data are provided thus this argument can be used. In CIL, they are not provided, so it is just a placeholder for API consistence but never used, and best practices are not to provide this argument and leave it as the default value.

        **Returns:**
        - **output_feature** (`Tensor`): the output feature tensor to be passed into heads. This is the main target of backpropagation.
        - **hidden_features** (`dict[str, Tensor]`): the hidden features (after activation) in each weighted layer. Key (`str`) is the weighted layer name, value (`Tensor`) is the hidden feature tensor. This is used for the continual learning algorithms that need to use the hidden features for various purposes.
        """


class HATMaskBackbone(CLBackbone):
    r"""The backbone network for HAT-based algorithms with learnable hard attention masks.

    HAT-based algorithms:

    - [**HAT (Hard Attention to the Task, 2018)**](http://proceedings.mlr.press/v80/serra18a) is an architecture-based continual learning approach that uses learnable hard attention masks to select the task-specific parameters.
    - [**Adaptive HAT (Adaptive Hard Attention to the Task, 2024)**](https://link.springer.com/chapter/10.1007/978-3-031-70352-2_9) is an architecture-based continual learning approach that improves [HAT (Hard Attention to the Task, 2018)](http://proceedings.mlr.press/v80/serra18a) by introducing new adaptive soft gradient clipping based on parameter importance and network sparsity.
    - **CBPHAT** is what I am working on, trying combining HAT (Hard Attention to the Task) algorithm with Continual Backpropagation (CBP) by leveraging the contribution utility as the parameter importance like in AdaHAT (Adaptive Hard Attention to the Task) algorithm.
    """

    def __init__(self, output_dim: int | None, gate: str) -> None:
        r"""Initialise the HAT mask backbone network with task embeddings and masks.

        **Args:**
        - **output_dim** (`int`): The output dimension which connects to CL output heads. The `input_dim` of output heads are expected to be the same as this `output_dim`. In some cases, this class is used for a block in the backbone network, which doesn't have the output dimension. In this case, it can be `None`.
        - **gate** (`str`): the type of gate function turning the real value task embeddings into attention masks, should be one of the following:
            - `sigmoid`: the sigmoid function.
        """
        CLBackbone.__init__(self, output_dim=output_dim)

        self.register_hat_mask_module_explicitly(
            gate=gate
        )  # we moved the registration of the modules to a separate method to solve a problem of multiple inheritance in terms of `nn.Module`

        HATMaskBackbone.sanity_check(self)

    def register_hat_mask_module_explicitly(self, gate: str) -> None:
        r"""Register all `nn.Module`s explicitly in this method. For `HATMaskBackbone`, they are task embedding for the current task and the masks.

        **Args:**
        - **gate** (`str`): the type of gate function turning the real value task embeddings into attention masks, should be one of the following:
            - `sigmoid`: the sigmoid function.
        """
        self.gate: str = gate
        r"""Store the type of gate function."""
        if gate == "sigmoid":
            self.gate_fn: nn.Module = nn.Sigmoid()
            r"""The gate function turning the real value task embeddings into attention masks."""

        self.task_embedding_t: nn.ModuleDict = nn.ModuleDict()
        r"""Store the task embedding for the current task. Keys are the layer names and values are the task embedding `nn.Embedding` for the layer. Each task embedding has size (1, number of units).
        
        We use `ModuleDict` rather than `dict` to make sure `LightningModule` can properly register these model parameters for the purpose of, like automatically transfering to device, being recorded in model summaries.
        
        we use `nn.Embedding` rather than `nn.Parameter` to store the task embedding for each layer, which is a type of `nn.Module` and can be accepted by `nn.ModuleDict`. (`nn.Parameter` cannot be accepted by `nn.ModuleDict`.)
        
        **This must be defined to cover each weighted layer (just as `self.weighted_layer_names` listed) in the backbone network.** Otherwise, the uncovered parts will keep updating for all tasks and become a source of catastrophic forgetting. """

    def initialise_task_embedding(self, mode: str) -> None:
        r"""Initialise the task embedding for the current task.

        **Args:**
        - **mode** (`str`): the initialisation mode for task embeddings, should be one of the following:
            1. 'N01' (default): standard normal distribution $N(0, 1)$.
            2. 'U-11': uniform distribution $U(-1, 1)$.
            3. 'U01': uniform distribution $U(0, 1)$.
            4. 'U-10': uniform distribution $U(-1, 0)$.
            5. 'last': inherit task embedding from last task.
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
        r"""Check the sanity of the arguments.

        **Raises:**
        - **ValueError**: when the `gate` is not one of the valid options.
        """

        if self.gate not in ["sigmoid"]:
            raise ValueError("The gate should be one of 'sigmoid'.")

    def get_mask(
        self,
        stage: str,
        s_max: float | None = None,
        batch_idx: int | None = None,
        num_batches: int | None = None,
        test_mask: dict[str, Tensor] | None = None,
    ) -> dict[str, Tensor]:
        r"""Get the hard attention mask used in `forward()` method for different stages.

        **Args:**
        - **stage** (`str`): the stage when applying the conversion, should be one of the following:
            1. 'train': training stage. If stage is 'train', get the mask from task embedding of current task through the gate function, which is scaled by an annealed scalar. See chapter 2.4 "Hard Attention Training" in [HAT paper](http://proceedings.mlr.press/v80/serra18a).
            2. ‘validation': validation stage. If stage is 'validation', get the mask from task embedding of current task through the gate function, which is scaled by `s_max`. (Note that in this stage, the binary mask hasn't been stored yet as the training is not over.)
            3. 'test': testing stage. If stage is 'test', apply the mask gate function is scaled by `s_max`, the large scaling making masks nearly binary.
        - **s_max** (`float`): the maximum scaling factor in the gate function. Doesn't apply to testing stage. See chapter 2.4 "Hard Attention Training" in [HAT paper](http://proceedings.mlr.press/v80/serra18a).
        - **batch_idx** (`int` | `None`): the current batch index. Applies only to training stage. For other stages, it is default `None`.
        - **num_batches** (`int` | `None`): the total number of batches. Applies only to training stage. For other stages, it is default `None`.
        - **test_mask** (`dict[str, Tensor]` | `None`): the binary mask used for test. Applies only to testing stage. For other stages, it is default `None`.

        **Returns:**
        - **mask** (`dict[str, Tensor]`): the hard attention (whose values are 0 or 1) mask. Key (`str`) is layer name, value (`Tensor`) is the mask tensor. The mask tensor has size (number of units).

        **Raises:**
        - **ValueError**: if the `batch_idx` and `batch_num` are not provided in 'train' stage; if the `s_max` is not provided in 'validation' stage; if the `task_id` is not provided in 'test' stage.
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
        if stage == "test" and (test_mask is None):
            raise ValueError(
                "The `task_mask` should be provided at testing stage, instead of the default value `None`."
            )

        mask = {}
        if stage == "train":
            for layer_name in self.weighted_layer_names:
                anneal_scalar = 1 / s_max + (s_max - 1 / s_max) * (batch_idx - 1) / (
                    num_batches - 1
                )  # see equation (3) in chapter 2.4 "Hard Attention Training" in [HAT paper](http://proceedings.mlr.press/v80/serra18a).
                mask[layer_name] = self.gate_fn(
                    self.task_embedding_t[layer_name].weight * anneal_scalar
                ).squeeze()
        elif stage == "validation":
            for layer_name in self.weighted_layer_names:
                mask[layer_name] = self.gate_fn(
                    self.task_embedding_t[layer_name].weight * s_max
                ).squeeze()
        elif stage == "test":
            mask = test_mask

        return mask

    def get_cumulative_mask(self) -> dict[str, Tensor]:
        r"""Get the cumulative mask till current task.

        **Returns:**
        - **cumulative_mask** (`dict[str, Tensor]`): the cumulative mask. Key (`str`) is layer name, value (`Tensor`) is the mask tensor. The mask tensor has size (number of units).
        """
        return self.cumulative_mask_for_previous_tasks

    def get_summative_mask(self) -> dict[str, Tensor]:
        r"""Get the summative mask till current task.

        **Returns:**
        - **summative_mask** (`dict[str, Tensor]`): the summative mask tensor. Key (`str`) is layer name, value (`Tensor`) is the mask tensor. The mask tensor has size (number of units).
        """
        return self.summative_mask_for_previous_tasks

    def get_layer_measure_parameter_wise(
        self,
        unit_wise_measure: dict[str, Tensor],
        layer_name: str,
        aggregation: str,
    ) -> Tensor:
        r"""Get the parameter-wise measure on the parameters right before the given layer.

        It is calculated from the given unit-wise measure. It aggregates two feature-sized vectors (corresponding the given layer and preceding layer) into a weight-wise matrix (corresponding the weights in between) and bias-wise vector (corresponding the bias of the given layer), using the given aggregation method. For example, given two feature-sized measure $m_{l,i}$ and $m_{l-1,j}$ and 'min' aggregation, the parameter-wise measure is then $\min \left(a_{l,i}, a_{l-1,j}\right)$, a matrix with respect to $i, j$.

        Note that if the given layer is the first layer with no preceding layer, we will get parameter-wise measure directly broadcasted from the unit-wise measure of given layer.

        This method is used in the calculation of parameter-wise measure in various HAT-based algorithms:

        - **HAT**: the parameter-wise measure is the binary mask for previous tasks from the unit-wise cumulative mask of previous tasks `self.cumulative_mask_for_previous_tasks`, which is $\min \left(a_{l,i}^{<t}, a_{l-1,j}^{<t}\right)$ in equation (2) in [HAT paper](http://proceedings.mlr.press/v80/serra18a).
        - **AdaHAT**: the parameter-wise measure is the parameter importance for previous tasks from the unit-wise summative mask of previous tasks `self.summative_mask_for_previous_tasks`, which is $\min \left(m_{l,i}^{<t,\text{sum}}, m_{l-1,j}^{<t,\text{sum}}\right)$ in equation (9) in [AdaHAT paper](https://link.springer.com/chapter/10.1007/978-3-031-70352-2_9).
        - **CBPHAT**: the parameter-wise measure is the parameter importance for previous tasks from the unit-wise importance of previous tasks `self.unit_importance_for_previous_tasks` based on contribution utility, which is $\min \left(I_{l,i}^{(t-1)}, I_{l-1,j}^{(t-1)}\right)$ in the adjustment rate formula in the paper draft.

        **Args:**
        - **unit_wise_measure** (`dict[str, Tensor]`): the unit-wise measure. Key is layer name, value is the unit-wise measure tensor. The measure tensor has size (number of units).
        - **layer_name** (`str`): the name of given layer.
        - **aggregation** (`str`): the aggregation method turning two feature-wise measures into weight-wise matrix, should be one of the following:
            - 'min': takes minimum of the two connected unit measures.
            - 'max': takes maximum of the two connected unit measures.

        **Returns:**
        - **weight_measure** (`Tensor`): the weight measure matrix, same size as the corresponding weights.
        - **bias_measure** (`Tensor`): the bias measure vector, same size as the corresponding bias.


        """

        # initialise the aggregation function
        if aggregation == "min":
            aggregation_func = torch.min
        elif aggregation == "max":
            aggregation_func = torch.max
        else:
            raise ValueError(f"The aggregation method {aggregation} is not supported.")

        # get the preceding layer name
        preceding_layer_name = self.preceding_layer_name(layer_name)

        # get weight size for expanding the measures
        layer = self.get_layer_by_name(layer_name)
        weight_size = layer.weight.size()

        # construct the weight-wise measure
        layer_measure = unit_wise_measure[layer_name]
        layer_measure_broadcast_size = (-1, 1) + tuple(
            1 for _ in range(len(weight_size) - 2)
        )  # since the size of mask tensor is (number of units), we extend it to (number of units, 1) and expand it to the weight size. The weight size has 2 dimensions in fully connected layers and 4 dimensions in convolutional layers

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
            )  # since the size of mask tensor is (number of units), we extend it to (1, number of units) and expand it to the weight size. The weight size has 2 dimensions in fully connected layers and 4 dimensions in convolutional layers
            preceding_layer_measure = unit_wise_measure[preceding_layer_name]
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
        test_mask: dict[str, Tensor] | None = None,
    ) -> tuple[Tensor, dict[str, Tensor], dict[str, Tensor]]:
        r"""The forward pass for data from task `task_id`. Task-specific mask for `task_id` are applied to the units in each layer.

        **Args:**
        - **input** (`Tensor`): The input tensor from data.
        - **stage** (`str`): the stage of the forward pass, should be one of the following:
            1. 'train': training stage.
            2. 'validation': validation stage.
            3. 'test': testing stage.
        - **s_max** (`float`): the maximum scaling factor in the gate function. See chapter 2.4 "Hard Attention Training" in [HAT paper](http://proceedings.mlr.press/v80/serra18a).
        - **batch_idx** (`int` | `None`): the current batch index. Applies only to training stage. For other stages, it is default `None`.
        - **num_batches** (`int` | `None`): the total number of batches. Applies only to training stage. For other stages, it is default `None`.
        - **test_mask** (`dict[str, Tensor]` | `None`): the binary mask used for test. Applies only to testing stage. For other stages, it is default `None`.

        **Returns:**
        - **output_feature** (`Tensor`): the output feature tensor to be passed into heads. This is the main target of backpropagation.
        - **mask** (`dict[str, Tensor]`): the mask for the current task. Key (`str`) is layer name, value (`Tensor`) is the mask tensor. The mask tensor has size (number of units).
        - **hidden_features** (`dict[str, Tensor]`): the hidden features (after activation) in each weighted layer. Key (`str`) is the weighted layer name, value (`Tensor`) is the hidden feature tensor. This is used for the continual learning algorithms that need to use the hidden features for various purposes. Although HAT algorithm does not need this, it is still provided for API consistence for other HAT-based algorithms inherited this `forward()` method of `HAT` class.

        """
        # this should be copied to all subclasses. Make sure it is called to get the mask for the current task from the task embedding in this stage
        mask = self.get_mask(
            stage,
            s_max=s_max,
            batch_idx=batch_idx,
            num_batches=num_batches,
            test_mask=test_mask,
        )
