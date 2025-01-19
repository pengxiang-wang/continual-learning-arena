"""
The submodule in `backbones` for CL backbone network bases.
"""

__all__ = ["CLBackbone", "HATMaskBackbone"]

import logging
import random

import torch
from torch import Tensor, nn
from typing_extensions import override

from clarena.utils import HATNetworkCapacity

# always get logger for built-in logging in each module
pylogger = logging.getLogger(__name__)


class CLBackbone(nn.Module):
    """The base class of continual learning backbone networks, inherited from `torch.nn.Module`."""

    def __init__(self, output_dim: int) -> None:
        """Initialise the CL backbone network.

        **Args:**
        - **output_dim** (`int`): The output dimension which connects to CL output heads. The `input_dim` of output heads are expected to be the same as this `output_dim`.
        """
        super().__init__()

        self.output_dim: int = output_dim
        """Store the output dimension of the backbone network."""

        self.task_id: int
        """Task ID counter indicating which task is being processed. Self updated during the task loop."""

        self.sanity_check()

    def sanity_check(self) -> None:
        """Check the sanity of the arguments.

        Nothing to check in the base class.
        """
        pass

    def setup_task_id(self, task_id: int) -> None:
        """Set up which task's dataset the CL experiment is on. This must be done before `forward()` method is called.

        **Args:**
        - **task_id** (`int`): the target task ID.
        """
        self.task_id = task_id

    @override  # since `nn.Module` uses it
    def forward(
        self,
        input: Tensor,
        stage: str,
        task_id: int | None = None,
    ) -> Tensor:
        """The forward pass for data from task `task_id`. In some backbones, the forward pass might be different for different tasks.It must be implemented by subclasses.

        **Args:**
        - **input** (`Tensor`): The input tensor from data.
        - **stage** (`str`): the stage of the forward pass, should be one of the following:
            1. 'train': training stage.
            2. 'validate': validation stage.
            3. 'test': testing stage.
        - **task_id** (`int`): the task ID where the data are from. If stage is 'train' or `validate`, it is usually from the current task `self.task_id`. If stage is 'test', it could be from any seen task. In TIL, the task IDs of test data are provided thus this argument can be used. In CIL, they are not provided, so it is just a placeholder for API consistence but never used, and best practices are not to provide this argument and leave it as the default value.

        **Returns:**
        - **feature** (`Tensor`): The output feature tensor to be passed into heads.
        """


class HATMaskBackbone(CLBackbone):
    """The backbone network for HAT algorithm with learnable hard attention masks.

    [HAT (Hard Attention to the Task, 2018)](http://proceedings.mlr.press/v80/serra18a) is an architecture-based continual learning approach that uses learnable hard attention masks to select the task-specific parameters.
    """

    def __init__(self, output_dim: int, gate: str) -> None:
        """Initialise the HAT mask backbone network with masks.

        **Args:**
        - **output_dim** (`int`): The output dimension which connects to CL output heads. The `input_dim` of output heads are expected to be the same as this `output_dim`.
        - **gate** (`str`): the type of gate function turning the real value task embeddings into attention masks, should be one of the following:
            - `sigmoid`: the sigmoid function.
        """
        super().__init__(output_dim)

        self.gate = gate
        """Store the type of gate function."""
        if gate == "sigmoid":
            self.gate_fn: nn.Module = nn.Sigmoid()
            """The gate function turning the real value task embeddings into attention masks."""

        self.masked_layers: nn.ModuleDict = nn.ModuleDict()
        """Store the layers with task embedding masking in the backbone network. Keys are the layer names and values are the layer modules. We use `ModuleDict` rather than dict just to make sure it can be recorded in model summaries."""

        self.task_embedding_t: nn.ModuleDict = nn.ModuleDict()
        """Store the task embedding for the current task. Keys are the layer names and values are the task embedding `nn.Parameter` for the layer. We use `ModuleDict` rather than dict just to make sure task embedding also as model parameter can be recorded in model summaries. 
        
        Must be defined to cover each layer in the backbone network. Otherwise, the uncovered parts will keep updating for all tasks and become a source of catastrophic forgetting. """

        self.masked_layer_order: list[str]
        """Store the layer names with task embedding masking in the order of forward pass. HAT gives task embedding to every possible layer. We provide this order to let the `clip_grad_mask()` method know which part of parameters are in between the layers. It must be provided in subclasses. """

        self.masks: dict[int, dict[str, Tensor]] = {}
        """Store the binary attention mask of each task gated from the task embedding. Keys are task IDs and values are the corresponding mask. Each mask is a dict where keys are layer names and values are the binary mask tensor for the layer."""

        self.cumulative_mask_t: dict[str, Tensor] = self.zeros_mask()
        """Store the cumulative binary attention mask of each task $\mathrm{M}^{<t}$, gated from the task embedding. Keys are task IDs and values are the corresponding mask. Each mask is a dict where keys are layer names and values are the binary mask tensor for the layer. It is initialised as zeros mask ($t = 1$). See equation (2) in chapter 3 in [AdaHAT paper](https://link.springer.com/chapter/10.1007/978-3-031-70352-2_9), or equation (5) in chapter 2.6 "Promoting Low Capacity Usage" in [HAT paper](http://proceedings.mlr.press/v80/serra18a). """

        self.summative_mask_t: dict[str, Tensor] = self.zeros_mask()
        """Store the summative binary attention mask of each task $\mathrm{M}^{<t,\text{sum}}$ gated from the task embedding. Keys are task IDs and values are the corresponding mask. Each mask is a dict where keys are layer names and values are the binary mask tensor for the layer. It is initialised as zeros mask ($t = 1$). See equation (7) in chapter 3.1 in [AdaHAT paper](https://link.springer.com/chapter/10.1007/978-3-031-70352-2_9). """

    def sanity_check(self) -> None:
        """Check the sanity of the arguments.

        **Raises:**
        - **ValueError**: when the `gate` is not one of the valid options.
        """

        if self.gate not in ["sigmoid"]:
            raise ValueError("The gate should be one of 'sigmoid'.")

        super().sanity_check()

    def zeros_mask(self) -> dict[str, Tensor]:
        """Return the zero mask for.

        **Returns:**
        - **zeros_mask** (`dict[str, Tensor]`): The zero mask for the current task. Key (`str`) is layer name, value (`Tensor`) is the mask tensor. 
        """

        for layer_name, te in self.task_embedding_t.items():
            self.mask[layer_name] = torch.zeros_like(te.weight)
        
        return {
            layer: torch.zeros_like(te) for layer, te in self.task_embeddings_t.items()
        }

    def store_mask(self, s_max: float) -> None:
        """Store the current task's binary mask into the masks dictionary, and also update the cumulative and summative masks.

        **Args:**
        - **s_max** (`float`): the maximum scaling factor in the gate function.
        """
        mask_t = {
            layer_name: self.gate_fn(te.weight * s_max)
            for layer_name, te in self.task_embedding_t.items()
        }
        self.masks[self.task_id] = mask_t
        self.cumulative_mask_t = {
            layer_name: torch.max(
                self.cumulative_mask_t[layer_name], mask_t[layer_name]
            )
            for layer_name in self.cumulative_mask_t.keys()
        }
        self.summative_mask_t = {
            layer_name: self.summative_mask_t[layer_name] + mask_t[layer_name]
            for layer_name in self.summative_mask_t.keys()
        }

    def get_mask(
        self,
        stage: str,
        s_max: float | None = None,
        batch_idx: int | None = None,
        num_batches: int | None = None,
        task_id: int | None = None,
    ) -> dict[str, Tensor]:
        """Convert current task embedding to masks through the scaled gate function.

        **Args:**
        - **stage** (`str`): the stage when applying the conversion, should be one of the following:
            1. 'train': training stage. If stage is 'train', get the mask from task embedding of current task through the gate function, which is scaled by an annealed scalar. See chapter 2.4 "Hard Attention Training" in [HAT paper](http://proceedings.mlr.press/v80/serra18a).
            2. ‘validate': validation stage. If stage is 'validate', get the mask from task embedding of current task through the gate function, which is scaled by `s_max`. (Note that in this stage, the binary mask hasn't been stored yet as the training is not all over.)
            3. 'test': testing stage. If stage is 'test', apply the mask gate function is scaled by `s_max`, the large scaling making masks nearly binary.
        - **s_max** (`float`): the maximum scaling factor in the gate function. See chapter 2.4 "Hard Attention Training" in [HAT paper](http://proceedings.mlr.press/v80/serra18a).
        - **batch_idx** (`int` | `None`): the current batch index. Applies only to training stage. For other stages, it is default `None`.
        - **num_batches** (`int` | `None`): the total number of batches. Applies only to training stage. For other stages, it is default `None`.
        - **task_id** (`int` | `None`): the task ID where the mask belongs to. Applies only to testing stage. For other stages, it is automatically the current task `self.task_id`.

        **Returns:**
        - **mask** (`dict[str, Tensor]`): the hard attention (whose values are 0 or 1) mask. Key (`str`) is layer name, value (`Tensor`) is the mask tensor. 

        **Raises:**
        - **ValueError**: if the `batch_idx` and `batch_num` are not provided in 'train' stage.
        """

        if stage == "train" and (batch_idx is None or num_batches is None):
            raise ValueError(
                "The `batch_idx` and `batch_num` should be provided in 'train' stage, instead of the default value `None`."
            )

        mask = {}
        if stage == "train":
            for layer, te in self.task_embeddings_t.items():
                anneal_scalar = 1 / s_max + (s_max - 1 / s_max) * (batch_idx - 1) / (
                    num_batches - 1
                )  # see equation (3) in chapter 2.4. Hard Attention Training in [HAT paper](http://proceedings.mlr.press/v80/serra18a).
                mask[layer] = self.gate_fn(te.weight * anneal_scalar)
        elif stage == "validate":
            mask[layer] = self.gate_fn(te.weight * s_max)
        elif stage == "test":
            mask = self.masks[task_id]

        return mask

    def get_cumulative_mask(self) -> dict[str, Tensor]:
        """Get the cumulative mask of the current task.

        **Returns:**
        - **cumulative_mask** (`dict[str, Tensor]`): the cumulative mask. Key (`str`) is layer name, value (`Tensor`) is the mask tensor. 
        """
        return self.cumulative_mask_t

    def get_summative_mask(self) -> dict[str, Tensor]:
        """Get the summative mask of the current task.

        **Returns:**
        - **summative_mask** (`dict[str, Tensor]`): the summative mask tensor. Key (`str`) is layer name, value (`Tensor`) is the mask tensor. 
        """
        return self.summative_mask_t

    def preceding_layer_name(self, layer_name: str) -> str:
        """Get the name of the preceding layer of the given layer from the stored `self.masked_layer_order`.

        **Args:**
        - **layer_name** (`str`): the name of the layer.

        **Returns:**
        - **preceding_layer_name** (`str`): the name of the preceding layer.

        **Raises:**
        - **ValueError**: if the `layer_name` is not in the masked layer order.
        """

        if layer_name not in self.masked_layer_order:
            raise ValueError(f"The layer name {layer_name} doesn't exist.")

        layer_index = self.masked_layer_order.index(layer_name)
        if layer_index == 0:
            return None
        return self.masked_layer_order[layer_index - 1]

    def get_layer_parameter_mask(self, layer_name: str) -> Tensor:
        """Get the binary mask for the parameters right before the given layer from the cumulative mask of previous tasks `self.cumulative_mask_t`, which is the matrix (in terms of $i$, $j$) of $\min \left(a_{l,i}^{<t}, a_{l-1,j}^{<t}\right)$ in equation (2) in [HAT paper](http://proceedings.mlr.press/v80/serra18a).

        For the parameters between the masked layer, the binary mask for the parameter is the minimum of the cumulative masks of the two units. If the given layer is the first layer, the binary mask for the parameter is the cumulative mask of the given layer.

        **Args:**
        - **layer_name** (`str`): the layer name.

        **Returns:**
        - **weight_mask** (`Tensor`): the weight mask matrix, same size as the corresponding weights.
        - **bias_mask** (`Tensor`): the bias mask vector, same size as the corresponding bias.
        """
        # get the preceding layer name
        preceding_layer_name = self.preceding_layer_name(layer_name)

        # get weight size for expanding the masks
        layer = self.masked_layers[layer_name]
        weight_size = layer.weight.size()

        # construct weight mask
        layer_mask = self.cumulative_mask_t[layer_name]
        layer_mask_broadcasted = layer_mask.expand(
            weight_size
        )  # expand the given layer mask to the weight size and broadcast

        if (
            preceding_layer_name
        ):  # if the layer is not the first layer, where the preceding layer exists
            preceding_layer_mask = self.cumulative_mask_t[
                self.preceding_layer_name(layer_name)
            ]
            preceding_layer_mask_broadcasted = preceding_layer_mask.expand(
                weight_size
            )  # expand the preceding layer mask to the weight size and broadcast
            weight_mask = torch.min(
                layer_mask_broadcasted, preceding_layer_mask_broadcasted
            )  # get the minimum of the two mask vectors, from expanded
        else:  # if the layer is the first layer
            weight_mask = layer_mask_broadcasted

        # construct bias mask
        bias_mask = layer_mask

        return weight_mask, bias_mask

    def get_layer_parameter_importance(self, layer_name: str) -> Tensor:
        """Get the importance of parameters right before the given layer from the summative mask of previous tasks `self.summative_mask_t`, which is the matrix (in terms of $i$, $j$) of $\min \left(m_{l,i}^{<t,\text{sum}}, m_{l-1,j}^{<t,\text{sum}}\right)$ in equation (9)in [AdaHAT paper](https://link.springer.com/chapter/10.1007/978-3-031-70352-2_9).

        For the parameters between the masked layer, the parameter importance is the minimum of the summative masks of the two units. If the given layer is the first layer, the parameter importance is the summative mask of the given layer.

        **Args:**
        - **layer_name** (`str`): the layer name.

        **Returns:**
        - **weight_importance** (`Tensor`): the weight importance matrix, same size as the corresponding weights.
        - **bias_importance** (`Tensor`): the bias importance vector, same size as the corresponding bias.
        """
        # get the preceding layer name
        preceding_layer_name = self.preceding_layer_name(layer_name)

        # get weight size for expanding the masks
        layer = self.masked_layers[layer_name]
        weight_size = layer.weight.size()

        # construct weight importance
        layer_importance = self.summative_mask_t[layer_name]
        layer_importance_broadcasted = layer_importance.expand(
            weight_size
        )  # expand the given layer importance to the weight size and broadcast

        if (
            preceding_layer_name
        ):  # if the layer is not the first layer, where the preceding layer exists
            preceding_layer_importance = self.summative_mask_t[
                self.preceding_layer_name(layer_name)
            ]
            preceding_layer_importance_broadcasted = preceding_layer_importance.expand(
                weight_size
            )  # expand the preceding layer importance to the weight size and broadcast
            weight_importance = torch.min(
                layer_importance_broadcasted, preceding_layer_importance_broadcasted
            )  # get the minimum of the two importance vectors, from expanded
        else:  # if the layer is the first layer
            weight_importance = layer_importance_broadcasted

        # construct bias importance
        bias_importance = layer_importance

        return weight_importance, bias_importance

    def clip_grad_by_adjustment(
        self,
        mode: str,
        alpha: float | None = None,
        epsilon: float | None = None,
        network_sparsity: dict[str, Tensor] | None = None,
    ) -> Tensor:
        """Clip the gradients by the adjustment rate.
        
        Note that as the task embedding fully covers every layer in the backbone network, no parameters are left out of this system. This applies not only the parameters in between layers with task embedding, but also those before the first layer. We designed it seperately in the codes.
        
        Network capacity is measured along with this method. Network capacity is defined as the average adjustment rate over all parameters. See chapter 4.1 in [AdaHAT paper](https://link.springer.com/chapter/10.1007/978-3-031-70352-2_9).

        **Args:**
        - **mode** (`str`): the mode of clipping, should be one of the following:
            1. 'hat': set the gradients of parameters linking to masked units to zero. This is the way that HAT does, which fixes the part of network for previous tasks completely. See equation (2) in chapter 2.3 "Network Training" in [HAT paper](http://proceedings.mlr.press/v80/serra18a).
            2. 'adahat_original': set the gradients of parameters linking to masked units to a soft adjustment rate in the original AdaHAT approach. This is the way that AdaHAT does, which allowes the part of network for previous tasks to be updated slightly. See equation (8) and (9) chapter 3.1 in [AdaHAT paper](https://link.springer.com/chapter/10.1007/978-3-031-70352-2_9).
            3. 'hat_random': set the gradients of parameters linking to masked units to random 0-1 values. See the Baselines section in chapter 4.1 in [AdaHAT paper](https://link.springer.com/chapter/10.1007/978-3-031-70352-2_9).
            4. 'hat_const_alpha': set the gradients of parameters linking to masked units to a constant value of `alpha`. See the Baselines section in chapter 4.1 in [AdaHAT paper](https://link.springer.com/chapter/10.1007/978-3-031-70352-2_9).
            5. 'hat_const_1': set the gradients of parameters linking to masked units to a constant value of 1, which means no gradient constraint on any parameter at all. See the Baselines section in chapter 4.1 in [AdaHAT paper](https://link.springer.com/chapter/10.1007/978-3-031-70352-2_9).
            6. 'adahat_no_sum': set the gradients of parameters linking to masked units to a soft adjustment rate in the original AdaHAT approach, but without considering the information of parameter importance i.e. summative mask. This is the way that one of the AdaHAT ablation study does. See chapter 4.3 in [AdaHAT paper](https://link.springer.com/chapter/10.1007/978-3-031-70352-2_9).
            7. 'adahat_no_reg': set the gradients of parameters linking to masked units to a soft adjustment rate in the original AdaHAT approach, but without considering the information of network sparsity i.e. mask sparsity regularisation value. This is the way that one of the AdaHAT ablation study does. See chapter 4.3 in [AdaHAT paper](https://link.springer.com/chapter/10.1007/978-3-031-70352-2_9).
        - **network_sparsity** (`dict[str, Tensor]` | `None`): The network sparsity i.e. the mask sparisty loss of each layer for the current task. It applies only to AdaHAT modes, as it is used to calculate the adjustment rate for the gradients.
        - **alpha** (`float` | `None`): the hyperparameter that control the overall intensity of gradient adjustment. It applies only to AdaHAT modes and `hat_const_alpha`.
        - **epsilon** (`float` | `None`): the small value to avoid division by zero appeared in equation (9) in [AdaHAT paper](https://link.springer.com/chapter/10.1007/978-3-031-70352-2_9).

        **Returns:**
        - **capacity** (`Tensor`): the calculated network capacity.
        
        """
        # placeholder for the adjustment rate to avoid the error of using it before assignment
        adjustment_rate_weight = 1
        adjustment_rate_bias = 1
        
        # initialise network capacity metric
        capacity = HATNetworkCapacity()

        # Calculate the adjustment rate for gradients of the parameters, both weights and biases (if exists)
        for layer_name in self.masked_layer_order:
            # loop over the layers
            layer = self.masked_layers[layer_name]

            if (
                mode == "hat" or "hat_random" or "hat_const_alpha" or "hat_const_1"
            ):  # HAT modes
                weight_mask, bias_mask = self.get_layer_parameter_mask(layer_name)

                if mode == "hat":
                    adjustment_rate_weight = 1 - weight_mask
                    adjustment_rate_bias = 1 - bias_mask

                elif mode == "hat_random":
                    adjustment_rate_weight = torch.rand_like(
                        weight_mask
                    ) * weight_mask + (1 - weight_mask)
                    adjustment_rate_bias = torch.rand_like(bias_mask) * bias_mask + (
                        1 - bias_mask
                    )

                elif mode == "hat_const_alpha":
                    adjustment_rate_weight = alpha * torch.ones_like(
                        weight_mask
                    ) * weight_mask + (1 - weight_mask)
                    adjustment_rate_bias = alpha * torch.ones_like(
                        bias_mask
                    ) * bias_mask + (1 - bias_mask)

                elif mode == "hat_const_1":
                    adjustment_rate_weight = torch.ones_like(
                        weight_mask
                    ) * weight_mask + (1 - weight_mask)
                    adjustment_rate_bias = torch.ones_like(bias_mask) * bias_mask + (
                        1 - bias_mask
                    )

            else:  # AdaHAT modes

                weight_importance, bias_importance = (
                    self.get_layer_parameter_importance(layer_name)
                )  # depend on parameter importance instead of parameter mask
                network_sparsity_layer = network_sparsity[layer_name]

                if mode == "adahat_original":

                    r_layer = alpha / (epsilon + network_sparsity_layer)
                    adjustment_rate_weight = torch.div(
                        r_layer, (weight_importance, r_layer)
                    )
                    adjustment_rate_bias = torch.div(
                        r_layer, (bias_importance, r_layer)
                    )

                elif mode == "adahat_no_sum":

                    r_layer = alpha / (epsilon + network_sparsity_layer)
                    adjustment_rate_weight = torch.div(r_layer, (self.task_id, r_layer))
                    adjustment_rate_bias = torch.div(r_layer, (self.task_id, r_layer))

                elif mode == "adahat_no_reg":

                    r_layer = alpha / (epsilon + 0.0)
                    adjustment_rate_weight = torch.div(
                        r_layer, (weight_importance, r_layer)
                    )
                    adjustment_rate_bias = torch.div(
                        r_layer, (bias_importance, r_layer)
                    )

            # apply the adjustment rate to the gradients
            layer.weight.grad.data *= adjustment_rate_weight
            if layer.bias:
                layer.bias.grad.data *= adjustment_rate_bias
                
            # update network capacity metric
            capacity.update(adjustment_rate_weight, adjustment_rate_bias)
                
        return capacity.compute()

    def compensate_task_embedding_gradients(
        self,
        clamp_threshold: float,
        s_max: float,
        batch_idx: int,
        num_batches: int,
    ) -> None:
        """Compensate the gradients of task embeddings during training. See chapter 2.5 "Embedding Gradient Compensation" in [HAT paper](http://proceedings.mlr.press/v80/serra18a).

        **Args:**
        - `clamp_threshold` (`float`):
        - **s_max** (`float`): the maximum scaling factor in the gate function. See chapter 2.4. Hard Attention Training in [HAT paper](http://proceedings.mlr.press/v80/serra18a).
        - **batch_idx** (`int`): the current training batch index.
        - **num_batches** (`int`): the total number of training batches.

        """

        for te in self.task_embedding_t.values():
            anneal_scalar = 1 / s_max + (s_max - 1 / s_max) * (batch_idx - 1) / (
                num_batches - 1
            )  # see equation (3) in chapter 2.4 "Hard Attention Training" in [HAT paper](http://proceedings.mlr.press/v80/serra18a)

            num = (
                torch.cosh(
                    torch.clamp(
                        anneal_scalar * te.weight.data,
                        -clamp_threshold,
                        clamp_threshold,
                    )
                )
                + 1
            )

            den = torch.cosh(te.weight.data) + 1

            compensation = s_max / anneal_scalar * num / den

            te.weight.grad.data *= compensation

    @override
    def forward(
        self,
        input: Tensor,
        stage: str,
        s_max: float | None = None,
        batch_idx: int | None = None,
        num_batches: int | None = None,
        task_id: int | None = None,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        """The forward pass for data from task `task_id`. Task-specific mask for `task_id` are applied to the neurons in each layer.

        **Args:**
        - **input** (`Tensor`): The input tensor from data.
        - **stage** (`str`): the stage of the forward pass, should be one of the following:
            1. 'train': training stage.
            2. 'validate': validation stage.
            3. 'test': testing stage.
        - **s_max** (`float`): the maximum scaling factor in the gate function. See chapter 2.4 "Hard Attention Training" in [HAT paper](http://proceedings.mlr.press/v80/serra18a).
        - **batch_idx** (`int` | `None`): the current batch index. Applies only to training stage. For other stages, it is default `None`.
        - **num_batches** (`int` | `None`): the total number of batches. Applies only to training stage. For other stages, it is default `None`.
        - **task_id** (`int`): the task ID where the data are from. Applies only to testing stage. For other stages, it is automatically the current task `self.task_id`. If stage is 'test', it could be from any seen task. In TIL, the task IDs of test data are provided thus this argument can be used. HAT algorithm works only for TIL.

        **Returns:**
        - **feature** (`Tensor`): the output feature tensor to be passed.
        - **mask** (`dict[str, Tensor]`): the mask for the current task. Key (`str`) is layer name, value (`Tensor`) is the mask tensor.
        """
        # this should be copied to all subclasses. Make sure it is called to get the mask for the current task from the task embedding in this stage
        mask = self.get_mask(
            stage,
            s_max=s_max,
            batch_idx=batch_idx,
            num_batches=num_batches,
            task_id=task_id,
        )

        # the forward pass should apply the mask at each neuron. See chapter 2.2 Architecture in [HAT paper](http://proceedings.mlr.press/v80/serra18a).
        feature = input

        return feature, mask
