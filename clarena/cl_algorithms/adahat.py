r"""
The submodule in `cl_algorithms` for [AdaHAT (Adaptive Hard Attention to the Task) algorithm](https://link.springer.com/chapter/10.1007/978-3-031-70352-2_9).
"""

__all__ = ["AdaHAT"]

import logging

import torch
from torch import Tensor

from clarena.backbones import HATMaskBackbone
from clarena.cl_algorithms import HAT
from clarena.cl_heads import HeadsCIL, HeadsTIL
from clarena.utils import HATNetworkCapacity

# always get logger for built-in logging in each module
pylogger = logging.getLogger(__name__)


class AdaHAT(HAT):
    r"""AdaHAT (Adaptive Hard Attention to the Task) algorithm.

    [Adaptive HAT (Adaptive Hard Attention to the Task, 2024)](https://link.springer.com/chapter/10.1007/978-3-031-70352-2_9) is an architecture-based continual learning approach that improves [HAT (Hard Attention to the Task, 2018)](http://proceedings.mlr.press/v80/serra18a) by introducing new adaptive soft gradient clipping based on parameter importance and network sparsity.

    We implement AdaHAT as a subclass of HAT algorithm, as AdaHAT has the same  `forward()`, `compensate_task_embedding_gradients()`, `training_step()`, `on_train_end()`,`validation_step()`, `test_step()` method as `HAT` class.
    """

    def __init__(
        self,
        backbone: HATMaskBackbone,
        heads: HeadsTIL | HeadsCIL,
        adjustment_mode: str,
        adjustment_intensity: float,
        s_max: float,
        clamp_threshold: float,
        mask_sparsity_reg_factor: float,
        mask_sparsity_reg_mode: str = "original",
        task_embedding_init_mode: str = "N01",
        epsilon: float = 0.1,
    ) -> None:
        r"""Initialise the AdaHAT algorithm with the network.

        **Args:**
        - **backbone** (`HATMaskBackbone`): must be a backbone network with HAT mask mechanism.
        - **heads** (`HeadsTIL` | `HeadsCIL`): output heads.
        - **adjustment_mode** (`str`): the strategy of adjustment i.e. the mode of gradient clipping, should be one of the following:
            1. 'adahat': set the gradients of parameters linking to masked units to a soft adjustment rate in the original AdaHAT approach. This is the way that AdaHAT does, which allowes the part of network for previous tasks to be updated slightly. See equation (8) and (9) chapter 3.1 in [AdaHAT paper](https://link.springer.com/chapter/10.1007/978-3-031-70352-2_9).
            2. 'adahat_no_sum': set the gradients of parameters linking to masked units to a soft adjustment rate in the original AdaHAT approach, but without considering the information of parameter importance i.e. summative mask. This is the way that one of the AdaHAT ablation study does. See chapter 4.3 in [AdaHAT paper](https://link.springer.com/chapter/10.1007/978-3-031-70352-2_9).
            3. 'adahat_no_reg': set the gradients of parameters linking to masked units to a soft adjustment rate in the original AdaHAT approach, but without considering the information of network sparsity i.e. mask sparsity regularisation value. This is the way that one of the AdaHAT ablation study does. See chapter 4.3 in [AdaHAT paper](https://link.springer.com/chapter/10.1007/978-3-031-70352-2_9).
        - **adjustment_intensity** (`float`): hyperparameter, control the overall intensity of gradient adjustment. It's the $\alpha$ in equation (9) in [AdaHAT paper](https://link.springer.com/chapter/10.1007/978-3-031-70352-2_9).
        - **s_max** (`float`): hyperparameter, the maximum scaling factor in the gate function. See chapter 2.4 "Hard Attention Training" in [HAT paper](http://proceedings.mlr.press/v80/serra18a).
        - **clamp_threshold** (`float`): the threshold for task embedding gradient compensation. See chapter 2.5 "Embedding Gradient Compensation" in [HAT paper](http://proceedings.mlr.press/v80/serra18a).
        - **mask_sparsity_reg_factor** (`float`): hyperparameter, the regularisation factor for mask sparsity.
        - **mask_sparsity_reg_mode** (`str`): the mode of mask sparsity regularisation, should be one of the following:
            1. 'original' (default): the original mask sparsity regularisation in HAT paper.
            2. 'cross': the cross version mask sparsity regularisation.
        - **task_embedding_init_mode** (`str`): the initialisation method for task embeddings, should be one of the following:
            1. 'N01' (default): standard normal distribution $N(0, 1)$.
            2. 'U-11': uniform distribution $U(-1, 1)$.
            3. 'U01': uniform distribution $U(0, 1)$.
            4. 'U-10': uniform distribution $U(-1, 0)$.
            5. 'last': inherit task embedding from last task.
        - **epsilon** (`float`): the value added to network sparsity to avoid division by zero appeared in equation (9) in [AdaHAT paper](https://link.springer.com/chapter/10.1007/978-3-031-70352-2_9).
        """
        HAT.__init__(
            self,
            backbone=backbone,
            heads=heads,
            adjustment_mode=adjustment_mode,
            s_max=s_max,
            clamp_threshold=clamp_threshold,
            mask_sparsity_reg_factor=mask_sparsity_reg_factor,
            mask_sparsity_reg_mode=mask_sparsity_reg_mode,
            task_embedding_init_mode=task_embedding_init_mode,
            alpha=None,
        )

        self.adjustment_intensity = adjustment_intensity
        r"""Store the adjustment intensity in equation (9) in [AdaHAT paper](https://link.springer.com/chapter/10.1007/978-3-031-70352-2_9)."""
        self.epsilon = epsilon
        """Store the small value to avoid division by zero appeared in equation (9) in [AdaHAT paper](https://link.springer.com/chapter/10.1007/978-3-031-70352-2_9)."""

        self.summative_mask_for_previous_tasks: dict[str, Tensor] = {}
        r"""Store the summative binary attention mask $\mathrm{M}^{<t,\text{sum}}$ previous tasks $1,\cdots, t-1$, gated from the task embedding. Keys are task IDs and values are the corresponding summative mask. Each cumulative mask is a dict where keys are layer names and values are the binary mask tensor for the layer. The mask tensor has size (number of units). """

        # set manual optimisation
        self.automatic_optimization = False

        AdaHAT.sanity_check(self)

    def sanity_check(self) -> None:
        r"""Check the sanity of the arguments.

        **Raises:**
        - **ValueError**: If the `adjustment_intensity` is not positive.
        """
        if self.adjustment_intensity <= 0:
            raise ValueError(
                f"The adjustment intensity should be positive, but got {self.adjustment_intensity}."
            )

    def on_train_start(self) -> None:
        r"""Additionally initialise the summative mask at the beginning of first task."""
        HAT.on_train_start(self)

        # initialise the summative mask at the beginning of first task. This should not be called in `__init__()` method as the `self.device` is not available at that time.
        if self.task_id == 1:
            for layer_name in self.backbone.weighted_layer_names:
                layer = self.backbone.get_layer_by_name(
                    layer_name
                )  # get the layer by its name
                num_units = layer.weight.shape[0]

                self.summative_mask_for_previous_tasks[layer_name] = torch.zeros(
                    num_units
                ).to(
                    self.device
                )  # the summative mask $\mathrm{M}^{<t,\text{sum}}$ is initialised as zeros mask ($t = 1$). See equation (7) in chapter 3.1 in [AdaHAT paper](https://link.springer.com/chapter/10.1007/978-3-031-70352-2_9).

    def clip_grad_by_adjustment(
        self,
        network_sparsity: dict[str, Tensor] | None = None,
    ) -> Tensor:
        r"""Clip the gradients by the adjustment rate.

        Note that as the task embedding fully covers every layer in the backbone network, no parameters are left out of this system. This applies not only the parameters in between layers with task embedding, but also those before the first layer. We designed it seperately in the codes.

        Network capacity is measured along with this method. Network capacity is defined as the average adjustment rate over all parameters. See chapter 4.1 in [AdaHAT paper](https://link.springer.com/chapter/10.1007/978-3-031-70352-2_9).

        **Args:**
        - **network_sparsity** (`dict[str, Tensor]` | `None`): The network sparsity i.e. the mask sparsity loss of each layer for the current task. It applies only to AdaHAT modes, as it is used to calculate the adjustment rate for the gradients.

        **Returns:**
        - **capacity** (`Tensor`): the calculated network capacity.
        """

        # initialise network capacity metric
        capacity = HATNetworkCapacity()

        # Calculate the adjustment rate for gradients of the parameters, both weights and biases (if exists)
        for layer_name in self.backbone.weighted_layer_names:

            layer = self.backbone.get_layer_by_name(
                layer_name
            )  # get the layer by its name

            # placeholder for the adjustment rate to avoid the error of using it before assignment
            adjustment_rate_weight = 1
            adjustment_rate_bias = 1

            weight_importance, bias_importance = (
                self.backbone.get_layer_measure_parameter_wise(
                    unit_wise_measure=self.summative_mask_for_previous_tasks,
                    layer_name=layer_name,
                    aggregation="min",
                )
            )  # AdaHAT depend on parameter importance instead of parameter mask like HAT

            network_sparsity_layer = network_sparsity[layer_name]

            if self.adjustment_mode == "adahat":
                r_layer = self.adjustment_intensity / (
                    self.epsilon + network_sparsity_layer
                )
                adjustment_rate_weight = torch.div(
                    r_layer, (weight_importance + r_layer)
                )
                adjustment_rate_bias = torch.div(r_layer, (bias_importance + r_layer))

            elif self.adjustment_mode == "adahat_no_sum":

                r_layer = self.adjustment_intensity / (
                    self.epsilon + network_sparsity_layer
                )
                adjustment_rate_weight = torch.div(r_layer, (self.task_id + r_layer))
                adjustment_rate_bias = torch.div(r_layer, (self.task_id + r_layer))

            elif self.adjustment_mode == "adahat_no_reg":

                r_layer = self.adjustment_intensity / (self.epsilon + 0.0)
                adjustment_rate_weight = torch.div(
                    r_layer, (weight_importance + r_layer)
                )
                adjustment_rate_bias = torch.div(r_layer, (bias_importance + r_layer))

            # apply the adjustment rate to the gradients
            layer.weight.grad.data *= adjustment_rate_weight
            if layer.bias is not None:
                layer.bias.grad.data *= adjustment_rate_bias

            # update network capacity metric
            capacity.update(adjustment_rate_weight, adjustment_rate_bias)

        return capacity.compute()

    def on_train_end(self) -> None:
        r"""Additionally update summative mask after training the task."""

        HAT.on_train_end(self)

        mask_t = self.masks[
            f"{self.task_id}"
        ]  # get stored mask for the current task again
        self.summative_mask_for_previous_tasks = {
            layer_name: self.summative_mask_for_previous_tasks[layer_name]
            + mask_t[layer_name]
            for layer_name in self.backbone.weighted_layer_names
        }
