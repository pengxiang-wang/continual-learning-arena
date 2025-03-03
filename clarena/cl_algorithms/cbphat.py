r"""
The submodule in `cl_algorithms` for CBPHAT algorithm.
"""

__all__ = ["CBPHAT"]

import logging
from typing import Any

import torch
from torch import Tensor

from clarena.backbones import HATMaskBackbone
from clarena.cl_algorithms import AdaHAT
from clarena.cl_heads import HeadsCIL, HeadsTIL
from clarena.utils import HATNetworkCapacity
from clarena.utils.transforms import min_max_normalise

# always get logger for built-in logging in each module
pylogger = logging.getLogger(__name__)


class CBPHAT(AdaHAT):
    r"""CBPHAT algorithm.

    CBPHAT is what I am working on, trying combining HAT (Hard Attention to the Task) algorithm with Continual Backpropagation (CBP) by leveraging the contribution utility as the parameter importance like in AdaHAT (Adaptive Hard Attention to the Task) algorithm.

    We implement CBPHAT as a subclass of AdaHAT algorithm because CBPHAT adopt the similar idea as AdaHAT.
    """

    def __init__(
        self,
        backbone: HATMaskBackbone,
        heads: HeadsTIL | HeadsCIL,
        adjustment_mode: str,
        adjustment_intensity: float,
        utility_decay_rate: float,
        s_max: float,
        clamp_threshold: float,
        mask_sparsity_reg_factor: float,
        mask_sparsity_reg_mode: str = "original",
        task_embedding_init_mode: str = "N01",
        epsilon: float = 0.1,
    ) -> None:
        r"""Initialise the CBPHAT algorithm with the network.

        **Args:**
        - **backbone** (`HATMaskBackbone`): must be a backbone network with HAT mask mechanism.
        - **heads** (`HeadsTIL` | `HeadsCIL`): output heads.
        - **adjustment_mode** (`str`): the strategy of adjustment i.e. the mode of gradient clipping, should be one of the following:
            1. 'cbphat': our original CBP mode.
        - **adjustment_intensity** (`float`): hyperparameter, control the overall intensity of gradient adjustment. It's the $\alpha$ in equation (9) in [AdaHAT paper](https://link.springer.com/chapter/10.1007/978-3-031-70352-2_9).
        - **utility_decay_rate** (`float`): the utility decay rate of units. It is the rate at which the utility of a unit decays over time.
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
        - **epsilon** (`float`): the value added to network sparsity to avoid zero appeared in equation (9) in [AdaHAT paper](https://link.springer.com/chapter/10.1007/978-3-031-70352-2_9).
        """
        AdaHAT.__init__(
            self,
            backbone=backbone,
            heads=heads,
            adjustment_mode=adjustment_mode,
            adjustment_intensity=adjustment_intensity,
            s_max=s_max,
            clamp_threshold=clamp_threshold,
            mask_sparsity_reg_factor=mask_sparsity_reg_factor,
            mask_sparsity_reg_mode=mask_sparsity_reg_mode,
            task_embedding_init_mode=task_embedding_init_mode,
            epsilon=epsilon,
        )

        self.utility_decay_rate: float = utility_decay_rate
        r"""Store the utility decay rate of units. """

        self.contribution_utility_t: dict[str, Tensor] = {}
        r"""Store the summative min-max scaled contribution utility of units. See $U$ in the paper draft. Keys are layer names and values are the utility tensor for the layer. The utility tensor is the same size as the feature tensor with size (number of units). """
        self.unit_importance_for_previous_tasks: dict[str, Tensor] = {}
        r"""Store the unit importance values of units for previous tasks (1, \cdots, self.task_id - 1). See the "screenshot" $I^{(t-1)}$ in the paper draft. Keys are layer names and values are the importance tensor for the layer. The importance tensor is the same size as the feature tensor with size (number of units). """
        self.age_t: dict[str, Tensor] = {}
        r"""Store the age of units. Keys are layer names and values are the age tensor for the layer for current task. The age tensor is the same size as the feature tensor with size (number of units). """

        # set manual optimisation
        self.automatic_optimization = False

        CBPHAT.sanity_check(self)

    def sanity_check(self) -> None:
        r"""Check the sanity of the arguments.

        **Raises:**
        - **ValueError**: If the utility decay rate is not in the range (0, 1].
        """
        if self.utility_decay_rate > 1 or self.utility_decay_rate <= 0:
            raise ValueError(
                f"The utility decay rate should be in the range (0, 1], but got {self.utility_decay_rate}."
            )

    def on_train_start(self) -> None:
        r"""Additionally initialise the utility, age and the CBPHAT unit importance for each layer as zeros."""
        AdaHAT.on_train_start(self)

        for layer_name in self.backbone.weighted_layer_names:
            layer = self.backbone.get_layer_by_name(
                layer_name
            )  # get the layer by its name
            num_units = layer.weight.shape[0]

            # initialise the utility and age at the beginning of first task
            self.contribution_utility_t[layer_name] = torch.zeros(num_units).to(
                self.device
            )
            self.age_t[layer_name] = torch.zeros(num_units).to(self.device)

            # initialise the unit importance at the beginning of first task. This should not be called in `__init__()` method as the `self.device` is not available at that time.
            if self.task_id == 1:
                self.unit_importance_for_previous_tasks[layer_name] = torch.zeros(
                    num_units
                ).to(
                    self.device
                )  # the unit importance $I^{(t-1)}$ is initialised as zeros mask ($t = 1$). See the paper draft.

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
                    unit_wise_measure=self.unit_importance_for_previous_tasks,
                    layer_name=layer_name,
                    aggregation="min",
                )
            )

            weight_summative_mask, bias_summative_mask = (
                self.backbone.get_layer_measure_parameter_wise(
                    unit_wise_measure=self.summative_mask_for_previous_tasks,
                    layer_name=layer_name,
                    aggregation="min",
                )
            )

            network_sparsity_layer = network_sparsity[layer_name]

            if self.adjustment_mode == "cbphat":

                r_layer = self.adjustment_intensity / (
                    self.epsilon + network_sparsity_layer
                )
                adjustment_rate_weight = torch.div(
                    r_layer, (weight_importance + weight_summative_mask + r_layer)
                )

                adjustment_rate_bias = torch.div(
                    r_layer, (bias_importance + bias_summative_mask + r_layer)
                )

            # apply the adjustment rate to the gradients
            layer.weight.grad.data *= adjustment_rate_weight
            if layer.bias is not None:
                layer.bias.grad.data *= adjustment_rate_bias

            # update network capacity metric
            capacity.update(adjustment_rate_weight, adjustment_rate_bias)

        return capacity.compute()

    def on_train_batch_end(
        self, outputs: dict[str, Any], batch: Any, batch_idx: int
    ) -> None:
        r"""Update the contribution utility and age of units after each training step.

        **Args:**
        - **outputs** (`dict[str, Any]`): the outputs of the training step, which is the returns of the `training_step()` method in the `CLAlgorithm`.
        - **batch** (`Any`): the training data batch.
        - **batch_idx** (`int`): the index of the current batch. This is for the file name of mask figures.
        """

        hidden_features = outputs["hidden_features"]
        mask = outputs["mask"]

        for layer_name in self.backbone.weighted_layer_names:
            # layer-wise operation
            layer = self.backbone.get_layer_by_name(
                layer_name
            )  # get the layer by its name

            # update age
            self.age_t[layer_name] += 1

            # calculate current contribution utility
            current_contribution_utility = (
                (
                    torch.mean(
                        torch.abs(hidden_features[layer_name]),
                        dim=[
                            i
                            for i in range(hidden_features[layer_name].dim())
                            if i != 1
                        ],  # average the features over batch samples
                    )
                    * torch.sum(
                        torch.abs(layer.weight),
                        dim=[
                            i for i in range(layer.weight.dim()) if i != 0
                        ],  # sum over the output dimension
                    )
                )
                * mask[layer_name]
            ).detach()
            current_contribution_utility = min_max_normalise(
                current_contribution_utility
            )  # normalise the utility to [0,1] to avoid linearly increasing utility

            # update utility
            self.contribution_utility_t[layer_name] = (
                self.utility_decay_rate * self.contribution_utility_t[layer_name]
                + current_contribution_utility
            )

    def on_train_end(self) -> None:
        r"""Additionally convert the contribution utility into importance and store (take screenshot of) it as unit importance for previous tasks at the end of a task training."""
        AdaHAT.on_train_end(
            self
        )  # store the mask and update cumulative and summative masks

        for layer_name in self.backbone.weighted_layer_names:
            self.unit_importance_for_previous_tasks[layer_name] += (
                self.contribution_utility_t[layer_name]
            ) / self.age_t[layer_name]
