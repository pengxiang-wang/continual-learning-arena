r"""
The submodule in `cl_algorithms` for SubtleAdaHAT algorithm.
"""

__all__ = ["SubtleAdaHAT"]

import logging
from typing import Any

import torch
from captum.attr import (
    InternalInfluence,
    LayerConductance,
    LayerDeepLift,
    LayerDeepLiftShap,
    LayerFeatureAblation,
    LayerGradCam,
    LayerGradientShap,
    LayerGradientXActivation,
    LayerIntegratedGradients,
    LayerLRP,
)
from torch import Tensor

from clarena.backbones import HATMaskBackbone
from clarena.cl_algorithms import AdaHAT
from clarena.cl_heads import HeadsCIL, HeadsTIL
from clarena.utils import HATNetworkCapacity
from clarena.utils.transforms import min_max_normalise

# always get logger for built-in logging in each module
pylogger = logging.getLogger(__name__)


class SubtleAdaHAT(AdaHAT):
    r"""SubtleAdaHAT algorithm.

    SubtleAdaHAT is what I am working on. It introduces a more unit-wise subtle importance in addition to the AdaHAT importance.

    We implement SubtleAdaHAT as a subclass of AdaHAT algorithm because SubtleAdaHAT adopt the similar idea as AdaHAT.
    """

    def __init__(
        self,
        backbone: HATMaskBackbone,
        heads: HeadsTIL | HeadsCIL,
        adjustment_mode: str,
        adjustment_intensity: float,
        adahat_importance_scaling_factor: float,
        subtle_importance_decay_rate: float,
        s_max: float,
        clamp_threshold: float,
        mask_sparsity_reg_factor: float,
        mask_sparsity_reg_mode: str = "original",
        task_embedding_init_mode: str = "N01",
        epsilon: float = 0.1,
        subtle_importance_type: str | None = None,
        subtle_importance_scaling_factor: float | None = None,
    ) -> None:
        r"""Initialise the SubtleAdaHAT algorithm with the network.

        **Args:**
        - **backbone** (`HATMaskBackbone`): must be a backbone network with HAT mask mechanism.
        - **heads** (`HeadsTIL` | `HeadsCIL`): output heads.
        - **adjustment_mode** (`str`): the strategy of adjustment i.e. the mode of gradient clipping, should be one of the following:
            1. 'subtle_importance_to_adahat' (default): adding a subtle importance to the AdaHAT importance.
        - **adjustment_intensity** (`float`): hyperparameter, control the overall intensity of gradient adjustment. It's the $\alpha$ in equation (9) in [AdaHAT paper](https://link.springer.com/chapter/10.1007/978-3-031-70352-2_9).
        - **subtle_importance_decay_rate** (`float`): the decay rate when accumulating subtle importance. It is the rate at which the utility of a unit decays over time.
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
        - **adahat_importance_scaling_factor** (`float`): the scaling factor for AdaHAT importance. The AdaHAT importance is multiplied by the scaling factor before applying the adjustment rate.
        - **subtle_importance_type** (`str` | `None`): the type of the unit-wise subtle importance added to AdaHAT importance. It applies only when adjustment_mode is 'subtle_importance_to_adahat', should be one of the following:
            1. 'input_weight_abs_sum':
            2. 'output_weight_abs_sum':
            3. 'input_weight_gradient_abs_sum':
            4. 'output_weight_gradient_abs_sum':
            5. 'activation_abs':
            6. 'input_weight_abs_sum_x_activation_abs':
            7. 'output_weight_abs_sum_x_activation_abs':
            8. 'gradient_x_activation_abs':
            9. 'input_weight_gradient_square_sum_x_activation_abs':
            10. 'output_weight_gradient_square_sum_x_activation_abs':
            11. 'conductance_abs':
            12. 'internal_influence_abs':
            13. 'gradcam_abs':
            14. 'deeplift_abs':
            15. 'deepliftshap_abs':
            16. 'gradientshap_abs':
            17. 'integrated_gradients_abs':
            18. 'feature_ablation_abs':
            19. 'lrp_abs':
            20. 'cbp_adaptation':
            21: 'cbp_adaptative_contribution':
        - **subtle_importance_scaling_factor** (`float` | `None`): the scaling factor for the subtle importance. The subtle importance is multiplied by the scaling factor before applying the adjustment rate. It applies only when adjustment_mode is 'subtle_importance_to_adahat'.
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
            adahat_importance_scaling_factor=adahat_importance_scaling_factor,
        )

        self.subtle_importance_type: str | None = subtle_importance_type
        r"""Store the type of the unit-wise subtle importance added to AdaHAT importance. """
        self.subtle_importance_scaling_factor: float | None = (
            subtle_importance_scaling_factor
        )
        r"""Store the scaling factor for the subtle importance. """

        self.subtle_importance_decay_rate: float = subtle_importance_decay_rate
        r"""Store the decay rate when accumulating subtle importance. """

        self.accumulated_subtle_importance: dict[str, Tensor] = {}
        r"""Store the min-max scaled ($[0, 1]$) accumulated unit-wise subtle importance of units. See $U_{l,i}$ in the paper draft. Keys are layer names and values are the utility tensor for the layer. The utility tensor is the same size as the feature tensor with size (number of units). """
        self.age_t: dict[str, Tensor] = {}
        r"""Store the age of units. Keys are layer names and values are the age tensor for the layer for current task. The age tensor is the same size as the feature tensor with size (number of units). """

        self.subtle_importance_for_previous_tasks: dict[str, Tensor] = {}
        r"""Store the unit-wise subtle importance values of units for previous tasks $(1, \cdots, t-1)$. See $I^{(<t)}$ in the paper draft. Keys are layer names and values are the importance tensor for the layer. The importance tensor is the same size as the feature tensor with size (number of units). """

        # set manual optimisation
        self.automatic_optimization = False

        SubtleAdaHAT.sanity_check(self)

    def sanity_check(self) -> None:
        r"""Check the sanity of the arguments.

        **Raises:**
        - **ValueError**: If the subtle importance decay rate is not in the range (0, 1].
        """
        if (
            self.subtle_importance_decay_rate > 1
            or self.subtle_importance_decay_rate <= 0
        ):
            raise ValueError(
                f"The subtle importance decay rate should be in the range (0, 1], but got {self.subtle_importance_decay_rate}."
            )

    def on_train_start(self) -> None:
        r"""Additionally initialise the utility, age and the SubtleAdaHAT unit-wise subtle importance for each layer as zeros."""
        AdaHAT.on_train_start(self)

        for layer_name in self.backbone.weighted_layer_names:
            layer = self.backbone.get_layer_by_name(
                layer_name
            )  # get the layer by its name
            num_units = layer.weight.shape[0]

            # initialise the accumulated subtle importance and age at the beginning of first task
            self.accumulated_subtle_importance[layer_name] = torch.zeros(num_units).to(
                self.device
            )
            self.age_t[layer_name] = torch.zeros(num_units).to(self.device)

            # initialise the unit-wise subtle importance at the beginning of first task. This should not be called in `__init__()` method as the `self.device` is not available at that time.
            if self.task_id == 1:
                self.subtle_importance_for_previous_tasks[layer_name] = torch.zeros(
                    num_units
                ).to(
                    self.device
                )  # the unit-wise subtle importance $I^{(t-1)}$ is initialised as zeros mask ($t = 1$). See the paper draft.

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

        # calculate the adjustment rate for gradients of the parameters, both weights and biases (if exists)
        for layer_name in self.backbone.weighted_layer_names:

            layer = self.backbone.get_layer_by_name(
                layer_name
            )  # get the layer by its name

            # placeholder for the adjustment rate to avoid the error of using it before assignment
            adjustment_rate_weight = 1
            adjustment_rate_bias = 1

            weight_subtle_importance, bias_subtle_importance = (
                self.backbone.get_layer_measure_parameter_wise(
                    unit_wise_measure=self.subtle_importance_for_previous_tasks,
                    layer_name=layer_name,
                    aggregation="min",
                )
            )

            weight_adahat_importance, bias_adahat_importance = (
                self.backbone.get_layer_measure_parameter_wise(
                    unit_wise_measure=self.summative_mask_for_previous_tasks,
                    layer_name=layer_name,
                    aggregation="min",
                )
            )

            # apply the scaling factor to the parameter importance
            weight_subtle_importance = (
                weight_subtle_importance * self.subtle_importance_scaling_factor
            )
            bias_subtle_importance = (
                bias_subtle_importance * self.subtle_importance_scaling_factor
            )
            weight_adahat_importance = (
                weight_adahat_importance * self.adahat_importance_scaling_factor
            )
            bias_adahat_importance = (
                bias_adahat_importance * self.adahat_importance_scaling_factor
            )

            network_sparsity_layer = network_sparsity[layer_name]

            if self.adjustment_mode == "subtle_importance_to_adahat":
                r_layer = self.adjustment_intensity / (
                    self.epsilon + network_sparsity_layer
                )
                adjustment_rate_weight = torch.div(
                    r_layer,
                    (weight_subtle_importance + weight_adahat_importance + r_layer),
                )

                adjustment_rate_bias = torch.div(
                    r_layer, (bias_subtle_importance + bias_adahat_importance + r_layer)
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
        r"""Update the accumulated subtle importance and age of units after each training step.

        **Args:**
        - **outputs** (`dict[str, Any]`): the outputs of the training step, which is the returns of the `training_step()` method in the `CLAlgorithm`.
        - **batch** (`Any`): the training data batch.
        - **batch_idx** (`int`): the index of the current batch. This is for the file name of mask figures.
        """

        # get potential useful information from training batch
        activations = outputs["activations"]
        mask = outputs["mask"]
        input = outputs["input"]
        target = outputs["target"]
        num_batches = self.trainer.num_training_batches

        for layer_name in self.backbone.weighted_layer_names:
            # layer-wise operation

            activation = activations[layer_name]
            m = mask[layer_name]
            num_units = activation.shape[1]

            # calculate unit-wise subtle importance of the training step. See $v_{l,i}$ in the paper draft.
            if self.subtle_importance_type == "input_weight_abs_sum":
                subtle_importance_step = (
                    self.get_subtle_importance_step_layer_weight_abs_sum(
                        layer_name=layer_name,
                        mask=m,
                        if_output_weight=False,
                        reciprocal=False,
                    )
                )
            elif self.subtle_importance_type == "output_weight_abs_sum":
                subtle_importance_step = (
                    self.get_subtle_importance_step_layer_weight_abs_sum(
                        layer_name=layer_name,
                        mask=m,
                        if_output_weight=True,
                        reciprocal=False,
                    )
                )
            elif self.subtle_importance_type == "input_weight_gradient_abs_sum":
                subtle_importance_step = (
                    self.get_subtle_importance_step_layer_weight_gradient_abs_sum(
                        layer_name=layer_name, mask=m, if_output_weight=False
                    )
                )
            elif self.subtle_importance_type == "output_weight_gradient_abs_sum":
                subtle_importance_step = (
                    self.get_subtle_importance_step_layer_weight_gradient_abs_sum(
                        layer_name=layer_name, mask=m, if_output_weight=True
                    )
                )
            elif self.subtle_importance_type == "activation_abs":
                subtle_importance_step = (
                    self.get_subtle_importance_step_layer_activation_abs(
                        activation=activation, mask=m
                    )
                )
            elif self.subtle_importance_type == "input_weight_abs_sum_x_activation_abs":
                subtle_importance_step = self.get_subtle_importance_step_layer_weight_abs_sum_x_activation_abs(
                    layer_name=layer_name,
                    activation=activation,
                    mask=m,
                    if_output_weight=False,
                )
            elif (
                self.subtle_importance_type == "output_weight_abs_sum_x_activation_abs"
            ):
                subtle_importance_step = self.get_subtle_importance_step_layer_weight_abs_sum_x_activation_abs(
                    layer_name=layer_name,
                    activation=activation,
                    mask=m,
                    if_output_weight=True,
                )
            elif self.subtle_importance_type == "gradient_x_activation_abs":
                subtle_importance_step = (
                    self.get_subtle_importance_step_layer_gradient_x_activation_abs(
                        layer_name=layer_name,
                        input=input,
                        target=target,
                        batch_idx=batch_idx,
                        num_batches=num_batches,
                        mask=m,
                    )
                )
            elif (
                self.subtle_importance_type
                == "input_weight_gradient_square_sum_x_activation_abs"
            ):
                subtle_importance_step = self.get_subtle_importance_step_layer_weight_abs_sum_x_activation_abs(
                    layer_name=layer_name,
                    activation=activation,
                    mask=m,
                    if_output_weight=False,
                )
            elif (
                self.subtle_importance_type
                == "output_weight_gradient_square_sum_x_activation_abs"
            ):
                subtle_importance_step = self.get_subtle_importance_step_layer_weight_abs_sum_x_activation_abs(
                    layer_name=layer_name,
                    activation=activation,
                    mask=m,
                    if_output_weight=True,
                )
            elif self.subtle_importance_type == "conductance_abs":
                subtle_importance_step = (
                    self.get_subtle_importance_step_layer_conductance_abs(
                        layer_name=layer_name,
                        input=input,
                        baselines=None,
                        target=target,
                        batch_idx=batch_idx,
                        num_batches=num_batches,
                        mask=m,
                    )
                )
            elif self.subtle_importance_type == "internal_influence_abs":
                subtle_importance_step = (
                    self.get_subtle_importance_step_layer_internal_influence_abs(
                        layer_name=layer_name,
                        input=input,
                        baselines=None,
                        target=target,
                        batch_idx=batch_idx,
                        num_batches=num_batches,
                        mask=m,
                    )
                )
            elif self.subtle_importance_type == "gradcam_abs":
                subtle_importance_step = (
                    self.get_subtle_importance_step_layer_gradcam_abs(
                        layer_name=layer_name,
                        input=input,
                        target=target,
                        batch_idx=batch_idx,
                        num_batches=num_batches,
                        mask=m,
                    )
                )
            elif self.subtle_importance_type == "deeplift_abs":
                subtle_importance_step = (
                    self.get_subtle_importance_step_layer_deeplift_abs(
                        layer_name=layer_name,
                        input=input,
                        baselines=None,
                        target=target,
                        batch_idx=batch_idx,
                        num_batches=num_batches,
                        mask=m,
                    )
                )
            elif self.subtle_importance_type == "deepliftshap_abs":
                subtle_importance_step = (
                    self.get_subtle_importance_step_layer_deepliftshap_abs(
                        layer_name=layer_name,
                        input=input,
                        baselines=None,
                        target=target,
                        batch_idx=batch_idx,
                        num_batches=num_batches,
                        mask=m,
                    )
                )
            elif self.subtle_importance_type == "gradientshap_abs":
                subtle_importance_step = (
                    self.get_subtle_importance_step_layer_gradientshap_abs(
                        layer_name=layer_name,
                        input=input,
                        baselines=None,
                        target=target,
                        batch_idx=batch_idx,
                        num_batches=num_batches,
                        mask=m,
                    )
                )
            elif self.subtle_importance_type == "integrated_gradients_abs":
                subtle_importance_step = (
                    self.get_subtle_importance_step_layer_integrated_gradients_abs(
                        layer_name=layer_name,
                        input=input,
                        baselines=None,
                        target=target,
                        batch_idx=batch_idx,
                        num_batches=num_batches,
                        mask=m,
                    )
                )
            elif self.subtle_importance_type == "feature_ablation_abs":
                subtle_importance_step = (
                    self.get_subtle_importance_step_layer_feature_ablation_abs(
                        layer_name=layer_name,
                        input=input,
                        layer_baselines=None,
                        target=target,
                        batch_idx=batch_idx,
                        num_batches=num_batches,
                        mask=m,
                    )
                )
            elif self.subtle_importance_type == "lrp_abs":
                subtle_importance_step = self.get_subtle_importance_step_layer_lrp_abs(
                    layer_name=layer_name,
                    input=input,
                    target=target,
                    batch_idx=batch_idx,
                    num_batches=num_batches,
                    mask=m,
                )
            elif self.subtle_importance_type == "cbp_adaptation":
                subtle_importance_step = (
                    self.get_subtle_importance_step_layer_weight_abs_sum(
                        layer_name=layer_name,
                        mask=m,
                        if_output_weight=False,
                        reciprocal=True,
                    )
                )
            elif self.subtle_importance_type == "cbp_adaptive_contribution":
                subtle_importance_step = (
                    self.get_subtle_importance_step_layer_cbp_adaptive_contribution(
                        layer_name=layer_name,
                        activation=activation,
                        mask=m,
                    )
                )
            print(subtle_importance_step)

            subtle_importance_step = min_max_normalise(
                subtle_importance_step
            )  # min-max scaling the utility to [0,1]. See in the paper draft.

            print(subtle_importance_step)

            # update accumulated subtle importance
            self.accumulated_subtle_importance[layer_name] = (
                self.subtle_importance_decay_rate
                * self.accumulated_subtle_importance[layer_name]
                + subtle_importance_step
            )

            # update age
            self.age_t[layer_name] += 1

    def on_train_end(self) -> None:
        r"""Additionally store (take screenshot of) it as unit-wise subtle importance for previous tasks at the end of a task training."""
        AdaHAT.on_train_end(
            self
        )  # store the mask and update cumulative and summative masks

        for layer_name in self.backbone.weighted_layer_names:
            self.subtle_importance_for_previous_tasks[layer_name] += (
                self.accumulated_subtle_importance[layer_name]
            ) / self.age_t[layer_name]

    def get_subtle_importance_step_layer_weight_abs_sum(
        self: str,
        layer_name: str,
        mask: Tensor,
        if_output_weight: bool,
        reciprocal: bool,
    ) -> Tensor:
        r"""Get the raw unit-wise subtle importance (before scaling) of a layer of a training step. See $v_l$ in the paper draft. This method uses the sum of absolute values of layer input or output weights.

        **Args:**
        - **layer_name** (`str`): the name of layer to get unit-wise subtle importance.
        - **mask** (`Tensor`): the mask tensor of the layer. It has the same size as the feature tensor with size (number of units).
        - **if_output_weight** (`bool`): whether to use the output weights or input weights.
        - **reciprocal** (`bool`): whether to take reciprocal.

        **Returns:**
        - **subtle_importance_step_layer** (`Tensor`): the unit-wise subtle importance of the layer of the training step.
        """
        layer = self.backbone.get_layer_by_name(layer_name)

        if not if_output_weight:
            weight_abs = torch.abs(layer.weight.data)
            weight_abs_sum = torch.sum(
                weight_abs,
                dim=[
                    i for i in range(weight_abs.dim()) if i != 0
                ],  # sum over the input dimension
            )
        else:
            weight_abs = torch.abs(self.next_layer(layer_name).weight.data)
            weight_abs_sum = torch.sum(
                weight_abs,
                dim=[
                    i for i in range(weight_abs.dim()) if i != 1
                ],  # sum over the output dimension
            )

        if reciprocal:
            weight_abs_sum_reciprocal = torch.reciprocal(weight_abs_sum)
            subtle_importance_step_layer = weight_abs_sum_reciprocal * mask
        else:
            subtle_importance_step_layer = weight_abs_sum * mask
        subtle_importance_step_layer = subtle_importance_step_layer.detach()

        return subtle_importance_step_layer

    def get_subtle_importance_step_layer_weight_gradient_abs_sum(
        self: str,
        layer_name: str,
        mask: Tensor,
        if_output_weight: bool,
    ) -> Tensor:
        r"""Get the raw unit-wise subtle importance (before scaling) of a layer of a training step. See $v_l$ in the paper draft. This method uses the sum of absolute values of gradients of the layer input or output weights.

        **Args:**
        - **layer_name** (`str`): the name of layer to get unit-wise subtle importance.
        - **mask** (`Tensor`): the mask tensor of the layer. It has the same size as the feature tensor with size (number of units).
        - **if_output_weight** (`bool`): whether to use the output weights or input weights.

        **Returns:**
        - **subtle_importance_step_layer** (`Tensor`): the unit-wise subtle importance of the layer of the training step.
        """
        layer = self.backbone.get_layer_by_name(layer_name)

        if not if_output_weight:
            gradient_abs = torch.abs(layer.weight.grad.data)
            gradient_abs_sum = torch.sum(
                gradient_abs,
                dim=[
                    i for i in range(gradient_abs.dim()) if i != 0
                ],  # sum over the input dimension
            )
        else:
            gradient_abs = torch.abs(self.next_layer(layer_name).weight.grad.data)
            gradient_abs_sum = torch.sum(
                gradient_abs,
                dim=[
                    i for i in range(gradient_abs.dim()) if i != 1
                ],  # sum over the output dimension
            )

        subtle_importance_step_layer = gradient_abs_sum * mask
        subtle_importance_step_layer = subtle_importance_step_layer.detach()

        return subtle_importance_step_layer

    def get_subtle_importance_step_layer_activation_abs(
        self: str,
        activation: Tensor,
        mask: Tensor,
    ) -> Tensor:
        r"""Get the raw unit-wise subtle importance (before scaling) of a layer of a training step. See $v_l$ in the paper draft. This method uses the absolute value of activation of the layer. This is our own implementation of [Layer Activation](https://captum.ai/api/layer.html#layer-activation) in Captum.

        **Args:**
        - **activation** (`Tensor`): the activation tensor of the layer. It has the same size of (number of units).
        - **mask** (`Tensor`): the mask tensor of the layer. It has the same size as the feature tensor with size (number of units).

        **Returns:**
        - **subtle_importance_step_layer** (`Tensor`): the unit-wise subtle importance of the layer of the training step.
        """
        activation_abs_batch_mean = torch.mean(
            torch.abs(activation),
            dim=[
                i for i in range(activation.dim()) if i != 1
            ],  # average the features over batch samples
        )
        subtle_importance_step_layer = activation_abs_batch_mean * mask
        subtle_importance_step_layer = subtle_importance_step_layer.detach()

        return subtle_importance_step_layer

    def get_subtle_importance_step_layer_weight_abs_sum_x_activation_abs(
        self: str,
        layer_name: str,
        activation: Tensor,
        mask: Tensor,
        if_output_weight: bool,
    ) -> Tensor:
        r"""Get the raw unit-wise subtle importance (before scaling) of a layer of a training step. See $v_l$ in the paper draft. This method uses the sum of absolute values of layer input / output weights multiplied by absolute values of activation. The input weights version is equal to the contribution utility in [CBP](https://www.nature.com/articles/s41586-024-07711-7).

        **Args:**
        - **layer_name** (`str`): the name of layer to get unit-wise subtle importance.
        - **activation** (`Tensor`): the activation tensor of the layer. It has the same size of (number of units).
        - **mask** (`Tensor`): the mask tensor of the layer. It has the same size as the feature tensor with size (number of units).
        - **if_output_weight** (`bool`): whether to use the output weights or input weights.

        **Returns:**
        - **subtle_importance_step_layer** (`Tensor`): the unit-wise subtle importance of the layer of the training step.
        """
        layer = self.backbone.get_layer_by_name(layer_name)

        if not if_output_weight:
            weight_abs = torch.abs(layer.weight.data)
            weight_abs_sum = torch.sum(
                weight_abs,
                dim=[
                    i for i in range(weight_abs.dim()) if i != 0
                ],  # sum over the input dimension
            )
        else:
            weight_abs = torch.abs(self.next_layer(layer_name).weight.data)
            weight_abs_sum = torch.sum(
                weight_abs,
                dim=[
                    i for i in range(weight_abs.dim()) if i != 1
                ],  # sum over the output dimension
            )

        activation_abs_batch_mean = torch.mean(
            torch.abs(activation),
            dim=[
                i for i in range(activation.dim()) if i != 1
            ],  # average the features over batch samples
        )

        subtle_importance_step_layer = weight_abs_sum * activation_abs_batch_mean * mask
        subtle_importance_step_layer = subtle_importance_step_layer.detach()

        return subtle_importance_step_layer

    def get_subtle_importance_step_layer_gradient_x_activation_abs(
        self: str,
        layer_name: str,
        input: Tensor | tuple[Tensor, ...],
        target: Tensor | None,
        batch_idx: int,
        num_batches: int,
        mask: Tensor,
    ) -> Tensor:
        r"""Get the raw unit-wise subtle importance (before scaling) of a layer of a training step. See $v_l$ in the paper draft. This method uses the absolute values of the gradient of layer activation multiplied by the activation. We implement this using [Layer Gradient X Activation](https://captum.ai/api/layer.html#layer-gradient-x-activation) in Captum.

        **Args:**
        - **layer_name** (`str`): the name of layer to get unit-wise subtle importance.
        - **input** (`Tensor` | `tuple[Tensor, ...]`): the input batch of the training step.
        - **target** (`Tensor` | `None`): the target batch of the training step.
        - **batch_idx** (`int`): the index of the current batch. This is an argument of the forward function during training.
        - **num_batches** (`int`): the number of batches in the training step. This is an argument of the forward function during training.
        - **mask** (`Tensor`): the mask tensor of the layer. It has the same size as the feature tensor with size (number of units).

        **Returns:**
        - **subtle_importance_step_layer** (`Tensor`): the unit-wise subtle importance of the layer of the training step.
        """
        layer = self.backbone.get_layer_by_name(layer_name)

        input = input.requires_grad_()

        # initialise the Layer Gradient X Activation object
        layer_gradient_x_activation = LayerGradientXActivation(
            forward_func=self.forward, layer=layer
        )

        self.set_forward_func_return_logits_only(True)
        # calculate layer attribution of the step
        attribution = layer_gradient_x_activation.attribute(
            inputs=input,
            target=target,
            additional_forward_args=("train", batch_idx, num_batches, self.task_id),
        )
        self.set_forward_func_return_logits_only(False)

        attribution_abs_step_layer = torch.abs(attribution)

        subtle_importance_step_layer = torch.mean(
            attribution_abs_step_layer,
            dim=[
                i for i in range(attribution.dim()) if i != 1
            ],  # average the features over batch samples
        )

        subtle_importance_step_layer = subtle_importance_step_layer * mask
        subtle_importance_step_layer = subtle_importance_step_layer.detach()

        return subtle_importance_step_layer

    def get_subtle_importance_step_layer_weight_gradient_square_sum_x_activation_abs(
        self: str,
        layer_name: str,
        activation: Tensor,
        mask: Tensor,
        if_output_weight: bool,
    ) -> Tensor:
        r"""Get the raw unit-wise subtle importance (before scaling) of a layer of a training step. See $v_l$ in the paper draft. This method uses the sum of layer weight gradient squares multiplied by absolute values of activation. The weight gradient square is equal to fisher information in [EWC](https://www.pnas.org/doi/10.1073/pnas.1611835114).

        **Args:**
        - **layer_name** (`str`): the name of layer to get unit-wise subtle importance.
        - **activation** (`Tensor`): the activation tensor of the layer. It has the same size of (number of units).
        - **mask** (`Tensor`): the mask tensor of the layer. It has the same size as the feature tensor with size (number of units).
        - **if_output_weight** (`bool`): whether to use the output weights or input weights.

        **Returns:**
        - **subtle_importance_step_layer** (`Tensor`): the unit-wise subtle importance of the layer of the training step.
        """
        layer = self.backbone.get_layer_by_name(layer_name)

        if not if_output_weight:
            gradient_square = layer.weight.grad.data**2
            gradient_square_sum = torch.sum(
                gradient_square,
                dim=[
                    i for i in range(gradient_square.dim()) if i != 0
                ],  # sum over the input dimension
            )
        else:
            gradient_square = self.next_layer(layer_name).weight.grad.data**2
            gradient_square_sum = torch.sum(
                gradient_square,
                dim=[
                    i for i in range(gradient_square.dim()) if i != 1
                ],  # sum over the output dimension
            )

        activation_abs_batch_mean = torch.mean(
            torch.abs(activation),
            dim=[
                i for i in range(activation.dim()) if i != 1
            ],  # average the features over batch samples
        )

        subtle_importance_step_layer = (
            gradient_square_sum * activation_abs_batch_mean * mask
        )
        subtle_importance_step_layer = subtle_importance_step_layer.detach()

        return subtle_importance_step_layer

    def get_subtle_importance_step_layer_conductance_abs(
        self: str,
        layer_name: str,
        input: Tensor | tuple[Tensor, ...],
        baselines: None | int | float | Tensor | tuple[int | float | Tensor, ...],
        target: Tensor | None,
        batch_idx: int,
        num_batches: int,
        mask: Tensor,
    ) -> Tensor:
        r"""Get the raw unit-wise subtle importance (before scaling) of a layer of a training step. See $v_l$ in the paper draft. This method uses the absolute values of [conductance](https://openreview.net/forum?id=SylKoo0cKm). We implement this using [Layer Conductance](https://captum.ai/api/layer.html#layer-conductance) in Captum.

        **Args:**
        - **layer_name** (`str`): the name of layer to get unit-wise subtle importance.
        - **input** (`Tensor` | `tuple[Tensor, ...]`): the input batch of the training step.
        - **baselines** (`None` | `int` | `float` | `Tensor` | `tuple[int | float | Tensor, ...]`): starting point from which integral is computed in this method. Please refer to the [Captum documentation](https://captum.ai/api/layer.html#captum.attr.LayerConductance.attribute) for more details.
        - **target** (`Tensor` | `None`): the target batch of the training step.
        - **batch_idx** (`int`): the index of the current batch. This is an argument of the forward function during training.
        - **num_batches** (`int`): the number of batches in the training step. This is an argument of the forward function during training.- **mask** (`Tensor`): the mask tensor of the layer. It has the same size as the feature tensor with size (number of units).

        **Returns:**
        - **subtle_importance_step_layer** (`Tensor`): the unit-wise subtle importance of the layer of the training step.
        """
        layer = self.backbone.get_layer_by_name(layer_name)

        # initialise the Layer Conductance object
        layer_conductance = LayerConductance(forward_func=self.forward, layer=layer)

        self.set_forward_func_return_logits_only(True)
        # calculate layer attribution of the step
        attribution = layer_conductance.attribute(
            inputs=input,
            baselines=baselines,
            target=target,
            additional_forward_args=("train", batch_idx, num_batches, self.task_id),
        )
        self.set_forward_func_return_logits_only(False)

        attribution_abs_batch_mean = torch.mean(
            torch.abs(attribution),
            dim=[
                i for i in range(attribution.dim()) if i != 1
            ],  # average the features over batch samples
        )

        subtle_importance_step_layer = attribution_abs_batch_mean * mask
        subtle_importance_step_layer = subtle_importance_step_layer.detach()

        return subtle_importance_step_layer

    def get_subtle_importance_step_layer_internal_influence_abs(
        self: str,
        layer_name: str,
        input: Tensor | tuple[Tensor, ...],
        baselines: None | int | float | Tensor | tuple[int | float | Tensor, ...],
        target: Tensor | None,
        batch_idx: int,
        num_batches: int,
        mask: Tensor,
    ) -> Tensor:
        r"""Get the raw unit-wise subtle importance (before scaling) of a layer of a training step. See $v_l$ in the paper draft. This method uses the absolute values of [internal influence](https://openreview.net/forum?id=SJPpHzW0-). We implement this using [Internal Influence](https://captum.ai/api/layer.html#internal-influence) in Captum.

        **Args:**
        - **layer_name** (`str`): the name of layer to get unit-wise subtle importance.
        - **input** (`Tensor` | `tuple[Tensor, ...]`): the input batch of the training step.
        - **baselines** (`None` | `int` | `float` | `Tensor` | `tuple[int | float | Tensor, ...]`): starting point from which integral is computed in this method. Please refer to the [Captum documentation](https://captum.ai/api/layer.html#captum.attr.InternalInfluence.attribute) for more details.
        - **target** (`Tensor` | `None`): the target batch of the training step.
        - **batch_idx** (`int`): the index of the current batch. This is an argument of the forward function during training.
        - **num_batches** (`int`): the number of batches in the training step. This is an argument of the forward function during training.
        - **mask** (`Tensor`): the mask tensor of the layer. It has the same size as the feature tensor with size (number of units).

        **Returns:**
        - **subtle_importance_step_layer** (`Tensor`): the unit-wise subtle importance of the layer of the training step.
        """
        layer = self.backbone.get_layer_by_name(layer_name)

        # initialise the Internal Influence object
        internal_influence = InternalInfluence(forward_func=self.forward, layer=layer)

        self.set_forward_func_return_logits_only(True)
        # calculate layer attribution of the step
        attribution = internal_influence.attribute(
            inputs=input,
            baselines=baselines,
            target=target,
            additional_forward_args=("train", batch_idx, num_batches, self.task_id),
        )
        self.set_forward_func_return_logits_only(False)

        attribution_abs_batch_mean = torch.mean(
            torch.abs(attribution),
            dim=[
                i for i in range(attribution.dim()) if i != 1
            ],  # average the features over batch samples
        )

        subtle_importance_step_layer = attribution_abs_batch_mean * mask
        subtle_importance_step_layer = subtle_importance_step_layer.detach()

        return subtle_importance_step_layer

    def get_subtle_importance_step_layer_gradcam_abs(
        self: str,
        layer_name: str,
        input: Tensor | tuple[Tensor, ...],
        target: Tensor | None,
        batch_idx: int,
        num_batches: int,
        mask: Tensor,
    ) -> Tensor:
        r"""Get the raw unit-wise subtle importance (before scaling) of a layer of a training step. See $v_l$ in the paper draft. This method uses the absolute values of [Grad-CAM](https://openreview.net/forum?id=SJPpHzW0-). We implement this using [Layer Grad-CAM](https://captum.ai/api/layer.html#gradcam) in Captum.

        **Args:**
        - **layer_name** (`str`): the name of layer to get unit-wise subtle importance.
        - **input** (`Tensor` | `tuple[Tensor, ...]`): the input batch of the training step.
        - **target** (`Tensor` | `None`): the target batch of the training step.
        - **batch_idx** (`int`): the index of the current batch. This is an argument of the forward function during training.
        - **num_batches** (`int`): the number of batches in the training step. This is an argument of the forward function during training.
        - **mask** (`Tensor`): the mask tensor of the layer. It has the same size as the feature tensor with size (number of units).

        **Returns:**
        - **subtle_importance_step_layer** (`Tensor`): the unit-wise subtle importance of the layer of the training step.
        """
        layer = self.backbone.get_layer_by_name(layer_name)

        # initialise the GradCAM object
        gradcam = LayerGradCam(forward_func=self.forward, layer=layer)

        self.set_forward_func_return_logits_only(True)
        # calculate layer attribution of the step
        attribution = gradcam.attribute(
            inputs=input,
            target=target,
            additional_forward_args=("train", batch_idx, num_batches, self.task_id),
        )
        self.set_forward_func_return_logits_only(False)

        attribution_abs_batch_mean = torch.mean(
            torch.abs(attribution),
            dim=[
                i for i in range(attribution.dim()) if i != 1
            ],  # average the features over batch samples
        )

        subtle_importance_step_layer = attribution_abs_batch_mean * mask
        subtle_importance_step_layer = subtle_importance_step_layer.detach()

        return subtle_importance_step_layer

    def get_subtle_importance_step_layer_deeplift_abs(
        self: str,
        layer_name: str,
        input: Tensor | tuple[Tensor, ...],
        baselines: None | int | float | Tensor | tuple[int | float | Tensor, ...],
        target: Tensor | None,
        batch_idx: int,
        num_batches: int,
        mask: Tensor,
    ) -> Tensor:
        r"""Get the raw unit-wise subtle importance (before scaling) of a layer of a training step. See $v_l$ in the paper draft. This method uses the absolute values of [DeepLift](https://proceedings.mlr.press/v70/shrikumar17a/shrikumar17a.pdf). We implement this using [Layer DeepLift](https://captum.ai/api/layer.html#layer-deeplift) in Captum.

        **Args:**
        - **layer_name** (`str`): the name of layer to get unit-wise subtle importance.
        - **input** (`Tensor` | `tuple[Tensor, ...]`): the input batch of the training step.
        - **baselines** (`None` | `int` | `float` | `Tensor` | `tuple[int | float | Tensor, ...]`): baselines define reference samples that are compared with the inputs. Please refer to the [Captum documentation](https://captum.ai/api/layer.html#captum.attr.LayerDeepLift.attribute) for more details.
        - **target** (`Tensor` | `None`): the target batch of the training step.
        - **batch_idx** (`int`): the index of the current batch. This is an argument of the forward function during training.
        - **num_batches** (`int`): the number of batches in the training step. This is an argument of the forward function during training.
        - **mask** (`Tensor`): the mask tensor of the layer. It has the same size as the feature tensor with size (number of units).

        **Returns:**
        - **subtle_importance_step_layer** (`Tensor`): the unit-wise subtle importance of the layer of the training step.
        """
        layer = self.backbone.get_layer_by_name(layer_name)

        # initialise the Layer DeepLift object
        layer_deeplift = LayerDeepLift(model=self, layer=layer)

        self.set_forward_func_return_logits_only(True)
        # calculate layer attribution of the step
        attribution = layer_deeplift.attribute(
            inputs=input,
            baselines=baselines,
            target=target,
            additional_forward_args=("train", batch_idx, num_batches, self.task_id),
        )
        self.set_forward_func_return_logits_only(False)

        attribution_abs_batch_mean = torch.mean(
            torch.abs(attribution),
            dim=[
                i for i in range(attribution.dim()) if i != 1
            ],  # average the features over batch samples
        )

        subtle_importance_step_layer = attribution_abs_batch_mean * mask
        subtle_importance_step_layer = subtle_importance_step_layer.detach()

        return subtle_importance_step_layer

    def get_subtle_importance_step_layer_deepliftshap_abs(
        self: str,
        layer_name: str,
        input: Tensor | tuple[Tensor, ...],
        baselines: None | int | float | Tensor | tuple[int | float | Tensor, ...],
        target: Tensor | None,
        batch_idx: int,
        num_batches: int,
        mask: Tensor,
    ) -> Tensor:
        r"""Get the raw unit-wise subtle importance (before scaling) of a layer of a training step. See $v_l$ in the paper draft. This method uses the absolute values of [DeepLift SHAP](https://proceedings.neurips.cc/paper_files/paper/2017/file/8a20a8621978632d76c43dfd28b67767-Paper.pdf). We implement this using [Layer DeepLiftShap](https://captum.ai/api/layer.html#layer-deepliftshap) in Captum.

        **Args:**
        - **layer_name** (`str`): the name of layer to get unit-wise subtle importance.
        - **input** (`Tensor` | `tuple[Tensor, ...]`): the input batch of the training step.
        - **baselines** (`None` | `int` | `float` | `Tensor` | `tuple[int | float | Tensor, ...]`): baselines define reference samples that are compared with the inputs. Please refer to the [Captum documentation](https://captum.ai/api/layer.html#captum.attr.LayerDeepLiftShap.attribute) for more details.
        - **target** (`Tensor` | `None`): the target batch of the training step.
        - **batch_idx** (`int`): the index of the current batch. This is an argument of the forward function during training.
        - **num_batches** (`int`): the number of batches in the training step. This is an argument of the forward function during training.
        - **mask** (`Tensor`): the mask tensor of the layer. It has the same size as the feature tensor with size (number of units).

        **Returns:**
        - **subtle_importance_step_layer** (`Tensor`): the unit-wise subtle importance of the layer of the training step.
        """
        layer = self.backbone.get_layer_by_name(layer_name)

        # initialise the Layer DeepLiftShap object
        layer_deepliftshap = LayerDeepLiftShap(model=self, layer=layer)

        self.set_forward_func_return_logits_only(True)
        # calculate layer attribution of the step
        attribution = layer_deepliftshap.attribute(
            inputs=input,
            baselines=baselines,
            target=target,
            additional_forward_args=("train", batch_idx, num_batches, self.task_id),
        )
        self.set_forward_func_return_logits_only(False)

        attribution_abs_batch_mean = torch.mean(
            torch.abs(attribution),
            dim=[
                i for i in range(attribution.dim()) if i != 1
            ],  # average the features over batch samples
        )

        subtle_importance_step_layer = attribution_abs_batch_mean * mask
        subtle_importance_step_layer = subtle_importance_step_layer.detach()

        return subtle_importance_step_layer

    def get_subtle_importance_step_layer_gradientshap_abs(
        self: str,
        layer_name: str,
        input: Tensor | tuple[Tensor, ...],
        baselines: None | int | float | Tensor | tuple[int | float | Tensor, ...],
        target: Tensor | None,
        batch_idx: int,
        num_batches: int,
        mask: Tensor,
    ) -> Tensor:
        r"""Get the raw unit-wise subtle importance (before scaling) of a layer of a training step. See $v_l$ in the paper draft. This method uses the absolute values of gradient SHAP. We implement this using [Layer GradientShap](https://captum.ai/api/layer.html#layer-gradientshap) in Captum.

        **Args:**
        - **layer_name** (`str`): the name of layer to get unit-wise subtle importance.
        - **input** (`Tensor` | `tuple[Tensor, ...]`): the input batch of the training step.
        - **baselines** (`None` | `int` | `float` | `Tensor` | `tuple[int | float | Tensor, ...]`): starting point from which expectation is computed. Please refer to the [Captum documentation](https://captum.ai/api/layer.html#captum.attr.LayerGradientShap.attribute) for more details.
        - **target** (`Tensor` | `None`): the target batch of the training step.
        - **batch_idx** (`int`): the index of the current batch. This is an argument of the forward function during training.
        - **num_batches** (`int`): the number of batches in the training step. This is an argument of the forward function during training.
        - **mask** (`Tensor`): the mask tensor of the layer. It has the same size as the feature tensor with size (number of units).

        **Returns:**
        - **subtle_importance_step_layer** (`Tensor`): the unit-wise subtle importance of the layer of the training step.
        """
        layer = self.backbone.get_layer_by_name(layer_name)

        # initialise the Layer GradientShap object
        layer_gradientshap = LayerGradientShap(forward_func=self.forward, layer=layer)

        self.set_forward_func_return_logits_only(True)
        # calculate layer attribution of the step
        attribution = layer_gradientshap.attribute(
            inputs=input,
            baselines=baselines,
            target=target,
            additional_forward_args=("train", batch_idx, num_batches, self.task_id),
        )
        self.set_forward_func_return_logits_only(False)

        attribution_abs_batch_mean = torch.mean(
            torch.abs(attribution),
            dim=[
                i for i in range(attribution.dim()) if i != 1
            ],  # average the features over batch samples
        )

        subtle_importance_step_layer = attribution_abs_batch_mean * mask
        subtle_importance_step_layer = subtle_importance_step_layer.detach()

        return subtle_importance_step_layer

    def get_subtle_importance_step_layer_integrated_gradients_abs(
        self: str,
        layer_name: str,
        input: Tensor | tuple[Tensor, ...],
        baselines: None | int | float | Tensor | tuple[int | float | Tensor, ...],
        target: Tensor | None,
        batch_idx: int,
        num_batches: int,
        mask: Tensor,
    ) -> Tensor:
        r"""Get the raw unit-wise subtle importance (before scaling) of a layer of a training step. See $v_l$ in the paper draft. This method uses the absolute values of [integrated gradients](https://proceedings.mlr.press/v70/sundararajan17a/sundararajan17a.pdf). We implement this using [Layer Integrated Gradients](https://captum.ai/api/layer.html#layer-integrated-gradients) in Captum.

        **Args:**
        - **layer_name** (`str`): the name of layer to get unit-wise subtle importance.
        - **input** (`Tensor` | `tuple[Tensor, ...]`): the input batch of the training step.
        - **baselines** (`None` | `int` | `float` | `Tensor` | `tuple[int | float | Tensor, ...]`): starting point from which integral is computed. Please refer to the [Captum documentation](https://captum.ai/api/layer.html#captum.attr.LayerIntegratedGradients.attribute) for more details.
        - **target** (`Tensor` | `None`): the target batch of the training step.
        - **batch_idx** (`int`): the index of the current batch. This is an argument of the forward function during training.
        - **num_batches** (`int`): the number of batches in the training step. This is an argument of the forward function during training.
        - **mask** (`Tensor`): the mask tensor of the layer. It has the same size as the feature tensor with size (number of units).

        **Returns:**
        - **subtle_importance_step_layer** (`Tensor`): the unit-wise subtle importance of the layer of the training step.
        """
        layer = self.backbone.get_layer_by_name(layer_name)

        # initialise the Layer Integrated Gradients object
        layer_integrated_gradients = LayerIntegratedGradients(
            forward_func=self.forward, layer=layer
        )

        self.set_forward_func_return_logits_only(True)
        # calculate layer attribution of the step
        attribution = layer_integrated_gradients.attribute(
            inputs=input,
            baselines=baselines,
            target=target,
            additional_forward_args=("train", batch_idx, num_batches, self.task_id),
        )
        self.set_forward_func_return_logits_only(False)

        attribution_abs_batch_mean = torch.mean(
            torch.abs(attribution),
            dim=[
                i for i in range(attribution.dim()) if i != 1
            ],  # average the features over batch samples
        )

        subtle_importance_step_layer = attribution_abs_batch_mean * mask
        subtle_importance_step_layer = subtle_importance_step_layer.detach()

        return subtle_importance_step_layer

    def get_subtle_importance_step_layer_feature_ablation_abs(
        self: str,
        layer_name: str,
        input: Tensor | tuple[Tensor, ...],
        layer_baselines: None | int | float | Tensor | tuple[int | float | Tensor, ...],
        target: Tensor | None,
        batch_idx: int,
        num_batches: int,
        mask: Tensor,
    ) -> Tensor:
        r"""Get the raw unit-wise subtle importance (before scaling) of a layer of a training step. See $v_l$ in the paper draft. This method uses the absolute values of [feature ablation](https://link.springer.com/chapter/10.1007/978-3-319-10590-1_53) attribution. We implement this using [Layer Feature Ablation](https://captum.ai/api/layer.html#layer-feature-ablation) in Captum.

        **Args:**
        - **layer_name** (`str`): the name of layer to get unit-wise subtle importance.
        - **input** (`Tensor` | `tuple[Tensor, ...]`): the input batch of the training step.
        - **layer_baselines** (`None` | `int` | `float` | `Tensor` | `tuple[int | float | Tensor, ...]`): reference values which replace each layer input / output value when ablated. Please refer to the [Captum documentation](https://captum.ai/api/layer.html#captum.attr.LayerFeatureAblation.attribute) for more details.
        - **target** (`Tensor` | `None`): the target batch of the training step.
        - **batch_idx** (`int`): the index of the current batch. This is an argument of the forward function during training.
        - **num_batches** (`int`): the number of batches in the training step. This is an argument of the forward function during training.
        - **mask** (`Tensor`): the mask tensor of the layer. It has the same size as the feature tensor with size (number of units).

        **Returns:**
        - **subtle_importance_step_layer** (`Tensor`): the unit-wise subtle importance of the layer of the training step.
        """
        layer = self.backbone.get_layer_by_name(layer_name)

        # initialise the Layer Feature Ablation object
        layer_feature_ablation = LayerFeatureAblation(
            forward_func=self.forward, layer=layer
        )

        self.set_forward_func_return_logits_only(True)
        # calculate layer attribution of the step
        attribution = layer_feature_ablation.attribute(
            inputs=input,
            layer_baselines=layer_baselines,
            target=target,
            additional_forward_args=("train", batch_idx, num_batches, self.task_id),
        )
        self.set_forward_func_return_logits_only(False)

        attribution_abs_batch_mean = torch.mean(
            torch.abs(attribution),
            dim=[
                i for i in range(attribution.dim()) if i != 1
            ],  # average the features over batch samples
        )

        subtle_importance_step_layer = attribution_abs_batch_mean * mask
        subtle_importance_step_layer = subtle_importance_step_layer.detach()

        return subtle_importance_step_layer

    def get_subtle_importance_step_layer_lrp_abs(
        self: str,
        layer_name: str,
        input: Tensor | tuple[Tensor, ...],
        target: Tensor | None,
        batch_idx: int,
        num_batches: int,
        mask: Tensor,
    ) -> Tensor:
        r"""Get the raw unit-wise subtle importance (before scaling) of a layer of a training step. See $v_l$ in the paper draft. This method uses the absolute values of [LRP](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0130140). We implement this using [Layer LRP](https://captum.ai/api/layer.html#layer-lrp) in Captum.

        **Args:**
        - **layer_name** (`str`): the name of layer to get unit-wise subtle importance.
        - **input** (`Tensor` | `tuple[Tensor, ...]`): the input batch of the training step.
        - **target** (`Tensor` | `None`): the target batch of the training step.
        - **batch_idx** (`int`): the index of the current batch. This is an argument of the forward function during training.
        - **num_batches** (`int`): the number of batches in the training step. This is an argument of the forward function during training.
        - **mask** (`Tensor`): the mask tensor of the layer. It has the same size as the feature tensor with size (number of units).

        **Returns:**
        - **subtle_importance_step_layer** (`Tensor`): the unit-wise subtle importance of the layer of the training step.
        """
        layer = self.backbone.get_layer_by_name(layer_name)

        # initialise the Layer DeepLift object
        layer_lrp = LayerLRP(model=self, layer=layer)

        # set model to evaluation mode to prevent updating the model parameters
        self.eval()

        self.set_forward_func_return_logits_only(True)
        # calculate layer attribution of the step
        attribution = layer_lrp.attribute(
            inputs=input,
            target=target,
            additional_forward_args=("train", batch_idx, num_batches, self.task_id),
        )
        self.set_forward_func_return_logits_only(False)

        attribution_abs_batch_mean = torch.mean(
            torch.abs(attribution),
            dim=[
                i for i in range(attribution.dim()) if i != 1
            ],  # average the features over batch samples
        )

        subtle_importance_step_layer = attribution_abs_batch_mean * mask
        subtle_importance_step_layer = subtle_importance_step_layer.detach()

        return subtle_importance_step_layer

    def get_subtle_importance_step_layer_cbp_adaptive_contribution(
        self: str,
        layer_name: str,
        activation: Tensor,
        mask: Tensor,
    ) -> Tensor:
        r"""Get the raw unit-wise subtle importance (before scaling) of a layer of a training step. See $v_l$ in the paper draft. This method uses the sum of absolute values of layer output weights multiplied by absolute values of activation, then divided by the reciprocal of sum of absolute values of layer input weights. It is equal to the adaptive contribution utility in [CBP](https://www.nature.com/articles/s41586-024-07711-7).

        **Args:**
        - **layer_name** (`str`): the name of layer to get unit-wise subtle importance.
        - **activation** (`Tensor`): the activation tensor of the layer. It has the same size of (number of units).
        - **mask** (`Tensor`): the mask tensor of the layer. It has the same size as the feature tensor with size (number of units).

        **Returns:**
        - **subtle_importance_step_layer** (`Tensor`): the unit-wise subtle importance of the layer of the training step.
        """
        layer = self.backbone.get_layer_by_name(layer_name)

        input_weight_abs = torch.abs(layer.weight.data)
        input_weight_abs_sum = torch.sum(
            input_weight_abs,
            dim=[
                i for i in range(input_weight_abs.dim()) if i != 0
            ],  # sum over the input dimension
        )
        input_weight_abs_sum_reciprocal = torch.reciprocal(input_weight_abs_sum)

        output_weight_abs = torch.abs(self.next_layer(layer_name).weight.data)
        output_weight_abs_sum = torch.sum(
            output_weight_abs,
            dim=[
                i for i in range(output_weight_abs.dim()) if i != 1
            ],  # sum over the output dimension
        )

        activation_abs_batch_mean = torch.mean(
            torch.abs(activation),
            dim=[
                i for i in range(activation.dim()) if i != 1
            ],  # average the features over batch samples
        )

        subtle_importance_step_layer = (
            output_weight_abs_sum
            * activation_abs_batch_mean
            * input_weight_abs_sum_reciprocal
            * mask
        )
        subtle_importance_step_layer = subtle_importance_step_layer.detach()

        return subtle_importance_step_layer
