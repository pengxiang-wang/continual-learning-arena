r"""
The submodule in `cl_algorithms` for [AdaHAT (Adaptive Hard Attention to the Task)](https://link.springer.com/chapter/10.1007/978-3-031-70352-2_9) algorithm.
"""

__all__ = ["AdaHAT"]

import logging
from typing import Any

import torch
from torch import Tensor

from clarena.backbones import HATMaskBackbone
from clarena.cl_algorithms import HAT
from clarena.heads import HeadsTIL
from clarena.utils.metrics import HATNetworkCapacityMetric

# always get logger for built-in logging in each module
pylogger = logging.getLogger(__name__)


class AdaHAT(HAT):
    r"""[AdaHAT (Adaptive Hard Attention to the Task)](https://link.springer.com/chapter/10.1007/978-3-031-70352-2_9) algorithm.

    An architecture-based continual learning approach that improves [HAT (Hard Attention to the Task)](http://proceedings.mlr.press/v80/serra18a) by introducing adaptive soft gradient clipping based on parameter importance and network sparsity.

    We implement AdaHAT as a subclass of HAT, as it shares the same `forward()`, `compensate_task_embedding_gradients()`, `training_step()`, `on_train_end()`, `validation_step()`, and `test_step()` methods as the `HAT` class.
    """

    def __init__(
        self,
        backbone: HATMaskBackbone,
        heads: HeadsTIL,
        adjustment_mode: str,
        adjustment_intensity: float,
        s_max: float,
        clamp_threshold: float,
        mask_sparsity_reg_factor: float,
        mask_sparsity_reg_mode: str = "original",
        task_embedding_init_mode: str = "N01",
        epsilon: float = 0.1,
        non_algorithmic_hparams: dict[str, Any] = {},
    ) -> None:
        r"""Initialize the AdaHAT algorithm with the network.

        **Args:**
        - **backbone** (`HATMaskBackbone`): must be a backbone network with the HAT mask mechanism.
        - **heads** (`HeadsTIL`): output heads. AdaHAT supports only TIL (Task-Incremental Learning).
        - **adjustment_mode** (`str`): the strategy of adjustment (i.e., the mode of gradient clipping), must be one of:
            1. 'adahat': set gradients of parameters linking to masked units to a soft adjustment rate in the original AdaHAT approach (allows slight updates on previous-task parameters). See Eqs. (8) and (9) in Sec. 3.1 of the [AdaHAT paper](https://link.springer.com/chapter/10.1007/978-3-031-70352-2_9).
            2. 'adahat_no_sum': as above but without parameter-importance (i.e., no summative mask). See Sec. 4.3 (ablation study) in the [AdaHAT paper](https://link.springer.com/chapter/10.1007/978-3-031-70352-2_9).
            3. 'adahat_no_reg': as above but without network sparsity (i.e., no mask sparsity regularization term). See Sec. 4.3 (ablation study) in the [AdaHAT paper](https://link.springer.com/chapter/10.1007/978-3-031-70352-2_9).
        - **adjustment_intensity** (`float`): hyperparameter, controls the overall intensity of gradient adjustment (the $\alpha$ in Eq. (9) of the [AdaHAT paper](https://link.springer.com/chapter/10.1007/978-3-031-70352-2_9)).
        - **s_max** (`float`): hyperparameter, the maximum scaling factor in the gate function. See Sec. 2.4 "Hard Attention Training" in the [HAT paper](http://proceedings.mlr.press/v80/serra18a).
        - **clamp_threshold** (`float`): the threshold for task embedding gradient compensation. See Sec. 2.5 "Embedding Gradient Compensation" in the [HAT paper](http://proceedings.mlr.press/v80/serra18a).
        - **mask_sparsity_reg_factor** (`float`): hyperparameter, the regularization factor for mask sparsity.
        - **mask_sparsity_reg_mode** (`str`): the mode of mask sparsity regularization, must be one of:
            1. 'original' (default): the original mask sparsity regularization in the [HAT paper](http://proceedings.mlr.press/v80/serra18a).
            2. 'cross': the cross version of mask sparsity regularization.
        - **task_embedding_init_mode** (`str`): the initialization mode for task embeddings, must be one of:
            1. 'N01' (default): standard normal distribution $N(0, 1)$.
            2. 'U-11': uniform distribution $U(-1, 1)$.
            3. 'U01': uniform distribution $U(0, 1)$.
            4. 'U-10': uniform distribution $U(-1, 0)$.
            5. 'last': inherit the task embedding from the last task.
        - **epsilon** (`float`): the value added to network sparsity to avoid division by zero (appearing in Eq. (9) of the [AdaHAT paper](https://link.springer.com/chapter/10.1007/978-3-031-70352-2_9)).
        - **non_algorithmic_hparams** (`dict[str, Any]`): non-algorithmic hyperparameters that are not related to the algorithm itself are passed to this `LightningModule` object from the config, such as optimizer and learning rate scheduler configurations. They are saved for Lightning APIs from `save_hyperparameters()` method. This is useful for the experiment configuration and reproducibility.
        """
        super().__init__(
            backbone=backbone,
            heads=heads,
            adjustment_mode=adjustment_mode,
            s_max=s_max,
            clamp_threshold=clamp_threshold,
            mask_sparsity_reg_factor=mask_sparsity_reg_factor,
            mask_sparsity_reg_mode=mask_sparsity_reg_mode,
            task_embedding_init_mode=task_embedding_init_mode,
            alpha=None,
            non_algorithmic_hparams=non_algorithmic_hparams,
        )

        self.adjustment_intensity: float = adjustment_intensity
        r"""The adjustment intensity in Eq. (9) of the [AdaHAT paper](https://link.springer.com/chapter/10.1007/978-3-031-70352-2_9)."""
        self.epsilon: float | None = epsilon
        r"""The small value to avoid division by zero (appearing in Eq. (9) of the [AdaHAT paper](https://link.springer.com/chapter/10.1007/978-3-031-70352-2_9))."""

        # save additional algorithmic hyperparameters
        self.save_hyperparameters("adjustment_intensity", "epsilon")

        self.summative_mask_for_previous_tasks: dict[str, Tensor] = {}
        r"""The summative binary attention mask $\mathrm{M}^{<t,\text{sum}}$ of previous tasks $1,\cdots, t-1$, gated from the task embedding. It is a dict where keys are layer names and values are the binary mask tensors for the layers. The mask tensor has size (number of units, )."""

        # set manual optimization
        self.automatic_optimization = False

        AdaHAT.sanity_check(self)

    def sanity_check(self) -> None:
        r"""Sanity check."""
        if self.adjustment_intensity <= 0:
            raise ValueError(
                f"The adjustment intensity should be positive, but got {self.adjustment_intensity}."
            )

    def on_train_start(self) -> None:
        r"""Additionally initialize the summative mask at the beginning of the first task."""
        super().on_train_start()

        # initialize the summative mask at the beginning of the first task. This should not be called in `__init__()` method because `self.device` is not available at that time
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
                )  # the summative mask $\mathrm{M}^{<t,\text{sum}}$ is initialized as a zeros mask for $t = 1$. See Eq. (7) in Sec. 3.1 of the [AdaHAT paper](https://link.springer.com/chapter/10.1007/978-3-031-70352-2_9)

    def clip_grad_by_adjustment(
        self,
        network_sparsity: dict[str, Tensor] | None = None,
    ) -> tuple[dict[str, Tensor], dict[str, Tensor], Tensor]:
        r"""Clip the gradients by the adjustment rate. See Eq. (8) in the [AdaHAT paper](https://link.springer.com/chapter/10.1007/978-3-031-70352-2_9).

        Note that because the task embedding fully covers every layer in the backbone network, no parameters are left out of this system. This applies not only to parameters between layers with task embeddings, but also to those before the first layer. We design it separately in the code.

        Network capacity is measured alongside this method. Network capacity is defined as the average adjustment rate over all parameters. See Sec. 4.1 in the [AdaHAT paper](https://link.springer.com/chapter/10.1007/978-3-031-70352-2_9).

        **Args:**
        - **network_sparsity** (`dict[str, Tensor]` | `None`): the network sparsity (i.e., the mask sparsity loss of each layer) for the current task. Keys are layer names and values are the network sparsity values. It is used to calculate the adjustment rate for gradients. Applies only to mode `adahat` and `adahat_no_sum`, not `adahat_no_reg`.

        **Returns:**
        - **adjustment_rate_weight** (`dict[str, Tensor]`): the adjustment rate for weights. Keys (`str`) are layer names and values (`Tensor`) are the adjustment rate tensors.
        - **adjustment_rate_bias** (`dict[str, Tensor]`): the adjustment rate for biases. Keys (`str`) are layer names and values (`Tensor`) are the adjustment rate tensors.
        - **capacity** (`Tensor`): the calculated network capacity.
        """

        # initialize network capacity metric
        capacity = HATNetworkCapacityMetric().to(self.device)
        adjustment_rate_weight = {}
        adjustment_rate_bias = {}

        # calculate the adjustment rate for gradients of the parameters, both weights and biases (if they exist). See Eq. (9) in the [AdaHAT paper](https://link.springer.com/chapter/10.1007/978-3-031-70352-2_9)
        for layer_name in self.backbone.weighted_layer_names:

            layer = self.backbone.get_layer_by_name(
                layer_name
            )  # get the layer by its name

            # placeholder for the adjustment rate to avoid the error of using it before assignment
            adjustment_rate_weight_layer = 1
            adjustment_rate_bias_layer = 1

            weight_importance, bias_importance = (
                self.backbone.get_layer_measure_parameter_wise(
                    neuron_wise_measure=self.summative_mask_for_previous_tasks,
                    layer_name=layer_name,
                    aggregation_mode="min",
                )
            )  # AdaHAT depends on parameter importance rather than parameter masks (as in HAT)

            network_sparsity_layer = network_sparsity[layer_name]

            if self.adjustment_mode == "adahat":
                r_layer = self.adjustment_intensity / (
                    self.epsilon + network_sparsity_layer
                )
                adjustment_rate_weight_layer = torch.div(
                    r_layer, (weight_importance + r_layer)
                )
                adjustment_rate_bias_layer = torch.div(
                    r_layer, (bias_importance + r_layer)
                )

            elif self.adjustment_mode == "adahat_no_sum":

                r_layer = self.adjustment_intensity / (
                    self.epsilon + network_sparsity_layer
                )
                adjustment_rate_weight_layer = torch.div(
                    r_layer, (self.task_id + r_layer)
                )
                adjustment_rate_bias_layer = torch.div(
                    r_layer, (self.task_id + r_layer)
                )

            elif self.adjustment_mode == "adahat_no_reg":

                r_layer = self.adjustment_intensity / (self.epsilon + 0.0)
                adjustment_rate_weight_layer = torch.div(
                    r_layer, (weight_importance + r_layer)
                )
                adjustment_rate_bias_layer = torch.div(
                    r_layer, (bias_importance + r_layer)
                )

            # apply the adjustment rate to the gradients
            layer.weight.grad.data *= adjustment_rate_weight_layer
            if layer.bias is not None:
                layer.bias.grad.data *= adjustment_rate_bias_layer

            # store the adjustment rate for logging
            adjustment_rate_weight[layer_name] = adjustment_rate_weight_layer
            if layer.bias is not None:
                adjustment_rate_bias[layer_name] = adjustment_rate_bias_layer

            # update network capacity metric
            capacity.update(adjustment_rate_weight_layer, adjustment_rate_bias_layer)

        return adjustment_rate_weight, adjustment_rate_bias, capacity.compute()

    def on_train_end(self) -> None:
        r"""Additionally update the summative mask after training the task."""
        super().on_train_end()

        mask_t = self.backbone.masks[
            f"{self.task_id}"
        ]  # get stored mask for the current task again

        # update the summative mask for previous tasks. See Eq. (7) in Sec. 3.1 of the [AdaHAT paper](https://link.springer.com/chapter/10.1007/978-3-031-70352-2_9)
        self.summative_mask_for_previous_tasks = {
            layer_name: self.summative_mask_for_previous_tasks[layer_name]
            + mask_t[layer_name]
            for layer_name in self.backbone.weighted_layer_names
        }
