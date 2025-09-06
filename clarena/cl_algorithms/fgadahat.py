r"""
The submodule in `cl_algorithms` for FG-AdaHAT algorithm.
"""

__all__ = ["FGAdaHAT"]

import logging
import math
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
from clarena.heads import HeadsTIL
from clarena.utils.metrics import HATNetworkCapacityMetric
from clarena.utils.transforms import min_max_normalize

# always get logger for built-in logging in each module
pylogger = logging.getLogger(__name__)


class FGAdaHAT(AdaHAT):
    r"""FG-AdaHAT (Fine-Grained Adaptive Hard Attention to the Task) algorithm.

    An architecture-based continual learning approach that improves [AdaHAT (Adaptive Hard Attention to the Task)](https://link.springer.com/chapter/10.1007/978-3-031-70352-2_9) by introducing fine-grained neuron-wise importance measures guiding the adaptive adjustment mechanism in AdaHAT.

    We implement FG-AdaHAT as a subclass of AdaHAT, as it reuses AdaHAT's summative mask and other components.
    """

    def __init__(
        self,
        backbone: HATMaskBackbone,
        heads: HeadsTIL,
        adjustment_intensity: float,
        importance_type: str,
        importance_summing_strategy: str,
        importance_scheduler_type: str,
        neuron_to_weight_importance_aggregation_mode: str,
        s_max: float,
        clamp_threshold: float,
        mask_sparsity_reg_factor: float,
        mask_sparsity_reg_mode: str = "original",
        base_importance: float = 0.01,
        base_mask_sparsity_reg: float = 0.1,
        base_linear: float = 10,
        filter_by_cumulative_mask: bool = False,
        filter_unmasked_importance: bool = True,
        step_multiply_training_mask: bool = True,
        task_embedding_init_mode: str = "N01",
        importance_summing_strategy_linear_step: float | None = None,
        importance_summing_strategy_exponential_rate: float | None = None,
        importance_summing_strategy_log_base: float | None = None,
        non_algorithmic_hparams: dict[str, Any] = {},
    ) -> None:
        r"""Initialize the FG-AdaHAT algorithm with the network.

        **Args:**
        - **backbone** (`HATMaskBackbone`): must be a backbone network with the HAT mask mechanism.
        - **heads** (`HeadsTIL`): output heads. FG-AdaHAT supports only TIL (Task-Incremental Learning).
        - **adjustment_intensity** (`float`): hyperparameter, controls the overall intensity of gradient adjustment (the $\alpha$ in the paper).
        - **importance_type** (`str`): the type of neuron-wise importance, must be one of:
            1. 'input_weight_abs_sum': sum of absolute input weights;
            2. 'output_weight_abs_sum': sum of absolute output weights;
            3. 'input_weight_gradient_abs_sum': sum of absolute gradients of the input weights (Input Gradients (IG) in the paper);
            4. 'output_weight_gradient_abs_sum': sum of absolute gradients of the output weights (Output Gradients (OG) in the paper);
            5. 'activation_abs': absolute activation;
            6. 'input_weight_abs_sum_x_activation_abs': sum of absolute input weights multiplied by absolute activation (Input Contribution Utility (ICU) in the paper);
            7. 'output_weight_abs_sum_x_activation_abs': sum of absolute output weights multiplied by absolute activation (Contribution Utility (CU) in the paper);
            8. 'gradient_x_activation_abs': absolute gradient (the saliency) multiplied by activation;
            9. 'input_weight_gradient_square_sum': sum of squared gradients of the input weights;
            10. 'output_weight_gradient_square_sum': sum of squared gradients of the output weights;
            11. 'input_weight_gradient_square_sum_x_activation_abs': sum of squared gradients of the input weights multiplied by absolute activation (Activation Fisher Information (AFI) in the paper);
            12. 'output_weight_gradient_square_sum_x_activation_abs': sum of squared gradients of the output weights multiplied by absolute activation;
            13. 'conductance_abs': absolute layer conductance;
            14. 'internal_influence_abs': absolute internal influence (Internal Influence (II) in the paper);
            15. 'gradcam_abs': absolute Grad-CAM;
            16. 'deeplift_abs': absolute DeepLIFT (DeepLIFT (DL) in the paper);
            17. 'deepliftshap_abs': absolute DeepLIFT-SHAP;
            18. 'gradientshap_abs': absolute Gradient-SHAP (Gradient SHAP (GS) in the paper);
            19. 'integrated_gradients_abs': absolute Integrated Gradients;
            20. 'feature_ablation_abs': absolute Feature Ablation (Feature Ablation (FA) in the paper);
            21. 'lrp_abs': absolute Layer-wise Relevance Propagation (LRP);
            22. 'cbp_adaptation': the adaptation function in [Continual Backpropagation (CBP)](https://www.nature.com/articles/s41586-024-07711-7);
            23. 'cbp_adaptive_contribution': the adaptive contribution function in [Continual Backpropagation (CBP)](https://www.nature.com/articles/s41586-024-07711-7);
        - **importance_summing_strategy** (`str`): the strategy to sum neuron-wise importance for previous tasks, must be one of:
            1. 'add_latest': add the latest neuron-wise importance to the summative importance;
            2. 'add_all': add all previous neuron-wise importance (including the latest) to the summative importance;
            3. 'add_average': add the average of all previous neuron-wise importance (including the latest) to the summative importance;
            4. 'linear_decrease': weigh the previous neuron-wise importance by a linear factor that decreases with the task ID;
            5. 'quadratic_decrease': weigh the previous neuron-wise importance that decreases quadratically with the task ID;
            6. 'cubic_decrease': weigh the previous neuron-wise importance that decreases cubically with the task ID;
            7. 'exponential_decrease': weigh the previous neuron-wise importance by an exponential factor that decreases with the task ID;
            8. 'log_decrease': weigh the previous neuron-wise importance by a logarithmic factor that decreases with the task ID;
            9. 'factorial_decrease': weigh the previous neuron-wise importance that decreases factorially with the task ID;
        - **importance_scheduler_type** (`str`): the scheduler for importance, i.e., the factor $c^t$ multiplied to parameter importance. Must be one of:
            1. 'linear_sparsity_reg': $c^t = (t+b_L) \cdot [R(M^t, M^{<t}) + b_R]$, where $R(M^t, M^{<t})$ is the mask sparsity regularization betwwen the current task and previous tasks, $b_L$ is the base linear factor (see argument `base_linear`), and $b_R$ is the base mask sparsity regularization factor (see argument `base_mask_sparsity_reg`);
            2. 'sparsity_reg': $c^t = [R(M^t, M^{<t}) + b_R]$;
            3. 'summative_mask_sparsity_reg': $c^t_{l,ij} = \left(\min \left(m^{<t, \text{sum}}_{l,i}, m^{<t, \text{sum}}_{l-1,j}\right)+b_L\right) \cdot [R(M^t, M^{<t}) + b_R]$.
        - **neuron_to_weight_importance_aggregation_mode** (`str`): aggregation mode from neuron-wise to weight-wise importance ($\text{Agg}(\cdot)$ in the paper), must be one of:
            1. 'min': take the minimum of neuron-wise importance for each weight;
            2. 'max': take the maximum of neuron-wise importance for each weight;
            3. 'mean': take the mean of neuron-wise importance for each weight.
        - **s_max** (`float`): hyperparameter, the maximum scaling factor in the gate function. See Sec. 2.4 "Hard Attention Training" in the [HAT paper](http://proceedings.mlr.press/v80/serra18a).
        - **clamp_threshold** (`float`): the threshold for task embedding gradient compensation. See Sec. 2.5 "Embedding Gradient Compensation" in the [HAT paper](http://proceedings.mlr.press/v80/serra18a).
        - **mask_sparsity_reg_factor** (`float`): hyperparameter, the regularization factor for mask sparsity.
        - **mask_sparsity_reg_mode** (`str`): the mode of mask sparsity regularization, must be one of:
            1. 'original' (default): the original mask sparsity regularization in the [HAT paper](http://proceedings.mlr.press/v80/serra18a).
            2. 'cross': the cross version of mask sparsity regularization.
        - **base_importance** (`float`): base value added to importance ($b_I$ in the paper). Default: 0.01.
        - **base_mask_sparsity_reg** (`float`): base value added to mask sparsity regularization factor in the importance scheduler ($b_R$ in the paper). Default: 0.1.
        - **base_linear** (`float`): base value added to the linear factor in the importance scheduler ($b_L$ in the paper). Default: 10.
        - **filter_by_cumulative_mask** (`bool`): whether to multiply the cumulative mask to the importance when calculating adjustment rate. Default: False.
        - **filter_unmasked_importance** (`bool`): whether to filter unmasked importance values (set to 0) at the end of task training. Default: False.
        - **step_multiply_training_mask** (`bool`): whether to multiply the training mask to the importance at each training step. Default: True.
        - **task_embedding_init_mode** (`str`): the initialization mode for task embeddings, must be one of:
            1. 'N01' (default): standard normal distribution $N(0, 1)$.
            2. 'U-11': uniform distribution $U(-1, 1)$.
            3. 'U01': uniform distribution $U(0, 1)$.
            4. 'U-10': uniform distribution $U(-1, 0)$.
            5. 'last': inherit the task embedding from the last task.
        - **importance_summing_strategy_linear_step** (`float` | `None`): linear step for the importance summing strategy (used when `importance_summing_strategy` is 'linear_decrease'). Must be > 0.
        - **importance_summing_strategy_exponential_rate** (`float` | `None`): exponential rate for the importance summing strategy (used when `importance_summing_strategy` is 'exponential_decrease'). Must be > 1.
        - **importance_summing_strategy_log_base** (`float` | `None`): base for the logarithm in the importance summing strategy (used when `importance_summing_strategy` is 'log_decrease'). Must be > 1.
        - **non_algorithmic_hparams** (`dict[str, Any]`): non-algorithmic hyperparameters that are not related to the algorithm itself are passed to this `LightningModule` object from the config, such as optimizer and learning rate scheduler configurations. They are saved for Lightning APIs from `save_hyperparameters()` method. This is useful for the experiment configuration and reproducibility.

        """
        super().__init__(
            backbone=backbone,
            heads=heads,
            adjustment_mode=None,  # use the own adjustment mechanism of FG-AdaHAT
            adjustment_intensity=adjustment_intensity,
            s_max=s_max,
            clamp_threshold=clamp_threshold,
            mask_sparsity_reg_factor=mask_sparsity_reg_factor,
            mask_sparsity_reg_mode=mask_sparsity_reg_mode,
            task_embedding_init_mode=task_embedding_init_mode,
            epsilon=base_mask_sparsity_reg,  # the epsilon is now the base mask sparsity regularization factor
            non_algorithmic_hparams=non_algorithmic_hparams,
        )

        # save additional algorithmic hyperparameters
        self.save_hyperparameters(
            "adjustment_intensity",
            "importance_type",
            "importance_summing_strategy",
            "importance_scheduler_type",
            "neuron_to_weight_importance_aggregation_mode",
            "s_max",
            "clamp_threshold",
            "mask_sparsity_reg_factor",
            "mask_sparsity_reg_mode",
            "base_importance",
            "base_mask_sparsity_reg",
            "base_linear",
            "filter_by_cumulative_mask",
            "filter_unmasked_importance",
            "step_multiply_training_mask",
        )

        self.importance_type: str | None = importance_type
        r"""The type of the neuron-wise importance added to AdaHAT importance."""

        self.importance_scheduler_type: str = importance_scheduler_type
        r"""The type of the importance scheduler."""
        self.neuron_to_weight_importance_aggregation_mode: str = (
            neuron_to_weight_importance_aggregation_mode
        )
        r"""The mode of aggregation from neuron-wise to weight-wise importance. """
        self.filter_by_cumulative_mask: bool = filter_by_cumulative_mask
        r"""The flag to filter importance by the cumulative mask when calculating the adjustment rate."""
        self.filter_unmasked_importance: bool = filter_unmasked_importance
        r"""The flag to filter unmasked importance values (set them to 0) at the end of task training."""
        self.step_multiply_training_mask: bool = step_multiply_training_mask
        r"""The flag to multiply the training mask to the importance at each training step."""

        # importance summing strategy
        self.importance_summing_strategy: str = importance_summing_strategy
        r"""The strategy to sum the neuron-wise importance for previous tasks."""
        if importance_summing_strategy_linear_step is not None:
            self.importance_summing_strategy_linear_step: float = (
                importance_summing_strategy_linear_step
            )
            r"""The linear step for the importance summing strategy (only when `importance_summing_strategy` is 'linear_decrease')."""
        if importance_summing_strategy_exponential_rate is not None:
            self.importance_summing_strategy_exponential_rate: float = (
                importance_summing_strategy_exponential_rate
            )
            r"""The exponential rate for the importance summing strategy (only when `importance_summing_strategy` is 'exponential_decrease'). """
        if importance_summing_strategy_log_base is not None:
            self.importance_summing_strategy_log_base: float = (
                importance_summing_strategy_log_base
            )
            r"""The base for the logarithm in the importance summing strategy (only when `importance_summing_strategy` is 'log_decrease'). """

        # base values
        self.base_importance: float = base_importance
        r"""The base value added to the importance to avoid zero. """
        self.base_mask_sparsity_reg: float = base_mask_sparsity_reg
        r"""The base value added to the mask sparsity regularization to avoid zero. """
        self.base_linear: float = base_linear
        r"""The base value added to the linear layer to avoid zero. """

        self.importances: dict[int, dict[str, Tensor]] = {}
        r"""The min-max scaled ($[0, 1]$) neuron-wise importance of units. It is $I^{\tau}_{l}$ in the paper. Keys are task IDs and values are the corresponding importance tensors. Each importance tensor is a dict where keys are layer names and values are the importance tensor for the layer. The utility tensor is the same size as the feature tensor with size (number of units, ). """
        self.summative_importance_for_previous_tasks: dict[str, Tensor] = {}
        r"""The summative neuron-wise importance values of units for previous tasks before the current task `self.task_id`. See $I^{<t}_{l}$ in the paper. Keys are layer names and values are the summative importance tensor for the layer. The summative importance tensor has the same size as the feature tensor with size (number of units, ). """

        self.num_steps_t: int
        r"""The number of training steps for the current task `self.task_id`."""
        # set manual optimization
        self.automatic_optimization = False

        FGAdaHAT.sanity_check(self)

    def sanity_check(self) -> None:
        r"""Sanity check."""

        # check importance type
        if self.importance_type not in [
            "input_weight_abs_sum",
            "output_weight_abs_sum",
            "input_weight_gradient_abs_sum",
            "output_weight_gradient_abs_sum",
            "activation_abs",
            "input_weight_abs_sum_x_activation_abs",
            "output_weight_abs_sum_x_activation_abs",
            "gradient_x_activation_abs",
            "input_weight_gradient_square_sum",
            "output_weight_gradient_square_sum",
            "input_weight_gradient_square_sum_x_activation_abs",
            "output_weight_gradient_square_sum_x_activation_abs",
            "conductance_abs",
            "internal_influence_abs",
            "gradcam_abs",
            "deeplift_abs",
            "deepliftshap_abs",
            "gradientshap_abs",
            "integrated_gradients_abs",
            "feature_ablation_abs",
            "lrp_abs",
            "cbp_adaptation",
            "cbp_adaptive_contribution",
        ]:
            raise ValueError(
                f"importance_type must be one of the predefined types, but got {self.importance_type}"
            )

        # check importance summing strategy
        if self.importance_summing_strategy not in [
            "add_latest",
            "add_all",
            "add_average",
            "linear_decrease",
            "quadratic_decrease",
            "cubic_decrease",
            "exponential_decrease",
            "log_decrease",
            "factorial_decrease",
        ]:
            raise ValueError(
                f"importance_summing_strategy must be one of the predefined strategies, but got {self.importance_summing_strategy}"
            )

        # check importance scheduler type
        if self.importance_scheduler_type not in [
            "linear_sparsity_reg",
            "sparsity_reg",
            "summative_mask_sparsity_reg",
        ]:
            raise ValueError(
                f"importance_scheduler_type must be one of the predefined types, but got {self.importance_scheduler_type}"
            )

        # check neuron to weight importance aggregation mode
        if self.neuron_to_weight_importance_aggregation_mode not in [
            "min",
            "max",
            "mean",
        ]:
            raise ValueError(
                f"neuron_to_weight_importance_aggregation_mode must be one of the predefined modes, but got {self.neuron_to_weight_importance_aggregation_mode}"
            )

        # check base values
        if self.base_importance < 0:
            raise ValueError(
                f"base_importance must be >= 0, but got {self.base_importance}"
            )
        if self.base_mask_sparsity_reg <= 0:
            raise ValueError(
                f"base_mask_sparsity_reg must be > 0, but got {self.base_mask_sparsity_reg}"
            )
        if self.base_linear <= 0:
            raise ValueError(f"base_linear must be > 0, but got {self.base_linear}")

    def on_train_start(self) -> None:
        r"""Initialize neuron importance accumulation variable for each layer as zeros, in addition to AdaHAT's summative mask initialization."""
        super().on_train_start()

        self.importances[self.task_id] = (
            {}
        )  # initialize the importance for the current task

        # initialize the neuron importance at the beginning of each task. This should not be called in `__init__()` method because `self.device` is not available at that time.
        for layer_name in self.backbone.weighted_layer_names:
            layer = self.backbone.get_layer_by_name(
                layer_name
            )  # get the layer by its name
            num_units = layer.weight.shape[0]

            # initialize the accumulated importance at the beginning of each task
            self.importances[self.task_id][layer_name] = torch.zeros(num_units).to(
                self.device
            )

            # reset the number of steps counter for the current task
            self.num_steps_t = 0

            # initialize the summative neuron-wise importance at the beginning of the first task
            if self.task_id == 1:
                self.summative_importance_for_previous_tasks[layer_name] = torch.zeros(
                    num_units
                ).to(
                    self.device
                )  # the summative neuron-wise importance for previous tasks $I^{<t}_{l}$ is initialized as zeros mask when $t=1$

    def clip_grad_by_adjustment(
        self,
        network_sparsity: dict[str, Tensor],
    ) -> tuple[dict[str, Tensor], dict[str, Tensor], Tensor]:
        r"""Clip the gradients by the adjustment rate. See Eq. (1) in the paper.

        Note that because the task embedding fully covers every layer in the backbone network, no parameters are left out of this system. This applies not only to parameters between layers with task embeddings, but also to those before the first layer. We design it separately in the code.

        Network capacity is measured alongside this method. Network capacity is defined as the average adjustment rate over all parameters. See Sec. 4.1 in the [AdaHAT paper](https://link.springer.com/chapter/10.1007/978-3-031-70352-2_9).

        **Args:**
        - **network_sparsity** (`dict[str, Tensor]`): the network sparsity (i.e., mask sparsity loss of each layer) for the current task. Keys are layer names and values are the network sparsity values. It is used to calculate the adjustment rate for gradients. In FG-AdaHAT, it is used to construct the importance scheduler.

        **Returns:**
        - **adjustment_rate_weight** (`dict[str, Tensor]`): the adjustment rate for weights. Keys (`str`) are layer names and values (`Tensor`) are the adjustment rate tensors.
        - **adjustment_rate_bias** (`dict[str, Tensor]`): the adjustment rate for biases. Keys (`str`) are layer names and values (`Tensor`) are the adjustment rate tensors.
        - **capacity** (`Tensor`): the calculated network capacity.
        """

        # initialize network capacity metric
        capacity = HATNetworkCapacityMetric().to(self.device)
        adjustment_rate_weight = {}
        adjustment_rate_bias = {}

        # calculate the adjustment rate for gradients of the parameters, both weights and biases (if they exist). See Eq. (2) in the paper
        for layer_name in self.backbone.weighted_layer_names:

            layer = self.backbone.get_layer_by_name(
                layer_name
            )  # get the layer by its name

            # placeholder for the adjustment rate to avoid the error of using it before assignment
            adjustment_rate_weight_layer = 1
            adjustment_rate_bias_layer = 1

            # aggregate the neuron-wise importance to weight-wise importance. Note that the neuron-wise importance has already been min-max scaled to $[0, 1]$ in the `on_train_batch_end()` method, added the base value, and filtered by the mask
            weight_importance, bias_importance = (
                self.backbone.get_layer_measure_parameter_wise(
                    neuron_wise_measure=self.summative_importance_for_previous_tasks,
                    layer_name=layer_name,
                    aggregation_mode=self.neuron_to_weight_importance_aggregation_mode,
                )
            )

            weight_mask, bias_mask = self.backbone.get_layer_measure_parameter_wise(
                neuron_wise_measure=self.cumulative_mask_for_previous_tasks,
                layer_name=layer_name,
                aggregation_mode="min",
            )

            # filter the weight importance by the cumulative mask
            if self.filter_by_cumulative_mask:
                weight_importance = weight_importance * weight_mask
                bias_importance = bias_importance * bias_mask

            network_sparsity_layer = network_sparsity[layer_name]

            # calculate importance scheduler (the factor of importance). See Eq. (3) in the paper
            factor = network_sparsity_layer + self.base_mask_sparsity_reg
            if self.importance_scheduler_type == "linear_sparsity_reg":
                factor = factor * (self.task_id + self.base_linear)
            elif self.importance_scheduler_type == "sparsity_reg":
                pass
            elif self.importance_scheduler_type == "summative_mask_sparsity_reg":
                factor = factor * (
                    self.summative_mask_for_previous_tasks + self.base_linear
                )

            # calculate the adjustment rate
            adjustment_rate_weight_layer = torch.div(
                self.adjustment_intensity,
                (factor * weight_importance + self.adjustment_intensity),
            )

            adjustment_rate_bias_layer = torch.div(
                self.adjustment_intensity,
                (factor * bias_importance + self.adjustment_intensity),
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

    def on_train_batch_end(
        self, outputs: dict[str, Any], batch: Any, batch_idx: int
    ) -> None:
        r"""Calculate the step-wise importance, update the accumulated importance and number of steps counter after each training step.

        **Args:**
        - **outputs** (`dict[str, Any]`): outputs of the training step (returns of `training_step()` in `CLAlgorithm`).
        - **batch** (`Any`): training data batch.
        - **batch_idx** (`int`): index of the current batch (for mask figure file name).
        """

        # get potential useful information from training batch
        activations = outputs["activations"]
        input = outputs["input"]
        target = outputs["target"]
        mask = outputs["mask"]
        num_batches = self.trainer.num_training_batches

        for layer_name in self.backbone.weighted_layer_names:
            # layer-wise operation

            activation = activations[layer_name]

            # calculate neuron-wise importance of the training step. See $I^{\tau}_l(\mathbf{x},y)$ (before Eqs. (5) and (6)) in the paper.
            if self.importance_type == "input_weight_abs_sum":
                importance_step = self.get_importance_step_layer_weight_abs_sum(
                    layer_name=layer_name,
                    if_output_weight=False,
                    reciprocal=False,
                )
            elif self.importance_type == "output_weight_abs_sum":
                importance_step = self.get_importance_step_layer_weight_abs_sum(
                    layer_name=layer_name,
                    if_output_weight=True,
                    reciprocal=False,
                )
            elif self.importance_type == "input_weight_gradient_abs_sum":
                importance_step = (
                    self.get_importance_step_layer_weight_gradient_abs_sum(
                        layer_name=layer_name, if_output_weight=False
                    )
                )
            elif self.importance_type == "output_weight_gradient_abs_sum":
                importance_step = (
                    self.get_importance_step_layer_weight_gradient_abs_sum(
                        layer_name=layer_name, if_output_weight=True
                    )
                )
            elif self.importance_type == "activation_abs":
                importance_step = self.get_importance_step_layer_activation_abs(
                    activation=activation
                )
            elif self.importance_type == "input_weight_abs_sum_x_activation_abs":
                importance_step = (
                    self.get_importance_step_layer_weight_abs_sum_x_activation_abs(
                        layer_name=layer_name,
                        activation=activation,
                        if_output_weight=False,
                    )
                )
            elif self.importance_type == "output_weight_abs_sum_x_activation_abs":
                importance_step = (
                    self.get_importance_step_layer_weight_abs_sum_x_activation_abs(
                        layer_name=layer_name,
                        activation=activation,
                        if_output_weight=True,
                    )
                )
            elif self.importance_type == "gradient_x_activation_abs":
                importance_step = (
                    self.get_importance_step_layer_gradient_x_activation_abs(
                        layer_name=layer_name,
                        input=input,
                        target=target,
                        batch_idx=batch_idx,
                        num_batches=num_batches,
                    )
                )
            elif self.importance_type == "input_weight_gradient_square_sum":
                importance_step = (
                    self.get_importance_step_layer_weight_gradient_square_sum(
                        layer_name=layer_name,
                        activation=activation,
                        if_output_weight=False,
                    )
                )
            elif self.importance_type == "output_weight_gradient_square_sum":
                importance_step = (
                    self.get_importance_step_layer_weight_gradient_square_sum(
                        layer_name=layer_name,
                        activation=activation,
                        if_output_weight=True,
                    )
                )
            elif (
                self.importance_type
                == "input_weight_gradient_square_sum_x_activation_abs"
            ):
                importance_step = self.get_importance_step_layer_weight_gradient_square_sum_x_activation_abs(
                    layer_name=layer_name,
                    activation=activation,
                    if_output_weight=False,
                )
            elif (
                self.importance_type
                == "output_weight_gradient_square_sum_x_activation_abs"
            ):
                importance_step = self.get_importance_step_layer_weight_gradient_square_sum_x_activation_abs(
                    layer_name=layer_name,
                    activation=activation,
                    if_output_weight=True,
                )
            elif self.importance_type == "conductance_abs":
                importance_step = self.get_importance_step_layer_conductance_abs(
                    layer_name=layer_name,
                    input=input,
                    baselines=None,
                    target=target,
                    batch_idx=batch_idx,
                    num_batches=num_batches,
                )
            elif self.importance_type == "internal_influence_abs":
                importance_step = self.get_importance_step_layer_internal_influence_abs(
                    layer_name=layer_name,
                    input=input,
                    baselines=None,
                    target=target,
                    batch_idx=batch_idx,
                    num_batches=num_batches,
                )
            elif self.importance_type == "gradcam_abs":
                importance_step = self.get_importance_step_layer_gradcam_abs(
                    layer_name=layer_name,
                    input=input,
                    target=target,
                    batch_idx=batch_idx,
                    num_batches=num_batches,
                )
            elif self.importance_type == "deeplift_abs":
                importance_step = self.get_importance_step_layer_deeplift_abs(
                    layer_name=layer_name,
                    input=input,
                    baselines=None,
                    target=target,
                    batch_idx=batch_idx,
                    num_batches=num_batches,
                )
            elif self.importance_type == "deepliftshap_abs":
                importance_step = self.get_importance_step_layer_deepliftshap_abs(
                    layer_name=layer_name,
                    input=input,
                    baselines=None,
                    target=target,
                    batch_idx=batch_idx,
                    num_batches=num_batches,
                )
            elif self.importance_type == "gradientshap_abs":
                importance_step = self.get_importance_step_layer_gradientshap_abs(
                    layer_name=layer_name,
                    input=input,
                    baselines=None,
                    target=target,
                    batch_idx=batch_idx,
                    num_batches=num_batches,
                )
            elif self.importance_type == "integrated_gradients_abs":
                importance_step = (
                    self.get_importance_step_layer_integrated_gradients_abs(
                        layer_name=layer_name,
                        input=input,
                        baselines=None,
                        target=target,
                        batch_idx=batch_idx,
                        num_batches=num_batches,
                    )
                )
            elif self.importance_type == "feature_ablation_abs":
                importance_step = self.get_importance_step_layer_feature_ablation_abs(
                    layer_name=layer_name,
                    input=input,
                    layer_baselines=None,
                    target=target,
                    batch_idx=batch_idx,
                    num_batches=num_batches,
                )
            elif self.importance_type == "lrp_abs":
                importance_step = self.get_importance_step_layer_lrp_abs(
                    layer_name=layer_name,
                    input=input,
                    target=target,
                    batch_idx=batch_idx,
                    num_batches=num_batches,
                )
            elif self.importance_type == "cbp_adaptation":
                importance_step = self.get_importance_step_layer_weight_abs_sum(
                    layer_name=layer_name,
                    if_output_weight=False,
                    reciprocal=True,
                )
            elif self.importance_type == "cbp_adaptive_contribution":
                importance_step = (
                    self.get_importance_step_layer_cbp_adaptive_contribution(
                        layer_name=layer_name,
                        activation=activation,
                    )
                )

            importance_step = min_max_normalize(
                importance_step
            )  # min-max scaling the utility to $[0, 1]$. See Eq. (5) in the paper

            # multiply the importance by the training mask. See Eq. (6) in the paper
            if self.step_multiply_training_mask:
                importance_step = importance_step * mask[layer_name]

            # update accumulated importance
            self.importances[self.task_id][layer_name] = (
                self.importances[self.task_id][layer_name] + importance_step
            )

        # update number of steps counter
        self.num_steps_t += 1

    def on_train_end(self) -> None:
        r"""Additionally calculate neuron-wise importance for previous tasks at the end of training each task."""
        super().on_train_end()  # store the mask and update cumulative and summative masks

        for layer_name in self.backbone.weighted_layer_names:

            # average the neuron-wise step importance. See Eq. (4) in the paper
            self.importances[self.task_id][layer_name] = (
                self.importances[self.task_id][layer_name]
            ) / self.num_steps_t

            # add the base importance. See Eq. (6) in the paper
            self.importances[self.task_id][layer_name] = (
                self.importances[self.task_id][layer_name] + self.base_importance
            )

            # filter unmasked importance
            if self.filter_unmasked_importance:
                self.importances[self.task_id][layer_name] = (
                    self.importances[self.task_id][layer_name]
                    * self.backbone.masks[f"{self.task_id}"][layer_name]
                )

            # calculate the summative neuron-wise importance for previous tasks. See Eq. (4) in the paper
            if self.importance_summing_strategy == "add_latest":
                self.summative_importance_for_previous_tasks[
                    layer_name
                ] += self.importances[self.task_id][layer_name]

            elif self.importance_summing_strategy == "add_all":
                for t in range(1, self.task_id + 1):
                    self.summative_importance_for_previous_tasks[
                        layer_name
                    ] += self.importances[t][layer_name]

            elif self.importance_summing_strategy == "add_average":
                for t in range(1, self.task_id + 1):
                    self.summative_importance_for_previous_tasks[layer_name] += (
                        self.importances[t][layer_name] / self.task_id
                    )
            else:
                self.summative_importance_for_previous_tasks[
                    layer_name
                ] = torch.zeros_like(
                    self.summative_importance_for_previous_tasks[layer_name]
                ).to(
                    self.device
                )  # starting adding from 0

                if self.importance_summing_strategy == "linear_decrease":
                    s = self.importance_summing_strategy_linear_step
                    for t in range(1, self.task_id + 1):
                        w_t = s * (self.task_id - t) + 1

                elif self.importance_summing_strategy == "quadratic_decrease":
                    for t in range(1, self.task_id + 1):
                        w_t = (self.task_id - t + 1) ** 2
                elif self.importance_summing_strategy == "cubic_decrease":
                    for t in range(1, self.task_id + 1):
                        w_t = (self.task_id - t + 1) ** 3
                elif self.importance_summing_strategy == "exponential_decrease":
                    for t in range(1, self.task_id + 1):
                        r = self.importance_summing_strategy_exponential_rate

                        w_t = r ** (self.task_id - t + 1)
                elif self.importance_summing_strategy == "log_decrease":
                    a = self.importance_summing_strategy_log_base
                    for t in range(1, self.task_id + 1):
                        w_t = math.log(self.task_id - t, a) + 1
                elif self.importance_summing_strategy == "factorial_decrease":
                    for t in range(1, self.task_id + 1):
                        w_t = math.factorial(self.task_id - t + 1)
                else:
                    raise ValueError
                self.summative_importance_for_previous_tasks[layer_name] += (
                    self.importances[t][layer_name] * w_t
                )

    def get_importance_step_layer_weight_abs_sum(
        self: str,
        layer_name: str,
        if_output_weight: bool,
        reciprocal: bool,
    ) -> Tensor:
        r"""Get the raw neuron-wise importance (before scaling) of a layer of a training step. See $I^{\tau}_l(\mathbf{x},y)$ (before Eqs. (5) and (6)) in the paper. This method uses the sum of absolute values of layer input or output weights.

        **Args:**
        - **layer_name** (`str`): the name of layer to get neuron-wise importance.
        - **if_output_weight** (`bool`): whether to use the output weights or input weights.
        - **reciprocal** (`bool`): whether to take reciprocal.

        **Returns:**
        - **importance_step_layer** (`Tensor`): the neuron-wise importance of the layer of the training step.
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
            importance_step_layer = weight_abs_sum_reciprocal
        else:
            importance_step_layer = weight_abs_sum
        importance_step_layer = importance_step_layer.detach()

        return importance_step_layer

    def get_importance_step_layer_weight_gradient_abs_sum(
        self: str,
        layer_name: str,
        if_output_weight: bool,
    ) -> Tensor:
        r"""Get the raw neuron-wise importance (before scaling) of a layer of a training step. See $I^{\tau}_l(\mathbf{x},y)$ (before Eqs. (5) and (6)) in the paper. This method uses the sum of absolute values of gradients of the layer input or output weights.

        **Args:**
        - **layer_name** (`str`): the name of layer to get neuron-wise importance.
        - **if_output_weight** (`bool`): whether to use the output weights or input weights.

        **Returns:**
        - **importance_step_layer** (`Tensor`): the neuron-wise importance of the layer of the training step.
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

        importance_step_layer = gradient_abs_sum
        importance_step_layer = importance_step_layer.detach()

        return importance_step_layer

    def get_importance_step_layer_activation_abs(
        self: str,
        activation: Tensor,
    ) -> Tensor:
        r"""Get the raw neuron-wise importance (before scaling) of a layer of a training step. See $I^{\tau}_l(\mathbf{x},y)$ (before Eqs. (5) and (6)) in the paper. This method uses the absolute value of activation of the layer. This is our own implementation of [Layer Activation](https://captum.ai/api/layer.html#layer-activation) in Captum.

        **Args:**
        - **activation** (`Tensor`): the activation tensor of the layer. It has the same size of (number of units, ).

        **Returns:**
        - **importance_step_layer** (`Tensor`): the neuron-wise importance of the layer of the training step.
        """
        activation_abs_batch_mean = torch.mean(
            torch.abs(activation),
            dim=[
                i for i in range(activation.dim()) if i != 1
            ],  # average the features over batch samples
        )
        importance_step_layer = activation_abs_batch_mean
        importance_step_layer = importance_step_layer.detach()

        return importance_step_layer

    def get_importance_step_layer_weight_abs_sum_x_activation_abs(
        self: str,
        layer_name: str,
        activation: Tensor,
        if_output_weight: bool,
    ) -> Tensor:
        r"""Get the raw neuron-wise importance (before scaling) of a layer of a training step. See $I^{\tau}_l(\mathbf{x},y)$ (before Eqs. (5) and (6)) in the paper. This method uses the sum of absolute values of layer input / output weights multiplied by absolute values of activation. The input weights version is equal to the contribution utility in [CBP](https://www.nature.com/articles/s41586-024-07711-7).

        **Args:**
        - **layer_name** (`str`): the name of layer to get neuron-wise importance.
        - **activation** (`Tensor`): the activation tensor of the layer. It has the same size of (number of units, ).
        - **if_output_weight** (`bool`): whether to use the output weights or input weights.

        **Returns:**
        - **importance_step_layer** (`Tensor`): the neuron-wise importance of the layer of the training step.
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

        importance_step_layer = weight_abs_sum * activation_abs_batch_mean
        importance_step_layer = importance_step_layer.detach()

        return importance_step_layer

    def get_importance_step_layer_gradient_x_activation_abs(
        self: str,
        layer_name: str,
        input: Tensor | tuple[Tensor, ...],
        target: Tensor | None,
        batch_idx: int,
        num_batches: int,
    ) -> Tensor:
        r"""Get the raw neuron-wise importance (before scaling) of a layer of a training step. See $I^{\tau}_l(\mathbf{x},y)$ (before Eqs. (5) and (6)) in the paper. This method uses the absolute values of the gradient of layer activation multiplied by the activation. We implement this using [Layer Gradient X Activation](https://captum.ai/api/layer.html#layer-gradient-x-activation) in Captum.

        **Args:**
        - **layer_name** (`str`): the name of layer to get neuron-wise importance.
        - **input** (`Tensor` | `tuple[Tensor, ...]`): the input batch of the training step.
        - **target** (`Tensor` | `None`): the target batch of the training step.
        - **batch_idx** (`int`): the index of the current batch. This is an argument of the forward function during training.
        - **num_batches** (`int`): the number of batches in the training step. This is an argument of the forward function during training.

        **Returns:**
        - **importance_step_layer** (`Tensor`): the neuron-wise importance of the layer of the training step.
        """
        layer = self.backbone.get_layer_by_name(layer_name)

        input = input.requires_grad_()

        # initialize the Layer Gradient X Activation object
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

        attribution_abs_batch_mean = torch.mean(
            torch.abs(attribution),
            dim=[
                i for i in range(attribution.dim()) if i != 1
            ],  # average the features over batch samples
        )

        importance_step_layer = attribution_abs_batch_mean
        importance_step_layer = importance_step_layer.detach()

        return importance_step_layer

    def get_importance_step_layer_weight_gradient_square_sum(
        self: str,
        layer_name: str,
        activation: Tensor,
        if_output_weight: bool,
    ) -> Tensor:
        r"""Get the raw neuron-wise importance (before scaling) of a layer of a training step. See $I^{\tau}_l(\mathbf{x},y)$ (before Eqs. (5) and (6)) in the paper. This method uses the sum of layer weight gradient squares. The weight gradient square is equal to fisher information in [EWC](https://www.pnas.org/doi/10.1073/pnas.1611835114).

        **Args:**
        - **layer_name** (`str`): the name of layer to get neuron-wise importance.
        - **activation** (`Tensor`): the activation tensor of the layer. It has the same size of (number of units, ).
        - **if_output_weight** (`bool`): whether to use the output weights or input weights.

        **Returns:**
        - **importance_step_layer** (`Tensor`): the neuron-wise importance of the layer of the training step.
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

        importance_step_layer = gradient_square_sum
        importance_step_layer = importance_step_layer.detach()

        return importance_step_layer

    def get_importance_step_layer_weight_gradient_square_sum_x_activation_abs(
        self: str,
        layer_name: str,
        activation: Tensor,
        if_output_weight: bool,
    ) -> Tensor:
        r"""Get the raw neuron-wise importance (before scaling) of a layer of a training step. See $I^{\tau}_l(\mathbf{x},y)$ (before Eqs. (5) and (6)) in the paper. This method uses the sum of layer weight gradient squares multiplied by absolute values of activation. The weight gradient square is equal to fisher information in [EWC](https://www.pnas.org/doi/10.1073/pnas.1611835114).

        **Args:**
        - **layer_name** (`str`): the name of layer to get neuron-wise importance.
        - **activation** (`Tensor`): the activation tensor of the layer. It has the same size of (number of units, ).
        - **if_output_weight** (`bool`): whether to use the output weights or input weights.

        **Returns:**
        - **importance_step_layer** (`Tensor`): the neuron-wise importance of the layer of the training step.
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

        importance_step_layer = gradient_square_sum * activation_abs_batch_mean
        importance_step_layer = importance_step_layer.detach()

        return importance_step_layer

    def get_importance_step_layer_conductance_abs(
        self: str,
        layer_name: str,
        input: Tensor | tuple[Tensor, ...],
        baselines: None | int | float | Tensor | tuple[int | float | Tensor, ...],
        target: Tensor | None,
        batch_idx: int,
        num_batches: int,
    ) -> Tensor:
        r"""Get the raw neuron-wise importance (before scaling) of a layer of a training step. See $I^{\tau}_l(\mathbf{x},y)$ (before Eqs. (5) and (6)) in the paper. This method uses the absolute values of [conductance](https://openreview.net/forum?id=SylKoo0cKm). We implement this using [Layer Conductance](https://captum.ai/api/layer.html#layer-conductance) in Captum.

        **Args:**
        - **layer_name** (`str`): the name of layer to get neuron-wise importance.
        - **input** (`Tensor` | `tuple[Tensor, ...]`): the input batch of the training step.
        - **baselines** (`None` | `int` | `float` | `Tensor` | `tuple[int | float | Tensor, ...]`): starting point from which integral is computed in this method. Please refer to the [Captum documentation](https://captum.ai/api/layer.html#captum.attr.LayerConductance.attribute) for more details.
        - **target** (`Tensor` | `None`): the target batch of the training step.
        - **batch_idx** (`int`): the index of the current batch. This is an argument of the forward function during training.
        - **num_batches** (`int`): the number of batches in the training step. This is an argument of the forward function during training.- **mask** (`Tensor`): the mask tensor of the layer. It has the same size as the feature tensor with size (number of units, ).

        **Returns:**
        - **importance_step_layer** (`Tensor`): the neuron-wise importance of the layer of the training step.
        """
        layer = self.backbone.get_layer_by_name(layer_name)

        # initialize the Layer Conductance object
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

        importance_step_layer = attribution_abs_batch_mean
        importance_step_layer = importance_step_layer.detach()

        return importance_step_layer

    def get_importance_step_layer_internal_influence_abs(
        self: str,
        layer_name: str,
        input: Tensor | tuple[Tensor, ...],
        baselines: None | int | float | Tensor | tuple[int | float | Tensor, ...],
        target: Tensor | None,
        batch_idx: int,
        num_batches: int,
    ) -> Tensor:
        r"""Get the raw neuron-wise importance (before scaling) of a layer of a training step. See $I^{\tau}_l(\mathbf{x},y)$ (before Eqs. (5) and (6)) in the paper. This method uses the absolute values of [internal influence](https://openreview.net/forum?id=SJPpHzW0-). We implement this using [Internal Influence](https://captum.ai/api/layer.html#internal-influence) in Captum.

        **Args:**
        - **layer_name** (`str`): the name of layer to get neuron-wise importance.
        - **input** (`Tensor` | `tuple[Tensor, ...]`): the input batch of the training step.
        - **baselines** (`None` | `int` | `float` | `Tensor` | `tuple[int | float | Tensor, ...]`): starting point from which integral is computed in this method. Please refer to the [Captum documentation](https://captum.ai/api/layer.html#captum.attr.InternalInfluence.attribute) for more details.
        - **target** (`Tensor` | `None`): the target batch of the training step.
        - **batch_idx** (`int`): the index of the current batch. This is an argument of the forward function during training.
        - **num_batches** (`int`): the number of batches in the training step. This is an argument of the forward function during training.

        **Returns:**
        - **importance_step_layer** (`Tensor`): the neuron-wise importance of the layer of the training step.
        """
        layer = self.backbone.get_layer_by_name(layer_name)

        # initialize the Internal Influence object
        internal_influence = InternalInfluence(forward_func=self.forward, layer=layer)

        # convert the target to long type to avoid error
        target = target.long() if target is not None else None

        self.set_forward_func_return_logits_only(True)
        # calculate layer attribution of the step
        attribution = internal_influence.attribute(
            inputs=input,
            baselines=baselines,
            target=target,
            additional_forward_args=("train", batch_idx, num_batches, self.task_id),
            n_steps=5,  # set 10 instead of default 50 to accelerate the computation
        )
        self.set_forward_func_return_logits_only(False)

        attribution_abs_batch_mean = torch.mean(
            torch.abs(attribution),
            dim=[
                i for i in range(attribution.dim()) if i != 1
            ],  # average the features over batch samples
        )

        importance_step_layer = attribution_abs_batch_mean
        importance_step_layer = importance_step_layer.detach()

        return importance_step_layer

    def get_importance_step_layer_gradcam_abs(
        self: str,
        layer_name: str,
        input: Tensor | tuple[Tensor, ...],
        target: Tensor | None,
        batch_idx: int,
        num_batches: int,
    ) -> Tensor:
        r"""Get the raw neuron-wise importance (before scaling) of a layer of a training step. See $I^{\tau}_l(\mathbf{x},y)$ (before Eqs. (5) and (6)) in the paper. This method uses the absolute values of [Grad-CAM](https://openreview.net/forum?id=SJPpHzW0-). We implement this using [Layer Grad-CAM](https://captum.ai/api/layer.html#gradcam) in Captum.

        **Args:**
        - **layer_name** (`str`): the name of layer to get neuron-wise importance.
        - **input** (`Tensor` | `tuple[Tensor, ...]`): the input batch of the training step.
        - **target** (`Tensor` | `None`): the target batch of the training step.
        - **batch_idx** (`int`): the index of the current batch. This is an argument of the forward function during training.
        - **num_batches** (`int`): the number of batches in the training step. This is an argument of the forward function during training.

        **Returns:**
        - **importance_step_layer** (`Tensor`): the neuron-wise importance of the layer of the training step.
        """
        layer = self.backbone.get_layer_by_name(layer_name)

        # initialize the GradCAM object
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

        importance_step_layer = attribution_abs_batch_mean
        importance_step_layer = importance_step_layer.detach()

        return importance_step_layer

    def get_importance_step_layer_deeplift_abs(
        self: str,
        layer_name: str,
        input: Tensor | tuple[Tensor, ...],
        baselines: None | int | float | Tensor | tuple[int | float | Tensor, ...],
        target: Tensor | None,
        batch_idx: int,
        num_batches: int,
    ) -> Tensor:
        r"""Get the raw neuron-wise importance (before scaling) of a layer of a training step. See $I^{\tau}_l(\mathbf{x},y)$ (before Eqs. (5) and (6)) in the paper. This method uses the absolute values of [DeepLift](https://proceedings.mlr.press/v70/shrikumar17a/shrikumar17a.pdf). We implement this using [Layer DeepLift](https://captum.ai/api/layer.html#layer-deeplift) in Captum.

        **Args:**
        - **layer_name** (`str`): the name of layer to get neuron-wise importance.
        - **input** (`Tensor` | `tuple[Tensor, ...]`): the input batch of the training step.
        - **baselines** (`None` | `int` | `float` | `Tensor` | `tuple[int | float | Tensor, ...]`): baselines define reference samples that are compared with the inputs. Please refer to the [Captum documentation](https://captum.ai/api/layer.html#captum.attr.LayerDeepLift.attribute) for more details.
        - **target** (`Tensor` | `None`): the target batch of the training step.
        - **batch_idx** (`int`): the index of the current batch. This is an argument of the forward function during training.
        - **num_batches** (`int`): the number of batches in the training step. This is an argument of the forward function during training.

        **Returns:**
        - **importance_step_layer** (`Tensor`): the neuron-wise importance of the layer of the training step.
        """
        layer = self.backbone.get_layer_by_name(layer_name)

        # initialize the Layer DeepLift object
        layer_deeplift = LayerDeepLift(model=self, layer=layer)

        # convert the target to long type to avoid error
        target = target.long() if target is not None else None

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

        importance_step_layer = attribution_abs_batch_mean
        importance_step_layer = importance_step_layer.detach()

        return importance_step_layer

    def get_importance_step_layer_deepliftshap_abs(
        self: str,
        layer_name: str,
        input: Tensor | tuple[Tensor, ...],
        baselines: None | int | float | Tensor | tuple[int | float | Tensor, ...],
        target: Tensor | None,
        batch_idx: int,
        num_batches: int,
    ) -> Tensor:
        r"""Get the raw neuron-wise importance (before scaling) of a layer of a training step. See $I^{\tau}_l(\mathbf{x},y)$ (before Eqs. (5) and (6)) in the paper. This method uses the absolute values of [DeepLift SHAP](https://proceedings.neurips.cc/paper_files/paper/2017/file/8a20a8621978632d76c43dfd28b67767-Paper.pdf). We implement this using [Layer DeepLiftShap](https://captum.ai/api/layer.html#layer-deepliftshap) in Captum.

        **Args:**
        - **layer_name** (`str`): the name of layer to get neuron-wise importance.
        - **input** (`Tensor` | `tuple[Tensor, ...]`): the input batch of the training step.
        - **baselines** (`None` | `int` | `float` | `Tensor` | `tuple[int | float | Tensor, ...]`): baselines define reference samples that are compared with the inputs. Please refer to the [Captum documentation](https://captum.ai/api/layer.html#captum.attr.LayerDeepLiftShap.attribute) for more details.
        - **target** (`Tensor` | `None`): the target batch of the training step.
        - **batch_idx** (`int`): the index of the current batch. This is an argument of the forward function during training.
        - **num_batches** (`int`): the number of batches in the training step. This is an argument of the forward function during training.

        **Returns:**
        - **importance_step_layer** (`Tensor`): the neuron-wise importance of the layer of the training step.
        """
        layer = self.backbone.get_layer_by_name(layer_name)

        # initialize the Layer DeepLiftShap object
        layer_deepliftshap = LayerDeepLiftShap(model=self, layer=layer)

        # convert the target to long type to avoid error
        target = target.long() if target is not None else None

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

        importance_step_layer = attribution_abs_batch_mean
        importance_step_layer = importance_step_layer.detach()

        return importance_step_layer

    def get_importance_step_layer_gradientshap_abs(
        self: str,
        layer_name: str,
        input: Tensor | tuple[Tensor, ...],
        baselines: None | int | float | Tensor | tuple[int | float | Tensor, ...],
        target: Tensor | None,
        batch_idx: int,
        num_batches: int,
    ) -> Tensor:
        r"""Get the raw neuron-wise importance (before scaling) of a layer of a training step. See $I^{\tau}_l(\mathbf{x},y)$ (before Eqs. (5) and (6)) in the paper. This method uses the absolute values of gradient SHAP. We implement this using [Layer GradientShap](https://captum.ai/api/layer.html#layer-gradientshap) in Captum.

        **Args:**
        - **layer_name** (`str`): the name of layer to get neuron-wise importance.
        - **input** (`Tensor` | `tuple[Tensor, ...]`): the input batch of the training step.
        - **baselines** (`None` | `int` | `float` | `Tensor` | `tuple[int | float | Tensor, ...]`): starting point from which expectation is computed. Please refer to the [Captum documentation](https://captum.ai/api/layer.html#captum.attr.LayerGradientShap.attribute) for more details. If `None`, the baselines are set to zero.
        - **target** (`Tensor` | `None`): the target batch of the training step.
        - **batch_idx** (`int`): the index of the current batch. This is an argument of the forward function during training.
        - **num_batches** (`int`): the number of batches in the training step. This is an argument of the forward function during training.

        **Returns:**
        - **importance_step_layer** (`Tensor`): the neuron-wise importance of the layer of the training step.
        """
        layer = self.backbone.get_layer_by_name(layer_name)

        if baselines is None:
            baselines = torch.zeros_like(
                input
            )  # baselines are mandatory for GradientShap API. We explicitly set them to zero

        # initialize the Layer GradientShap object
        layer_gradientshap = LayerGradientShap(forward_func=self.forward, layer=layer)

        # convert the target to long type to avoid error
        target = target.long() if target is not None else None

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

        importance_step_layer = attribution_abs_batch_mean
        importance_step_layer = importance_step_layer.detach()

        return importance_step_layer

    def get_importance_step_layer_integrated_gradients_abs(
        self: str,
        layer_name: str,
        input: Tensor | tuple[Tensor, ...],
        baselines: None | int | float | Tensor | tuple[int | float | Tensor, ...],
        target: Tensor | None,
        batch_idx: int,
        num_batches: int,
    ) -> Tensor:
        r"""Get the raw neuron-wise importance (before scaling) of a layer of a training step. See $I^{\tau}_l(\mathbf{x},y)$ (before Eqs. (5) and (6)) in the paper. This method uses the absolute values of [integrated gradients](https://proceedings.mlr.press/v70/sundararajan17a/sundararajan17a.pdf). We implement this using [Layer Integrated Gradients](https://captum.ai/api/layer.html#layer-integrated-gradients) in Captum.

        **Args:**
        - **layer_name** (`str`): the name of layer to get neuron-wise importance.
        - **input** (`Tensor` | `tuple[Tensor, ...]`): the input batch of the training step.
        - **baselines** (`None` | `int` | `float` | `Tensor` | `tuple[int | float | Tensor, ...]`): starting point from which integral is computed. Please refer to the [Captum documentation](https://captum.ai/api/layer.html#captum.attr.LayerIntegratedGradients.attribute) for more details.
        - **target** (`Tensor` | `None`): the target batch of the training step.
        - **batch_idx** (`int`): the index of the current batch. This is an argument of the forward function during training.
        - **num_batches** (`int`): the number of batches in the training step. This is an argument of the forward function during training.

        **Returns:**
        - **importance_step_layer** (`Tensor`): the neuron-wise importance of the layer of the training step.
        """
        layer = self.backbone.get_layer_by_name(layer_name)

        # initialize the Layer Integrated Gradients object
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

        importance_step_layer = attribution_abs_batch_mean
        importance_step_layer = importance_step_layer.detach()

        return importance_step_layer

    def get_importance_step_layer_feature_ablation_abs(
        self: str,
        layer_name: str,
        input: Tensor | tuple[Tensor, ...],
        layer_baselines: None | int | float | Tensor | tuple[int | float | Tensor, ...],
        target: Tensor | None,
        batch_idx: int,
        num_batches: int,
        if_captum: bool = False,
    ) -> Tensor:
        r"""Get the raw neuron-wise importance (before scaling) of a layer of a training step. See $I^{\tau}_l(\mathbf{x},y)$ (before Eqs. (5) and (6)) in the paper. This method uses the absolute values of [feature ablation](https://link.springer.com/chapter/10.1007/978-3-319-10590-1_53) attribution. We implement this using [Layer Feature Ablation](https://captum.ai/api/layer.html#layer-feature-ablation) in Captum.

        **Args:**
        - **layer_name** (`str`): the name of layer to get neuron-wise importance.
        - **input** (`Tensor` | `tuple[Tensor, ...]`): the input batch of the training step.
        - **layer_baselines** (`None` | `int` | `float` | `Tensor` | `tuple[int | float | Tensor, ...]`): reference values which replace each layer input / output value when ablated. Please refer to the [Captum documentation](https://captum.ai/api/layer.html#captum.attr.LayerFeatureAblation.attribute) for more details.
        - **target** (`Tensor` | `None`): the target batch of the training step.
        - **batch_idx** (`int`): the index of the current batch. This is an argument of the forward function during training.
        - **num_batches** (`int`): the number of batches in the training step. This is an argument of the forward function during training.
        - **if_captum** (`bool`): whether to use Captum or not. If `True`, we use Captum to calculate the feature ablation. If `False`, we use our implementation. Default is `False`, because our implementation is much faster.

        **Returns:**
        - **importance_step_layer** (`Tensor`): the neuron-wise importance of the layer of the training step.
        """
        layer = self.backbone.get_layer_by_name(layer_name)

        if not if_captum:
            # 1. Baseline logits (take first element of forward output)
            baseline_out, _, _ = self.forward(
                input, "train", batch_idx, num_batches, self.task_id
            )
            if target is not None:
                baseline_scores = baseline_out.gather(1, target.view(-1, 1)).squeeze(1)
            else:
                baseline_scores = baseline_out.sum(dim=1)

            # 2. Capture layer’s output shape
            activs = {}
            handle = layer.register_forward_hook(
                lambda module, inp, out: activs.setdefault("output", out.detach())
            )
            _, _, _ = self.forward(input, "train", batch_idx, num_batches, self.task_id)
            handle.remove()
            layer_output = activs["output"]  # shape (B, F, ...)

            # 3. Build baseline tensor matching that shape
            if layer_baselines is None:
                baseline_tensor = torch.zeros_like(layer_output)
            elif isinstance(layer_baselines, (int, float)):
                baseline_tensor = torch.full_like(layer_output, layer_baselines)
            elif isinstance(layer_baselines, Tensor):
                if layer_baselines.shape == layer_output.shape:
                    baseline_tensor = layer_baselines
                elif layer_baselines.shape == layer_output.shape[1:]:
                    baseline_tensor = layer_baselines.unsqueeze(0).repeat(
                        layer_output.size(0), *([1] * layer_baselines.ndim)
                    )
                else:
                    raise ValueError(...)
            else:
                raise ValueError(...)

            B, F = layer_output.size(0), layer_output.size(1)

            # 4. Create a “mega-batch” replicating the input F times
            if isinstance(input, tuple):
                mega_inputs = tuple(
                    t.unsqueeze(0).repeat(F, *([1] * t.ndim)).view(-1, *t.shape[1:])
                    for t in input
                )
            else:
                mega_inputs = (
                    input.unsqueeze(0)
                    .repeat(F, *([1] * input.ndim))
                    .view(-1, *input.shape[1:])
                )

            # 5. Equally replicate the baseline tensor
            mega_baseline = (
                baseline_tensor.unsqueeze(0)
                .repeat(F, *([1] * baseline_tensor.ndim))
                .view(-1, *baseline_tensor.shape[1:])
            )

            # 6. Precompute vectorized indices
            device = layer_output.device
            positions = torch.arange(F * B, device=device)  # [0,1,...,F*B-1]
            feat_idx = torch.arange(F, device=device).repeat_interleave(
                B
            )  # [0,0,...,1,1,...,F-1]

            # 7. One hook to zero out each channel slice across the mega-batch
            def mega_ablate_hook(module, inp, out):
                out_mod = out.clone()
                # for each sample in mega-batch, zero its corresponding channel
                out_mod[positions, feat_idx] = mega_baseline[positions, feat_idx]
                return out_mod

            h = layer.register_forward_hook(mega_ablate_hook)
            out_all, _, _ = self.forward(
                mega_inputs, "train", batch_idx, num_batches, self.task_id
            )
            h.remove()

            # 8. Recover scores, reshape [F*B] → [F, B], diff & mean
            if target is not None:
                tgt_flat = target.unsqueeze(0).repeat(F, 1).view(-1)
                scores_all = out_all.gather(1, tgt_flat.view(-1, 1)).squeeze(1)
            else:
                scores_all = out_all.sum(dim=1)

            scores_all = scores_all.view(F, B)
            diffs = torch.abs(baseline_scores.unsqueeze(0) - scores_all)
            importance_step_layer = diffs.mean(dim=1).detach()  # [F]

            return importance_step_layer

        else:
            # initialize the Layer Feature Ablation object
            layer_feature_ablation = LayerFeatureAblation(
                forward_func=self.forward, layer=layer
            )

            # calculate layer attribution of the step
            self.set_forward_func_return_logits_only(True)
            attribution = layer_feature_ablation.attribute(
                inputs=input,
                layer_baselines=layer_baselines,
                # target=target, # disable target to enable perturbations_per_eval
                additional_forward_args=("train", batch_idx, num_batches, self.task_id),
                perturbations_per_eval=128,  # to accelerate the computation
            )
            self.set_forward_func_return_logits_only(False)

            attribution_abs_batch_mean = torch.mean(
                torch.abs(attribution),
                dim=[
                    i for i in range(attribution.dim()) if i != 1
                ],  # average the features over batch samples
            )

        importance_step_layer = attribution_abs_batch_mean
        importance_step_layer = importance_step_layer.detach()

        return importance_step_layer

    def get_importance_step_layer_lrp_abs(
        self: str,
        layer_name: str,
        input: Tensor | tuple[Tensor, ...],
        target: Tensor | None,
        batch_idx: int,
        num_batches: int,
    ) -> Tensor:
        r"""Get the raw neuron-wise importance (before scaling) of a layer of a training step. See $I^{\tau}_l(\mathbf{x},y)$ (before Eqs. (5) and (6)) in the paper. This method uses the absolute values of [LRP](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0130140). We implement this using [Layer LRP](https://captum.ai/api/layer.html#layer-lrp) in Captum.

        **Args:**
        - **layer_name** (`str`): the name of layer to get neuron-wise importance.
        - **input** (`Tensor` | `tuple[Tensor, ...]`): the input batch of the training step.
        - **target** (`Tensor` | `None`): the target batch of the training step.
        - **batch_idx** (`int`): the index of the current batch. This is an argument of the forward function during training.
        - **num_batches** (`int`): the number of batches in the training step. This is an argument of the forward function during training.

        **Returns:**
        - **importance_step_layer** (`Tensor`): the neuron-wise importance of the layer of the training step.
        """
        layer = self.backbone.get_layer_by_name(layer_name)

        # initialize the Layer LRP object
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

        importance_step_layer = attribution_abs_batch_mean
        importance_step_layer = importance_step_layer.detach()

        return importance_step_layer

    def get_importance_step_layer_cbp_adaptive_contribution(
        self: str,
        layer_name: str,
        activation: Tensor,
    ) -> Tensor:
        r"""Get the raw neuron-wise importance (before scaling) of a layer of a training step. See $I^{\tau}_l(\mathbf{x},y)$ (before Eqs. (5) and (6)) in the paper. This method uses the sum of absolute values of layer output weights multiplied by absolute values of activation, then divided by the reciprocal of sum of absolute values of layer input weights. It is equal to the adaptive contribution utility in [CBP](https://www.nature.com/articles/s41586-024-07711-7).

        **Args:**
        - **layer_name** (`str`): the name of layer to get neuron-wise importance.
        - **activation** (`Tensor`): the activation tensor of the layer. It has the same size of (number of units, ).

        **Returns:**
        - **importance_step_layer** (`Tensor`): the neuron-wise importance of the layer of the training step.
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

        importance_step_layer = (
            output_weight_abs_sum
            * activation_abs_batch_mean
            * input_weight_abs_sum_reciprocal
        )
        importance_step_layer = importance_step_layer.detach()

        return importance_step_layer
