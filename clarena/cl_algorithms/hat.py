r"""
The submodule in `cl_algorithms` for [HAT (Hard Attention to the Task) algorithm](http://proceedings.mlr.press/v80/serra18a).
"""

__all__ = ["HAT"]

import logging
from typing import Any

import torch
from torch import Tensor
from torch.utils.data import DataLoader

from clarena.backbones import HATMaskBackbone
from clarena.cl_algorithms import CLAlgorithm
from clarena.cl_algorithms.regularizers import HATMaskSparsityReg
from clarena.heads import HeadsTIL
from clarena.utils.metrics import HATNetworkCapacityMetric

# always get logger for built-in logging in each module
pylogger = logging.getLogger(__name__)


class HAT(CLAlgorithm):
    r"""[HAT (Hard Attention to the Task)](http://proceedings.mlr.press/v80/serra18a) algorithm.

    An architecture-based continual learning approach that uses learnable hard attention masks to select task-specific parameters.
    """

    def __init__(
        self,
        backbone: HATMaskBackbone,
        heads: HeadsTIL,
        adjustment_mode: str,
        s_max: float,
        clamp_threshold: float,
        mask_sparsity_reg_factor: float,
        mask_sparsity_reg_mode: str = "original",
        task_embedding_init_mode: str = "N01",
        alpha: float | None = None,
        non_algorithmic_hparams: dict[str, Any] = {},
    ) -> None:
        r"""Initialize the HAT algorithm with the network.

        **Args:**
        - **backbone** (`HATMaskBackbone`): must be a backbone network with the HAT mask mechanism.
        - **heads** (`HeadsTIL`): output heads. HAT only supports TIL (Task-Incremental Learning).
        - **adjustment_mode** (`str`): the strategy of adjustment (i.e., the mode of gradient clipping), must be one of:
            1. 'hat': set gradients of parameters linking to masked units to zero. This is how HAT fixes the part of the network for previous tasks completely. See Eq. (2) in Sec. 2.3 "Network Training" in the [HAT paper](http://proceedings.mlr.press/v80/serra18a).
            2. 'hat_random': set gradients of parameters linking to masked units to random 0–1 values. See "Baselines" in Sec. 4.1 in the [AdaHAT paper](https://link.springer.com/chapter/10.1007/978-3-031-70352-2_9).
            3. 'hat_const_alpha': set gradients of parameters linking to masked units to a constant value `alpha`. See "Baselines" in Sec. 4.1 in the [AdaHAT paper](https://link.springer.com/chapter/10.1007/978-3-031-70352-2_9).
            4. 'hat_const_1': set gradients of parameters linking to masked units to a constant value of 1 (i.e., no gradient constraint). See "Baselines" in Sec. 4.1 in the [AdaHAT paper](https://link.springer.com/chapter/10.1007/978-3-031-70352-2_9).
        - **s_max** (`float`): hyperparameter, the maximum scaling factor in the gate function. See Sec. 2.4 "Hard Attention Training" in the [HAT paper](http://proceedings.mlr.press/v80/serra18a).
        - **clamp_threshold** (`float`): the threshold for task embedding gradient compensation. See Sec. 2.5 "Embedding Gradient Compensation" in the [HAT paper](http://proceedings.mlr.press/v80/serra18a).
        - **mask_sparsity_reg_factor** (`float`): hyperparameter, the regularization factor for mask sparsity.
        - **mask_sparsity_reg_mode** (`str`): the mode of mask sparsity regularization, must be one of:
            1. 'original' (default): the original mask sparsity regularization in the HAT paper.
            2. 'cross': the cross version of mask sparsity regularization.
        - **task_embedding_init_mode** (`str`): the initialization mode for task embeddings, must be one of:
            1. 'N01' (default): standard normal distribution $N(0, 1)$.
            2. 'U-11': uniform distribution $U(-1, 1)$.
            3. 'U01': uniform distribution $U(0, 1)$.
            4. 'U-10': uniform distribution $U(-1, 0)$.
            5. 'last': inherit the task embedding from the last task.
        - **alpha** (`float` | `None`): the `alpha` in the 'HAT-const-alpha' mode. Applies only when `adjustment_mode` is 'hat_const_alpha'.
        - **non_algorithmic_hparams** (`dict[str, Any]`): non-algorithmic hyperparameters that are not related to the algorithm itself are passed to this `LightningModule` object from the config, such as optimizer and learning rate scheduler configurations. They are saved for Lightning APIs from `save_hyperparameters()` method. This is useful for the experiment configuration and reproducibility.

        """
        super().__init__(
            backbone=backbone,
            heads=heads,
            non_algorithmic_hparams=non_algorithmic_hparams,
        )

        # save additional algorithmic hyperparameters
        self.save_hyperparameters(
            "adjustment_mode",
            "s_max",
            "clamp_threshold",
            "mask_sparsity_reg_factor",
            "mask_sparsity_reg_mode",
            "task_embedding_init_mode",
            "alpha",
        )

        self.adjustment_mode: str = adjustment_mode
        r"""The adjustment mode for gradient clipping."""
        self.s_max: float = s_max
        r"""The hyperparameter s_max."""
        self.clamp_threshold: float = clamp_threshold
        r"""The clamp threshold for task embedding gradient compensation."""
        self.mask_sparsity_reg_factor: float = mask_sparsity_reg_factor
        r"""The mask sparsity regularization factor."""
        self.mask_sparsity_reg_mode: str = mask_sparsity_reg_mode
        r"""The mask sparsity regularization mode."""
        self.mark_sparsity_reg: HATMaskSparsityReg = HATMaskSparsityReg(
            factor=mask_sparsity_reg_factor, mode=mask_sparsity_reg_mode
        )
        r"""The mask sparsity regularizer."""
        self.task_embedding_init_mode: str = task_embedding_init_mode
        r"""The task embedding initialization mode."""
        self.alpha: float | None = alpha
        r"""The hyperparameter alpha for `hat_const_alpha`."""
        # self.epsilon: float | None = None
        # r"""HAT doesn't use epsilon for `hat_const_alpha`. It is kept for consistency with `epsilon` in `clip_grad_by_adjustment()` in `HATMaskBackbone`."""

        self.cumulative_mask_for_previous_tasks: dict[str, Tensor] = {}
        r"""The cumulative binary attention mask $\mathrm{M}^{<t}$ of previous tasks $1,\cdots, t-1$, gated from the task embedding ($t$ is `self.task_id`). It is a dict where keys are layer names and values are the binary mask tensors for the layers. The mask tensor has size (number of units, ). """

        # set manual optimization
        self.automatic_optimization = False

        HAT.sanity_check(self)

    def sanity_check(self) -> None:
        r"""Sanity check."""

        # check the backbone and heads
        if not isinstance(self.backbone, HATMaskBackbone):
            raise ValueError("The backbone should be an instance of `HATMaskBackbone`.")
        if not isinstance(self.heads, HeadsTIL):
            raise ValueError("The heads should be an instance of `HeadsTIL`.")

        # check marker sparsity regularization mode
        if self.mask_sparsity_reg_mode not in ["original", "cross"]:
            raise ValueError(
                "The mask_sparsity_reg_mode should be one of 'original', 'cross'."
            )

        # check task embedding initialization mode
        if self.task_embedding_init_mode not in [
            "N01",
            "U01",
            "U-10",
            "masked",
            "unmasked",
        ]:
            raise ValueError(
                "The task_embedding_init_mode should be one of 'N01', 'U01', 'U-10', 'masked', 'unmasked'."
            )

        # check adjustment mode `hat_const_alpha`
        if self.adjustment_mode == "hat_const_alpha" and self.alpha is None:
            raise ValueError(
                "Alpha should be given when the adjustment_mode is 'hat_const_alpha'."
            )

    def on_train_start(self) -> None:
        r"""Initialize the task embedding before training the next task and initialize the cumulative mask at the beginning of the first task."""

        self.backbone.initialize_task_embedding(mode=self.task_embedding_init_mode)

        self.backbone.initialize_independent_bn()

        # initialize the cumulative mask for the first task at the beginning of the first task. This should not be called in `__init__()` because `self.device` is not available at that time.
        if self.task_id == 1:
            for layer_name in self.backbone.weighted_layer_names:
                layer = self.backbone.get_layer_by_name(
                    layer_name
                )  # get the layer by its name
                num_units = layer.weight.shape[0]

                self.cumulative_mask_for_previous_tasks[layer_name] = torch.zeros(
                    num_units
                ).to(
                    self.device
                )  # the cumulative mask $\mathrm{M}^{<t}$ is initialized as a zeros mask ($t = 1$). See Eq. (2) in Sec. 3 in the [AdaHAT paper](https://link.springer.com/chapter/10.1007/978-3-031-70352-2_9), or Eq. (5) in Sec. 2.6 "Promoting Low Capacity Usage" in the [HAT paper](http://proceedings.mlr.press/v80/serra18a)

                # self.neuron_first_task[layer_name] = [None] * num_units

    def clip_grad_by_adjustment(
        self,
        **kwargs,
    ) -> tuple[dict[str, Tensor], dict[str, Tensor], Tensor]:
        r"""Clip the gradients by the adjustment rate. See Eq. (2) in Sec. 2.3 "Network Training" in the [HAT paper](http://proceedings.mlr.press/v80/serra18a).

        Note that because the task embedding fully covers every layer in the backbone network, no parameters are left out of this system.
        This applies not only to parameters between layers with task embeddings, but also to those before the first layer. We design it separately in the code.

        Network capacity is measured alongside this method. Network capacity is defined as the average adjustment rate over all parameters.
        See Sec. 4.1 in the [AdaHAT paper](https://link.springer.com/chapter/10.1007/978-3-031-70352-2_9).

        **Returns:**
        - **adjustment_rate_weight** (`dict[str, Tensor]`): the adjustment rate for weights. Keys (`str`) are layer names and values (`Tensor`) are the adjustment rate tensors.
        - **adjustment_rate_bias** (`dict[str, Tensor]`): the adjustment rate for biases. Keys (`str`) are layer name and values (`Tensor`) are the adjustment rate tensors.
        - **capacity** (`Tensor`): the calculated network capacity.
        """

        # initialize network capacity metric
        capacity = HATNetworkCapacityMetric().to(self.device)
        adjustment_rate_weight = {}
        adjustment_rate_bias = {}

        # calculate the adjustment rate for gradients of the parameters, both weights and biases (if they exist)
        for layer_name in self.backbone.weighted_layer_names:

            layer = self.backbone.get_layer_by_name(
                layer_name
            )  # get the layer by its name

            # placeholder for the adjustment rate to avoid the error of using it before assignment
            adjustment_rate_weight_layer = 1
            adjustment_rate_bias_layer = 1

            weight_mask, bias_mask = self.backbone.get_layer_measure_parameter_wise(
                neuron_wise_measure=self.cumulative_mask_for_previous_tasks,
                layer_name=layer_name,
                aggregation_mode="min",
            )

            if self.adjustment_mode == "hat":
                adjustment_rate_weight_layer = 1 - weight_mask
                adjustment_rate_bias_layer = 1 - bias_mask

            elif self.adjustment_mode == "hat_random":
                adjustment_rate_weight_layer = torch.rand_like(
                    weight_mask
                ) * weight_mask + (1 - weight_mask)
                adjustment_rate_bias_layer = torch.rand_like(bias_mask) * bias_mask + (
                    1 - bias_mask
                )

            elif self.adjustment_mode == "hat_const_alpha":
                adjustment_rate_weight_layer = self.alpha * torch.ones_like(
                    weight_mask
                ) * weight_mask + (1 - weight_mask)
                adjustment_rate_bias_layer = self.alpha * torch.ones_like(
                    bias_mask
                ) * bias_mask + (1 - bias_mask)

            elif self.adjustment_mode == "hat_const_1":
                adjustment_rate_weight_layer = torch.ones_like(
                    weight_mask
                ) * weight_mask + (1 - weight_mask)
                adjustment_rate_bias_layer = torch.ones_like(bias_mask) * bias_mask + (
                    1 - bias_mask
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

    def compensate_task_embedding_gradients(
        self,
        batch_idx: int,
        num_batches: int,
    ) -> None:
        r"""Compensate the gradients of task embeddings during training. See Sec. 2.5 "Embedding Gradient Compensation" in the [HAT paper](http://proceedings.mlr.press/v80/serra18a).

        **Args:**
        - **batch_idx** (`int`): the current training batch index.
        - **num_batches** (`int`): the total number of training batches.
        """

        for te in self.backbone.task_embedding_t.values():
            anneal_scalar = 1 / self.s_max + (self.s_max - 1 / self.s_max) * (
                batch_idx - 1
            ) / (
                num_batches - 1
            )  # see Eq. (3) in Sec. 2.4 "Hard Attention Training" in the [HAT paper](http://proceedings.mlr.press/v80/serra18a)

            num = (
                torch.cosh(
                    torch.clamp(
                        anneal_scalar * te.weight.data,
                        -self.clamp_threshold,
                        self.clamp_threshold,
                    )
                )
                + 1
            )

            den = torch.cosh(te.weight.data) + 1

            compensation = self.s_max / anneal_scalar * num / den

            te.weight.grad.data *= compensation

    def forward(
        self,
        input: torch.Tensor,
        stage: str,
        task_id: int | None = None,
        batch_idx: int | None = None,
        num_batches: int | None = None,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        r"""The forward pass for data from task `task_id`. Note that it is nothing to do with `forward()` method in `nn.Module`.

        **Args:**
        - **input** (`Tensor`): The input tensor from data.
        - **stage** (`str`): the stage of the forward pass; one of:
            1. 'train': training stage.
            2. 'validation': validation stage.
            3. 'test': testing stage.
        - **task_id** (`int`| `None`): the task ID where the data are from. If the stage is 'train' or 'validation', it should be the current task `self.task_id`. If stage is 'test', it could be from any seen task. In TIL, the task IDs of test data are provided thus this argument can be used. HAT algorithm works only for TIL.
        - **batch_idx** (`int` | `None`): the current batch index. Applies only to training stage. For other stages, it is default `None`.
        - **num_batches** (`int` | `None`): the total number of batches. Applies only to training stage. For other stages, it is default `None`.

        **Returns:**
        - **logits** (`Tensor`): the output logits tensor.
        - **mask** (`dict[str, Tensor]`): the mask for the current task. Key (`str`) is layer name, value (`Tensor`) is the mask tensor. The mask tensor has size (number of units, ).
        - **activations** (`dict[str, Tensor]`): the hidden features (after activation) in each weighted layer. Key (`str`) is the weighted layer name, value (`Tensor`) is the hidden feature tensor. This is used for the continual learning algorithms that need to use the hidden features for various purposes. Although HAT algorithm does not need this, it is still provided for API consistence for other HAT-based algorithms inherited this `forward()` method of `HAT` class.
        """
        feature, mask, activations = self.backbone(
            input,
            stage=stage,
            s_max=self.s_max if stage == "train" or stage == "validation" else None,
            batch_idx=batch_idx if stage == "train" else None,
            num_batches=num_batches if stage == "train" else None,
            test_task_id=task_id if stage == "test" else None,
        )
        logits = self.heads(feature, task_id)

        return (
            logits
            if self.if_forward_func_return_logits_only
            else (logits, mask, activations)
        )

    def training_step(self, batch: Any, batch_idx: int) -> dict[str, Tensor]:
        r"""Training step for current task `self.task_id`.

        **Args:**
        - **batch** (`Any`): a batch of training data.
        - **batch_idx** (`int`): the index of the batch. Used for calculating annealed scalar in HAT. See Sec. 2.4 "Hard Attention Training" in the [HAT paper](http://proceedings.mlr.press/v80/serra18a).

        **Returns:**
        - **outputs** (`dict[str, Tensor]`): a dictionary containing loss and other metrics from this training step. Keys (`str`) are metric names, and values (`Tensor`) are the metrics. Must include the key 'loss' (total loss) in the case of automatic optimization, according to PyTorch Lightning. For HAT, it includes 'mask' and 'capacity' for logging.
        """
        x, y = batch

        # zero the gradients before forward pass in manual optimization mode
        opt = self.optimizers()
        opt.zero_grad()

        # classification loss
        num_batches = self.trainer.num_training_batches
        logits, mask, activations = self.forward(
            x,
            stage="train",
            batch_idx=batch_idx,
            num_batches=num_batches,
            task_id=self.task_id,
        )
        loss_cls = self.criterion(logits, y)

        # regularization loss. See Sec. 2.6 "Promoting Low Capacity Usage" in the [HAT paper](http://proceedings.mlr.press/v80/serra18a)
        loss_reg, network_sparsity = self.mark_sparsity_reg(
            mask, self.cumulative_mask_for_previous_tasks
        )

        # total loss. See Eq. (4) in Sec. 2.6 "Promoting Low Capacity Usage" in the [HAT paper](http://proceedings.mlr.press/v80/serra18a)
        loss = loss_cls + loss_reg

        # backward step (manually)
        self.manual_backward(loss)  # calculate the gradients
        # HAT hard-clips gradients using the cumulative masks. See Eq. (2) in Sec. 2.3 "Network Training" in the HAT paper.
        # Network capacity is computed along with this process (defined as the average adjustment rate over all parameters; see Sec. 4.1 in the [AdaHAT paper](https://link.springer.com/chapter/10.1007/978-3-031-70352-2_9)).

        adjustment_rate_weight, adjustment_rate_bias, capacity = (
            self.clip_grad_by_adjustment(
                network_sparsity=network_sparsity,  # passed for compatibility with AdaHAT, which inherits this method
            )
        )
        # compensate the gradients of task embedding. See Sec. 2.5 "Embedding Gradient Compensation" in the [HAT paper](http://proceedings.mlr.press/v80/serra18a)
        self.compensate_task_embedding_gradients(
            batch_idx=batch_idx,
            num_batches=num_batches,
        )
        # update parameters with the modified gradients
        opt.step()

        # accuracy of the batch
        acc = (logits.argmax(dim=1) == y).float().mean()

        return {
            "loss": loss,  # return loss is essential for training step, or backpropagation will fail
            "loss_cls": loss_cls,
            "loss_reg": loss_reg,
            "acc": acc,
            "activations": activations,
            "logits": logits,
            "mask": mask,  # return other metrics for lightning loggers callback to handle at `on_train_batch_end()`
            "input": x,  # return the input batch for Captum to use
            "target": y,  # return the target batch for Captum to use
            "adjustment_rate_weight": adjustment_rate_weight,  # return the adjustment rate for weights and biases for logging
            "adjustment_rate_bias": adjustment_rate_bias,
            "capacity": capacity,  # return the network capacity for logging
        }

    def on_train_end(self) -> None:
        r"""The mask and update the cumulative mask after training the task."""

        # store the mask for the current task
        mask_t = self.backbone.store_mask()

        # store the batch normalization if necessary
        self.backbone.store_bn()

        # update the cumulative mask. See the first Eq. in Sec 2.3 "Network Training" in the [HAT paper](http://proceedings.mlr.press/v80/serra18a)
        self.cumulative_mask_for_previous_tasks = {
            layer_name: torch.max(
                self.cumulative_mask_for_previous_tasks[layer_name], mask_t[layer_name]
            )
            for layer_name in self.backbone.weighted_layer_names
        }

    def validation_step(self, batch: Any) -> dict[str, Tensor]:
        r"""Validation step for current task `self.task_id`.

        **Args:**
        - **batch** (`Any`): a batch of validation data.

        **Returns:**
        - **outputs** (`dict[str, Tensor]`): a dictionary contains loss and other metrics from this validation step. Keys (`str`) are the metrics names, and values (`Tensor`) are the metrics.
        """
        x, y = batch
        logits, _, _ = self.forward(x, stage="validation", task_id=self.task_id)
        loss_cls = self.criterion(logits, y)
        acc = (logits.argmax(dim=1) == y).float().mean()

        return {
            "loss_cls": loss_cls,
            "acc": acc,  # Return metrics for lightning loggers callback to handle at `on_validation_batch_end()`
        }

    def test_step(
        self, batch: DataLoader, batch_idx: int, dataloader_idx: int = 0
    ) -> dict[str, Tensor]:
        r"""Test step for current task `self.task_id`, which tests for all seen tasks indexed by `dataloader_idx`.

        **Args:**
        - **batch** (`Any`): a batch of test data.
        - **dataloader_idx** (`int`): the task ID of seen tasks to be tested. A default value of 0 is given otherwise the LightningModule will raise a `RuntimeError`.

        **Returns:**
        - **outputs** (`dict[str, Tensor]`): a dictionary contains loss and other metrics from this test step. Keys (`str`) are the metrics names, and values (`Tensor`) are the metrics.
        """
        test_task_id = self.get_test_task_id_from_dataloader_idx(dataloader_idx)

        x, y = batch
        logits, _, _ = self.forward(
            x,
            stage="test",
            task_id=test_task_id,
        )  # use the corresponding head and mask to test (instead of the current task `self.task_id`)
        loss_cls = self.criterion(logits, y)
        acc = (logits.argmax(dim=1) == y).float().mean()

        return {
            "loss_cls": loss_cls,
            "acc": acc,  # Return metrics for lightning loggers callback to handle at `on_test_batch_end()`
        }
