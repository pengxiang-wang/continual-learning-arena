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
from clarena.cl_algorithms.regularisers import HATMaskSparsityReg
from clarena.cl_heads import HeadsCIL, HeadsTIL
from clarena.utils import HATNetworkCapacity

# always get logger for built-in logging in each module
pylogger = logging.getLogger(__name__)


class HAT(CLAlgorithm):
    r"""HAT (Hard Attention to the Task) algorithm.

    [HAT (Hard Attention to the Task, 2018)](http://proceedings.mlr.press/v80/serra18a) is an architecture-based continual learning approach that uses learnable hard attention masks to select the task-specific parameters.

    """

    def __init__(
        self,
        backbone: HATMaskBackbone,
        heads: HeadsTIL | HeadsCIL,
        adjustment_mode: str,
        s_max: float,
        clamp_threshold: float,
        mask_sparsity_reg_factor: float,
        mask_sparsity_reg_mode: str = "original",
        task_embedding_init_mode: str = "N01",
        alpha: float | None = None,
    ) -> None:
        r"""Initialise the HAT algorithm with the network.

        **Args:**
        - **backbone** (`HATMaskBackbone`): must be a backbone network with HAT mask mechanism.
        - **heads** (`HeadsTIL` | `HeadsCIL`): output heads.
        - **adjustment_mode** (`str`): the strategy of adjustment i.e. the mode of gradient clipping, should be one of the following:
            1. 'hat': set the gradients of parameters linking to masked units to zero. This is the way that HAT does, which fixes the part of network for previous tasks completely. See equation (2) in chapter 2.3 "Network Training" in [HAT paper](http://proceedings.mlr.press/v80/serra18a).
            2. 'hat_random': set the gradients of parameters linking to masked units to random 0-1 values. See the "Baselines" section in chapter 4.1 in [AdaHAT paper](https://link.springer.com/chapter/10.1007/978-3-031-70352-2_9).
            3. 'hat_const_alpha': set the gradients of parameters linking to masked units to a constant value of `alpha`. See the "Baselines" section in chapter 4.1 in [AdaHAT paper](https://link.springer.com/chapter/10.1007/978-3-031-70352-2_9).
            4. 'hat_const_1': set the gradients of parameters linking to masked units to a constant value of 1, which means no gradient constraint on any parameter at all. See the "Baselines" section in chapter 4.1 in [AdaHAT paper](https://link.springer.com/chapter/10.1007/978-3-031-70352-2_9).
        - **s_max** (`float`): hyperparameter, the maximum scaling factor in the gate function. See chapter 2.4 "Hard Attention Training" in [HAT paper](http://proceedings.mlr.press/v80/serra18a).
        - **clamp_threshold** (`float`): the threshold for task embedding gradient compensation. See chapter 2.5 "Embedding Gradient Compensation" in [HAT paper](http://proceedings.mlr.press/v80/serra18a).
        - **mask_sparsity_reg_factor** (`float`): hyperparameter, the regularisation factor for mask sparsity.
        - **mask_sparsity_reg_mode** (`str`): the mode of mask sparsity regularisation, should be one of the following:
            1. 'original' (default): the original mask sparsity regularisation in HAT paper.
            2. 'cross': the cross version mask sparsity regularisation.
        - **task_embedding_init_mode** (`str`): the initialisation mode for task embeddings, should be one of the following:
            1. 'N01' (default): standard normal distribution $N(0, 1)$.
            2. 'U-11': uniform distribution $U(-1, 1)$.
            3. 'U01': uniform distribution $U(0, 1)$.
            4. 'U-10': uniform distribution $U(-1, 0)$.
            5. 'last': inherit task embedding from last task.
        - **alpha** (`float` | `None`): the `alpha` in the 'HAT-const-alpha' mode. See the "Baselines" section in chapter 4.1 in [AdaHAT paper](https://link.springer.com/chapter/10.1007/978-3-031-70352-2_9). It applies only when adjustment_mode is 'hat_const_alpha'.
        """
        CLAlgorithm.__init__(self, backbone=backbone, heads=heads)

        self.adjustment_mode = adjustment_mode
        r"""Store the adjustment mode for gradient clipping."""
        self.s_max = s_max
        r"""Store s_max. """
        self.clamp_threshold = clamp_threshold
        r"""Store the clamp threshold for task embedding gradient compensation."""
        self.mask_sparsity_reg_factor = mask_sparsity_reg_factor
        r"""Store the mask sparsity regularisation factor."""
        self.mask_sparsity_reg_mode = mask_sparsity_reg_mode
        r"""Store the mask sparsity regularisation mode."""
        self.mark_sparsity_reg = HATMaskSparsityReg(
            factor=mask_sparsity_reg_factor, mode=mask_sparsity_reg_mode
        )
        r"""Initialise and store the mask sparsity regulariser."""
        self.task_embedding_init_mode = task_embedding_init_mode
        r"""Store the task embedding initialisation mode."""
        self.alpha = alpha if adjustment_mode == "hat_const_alpha" else None
        r"""Store the alpha for `hat_const_alpha`."""
        self.epsilon = None
        r"""HAT doesn't use the epsilon for `hat_const_alpha`. We still set it here to be consistent with the `epsilon` in `clip_grad_by_adjustment()` method in `HATMaskBackbone`."""

        self.masks: dict[str, dict[str, Tensor]] = {}
        r"""Store the binary attention mask of each previous task gated from the task embedding. Keys are task IDs (string type) and values are the corresponding mask. Each mask is a dict where keys are layer names and values are the binary mask tensor for the layer. The mask tensor has size (number of units). """

        self.cumulative_mask_for_previous_tasks: dict[str, Tensor] = {}
        r"""Store the cumulative binary attention mask $\mathrm{M}^{<t}$ of previous tasks $1,\cdots, t-1$, gated from the task embedding. Keys are task IDs and values are the corresponding cumulative mask. Each cumulative mask is a dict where keys are layer names and values are the binary mask tensor for the layer. The mask tensor has size (number of units). """

        # set manual optimisation
        self.automatic_optimization = False

        HAT.sanity_check(self)

    def sanity_check(self) -> None:
        r"""Check the sanity of the arguments.

        **Raises:**
        - **ValueError**: when backbone is not designed for HAT, or the `mask_sparsity_reg_mode` or `task_embedding_init_mode` is not one of the valid options. Also, if `alpha` is not given when `adjustment_mode` is 'hat_const_alpha'.
        """
        if not isinstance(self.backbone, HATMaskBackbone):
            raise ValueError("The backbone should be an instance of HATMaskBackbone.")

        if self.mask_sparsity_reg_mode not in ["original", "cross"]:
            raise ValueError(
                "The mask_sparsity_reg_mode should be one of 'original', 'cross'."
            )
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

        if self.adjustment_mode == "hat_const_alpha" and self.alpha is None:
            raise ValueError(
                "Alpha should be given when the adjustment_mode is 'hat_const_alpha'."
            )

    def on_train_start(self) -> None:
        r"""Initialise the task embedding before training the next task and initialise the cumulative mask at the beginning of first task."""

        self.backbone.initialise_task_embedding(mode=self.task_embedding_init_mode)

        # initialise the cumulative mask at the beginning of first task. This should not be called in `__init__()` method as the `self.device` is not available at that time.
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
                )  # the cumulative mask $\mathrm{M}^{<t}$ is initialised as zeros mask ($t = 1$). See equation (2) in chapter 3 in [AdaHAT paper](https://link.springer.com/chapter/10.1007/978-3-031-70352-2_9), or equation (5) in chapter 2.6 "Promoting Low Capacity Usage" in [HAT paper](http://proceedings.mlr.press/v80/serra18a).

    def clip_grad_by_adjustment(
        self,
        **kwargs,
    ) -> Tensor:
        r"""Clip the gradients by the adjustment rate.

        Note that as the task embedding fully covers every layer in the backbone network, no parameters are left out of this system. This applies not only the parameters in between layers with task embedding, but also those before the first layer. We designed it seperately in the codes.

        Network capacity is measured along with this method. Network capacity is defined as the average adjustment rate over all parameters. See chapter 4.1 in [AdaHAT paper](https://link.springer.com/chapter/10.1007/978-3-031-70352-2_9).


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

            weight_mask, bias_mask = self.backbone.get_layer_measure_parameter_wise(
                unit_wise_measure=self.cumulative_mask_for_previous_tasks,
                layer_name=layer_name,
                aggregation="min",
            )

            if self.adjustment_mode == "hat":
                adjustment_rate_weight = 1 - weight_mask
                adjustment_rate_bias = 1 - bias_mask

            elif self.adjustment_mode == "hat_random":
                adjustment_rate_weight = torch.rand_like(weight_mask) * weight_mask + (
                    1 - weight_mask
                )
                adjustment_rate_bias = torch.rand_like(bias_mask) * bias_mask + (
                    1 - bias_mask
                )

            elif self.adjustment_mode == "hat_const_alpha":
                adjustment_rate_weight = self.alpha * torch.ones_like(
                    weight_mask
                ) * weight_mask + (1 - weight_mask)
                adjustment_rate_bias = self.alpha * torch.ones_like(
                    bias_mask
                ) * bias_mask + (1 - bias_mask)

            elif self.adjustment_mode == "hat_const_1":
                adjustment_rate_weight = torch.ones_like(weight_mask) * weight_mask + (
                    1 - weight_mask
                )
                adjustment_rate_bias = torch.ones_like(bias_mask) * bias_mask + (
                    1 - bias_mask
                )

            # apply the adjustment rate to the gradients
            layer.weight.grad.data *= adjustment_rate_weight
            if layer.bias is not None:
                layer.bias.grad.data *= adjustment_rate_bias

            # update network capacity metric
            capacity.update(adjustment_rate_weight, adjustment_rate_bias)

        return capacity.compute()

    def compensate_task_embedding_gradients(
        self,
        batch_idx: int,
        num_batches: int,
    ) -> None:
        r"""Compensate the gradients of task embeddings during training. See chapter 2.5 "Embedding Gradient Compensation" in [HAT paper](http://proceedings.mlr.press/v80/serra18a).

        **Args:**
        - **batch_idx** (`int`): the current training batch index.
        - **num_batches** (`int`): the total number of training batches.
        """

        for te in self.backbone.task_embedding_t.values():
            anneal_scalar = 1 / self.s_max + (self.s_max - 1 / self.s_max) * (
                batch_idx - 1
            ) / (
                num_batches - 1
            )  # see equation (3) in chapter 2.4 "Hard Attention Training" in [HAT paper](http://proceedings.mlr.press/v80/serra18a)

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
        batch_idx: int | None = None,
        num_batches: int | None = None,
        task_id: int | None = None,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        r"""The forward pass for data from task `task_id`. Note that it is nothing to do with `forward()` method in `nn.Module`.

        **Args:**
        - **input** (`Tensor`): The input tensor from data.
        - **stage** (`str`): the stage of the forward pass, should be one of the following:
            1. 'train': training stage.
            2. 'validation': validation stage.
            3. 'test': testing stage.
        - **batch_idx** (`int` | `None`): the current batch index. Applies only to training stage. For other stages, it is default `None`.
        - **num_batches** (`int` | `None`): the total number of batches. Applies only to training stage. For other stages, it is default `None`.
        - **task_id** (`int`| `None`): the task ID where the data are from. If the stage is 'train' or 'validation', it should be the current task `self.task_id`. If stage is 'test', it could be from any seen task. In TIL, the task IDs of test data are provided thus this argument can be used. HAT algorithm works only for TIL.

        **Returns:**
        - **logits** (`Tensor`): the output logits tensor.
        - **mask** (`dict[str, Tensor]`): the mask for the current task. Key (`str`) is layer name, value (`Tensor`) is the mask tensor. The mask tensor has size (number of units).
        - **hidden_features** (`dict[str, Tensor]`): the hidden features (after activation) in each weighted layer. Key (`str`) is the weighted layer name, value (`Tensor`) is the hidden feature tensor. This is used for the continual learning algorithms that need to use the hidden features for various purposes. Although HAT algorithm does not need this, it is still provided for API consistence for other HAT-based algorithms inherited this `forward()` method of `HAT` class.
        """
        feature, mask, hidden_features = self.backbone(
            input,
            stage=stage,
            s_max=self.s_max if stage == "train" or stage == "validation" else None,
            batch_idx=batch_idx if stage == "train" else None,
            num_batches=num_batches if stage == "train" else None,
            test_mask=self.masks[f"{task_id}"] if stage == "test" else None,
        )
        logits = self.heads(feature, task_id)

        return logits, mask, hidden_features

    def training_step(self, batch: Any, batch_idx: int) -> dict[str, Tensor]:
        r"""Training step for current task `self.task_id`.

        **Args:**
        - **batch** (`Any`): a batch of training data.
        - **batch_idx** (`int`): the index of the batch. Used for calculating annealed scalar in HAT. See chapter 2.4 "Hard Attention Training" in [HAT paper](http://proceedings.mlr.press/v80/serra18a).

        **Returns:**
        - **outputs** (`dict[str, Tensor]`): a dictionary contains loss and other metrics from this training step. Key (`str`) is the metrics name, value (`Tensor`) is the metrics. Must include the key 'loss' which is total loss in the case of automatic optimization, according to PyTorch Lightning docs. For HAT, it includes 'mask' and 'capacity' for logging.
        """
        x, y = batch

        # zero the gradients before forward pass in manual optimisation mode
        opt = self.optimizers()
        opt.zero_grad()

        # classification loss
        num_batches = self.trainer.num_training_batches
        logits, mask, hidden_features = self.forward(
            x,
            stage="train",
            batch_idx=batch_idx,
            num_batches=num_batches,
            task_id=self.task_id,
        )
        loss_cls = self.criterion(logits, y)

        # regularisation loss. See chapter 2.6 "Promoting Low Capacity Usage" in [HAT paper](http://proceedings.mlr.press/v80/serra18a).
        loss_reg, network_sparsity = self.mark_sparsity_reg(
            mask, self.cumulative_mask_for_previous_tasks
        )

        # total loss
        loss = loss_cls + loss_reg

        # backward step (manually)
        self.manual_backward(loss)  # calculate the gradients
        # HAT hard clip gradients by the cumulative masks. See equation (2) inchapter 2.3 "Network Training" in [HAT paper](http://proceedings.mlr.press/v80/serra18a). Network capacity is calculated along with this process. Network capacity is defined as the average adjustment rate over all paramaters. See chapter 4.1 in [AdaHAT paper](https://link.springer.com/chapter/10.1007/978-3-031-70352-2_9).
        capacity = self.clip_grad_by_adjustment(
            network_sparsity=network_sparsity,  # pass a keyword argument network sparsity here to make it compatible with AdaHAT. AdaHAT inherits this `training_step()` method.
        )
        # compensate the gradients of task embedding. See chapter 2.5 "Embedding Gradient Compensation" in [HAT paper](http://proceedings.mlr.press/v80/serra18a).
        self.compensate_task_embedding_gradients(
            batch_idx=batch_idx,
            num_batches=num_batches,
        )
        # update parameters with the modified gradients
        opt.step()

        # accuracy of the batch
        acc = (logits.argmax(dim=1) == y).float().mean()

        return {
            "loss": loss,  # Return loss is essential for training step, or backpropagation will fail
            "loss_cls": loss_cls,
            "loss_reg": loss_reg,
            "acc": acc,
            "hidden_features": hidden_features,
            "mask": mask,  # Return other metrics for lightning loggers callback to handle at `on_train_batch_end()`
            "capacity": capacity,
        }

    def on_train_end(self) -> None:
        r"""Store the mask and update cumulative mask after training the task."""

        # store the mask for the current task
        mask_t = {
            layer_name: self.backbone.gate_fn(
                self.backbone.task_embedding_t[layer_name].weight * self.s_max
            )
            .squeeze()
            .detach()
            for layer_name in self.backbone.weighted_layer_names
        }

        self.masks[f"{self.task_id}"] = mask_t

        # update the cumulative and summative masks
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
        - **outputs** (`dict[str, Tensor]`): a dictionary contains loss and other metrics from this validation step. Key (`str`) is the metrics name, value (`Tensor`) is the metrics.
        """
        x, y = batch
        logits, mask, hidden_features = self.forward(
            x, stage="validation", task_id=self.task_id
        )
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
        - **outputs** (`dict[str, Tensor]`): a dictionary contains loss and other metrics from this test step. Key (`str`) is the metrics name, value (`Tensor`) is the metrics.
        """
        test_task_id = dataloader_idx + 1

        x, y = batch
        logits, mask, hidden_features = self.forward(
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
