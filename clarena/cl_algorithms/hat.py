r"""
The submodule in `cl_algorithms` for [HAT (Hard Attention to the Task) algorithm](http://proceedings.mlr.press/v80/serra18a) and [AdaHAT (Adaptive Hard Attention to the Task) algorithm](https://link.springer.com/chapter/10.1007/978-3-031-70352-2_9).
"""

__all__ = ["HAT"]

import logging
from typing import Any

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader

from clarena.backbones import HATMaskBackbone
from clarena.cl_algorithms import CLAlgorithm
from clarena.cl_algorithms.regularisers import HATMaskSparsityReg
from clarena.cl_heads import HeadsCIL, HeadsTIL

# always get logger for built-in logging in each module
pylogger = logging.getLogger(__name__)


class HAT(CLAlgorithm):
    r"""HAT (Hard Attention to the Task) and AdaHAT (Adaptive Hard Attention to the Task) algorithm.

    [HAT (Hard Attention to the Task, 2018)](http://proceedings.mlr.press/v80/serra18a) is an architecture-based continual learning approach that uses learnable hard attention masks to select the task-specific parameters.

    [Adaptive HAT (Adaptive Hard Attention to the Task, 2024)](https://link.springer.com/chapter/10.1007/978-3-031-70352-2_9) is an architecture-based continual learning approach that improves HAT (Hard Attention to the Task, 2018) by introducing new adaptive soft gradient clipping based on parameter importance and network sparsity.
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
        task_embedding_init: str = "N01",
        adjustment_intensity: float | None = None,
        epsilon: float = 0.1,
    ) -> None:
        r"""Initialise the HAT algorithm with the network.

        **Args:**
        - **backbone** (`HATMaskBackbone`): must be a backbone network with HAT mask mechanism.
        - **heads** (`HeadsTIL` | `HeadsCIL`): output heads.
        - **adjustment_mode** (`str`): the strategy of adjustment i.e. the mode of gradient clipping, should be one of the following:
            1. 'hat': set the gradients of parameters linking to masked units to zero. This is the way that HAT does, which fixes the part of network for previous tasks completely. See equation (2) in chapter 2.3 "Network Training" in [HAT paper](http://proceedings.mlr.press/v80/serra18a).
            2. 'adahat': set the gradients of parameters linking to masked units to a soft adjustment rate in the original AdaHAT approach. This is the way that AdaHAT does, which allowes the part of network for previous tasks to be updated slightly. See equation (8) and (9) chapter 3.1 in [AdaHAT paper](https://link.springer.com/chapter/10.1007/978-3-031-70352-2_9).
            3. 'hat_random': set the gradients of parameters linking to masked units to random 0-1 values. See the Baselines section in chapter 4.1 in [AdaHAT paper](https://link.springer.com/chapter/10.1007/978-3-031-70352-2_9).
            4. 'hat_const_alpha': set the gradients of parameters linking to masked units to a constant value of `alpha`. See the Baselines section in chapter 4.1 in [AdaHAT paper](https://link.springer.com/chapter/10.1007/978-3-031-70352-2_9).
            5. 'hat_const_1': set the gradients of parameters linking to masked units to a constant value of 1, which means no gradient constraint on any parameter at all. See the Baselines section in chapter 4.1 in [AdaHAT paper](https://link.springer.com/chapter/10.1007/978-3-031-70352-2_9).
            6. 'adahat_no_sum': set the gradients of parameters linking to masked units to a soft adjustment rate in the original AdaHAT approach, but without considering the information of parameter importance i.e. summative mask. This is the way that one of the AdaHAT ablation study does. See chapter 4.3 in [AdaHAT paper](https://link.springer.com/chapter/10.1007/978-3-031-70352-2_9).
            7. 'adahat_no_reg': set the gradients of parameters linking to masked units to a soft adjustment rate in the original AdaHAT approach, but without considering the information of network sparsity i.e. mask sparsity regularisation value. This is the way that one of the AdaHAT ablation study does. See chapter 4.3 in [AdaHAT paper](https://link.springer.com/chapter/10.1007/978-3-031-70352-2_9).
        - **s_max** (`float`): hyperparameter, the maximum scaling factor in the gate function. See chapter 2.4 "Hard Attention Training" in [HAT paper](http://proceedings.mlr.press/v80/serra18a).
        - **clamp_threshold** (`float`): the threshold for task embedding gradient compensation. See chapter 2.5 "Embedding Gradient Compensation" in [HAT paper](http://proceedings.mlr.press/v80/serra18a).
        - **mask_sparsity_reg_factor** (`float`): hyperparameter, the regularisation factor for mask sparsity.
        - **mask_sparsity_reg_mode** (`str`): the mode of mask sparsity regularisation, should be one of the following:
            1. 'original' (default): the original mask sparsity regularisation in HAT paper.
            2. 'cross': the cross version mask sparsity regularisation.
        - task_embedding_init (`str`): the initialisation method for task embeddings, should be one of the following:
            1. 'N01' (default): standard normal distribution $N(0, 1)$.
            2. 'U-11':uniform distribution $U(-1, 1)$.
            3. 'U01': uniform distribution $U(0, 1)$.
            4. 'U-10': uniform distribution $U(-1, 0)$.
            5. 'last': inherit inherit task embedding from last task.
        - **adjustment_intensity** (`float` | `None`): hyperparameter, control the overall intensity of gradient adjustment. It applies only to AdaHAT modes and `hat_const_alpha`. It's the `alpha` in equation (9) and the `alpha` in the "HAT-const-alpha" equation in chapter 4.1 in [AdaHAT paper](https://link.springer.com/chapter/10.1007/978-3-031-70352-2_9).
        - **epsilon** (`float`): the small value to avoid division by zero appeared in equation (9) in [AdaHAT paper](https://link.springer.com/chapter/10.1007/978-3-031-70352-2_9). It applies only to AdaHAT modes.
        """
        super().__init__(backbone=backbone, heads=heads)

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
        self.task_embedding_init = task_embedding_init
        r"""Store the task embedding initialisation method."""
        self.adjustment_intensity = adjustment_intensity
        r"""Store the adjustment intensity for `hat_const_alpha`."""
        self.epsilon = epsilon
        """Store the small value to avoid division by zero appeared in equation (9) in [AdaHAT paper](https://link.springer.com/chapter/10.1007/978-3-031-70352-2_9)."""

        # set manual optimisation
        self.automatic_optimization = False

        self.sanity_check_HAT()

    def sanity_check_HAT(self) -> None:
        r"""Check the sanity of the arguments.

        **Raises:**
        - **ValueError**: when backbone is not designed for HAT, or the `mask_sparsity_reg_mode` or `task_embedding_init` is one of the valid options.
        """
        if not isinstance(self.backbone, HATMaskBackbone):
            raise ValueError("The backbone should be an instance of HATMaskBackbone.")

        if self.mask_sparsity_reg_mode not in ["original", "cross"]:
            raise ValueError(
                "The mask_sparsity_reg_mode should be one of 'original', 'cross'."
            )
        if self.task_embedding_init not in ["N01", "U01", "U-10", "masked", "unmasked"]:
            raise ValueError(
                "The task_embedding_init should be one of 'N01', 'U01', 'U-10', 'masked', 'unmasked'."
            )

    def on_train_start(self) -> None:
        r"""Initialise the task embedding before training the next task."""

        for te in self.backbone.task_embedding_t.values():
            if self.task_embedding_init == "N01":
                nn.init.normal_(te.weight, 0, 1)
            elif self.task_embedding_init == "U-11":
                nn.init.uniform_(te.weight, -1, 1)
            elif self.task_embedding_init == "U01":
                nn.init.uniform_(te.weight, 0, 1)
            elif self.task_embedding_init == "U-10":
                nn.init.uniform_(te.weight, -1, 0)
            elif self.task_embedding_init == "last":
                pass

        # initialise the cumulative and summative mask at the beginning of first task. This should not be called in `__init__()` method as the `self.device` is not available at that time.
        if self.task_id == 1:
            self.backbone.initialise_cumulative_and_summative_mask(device=self.device)

    def forward(
        self,
        input: torch.Tensor,
        stage: str,
        s_max: float | None = None,
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
        - **s_max** (`float`): the maximum scaling factor in the gate function. See chapter 2.4 "Hard Attention Training" in [HAT paper](http://proceedings.mlr.press/v80/serra18a).
        - **batch_idx** (`int` | `None`): the current batch index. Applies only to training stage. For other stages, it is default `None`.
        - **num_batches** (`int` | `None`): the total number of batches. Applies only to training stage. For other stages, it is default `None`.
        - **task_id** (`int`| `None`): the task ID where the data are from. Applies only to testing stage. For other stages, it is automatically the current task `self.task_id`. If stage is 'test', it could be from any seen task. In TIL, the task IDs of test data are provided thus this argument can be used. HAT algorithm works only for TIL.

        **Returns:**
        - **logits** (`Tensor`): the output logits tensor.
        - **mask** (`dict[str, Tensor]`): the mask for the current task. Key (`str`) is layer name, value (`Tensor`) is the mask tensor.
        """
        feature, mask = self.backbone(
            input,
            stage=stage,
            s_max=s_max,
            batch_idx=batch_idx,
            num_batches=num_batches,
            task_id=task_id,
        )
        logits = self.heads(feature, task_id)

        return logits, mask

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
        logits, mask = self.forward(
            x,
            stage="train",
            s_max=self.s_max,
            batch_idx=batch_idx,
            num_batches=num_batches,
            task_id=self.task_id,
        )
        loss_cls = self.criterion(logits, y)

        # regularisation loss. See chapter 2.6 "Promoting Low Capacity Usage" in [HAT paper](http://proceedings.mlr.press/v80/serra18a).
        previous_cumulative_mask = self.backbone.get_cumulative_mask()
        loss_reg, network_sparsity = self.mark_sparsity_reg(
            mask, previous_cumulative_mask
        )

        # total loss
        loss = loss_cls + loss_reg

        # backward step (manually)
        self.manual_backward(loss)  # calculate the gradients
        # HAT hard clip gradients by the cumulative masks. See equation (2) inchapter 2.3 "Network Training" in [HAT paper](http://proceedings.mlr.press/v80/serra18a). Network capacity is calculated along with this process. Network capacity is defined as the average adjustment rate over all paramaters. See chapter 4.1 in [AdaHAT paper](https://link.springer.com/chapter/10.1007/978-3-031-70352-2_9).
        capacity = self.backbone.clip_grad_by_adjustment(
            mode=self.adjustment_mode,
            network_sparsity=network_sparsity,
            adjustment_intensity=self.adjustment_intensity,
            epsilon=self.epsilon,
        )
        # compensate the gradients of task embedding. See chapter 2.5 "Embedding Gradient Compensation" in [HAT paper](http://proceedings.mlr.press/v80/serra18a).
        self.backbone.compensate_task_embedding_gradients(
            clamp_threshold=self.clamp_threshold,
            s_max=self.s_max,
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
            "mask": mask,  # Return other metrics for lightning loggers callback to handle at `on_train_batch_end()`
            "capacity": capacity,
        }

    def on_train_end(self) -> None:
        r"""Store the mask and update cumulative and summative masks after training the task."""
        self.backbone.store_mask(s_max=self.s_max)

    def validation_step(self, batch: Any) -> dict[str, Tensor]:
        r"""Validation step for current task `self.task_id`.

        **Args:**
        - **batch** (`Any`): a batch of validation data.

        **Returns:**
        - **outputs** (`dict[str, Tensor]`): a dictionary contains loss and other metrics from this validation step. Key (`str`) is the metrics name, value (`Tensor`) is the metrics.
        """
        x, y = batch
        logits, mask = self.forward(
            x, stage="validation", s_max=self.s_max, task_id=self.task_id
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
        logits, mask = self.forward(
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
