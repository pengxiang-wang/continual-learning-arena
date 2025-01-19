"""
The submodule in `cl_algorithms` for [HAT (Hard Attention to the Task) algorithm](http://proceedings.mlr.press/v80/serra18a).
"""

__all__ = ["HAT"]

import logging
from typing import Any

import torch
from lightning import Trainer
from torch import Tensor, nn
from torch.utils.data import DataLoader

from clarena.backbones.base import HATMaskBackbone
from clarena.cl_algorithms import CLAlgorithm
from clarena.cl_algorithms.regularisers import HATMaskSparsityReg

# always get logger for built-in logging in each module
pylogger = logging.getLogger(__name__)


class HAT(CLAlgorithm):
    """HAT (Hard Attention to the Task) algorithm.

    [HAT (Hard Attention to the Task, 2018)](http://proceedings.mlr.press/v80/serra18a) is an architecture-based continual learning approach that uses learnable hard attention masks to select the task-specific parameters.
    """

    def __init__(
        self,
        backbone: torch.nn.Module,
        heads: torch.nn.Module,
        s_max: float,
        clamp_threshold: float,
        mask_sparsity_factor: float,
        mask_sparsity_mode: str = "original",
        task_embedding_init: str = "N01",
    ) -> None:
        """Initialise the HAT algorithm with the network.

        **Args:**
        - **backbone** (`CLBackbone`): backbone network.
        - **heads** (`HeadsTIL` | `HeadsCIL`): output heads.
        - **s_max** (`float`): hyperparameter, the maximum scaling factor in the gate function. See chapter 2.4 "Hard Attention Training" in [HAT paper](http://proceedings.mlr.press/v80/serra18a). 
        - **clamp_threshold** (`float`): the threshold for task embedding gradient compensation. See chapter 2.5 "Embedding Gradient Compensation" in [HAT paper](http://proceedings.mlr.press/v80/serra18a).
        - **mask_sparsity_factor** (`float`): hyperparameter, the regularisation factor for mask sparsity.
        - **mask_sparsity_mode** (`str`): the mode of mask sparsity regularisation, should be one of the following:
            1. 'original' (default): the original mask sparsity regularisation in HAT paper.
            2. 'cross': the cross version mask sparsity regularisation.
        - task_embedding_init (`str`): the initialisation method for task embeddings, should be one of the following:
            1. 'N01' (default): standard normal distribution $N(0, 1)$.
            2. 'U-11':uniform distribution $U(-1, 1)$. 
            3. 'U01': uniform distribution $U(0, 1)$.
            4. 'U-10': uniform distribution $U(-1, 0)$.
            5. 'last': inherit inherit task embedding from last task.
        """
        super().__init__(backbone=backbone, heads=heads)

        self.s_max = s_max
        """Store s_max. """
        self.clamp_threshold = clamp_threshold
        """Store the clamp threshold for task embedding gradient compensation."""
        self.mask_sparsity_factor = mask_sparsity_factor
        """Store the mask sparsity regularisation factor."""
        self.mask_sparsity_mode = mask_sparsity_mode
        """Store the mask sparsity regularisation mode."""
        self.mark_sparsity_reg = HATMaskSparsityReg(
            factor=mask_sparsity_factor, mode=mask_sparsity_mode
        )
        """Initialise and store the mask sparsity regulariser."""
        self.task_embedding_init = task_embedding_init
        """Store the task embedding initialisation method."""

        # Set manual optimisation
        self.automatic_optimization = False

    def sanity_check(self) -> None:
        """Check the sanity of the arguments.

        **Raises:**
        - **ValueError**: when backbone is not designed for HAT, or the `mask_sparsity_mode` or `task_embedding_init` is one of the valid options.
        """
        if not isinstance(self.backbone, HATMaskBackbone):
            raise ValueError("The backbone should be an instance of HATMaskBackbone.")

        if self.mask_sparsity_mode not in ["original", "cross"]:
            raise ValueError(
                "The mask_sparsity_mode should be one of 'original', 'cross'."
            )
        if self.task_embedding_init not in ["N01", "U01", "U-10", "masked", "unmasked"]:
            raise ValueError(
                "The task_embedding_init should be one of 'N01', 'U01', 'U-10', 'masked', 'unmasked'."
            )

        super().sanity_check()

    def on_train_start(self, trainer: Trainer, pl_module: CLAlgorithm) -> None:
        """Initialise the task embedding before training the next task."""
    
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

    def forward(
        self,
        input: torch.Tensor,
        stage: str,
        s_max: float | None = None,
        batch_idx: int | None = None,
        num_batches: int | None = None,
        task_id: int | None = None,
    ) -> Tensor:
        """The forward pass for data from task `task_id`. Note that it is nothing to do with `forward()` method in `nn.Module`.

        **Args:**
        - **input** (`Tensor`): The input tensor from data.
        - **stage** (`str`): the stage of the forward pass, should be one of the following:
            1. 'train': training stage.
            2. 'validate': validation stage.
            3. 'test': testing stage.
        - **s_max** (`float`): the maximum scaling factor in the gate function. See chapter 2.4 "Hard Attention Training" in [HAT paper](http://proceedings.mlr.press/v80/serra18a).
        - **batch_idx** (`int` | `None`): the current batch index. Applies only to training stage. For other stages, it is default `None`.
        - **num_batches** (`int` | `None`): the total number of batches. Applies only to training stage. For other stages, it is default `None`.
        - **task_id** (`int`| `None`): the task ID where the data are from. Applies only to testing stage. For other stages, it is automatically the current task `self.task_id`. If stage is 'test', it could be from any seen task. In TIL, the task IDs of test data are provided thus this argument can be used. HAT algorithm works only for TIL.

        **Returns:**
        - **logits** (`Tensor`): the output logits tensor.
        """
        feature = self.backbone(
            input,
            stage=stage,
            s_max=s_max,
            batch_idx=batch_idx,
            num_batches=num_batches,
            task_id=task_id,
        )
        logits = self.heads(feature, task_id)
        
        return logits

    def training_step(self, batch: Any, batch_idx: int) -> dict[str, Tensor]:
        """Training step for current task `self.task_id`.

        **Args:**
        - **batch** (`Any`): a batch of training data.
        - **batch_idx** (`int`): the index of the batch. Used for calculating annealed scalar in HAT. See chapter 2.4 "Hard Attention Training" in [HAT paper](http://proceedings.mlr.press/v80/serra18a). 
        
        **Returns:**
        - **outputs** (`dict[str, Tensor]`): a dictionary contains loss and other metrics from this training step. Key (`str`) is the metrics name, value (`Tensor`) is the metrics. Must include the key 'loss' which is total loss in the case of automatic optimization, according to PyTorch Lightning docs.
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
        )
        loss_cls = self.criterion(logits, y)

        # regularisation loss. See chapter 2.6 "Promoting Low Capacity Usage" in [HAT paper](http://proceedings.mlr.press/v80/serra18a).
        previous_cumulative_mask = self.backbone.get_cumulative_mask()
        loss_reg = self.mark_sparsity_reg(mask, previous_cumulative_mask)

        # total loss
        loss = loss_cls + loss_reg
        
        # accuracy of the batch
        acc = (logits.argmax(dim=1) == y).float().mean()

        # backward step (manually) 
        self.manual_backward(loss) # calculate the gradients
        # HAT hard clip gradients by the cumulative masks. See equation (2) inchapter 2.3 "Network Training" in [HAT paper](http://proceedings.mlr.press/v80/serra18a). Network capacity is calculated along with this process. Network capacity is defined as the average adjustment rate over all paramaters. See chapter 4.1 in [AdaHAT paper](https://link.springer.com/chapter/10.1007/978-3-031-70352-2_9). 
        capacity = self.backbone.clip_grad_by_adjustment(mode="hat") 
        # compensate the gradients of task embedding. See chapter 2.5 "Embedding Gradient Compensation" in [HAT paper](http://proceedings.mlr.press/v80/serra18a).
        self.backbone.compensate_task_embedding_gradients(
            clamp_threshold=self.clamp_threshold,
            s_max=self.s_max,
            batch_idx=batch_idx,
            num_batches=num_batches,
        )
        # update parameters with the modified gradients
        opt.step()

        return {
            "loss": loss,  # Return loss is essential for training step, or backpropagation will fail
            "loss_cls": loss_cls,
            "acc": acc,
            "mask": mask,  # Return other metrics for lightning loggers callback to handle at `on_train_batch_end()`
            "capacity": capacity,
        }

    def validation_step(self, batch: Any) -> dict[str, Tensor]:
        """Validation step for current task `self.task_id`.

        **Args:**
        - **batch** (`Any`): a batch of validation data.
    
        **Returns:**
        - **outputs** (`dict[str, Tensor]`): a dictionary contains loss and other metrics from this validation step. Key (`str`) is the metrics name, value (`Tensor`) is the metrics. 
        """
        x, y = batch
        logits = self.forward(x, stage="validate")
        loss_cls = self.criterion(logits, y)
        acc = (logits.argmax(dim=1) == y).float().mean()

        return {
            "loss_cls": loss_cls,
            "acc": acc,  # Return metrics for lightning loggers callback to handle at `on_validation_batch_end()`
        }

    def test_step(self, batch: DataLoader, batch_idx: int, dataloader_idx: int = 0) -> dict[str, Tensor]:
        """Test step for current task `self.task_id`, which tests for all seen tasks indexed by `dataloader_idx`.

        **Args:**
        - **batch** (`Any`): a batch of test data.
        - **dataloader_idx** (`int`): the task ID of seen tasks to be tested. A default value of 0 is given otherwise the LightningModule will raise a `RuntimeError`.
        
        **Returns:**
        - **outputs** (`dict[str, Tensor]`): a dictionary contains loss and other metrics from this test step. Key (`str`) is the metrics name, value (`Tensor`) is the metrics. 
        """
        test_task_id = dataloader_idx + 1

        x, y = batch
        logits = self.forward(
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
