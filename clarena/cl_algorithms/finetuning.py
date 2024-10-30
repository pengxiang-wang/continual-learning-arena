"""
The submodule in `cl_algorithms` for Finetuning algorithm.
"""

__all__ = ["Finetuning"]

import logging
from typing import Any

import torch
from torch.utils.data import DataLoader

from clarena.cl_algorithms import CLAlgorithm

# always get logger for built-in logging in each module
pylogger = logging.getLogger(__name__)


class Finetuning(CLAlgorithm):
    """Finetuning algorithm.

    It is the most naive way for task-incremental learning. It simply initialises the backbone from the last task when training new task.
    """

    def __init__(
        self,
        backbone: torch.nn.Module,
        heads: torch.nn.Module,
    ) -> None:
        """Initialise the Finetuning algorithm with the network. It has no additional hyperparamaters.

        **Args:**
        - **backbone** (`CLBackbone`): backbone network.
        - **heads** (`HeadsTIL` | `HeadsCIL`): output heads.
        """
        super().__init__(backbone=backbone, heads=heads)

    def training_step(self, batch: Any):
        """Training step for current task `self.task_id`.

        **Args:**
        - **batch** (`Any`): a batch of training data.
        """
        x, y = batch
        logits = self.forward(x, self.task_id)
        loss_cls = self.criterion(logits, y)
        loss = loss_cls
        acc = (logits.argmax(dim=1) == y).float().mean()

        # Return loss is essential for training step, or backpropagation will fail
        # Return other metrics for lightning loggers callback to handle at `on_train_batch_end()`
        return {
            "loss": loss,
            "loss_cls": loss_cls,
            "acc": acc,
        }

    def validation_step(self, batch: Any):
        """Validation step for current task `self.task_id`.

        **Args:**
        - **batch** (`Any`): a batch of validation data.
        """
        x, y = batch
        logits = self.forward(x, self.task_id)
        loss_cls = self.criterion(logits, y)
        acc = (logits.argmax(dim=1) == y).float().mean()

        # Return metrics for lightning loggers callback to handle at `on_validation_batch_end()`
        return {
            "loss_cls": loss_cls,
            "acc": acc,
        }

    def test_step(self, batch: DataLoader, batch_idx: int, dataloader_idx: int = 0):
        """Test step for current task `self.task_id`, which tests for all seen tasks indexed by `dataloader_idx`.

        **Args:**
        - **batch** (`Any`): a batch of test data.
        - **dataloader_idx** (`int`): the task ID of seen tasks to be tested. A default value of 0 is given otherwise the LightningModule will raise a `RuntimeError`.
        """
        test_task_id = dataloader_idx + 1

        x, y = batch
        logits = self.forward(
            x, test_task_id
        )  # use the corresponding head to test (instead of the current task `self.task_id`)
        loss_cls = self.criterion(logits, y)
        acc = (logits.argmax(dim=1) == y).float().mean()

        # Return metrics for lightning loggers callback to handle at `on_test_batch_end()`
        return {
            "loss_cls": loss_cls,
            "acc": acc,
        }
