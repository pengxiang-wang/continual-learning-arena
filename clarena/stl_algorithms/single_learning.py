r"""
The submodule in `stl_algorithms` for single learning algorithm.
"""

__all__ = ["SingleLearning"]

import logging
from typing import Any

from torch import Tensor
from torch.utils.data import DataLoader

from clarena.backbones import Backbone
from clarena.heads import HeadSTL
from clarena.stl_algorithms import STLAlgorithm

# always get logger for built-in logging in each module
pylogger = logging.getLogger(__name__)


class SingleLearning(STLAlgorithm):
    r"""Single learning algorithm.

    The most naive way for single-task learning. It directly trains the task.
    """

    def __init__(
        self,
        backbone: Backbone,
        head: HeadSTL,
    ) -> None:
        r"""Initialize the SingleLearning algorithm with the network. It has no additional hyperparameters.

        **Args:**
        - **backbone** (`CLBackbone`): backbone network.
        - **head** (`HeadsMTL`): output head.
        """
        super().__init__(backbone=backbone, head=head)

    def training_step(self, batch: Any) -> dict[str, Tensor]:
        r"""Training step for single learning.

        **Args:**
        - **batch** (`Any`): a batch of training data.

        **Returns:**
        - **outputs** (`dict[str, Tensor]`): a dictionary contains loss and accuracy from this training step. Keys (`str`) are the metrics names, and values (`Tensor`) are the metrics. Must include the key 'loss' which is total loss in the case of automatic optimization, according to PyTorch Lightning docs.
        """
        x, y = batch

        logits, activations = self.forward(x, stage="train")

        # classification loss
        loss_cls = self.criterion(logits, y)

        # total loss
        loss = loss_cls

        # accuracy of the batch
        acc = (logits.argmax(dim=1) == y).float().mean()

        return {
            "loss": loss,  # return loss is essential for training step, or backpropagation will fail
            "loss_cls": loss_cls,
            "acc": acc,  # Return other metrics for lightning loggers callback to handle at `on_train_batch_end()`
            "activations": activations,
        }

    def validation_step(self, batch: DataLoader, batch_idx: int) -> dict[str, Tensor]:
        r"""Validation step for single learning.

        **Args:**
        - **batch** (`Any`): a batch of validation data.

        **Returns:**
        - **outputs** (`dict[str, Tensor]`): a dictionary contains loss and accuracy from this validation step. Keys (`str`) are the metrics names, and values (`Tensor`) are the metrics.
        """

        x, y = batch

        # the batch is from the same task, so no need to divide the input batch by tasks
        logits, activations = self.forward(x, stage="validation")

        loss_cls = self.criterion(logits, y)
        acc = (logits.argmax(dim=1) == y).float().mean()

        # Return metrics for lightning loggers callback to handle at `on_validation_batch_end()`
        return {
            "loss_cls": loss_cls,
            "acc": acc,
        }

    def test_step(self, batch: DataLoader, batch_idx: int) -> dict[str, Tensor]:
        r"""Test step for single learning.

        **Args:**
        - **batch** (`Any`): a batch of test data.

        **Returns:**
        - **outputs** (`dict[str, Tensor]`): a dictionary contains loss and accuracy from this test step. Keys (`str`) are the metrics names, and values (`Tensor`) are the metrics.
        """

        x, y = batch

        logits, activations = self.forward(x, stage="test")
        loss_cls = self.criterion(logits, y)
        acc = (logits.argmax(dim=1) == y).float().mean()

        # Return metrics for lightning loggers callback to handle at `on_test_batch_end()`
        return {
            "loss_cls": loss_cls,
            "acc": acc,
        }
