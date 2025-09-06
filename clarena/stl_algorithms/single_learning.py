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
        non_algorithmic_hparams: dict[str, Any] = {},
    ) -> None:
        r"""Initialize the SingleLearning algorithm with the network. It has no additional hyperparameters.

        **Args:**
        - **backbone** (`Backbone`): backbone network.
        - **head** (`HeadSTL`): output head.
        - **non_algorithmic_hparams** (`dict[str, Any]`): non-algorithmic hyperparameters that are not related to the algorithm itself are passed to this `LightningModule` object from the config, such as optimizer and learning rate scheduler configurations. They are saved for Lightning APIs from `save_hyperparameters()` method. This is useful for the experiment configuration and reproducibility.
        """
        super().__init__(
            backbone=backbone,
            head=head,
            non_algorithmic_hparams=non_algorithmic_hparams,
        )

    def training_step(self, batch: Any) -> dict[str, Tensor]:
        r"""Training step.

        **Args:**
        - **batch** (`Any`): a batch of training data.

        **Returns:**
        - **outputs** (`dict[str, Tensor]`): a dictionary contains loss and accuracy from this training step. Keys (`str`) are the metrics names, and values (`Tensor`) are the metrics. Must include the key 'loss' which is total loss in the case of automatic optimization, according to PyTorch Lightning docs.
        """
        x, y = batch

        # classification loss
        logits, activations = self.forward(x, stage="train")
        loss_cls = self.criterion(logits, y)

        # total loss
        loss = loss_cls

        # predicted labels
        preds = logits.argmax(dim=1)

        # accuracy of the batch
        acc = (preds == y).float().mean()

        return {
            "preds": preds,
            "loss": loss,  # return loss is essential for training step, or backpropagation will fail
            "loss_cls": loss_cls,
            "acc": acc,  # Return other metrics for lightning loggers callback to handle at `on_train_batch_end()`
            "activations": activations,
        }

    def validation_step(self, batch: Any) -> dict[str, Tensor]:
        r"""Validation step.

        **Args:**
        - **batch** (`Any`): a batch of validation data.

        **Returns:**
        - **outputs** (`dict[str, Tensor]`): a dictionary contains loss and accuracy from this validation step. Keys (`str`) are the metrics names, and values (`Tensor`) are the metrics.
        """

        x, y = batch
        logits, _ = self.forward(x, stage="validation")
        loss_cls = self.criterion(logits, y)
        preds = logits.argmax(dim=1)
        acc = (preds == y).float().mean()

        # Return metrics for lightning loggers callback to handle at `on_validation_batch_end()`
        return {
            "preds": preds,
            "loss_cls": loss_cls,
            "acc": acc,
        }

    def test_step(self, batch: DataLoader) -> dict[str, Tensor]:
        r"""Test step.

        **Args:**
        - **batch** (`Any`): a batch of test data.

        **Returns:**
        - **outputs** (`dict[str, Tensor]`): a dictionary contains loss and accuracy from this test step. Keys (`str`) are the metrics names, and values (`Tensor`) are the metrics.
        """

        x, y = batch
        logits, _ = self.forward(x, stage="test")
        loss_cls = self.criterion(logits, y)
        preds = logits.argmax(dim=1)
        acc = (preds == y).float().mean()

        # Return metrics for lightning loggers callback to handle at `on_test_batch_end()`
        return {
            "preds": preds,
            "loss_cls": loss_cls,
            "acc": acc,
        }
