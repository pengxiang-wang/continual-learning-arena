r"""
The submodule in `cl_algorithms` for Finetuning algorithm.
"""

__all__ = ["Finetuning"]

import logging
from typing import Any

import torch
from torch import Tensor
from torch.utils.data import DataLoader

from clarena.backbones import CLBackbone
from clarena.cl_algorithms import CLAlgorithm
from clarena.cl_heads import HeadsCIL, HeadsTIL

# always get logger for built-in logging in each module
pylogger = logging.getLogger(__name__)


class Finetuning(CLAlgorithm):
    r"""Finetuning algorithm.

    It is the most naive way for task-incremental learning. It simply initialises the backbone from the last task when training new task.
    """

    def __init__(
        self,
        backbone: CLBackbone,
        heads: HeadsTIL | HeadsCIL,
    ) -> None:
        r"""Initialise the Finetuning algorithm with the network. It has no additional hyperparamaters.

        **Args:**
        - **backbone** (`CLBackbone`): backbone network.
        - **heads** (`HeadsTIL` | `HeadsCIL`): output heads.
        """
        CLAlgorithm.__init__(self, backbone=backbone, heads=heads)

    def forward(self, input: Tensor, stage: str, task_id: int | None = None) -> Tensor:
        r"""The forward pass for data from task `task_id`. Note that it is nothing to do with `forward()` method in `nn.Module`.

        **Args:**
        - **input** (`Tensor`): The input tensor from data.
        - **stage** (`str`): the stage of the forward pass, should be one of the following:
            1. 'train': training stage.
            2. 'validation': validation stage.
            3. 'test': testing stage.
        - **task_id** (`int`): the task ID where the data are from. If stage is 'train' or `validation`, it is usually from the current task `self.task_id`. If stage is 'test', it could be from any seen task. In TIL, the task IDs of test data are provided thus this argument can be used. In CIL, they are not provided, so it is just a placeholder for API consistence but never used, and best practices are not to provide this argument and leave it as the default value. Finetuning algorithm works both for TIL and CIL.

        **Returns:**
        - **logits** (`Tensor`): the output logits tensor.
        - **hidden_features** (`dict[str, Tensor]`): the hidden features (after activation) in each weighted layer. Key (`str`) is the weighted layer name, value (`Tensor`) is the hidden feature tensor. This is used for the continual learning algorithms that need to use the hidden features for various purposes. Although Finetuning algorithm does not need this, it is still provided for API consistence for other algorithms inherited this `forward()` method of `Finetuning` class.
        """
        feature, hidden_features = self.backbone(input, stage=stage, task_id=task_id)
        logits = self.heads(feature, task_id)
        return logits, hidden_features

    def training_step(self, batch: Any) -> dict[str, Tensor]:
        """Training step for current task `self.task_id`.

        **Args:**
        - **batch** (`Any`): a batch of training data.

        **Returns:**
        - **outputs** (`dict[str, Tensor]`): a dictionary contains loss and other metrics from this training step. Key (`str`) is the metrics name, value (`Tensor`) is the metrics. Must include the key 'loss' which is total loss in the case of automatic optimization, according to PyTorch Lightning docs.
        """
        x, y = batch

        # classification loss
        logits, hidden_features = self.forward(x, stage="train", task_id=self.task_id)
        loss_cls = self.criterion(logits, y)

        # total loss
        loss = loss_cls

        # accuracy of the batch
        acc = (logits.argmax(dim=1) == y).float().mean()

        return {
            "loss": loss,  # Return loss is essential for training step, or backpropagation will fail
            "loss_cls": loss_cls,
            "acc": acc,  # Return other metrics for lightning loggers callback to handle at `on_train_batch_end()`
            "hidden_features": hidden_features,
        }

    def validation_step(self, batch: Any) -> dict[str, Tensor]:
        r"""Validation step for current task `self.task_id`.

        **Args:**
        - **batch** (`Any`): a batch of validation data.

        **Returns:**
        - **outputs** (`dict[str, Tensor]`): a dictionary contains loss and other metrics from this validation step. Key (`str`) is the metrics name, value (`Tensor`) is the metrics.
        """
        x, y = batch
        logits, hidden_features = self.forward(
            x,
            stage="validation",
            task_id=self.task_id,
        )
        loss_cls = self.criterion(logits, y)
        acc = (logits.argmax(dim=1) == y).float().mean()

        # Return metrics for lightning loggers callback to handle at `on_validation_batch_end()`
        return {
            "loss_cls": loss_cls,
            "acc": acc,
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
        logits, hidden_features = self.forward(
            x, stage="test", task_id=test_task_id
        )  # use the corresponding head to test (instead of the current task `self.task_id`)
        loss_cls = self.criterion(logits, y)
        acc = (logits.argmax(dim=1) == y).float().mean()

        # Return metrics for lightning loggers callback to handle at `on_test_batch_end()`
        return {
            "loss_cls": loss_cls,
            "acc": acc,
        }
