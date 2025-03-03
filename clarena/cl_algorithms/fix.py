r"""
The submodule in `cl_algorithms` for Fix algorithm.
"""

__all__ = ["Fix"]

import logging
from typing import Any

from torch import Tensor

from clarena.backbones import CLBackbone
from clarena.cl_algorithms import Finetuning
from clarena.cl_heads import HeadsCIL, HeadsTIL

# always get logger for built-in logging in each module
pylogger = logging.getLogger(__name__)


class Fix(Finetuning):
    r"""Fix algorithm.

    It is another naive way for task-incremental learning aside from Finetuning. It serves as kind of toy algorithm when discussing stability-plasticity dilemma in continual learning. It simply fixes the backbone forever after training first task.

    We implement Fix as a subclass of Finetuning algorithm, as Fix has the same `forward()`, `validation_step()` and `test_step()` method as `Finetuning` class.
    """

    def __init__(
        self,
        backbone: CLBackbone,
        heads: HeadsTIL | HeadsCIL,
    ) -> None:
        r"""Initialise the Fix algorithm with the network. It has no additional hyperparamaters.

        **Args:**
        - **backbone** (`CLBackbone`): backbone network.
        - **heads** (`HeadsTIL` | `HeadsCIL`): output heads.
        """
        Finetuning.__init__(self, backbone=backbone, heads=heads)

    def training_step(self, batch: Any) -> dict[str, Tensor]:
        """Training step for current task `self.task_id`.

        **Args:**
        - **batch** (`Any`): a batch of training data.

        **Returns:**
        - **outputs** (`dict[str, Tensor]`): a dictionary contains loss and other metrics from this training step. Key (`str`) is the metrics name, value (`Tensor`) is the metrics. Must include the key 'loss' which is total loss in the case of automatic optimization, according to PyTorch Lightning docs.
        """
        x, y = batch

        if self.task_id != 1:
            # Fix the backbone after training the first task
            for param in self.backbone.parameters():
                param.requires_grad = False

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
            "acc": acc,
            "hidden_features": hidden_features,
        }
