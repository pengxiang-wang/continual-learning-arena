r"""
The submodule in `cl_algorithms` for Random algorithm.
"""

__all__ = ["Random"]

import logging
from typing import Any

from torch import Tensor

from clarena.backbones import CLBackbone
from clarena.cl_algorithms import Finetuning
from clarena.cl_heads import HeadsCIL, HeadsTIL

# always get logger for built-in logging in each module
pylogger = logging.getLogger(__name__)


class Random(Finetuning):
    r"""Random stratified model. It pass the training step and simply use the randomly initialized model to predict the test data.

    This serves as a reference model to compute forgetting rate. See chapter 4 in [HAT (Hard Attention to the Task, 2018)](http://proceedings.mlr.press/v80/serra18a).


    We implement Random as a subclass of Finetuning algorithm, as Random has the same `forward()`, `validation_step()` and `test_step()` method as `Finetuning` class.
    """

    def __init__(
        self,
        backbone: CLBackbone,
        heads: HeadsTIL | HeadsCIL,
    ) -> None:
        r"""Initialise the Random algorithm with the network. It has no additional hyperparameters.

        **Args:**
        - **backbone** (`CLBackbone`): backbone network.
        - **heads** (`HeadsTIL` | `HeadsCIL`): output heads.
        """
        Finetuning.__init__(self, backbone=backbone, heads=heads)

        # set manual optimisation
        self.automatic_optimization = False

    def training_step(self, batch: Any) -> dict[str, Tensor]:
        """Pass the training step for current task `self.task_id`.

        **Args:**
        - **batch** (`Any`): a batch of training data.

        **Returns:**
        - **outputs** (`dict[str, Tensor]`): a dictionary contains loss and other metrics from this training step. Key (`str`) is the metrics name, value (`Tensor`) is the metrics. Must include the key 'loss' which is total loss in the case of automatic optimization, according to PyTorch Lightning docs.
        """
        x, y = batch

        # classification loss
        logits, activations = self.forward(x, stage="train", task_id=self.task_id)
        loss_cls = self.criterion(logits, y)

        # total loss
        loss = loss_cls

        # accuracy of the batch
        acc = (logits.argmax(dim=1) == y).float().mean()

        # note that we have set automatic_optimization = False, here we do not include the optimization step like `self.optimizers().step()` to make sure the model is not updated during training step. This is the key point of Random algorithm.

        return {
            "loss": loss,  # Return loss is essential for training step, or backpropagation will fail
            "loss_cls": loss_cls,
            "acc": acc,
            "activations": activations,
        }
