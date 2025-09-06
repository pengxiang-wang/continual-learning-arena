r"""
The submodule in `cl_algorithms` for Random algorithm.
"""

__all__ = ["Random"]

import logging
from typing import Any

from torch import Tensor
import torch

from clarena.backbones import CLBackbone
from clarena.cl_algorithms import Finetuning
from clarena.heads import HeadsCIL, HeadsTIL

# always get logger for built-in logging in each module
pylogger = logging.getLogger(__name__)


class Random(Finetuning):
    r"""Random stratified model.

    Pass the training step and simply use the randomly initialized model to predict the test data. This serves as a reference model to compute forgetting rate. See chapter 4 in [HAT (Hard Attention to the Task) paper](http://proceedings.mlr.press/v80/serra18a).


    We implement Random as a subclass of Finetuning algorithm, as Random has the same `forward()`, `validation_step()` and `test_step()` method as `Finetuning` class.
    """

    def __init__(
        self,
        backbone: CLBackbone,
        heads: HeadsTIL | HeadsCIL,
        non_algorithmic_hparams: dict[str, Any] = {},
    ) -> None:
        r"""Initialize the Random algorithm with the network. It has no additional hyperparameters.

        **Args:**
        - **backbone** (`CLBackbone`): backbone network.
        - **heads** (`HeadsTIL` | `HeadsCIL`): output heads.
        - **non_algorithmic_hparams** (`dict[str, Any]`): non-algorithmic hyperparameters that are not related to the algorithm itself are passed to this `LightningModule` object from the config, such as optimizer and learning rate scheduler configurations. They are saved for Lightning APIs from `save_hyperparameters()` method. This is useful for the experiment configuration and reproducibility.

        """
        super().__init__(backbone=backbone, heads=heads, non_algorithmic_hparams=non_algorithmic_hparams)

        # set manual optimization
        self.automatic_optimization = False

        # ensure we only freeze/switch to eval once
        self._frozen_applied: bool = False

    def training_step(self, batch: Any) -> dict[str, Tensor]:
        """Pass the training step for current task `self.task_id`.

        **Args:**
        - **batch** (`Any`): a batch of training data.

        **Returns:**
        - **outputs** (`dict[str, Tensor]`): a dictionary contains loss and other metrics from this training step. Keys (`str`) are the metrics names, and values (`Tensor`) are the metrics. Must include the key 'loss' which is total loss in the case of automatic optimization, according to PyTorch Lightning docs.
        """
        x, y = batch

        # freeze all parameters and stop BN/Dropout updates once
        if not self._frozen_applied:
            for p in self.parameters():
                p.requires_grad = False
            self.eval()
            self._frozen_applied = True
            pylogger.info("Random: parameters frozen and module set to eval; no training will occur.")

        # run forward and metrics without autograd
        with torch.inference_mode():
            logits, activations = self.forward(x, stage="train", task_id=self.task_id)
            loss_cls = self.criterion(logits, y)
            loss = loss_cls
            acc = (logits.argmax(dim=1) == y).float().mean()

        # note: no optimizer step, by design of Random algorithm.
        
        return {
            "loss": loss,  # return loss is essential for training step, or backpropagation will fail
            "loss_cls": loss_cls,
            "acc": acc,
            "activations": activations,
        }

