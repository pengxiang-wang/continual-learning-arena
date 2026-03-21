r"""
The submodule in `cl_algorithms` for Fix algorithm.
"""

__all__ = ["Fix"]

import logging
from typing import Any

import torch
from torch import Tensor

from clarena.backbones import CLBackbone
from clarena.cl_algorithms import Finetuning
from clarena.heads import HeadDIL, HeadsCIL, HeadsTIL

# always get logger for built-in logging in each module
pylogger = logging.getLogger(__name__)


class Fix(Finetuning):
    r"""Fix algorithm.

    Another naive way for task-incremental learning aside from Finetuning. It trains only on the first task and stops optimization completely for all later tasks. It serves as kind of toy algorithm when discussing stability-plasticity dilemma in continual learning.

    We implement `Fix` as a subclass of `Finetuning`, as it shares `forward()`, `validation_step()`, and `test_step()` with `Finetuning`.
    """

    def __init__(
        self,
        backbone: CLBackbone,
        heads: HeadsTIL | HeadsCIL | HeadDIL,
        non_algorithmic_hparams: dict[str, Any] = {},
        **kwargs,
    ) -> None:
        r"""Initialize the Fix algorithm with the network.

        **Args:**
        - **backbone** (`CLBackbone`): backbone network.
        - **heads** (`HeadsTIL` | `HeadsCIL` | `HeadDIL`): output heads.
        - **non_algorithmic_hparams** (`dict[str, Any]`): non-algorithmic hyperparameters that are not related to the algorithm itself are passed to this `LightningModule` object from the config, such as optimizer and learning rate scheduler configurations. They are saved for Lightning APIs from `save_hyperparameters()` method. This is useful for the experiment configuration and reproducibility.
        - **kwargs**: Reserved for multiple inheritance.

        """
        super().__init__(
            backbone=backbone,
            heads=heads,
            non_algorithmic_hparams=non_algorithmic_hparams,
            **kwargs,
        )

        self.automatic_optimization = False
        r"""Fix uses manual optimization so later tasks can skip optimizer steps entirely."""

        self._fixed_after_first_task: bool = False
        r"""Whether the model has been switched into the fixed-no-training regime after task 1."""

    def _enforce_fixed_module_modes(self) -> None:
        r"""Keep the model in eval mode after task 1 to avoid BN/Dropout state updates."""
        if not self._fixed_after_first_task:
            return

        self.backbone.eval()
        self.heads.eval()

    def _activate_fix_policy(self) -> None:
        r"""Activate the no-training regime after task 1."""
        if self.task_id == 1 or self._fixed_after_first_task:
            self._enforce_fixed_module_modes()
            return

        self._fixed_after_first_task = True
        self._enforce_fixed_module_modes()
        pylogger.info(
            "Fix: optimizer updates are disabled after task 1; later tasks run in eval mode only."
        )

    def train(self, mode: bool = True):
        r"""Override `nn.Module.train` to keep the fixed model in eval mode."""
        super().train(mode)
        self._enforce_fixed_module_modes()
        return self

    def training_step(self, batch: Any) -> dict[str, Tensor]:
        """Training step for current task `self.task_id`.

        **Args:**
        - **batch** (`Any`): a batch of training data.

        **Returns:**
        - **outputs** (`dict[str, Tensor]`): a dictionary contains loss and other metrics from this training step. Keys (`str`) are the metrics names, and values (`Tensor`) are the metrics. Must include the key 'loss' which is total loss in the case of automatic optimization, according to PyTorch Lightning docs.
        """
        x, y = batch

        if self.task_id == 1:
            opt = self.optimizers()
            opt.zero_grad()

            self.backbone.train()
            self.heads.train()
            logits, activations = self.forward(x, stage="train", task_id=self.task_id)
            loss_cls = self.criterion(logits, y)
            loss = loss_cls
            self.manual_backward(loss)
            opt.step()
            preds = logits.argmax(dim=1)
            acc = (preds == y).float().mean()
        else:
            self._activate_fix_policy()

            with torch.inference_mode():
                logits, activations = self.forward(
                    x, stage="train", task_id=self.task_id
                )
                loss_cls = self.criterion(logits, y)
                loss = loss_cls
                preds = logits.argmax(dim=1)
                acc = (preds == y).float().mean()

        return {
            "preds": preds,
            "loss": loss,  # return loss is essential for training step, or backpropagation will fail
            "loss_cls": loss_cls,
            "acc": acc,
            "activations": activations,
        }
