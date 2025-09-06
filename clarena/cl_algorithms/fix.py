r"""
The submodule in `cl_algorithms` for Fix algorithm.
"""

__all__ = ["Fix"]

import logging
from typing import Any

from torch import Tensor

from clarena.backbones import CLBackbone
from clarena.cl_algorithms import Finetuning
from clarena.heads import HeadsCIL, HeadsTIL

# always get logger for built-in logging in each module
pylogger = logging.getLogger(__name__)


class Fix(Finetuning):
    r"""Fix algorithm.

    Another naive way for task-incremental learning aside from Finetuning. It simply fixes the backbone forever after training first task. It serves as kind of toy algorithm when discussing stability-plasticity dilemma in continual learning.

    We implement `Fix` as a subclass of `Finetuning`, as it shares `forward()`, `validation_step()`, and `test_step()` with `Finetuning`.
    """

    def __init__(
        self,
        backbone: CLBackbone,
        heads: HeadsTIL | HeadsCIL,
        non_algorithmic_hparams: dict[str, Any] = {},
    ) -> None:
        r"""Initialize the Fix algorithm with the network. It has no additional hyperparameters.

        **Args:**
        - **backbone** (`CLBackbone`): backbone network.
        - **heads** (`HeadsTIL` | `HeadsCIL`): output heads.
        - **non_algorithmic_hparams** (`dict[str, Any]`): non-algorithmic hyperparameters that are not related to the algorithm itself are passed to this `LightningModule` object from the config, such as optimizer and learning rate scheduler configurations. They are saved for Lightning APIs from `save_hyperparameters()` method. This is useful for the experiment configuration and reproducibility.

        """
        super().__init__(
            backbone=backbone,
            heads=heads,
            non_algorithmic_hparams=non_algorithmic_hparams,
        )

        # freeze only once after task 1
        self._backbone_frozen: bool = False

    def training_step(self, batch: Any) -> dict[str, Tensor]:
        """Training step for current task `self.task_id`.

        **Args:**
        - **batch** (`Any`): a batch of training data.

        **Returns:**
        - **outputs** (`dict[str, Tensor]`): a dictionary contains loss and other metrics from this training step. Keys (`str`) are the metrics names, and values (`Tensor`) are the metrics. Must include the key 'loss' which is total loss in the case of automatic optimization, according to PyTorch Lightning docs.
        """
        x, y = batch

        if self.task_id != 1:
            # freeze the backbone once after the first task; also stop BN/Dropout updates
            if not self._backbone_frozen:
                for p in self.backbone.parameters():
                    p.requires_grad = False
                self.backbone.eval()
                self._backbone_frozen = True
                pylogger.info("Fix: backbone frozen after task 1 (set to eval mode).")
        else:
            # ensure backbone is trainable during the first task
            self.backbone.train()

        # classification loss
        logits, activations = self.forward(x, stage="train", task_id=self.task_id)
        loss_cls = self.criterion(logits, y)

        # total loss
        loss = loss_cls

        # accuracy of the batch
        acc = (logits.argmax(dim=1) == y).float().mean()

        return {
            "loss": loss,  # return loss is essential for training step, or backpropagation will fail
            "loss_cls": loss_cls,
            "acc": acc,
            "activations": activations,
        }
