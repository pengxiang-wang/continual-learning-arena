r"""
The submodule in `cl_algorithms` for Independent learning algorithm.
"""

__all__ = ["Independent"]

import logging
from copy import deepcopy

from torch import Tensor
from torch.utils.data import DataLoader

from clarena.backbones import CLBackbone
from clarena.cl_algorithms import Finetuning
from clarena.cl_heads import HeadsCIL, HeadsTIL

# always get logger for built-in logging in each module
pylogger = logging.getLogger(__name__)


class Independent(Finetuning):
    r"""Independent learning algorithm.

    It is another naive way for task-incremental learning aside from Finetuning. It assigns a new independent model for each task. This is a simple way to avoid catastrophic forgetting at the extreme cost of memory. It achieves the theoretical upper bound of performance in continual learning.

    We implement Independent as a subclass of Finetuning algorithm, as Independent has the same `forward()`, `training_step()`, `validation_step()` and `test_step()` method as `Finetuning` class.
    """

    def __init__(
        self,
        backbone: CLBackbone,
        heads: HeadsTIL | HeadsCIL,
    ) -> None:
        r"""Initialise the Independent algorithm with the network. It has no additional hyperparamaters.

        **Args:**
        - **backbone** (`CLBackbone`): backbone network.
        - **heads** (`HeadsTIL` | `HeadsCIL`): output heads.
        """
        Finetuning.__init__(self, backbone=backbone, heads=heads)

        self.original_backbone_state_dict: CLBackbone = deepcopy(backbone.state_dict())
        r"""Store the original backbone network state dict as the source of creating new independent backbone. """

        self.backbones: dict[str, CLBackbone] = {}
        r"""Store the list of independent backbones for each task. Keys are task IDs (string type) and values are the corresponding backbone. """

    def on_fit_start(self) -> None:
        r"""Initialise an independent backbone for `self.task_id`, duplicated from the original backbone."""
        self.backbone.load_state_dict(self.original_backbone_state_dict)

    def on_train_end(self) -> None:
        r"""Store the trained independent backbone for `self.task_id`."""
        self.backbones[f"{self.task_id}"] = deepcopy(self.backbone)

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
        test_task_id = self.get_test_task_id_from_dataloader_idx(dataloader_idx)

        x, y = batch
        backbone = self.backbones[
            f"{test_task_id}"
        ]  # use the corresponding independenet backbone for the test task
        feature, activations = backbone(x, stage="test", task_id=test_task_id)
        logits = self.heads(feature, test_task_id)
        # use the corresponding head to test (instead of the current task `self.task_id`)
        loss_cls = self.criterion(logits, y)
        acc = (logits.argmax(dim=1) == y).float().mean()

        # Return metrics for lightning loggers callback to handle at `on_test_batch_end()`
        return {
            "loss_cls": loss_cls,
            "acc": acc,
        }
