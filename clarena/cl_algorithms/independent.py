r"""
The submodule in `cl_algorithms` for Independent learning algorithm.
"""

__all__ = ["Independent"]

import logging
from copy import deepcopy
from typing import Any

from torch import Tensor
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader

from clarena.backbones import CLBackbone
from clarena.cl_algorithms import Finetuning, UnlearnableCLAlgorithm
from clarena.heads import HeadDIL, HeadsCIL, HeadsTIL

# always get logger for built-in logging in each module
pylogger = logging.getLogger(__name__)


class Independent(Finetuning):
    r"""Independent learning algorithm.

    Another naive way for task-incremental learning aside from Finetuning. It assigns a new independent model for each task. This is a simple way to avoid catastrophic forgetting at the extreme cost of memory. It achieves the theoretical upper bound of performance in continual learning.

    We implement Independent as a subclass of Finetuning algorithm, as Independent has the same `forward()`, `training_step()`, `validation_step()` and `test_step()` method as `Finetuning` class.
    """

    def __init__(
        self,
        backbone: CLBackbone,
        heads: HeadsTIL | HeadsCIL | HeadDIL,
        non_algorithmic_hparams: dict[str, Any] = {},
        **kwargs,
    ) -> None:
        r"""Initialize the Independent algorithm with the network. It has no additional hyperparameters.

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

        self.original_backbone_state_dict: dict = deepcopy(backbone.state_dict())
        r"""The original backbone state dict before training on any task. Used to initialize new independent backbones for new tasks."""

        self.backbones: dict[int, CLBackbone] = {}
        r"""The list of independent backbones for each task. Keys are task IDs and values are the corresponding backbones. """

        self.backbone_valid_task_ids: set[int] = set()
        r"""The list of task IDs that have valid backbones."""

    def on_train_start(self):
        r"""At the start of training for current task `self.task_id`, load the original backbone state dict to create a new independent backbone for the current task."""
        self.backbone.load_state_dict(self.original_backbone_state_dict)

    def on_train_end(self) -> None:
        r"""Save the trained independent backbone for `self.task_id`."""
        self.backbones[self.task_id] = deepcopy(self.backbone)
        self.backbone_valid_task_ids.add(self.task_id)

    def test_step(
        self, batch: DataLoader, batch_idx: int, dataloader_idx: int = 0
    ) -> dict[str, Tensor]:
        r"""Test step for current task `self.task_id`, which tests for all seen tasks indexed by `dataloader_idx`.

        **Args:**
        - **batch** (`Any`): a batch of test data.
        - **dataloader_idx** (`int`): the task ID of seen tasks to be tested. A default value of 0 is given otherwise the LightningModule will raise a `RuntimeError`.

        **Returns:**
        - **outputs** (`dict[str, Tensor]`): a dictionary contains loss and other metrics from this test step. Keys (`str`) are the metrics names, and values (`Tensor`) are the metrics.
        """
        test_task_id = self.get_test_task_id_from_dataloader_idx(dataloader_idx)

        x, y = batch
        backbone = self.backbones[
            test_task_id
        ]  # use the corresponding independenet backbone for the test task
        feature, _ = backbone(x, stage="test", task_id=test_task_id)
        logits = self.heads(feature, test_task_id)
        # use the corresponding head to test (instead of the current task `self.task_id`)
        loss_cls = self.criterion(logits, y)
        acc = (logits.argmax(dim=1) == y).float().mean()

        # Return metrics for lightning loggers callback to handle at `on_test_batch_end()`
        return {
            "loss_cls": loss_cls,
            "acc": acc,
        }


class UnlearnableIndependent(UnlearnableCLAlgorithm, Independent):
    r"""Unlearnable Independent learning algorithm.

    This is a variant of Independent that supports unlearning. It has the same functionality as Independent, but it also supports unlearning requests and permanent tasks.
    """

    def __init__(
        self,
        backbone: CLBackbone,
        heads: HeadsTIL | HeadsCIL | HeadDIL,
        non_algorithmic_hparams: dict[str, Any] = {},
        disable_unlearning: bool = False,
    ) -> None:
        r"""Initialize the Independent algorithm with the network. It has no additional hyperparameters.

        **Args:**
        - **backbone** (`CLBackbone`): backbone network.
        - **heads** (`HeadsTIL` | `HeadsCIL` | `HeadDIL`): output heads.
        - **disable_unlearning** (`bool`): whether to disable unlearning. This is used in reference experiments following continual learning pipeline. Default is `False`.

        """
        super().__init__(
            backbone=backbone,
            heads=heads,
            non_algorithmic_hparams=non_algorithmic_hparams,
            disable_unlearning=disable_unlearning,
        )

    def aggregated_backbone_output(self, input: Tensor) -> Tensor:
        r"""Get the aggregated backbone output for the input data. All parts of backbones should be aggregated together.

        This output feature is used for measuring unlearning metrics, such as Distribution Distance (DD). An aggregated output involving every part of the backbone is needed to ensure the fairness of the metric.

        **Args:**
        - **input** (`Tensor`): the input tensor from data.

        **Returns:**
        - **output** (`Tensor`): the aggregated backbone output tensor.
        """
        feature = 0

        for t in self.backbone_valid_task_ids:
            feature_t = self.backbones[t](input, stage="unlearning_test")[0]
            feature += feature_t
        feature = feature / len(self.backbone_valid_task_ids)

        return feature
