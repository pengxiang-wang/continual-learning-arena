r"""
The submodule in `cl_algorithms` for Finetuning algorithm.
"""

__all__ = ["Finetuning", "AmnesiacFinetuning"]

import logging
from copy import deepcopy
from typing import Any

import torch
from torch import Tensor
from torch.utils.data import DataLoader

from clarena.backbones import CLBackbone
from clarena.cl_algorithms import AmnesiacCLAlgorithm, CLAlgorithm
from clarena.heads import HeadDIL, HeadsCIL, HeadsTIL

# always get logger for built-in logging in each module
pylogger = logging.getLogger(__name__)


class Finetuning(CLAlgorithm):
    r"""Finetuning algorithm.

    The most naive way for task-incremental learning. It simply initializes the backbone from the last task when training new task.
    """

    def __init__(
        self,
        backbone: CLBackbone,
        heads: HeadsTIL | HeadsCIL | HeadDIL,
        non_algorithmic_hparams: dict[str, Any] = {},
        **kwargs,
    ) -> None:
        r"""Initialize the Finetuning algorithm with the network. It has no additional hyperparameters.

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

    def training_step(self, batch: Any) -> dict[str, Tensor]:
        """Training step for current task `self.task_id`.

        **Args:**
        - **batch** (`Any`): a batch of training data.

        **Returns:**
        - **outputs** (`dict[str, Tensor]`): a dictionary contains loss and other metrics from this training step. Keys (`str`) are the metrics names, and values (`Tensor`) are the metrics. Must include the key 'loss' which is total loss in the case of automatic optimization, according to PyTorch Lightning docs.
        """
        x, y = batch

        # classification loss
        logits, activations = self.forward(x, stage="train", task_id=self.task_id)
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
        r"""Validation step for current task `self.task_id`.

        **Args:**
        - **batch** (`Any`): a batch of validation data.

        **Returns:**
        - **outputs** (`dict[str, Tensor]`): a dictionary contains loss and other metrics from this validation step. Key (`str`) are the metrics names, value (`Tensor`) are the metrics.
        """
        x, y = batch
        logits, _ = self.forward(x, stage="validation", task_id=self.task_id)
        loss_cls = self.criterion(logits, y)
        preds = logits.argmax(dim=1)
        acc = (preds == y).float().mean()

        # Return metrics for lightning loggers callback to handle at `on_validation_batch_end()`
        return {
            "preds": preds,
            "loss_cls": loss_cls,
            "acc": acc,
        }

    def test_step(
        self, batch: DataLoader, batch_idx: int, dataloader_idx: int = 0
    ) -> dict[str, Tensor]:
        r"""Test step for current task `self.task_id`, which tests all seen tasks indexed by `dataloader_idx`.

        **Args:**
        - **batch** (`Any`): a batch of test data.
        - **dataloader_idx** (`int`): the task ID of seen tasks to be tested. A default value of 0 is given otherwise the LightningModule will raise a `RuntimeError`.

        **Returns:**
        - **outputs** (`dict[str, Tensor]`): a dictionary contains loss and other metrics from this test step. Key (`str`) are the metrics name, value (`Tensor`) are the metrics.
        """
        test_task_id = self.get_test_task_id_from_dataloader_idx(dataloader_idx)

        x, y = batch

        logits, _ = self.forward(
            x, stage="test", task_id=test_task_id
        )  # use the corresponding head to test (instead of the current task `self.task_id`)
        loss_cls = self.criterion(logits, y)
        preds = logits.argmax(dim=1)
        acc = (preds == y).float().mean()

        # Return metrics for lightning loggers callback to handle at `on_test_batch_end()`
        return {
            "preds": preds,
            "loss_cls": loss_cls,
            "acc": acc,
        }


class AmnesiacFinetuning(AmnesiacCLAlgorithm, Finetuning):
    r"""Amnesiac Finetuning algorithm."""

    def __init__(
        self,
        backbone: CLBackbone,
        heads: HeadsTIL | HeadsCIL | HeadDIL,
        non_algorithmic_hparams: dict[str, Any] = {},
        disable_unlearning: bool = False,
        **kwargs,
    ) -> None:
        r"""Initialize the Amnesiac Finetuning algorithm with the network.


        **Args:**
        - **backbone** (`CLBackbone`): backbone network.
        - **heads** (`HeadsTIL` | `HeadsCIL` | `HeadDIL`): output heads.
        - **non_algorithmic_hparams** (`dict[str, Any]`): non-algorithmic hyperparameters that are not related to the algorithm itself are passed to this `LightningModule` object from the config, such as optimizer and learning rate scheduler configurations. They are saved for Lightning APIs from `save_hyperparameters()` method. This is useful for the experiment configuration and reproducibility.
        - **disable_unlearning** (`bool`): whether to disable the unlearning functionality. Default is `False`.
        - **kwargs**: Reserved for multiple inheritance.
        """
        super().__init__(
            backbone=backbone,
            heads=heads,
            non_algorithmic_hparams=non_algorithmic_hparams,
            disable_unlearning=disable_unlearning,
            **kwargs,
        )

    def on_train_start(self) -> None:
        """Record backbone parameters before training current task."""
        Finetuning.on_train_start(self)
        AmnesiacCLAlgorithm.on_train_start(self)

    def on_train_end(self) -> None:
        """Record backbone parameters before training current task."""
        Finetuning.on_train_end(self)
        AmnesiacCLAlgorithm.on_train_end(self)


# class UnlearnableFinetuning(UnlearnableCLAlgorithm, Finetuning):
#     r"""Unlearnable Finetuning algorithm.

#     Idea:
#     - For each task t, record parameter delta: Δ_t = θ_post - θ_pre
#     - When forgetting task k, rollback: θ ← θ - Δ_k
#     - Optionally reset the corresponding task head (TIL/DIL) to remove that task’s head knowledge.
#     """

#     def __init__(
#         self,
#         backbone: CLBackbone,
#         heads: HeadsTIL | HeadsCIL | HeadDIL,
#         non_algorithmic_hparams: dict[str, Any] = {},
#         disable_unlearning: bool = False,
#         **kwargs,
#     ) -> None:
#         super().__init__(
#             backbone=backbone,
#             heads=heads,
#             non_algorithmic_hparams=non_algorithmic_hparams,
#             disable_unlearning=disable_unlearning,
#             **kwargs,
#         )

#         # --- delta rollback buffers ---
#         self._pre_task_params: dict[str, Tensor] | None = None
#         r"""Backbone parameter snapshot before training current task (θ_pre)."""

#         self.task_deltas: dict[int, dict[str, Tensor]] = {}
#         r"""Per-task parameter delta Δ_t = θ_post - θ_pre, keyed by task_id."""

#     def on_train_start(self) -> None:
#         """Record θ_pre before training current task."""
#         super().on_train_start()

#         if self.disable_unlearning:
#             return

#         self._pre_task_params = {
#             n: p.detach().clone()
#             for n, p in self.backbone.named_parameters()
#             if p.requires_grad
#         }

#     def on_train_end(self) -> None:
#         """Record Δ_t after training current task."""
#         super().on_train_end()

#         if self.disable_unlearning:
#             return

#         if self._pre_task_params is None:
#             return

#         delta_t: dict[str, Tensor] = {}
#         for n, p in self.backbone.named_parameters():
#             if not p.requires_grad:
#                 continue
#             delta_t[n] = (p.detach() - self._pre_task_params[n]).clone()

#         self.task_deltas[self.task_id] = delta_t
#         self._pre_task_params = None

#     @torch.no_grad()
#     def unlearn_task(self, task_id: int) -> None:
#         r"""Forget task `task_id` by delta rollback: θ ← θ - Δ_task_id."""
#         if self.disable_unlearning:
#             return

#         if task_id not in self.task_deltas:
#             raise ValueError(f"No stored delta for task {task_id}.")

#         delta = self.task_deltas[task_id]
#         for n, p in self.backbone.named_parameters():
#             if not p.requires_grad:
#                 continue
#             if n in delta:
#                 p.sub_(delta[n])

#         # Best-effort: reset head parameters for that task (TIL/DIL)
#         self._reset_head_if_possible(task_id)

#     def _reset_head_if_possible(self, task_id: int) -> None:
#         """Best-effort reset for task-specific head (TIL/DIL)."""
#         # Try heads.get_head(task_id)
#         try:
#             head = self.heads.get_head(task_id)
#             if hasattr(head, "reset_parameters"):
#                 head.reset_parameters()
#             return
#         except Exception:
#             pass

#         # Fallback: HeadsTIL-like dict storage: heads.heads[str(task_id)]
#         try:
#             if hasattr(self.heads, "heads"):
#                 key = f"{task_id}"
#                 if key in self.heads.heads:
#                     head = self.heads.heads[key]
#                     if hasattr(head, "reset_parameters"):
#                         head.reset_parameters()
#         except Exception:
#             pass

#     def aggregated_backbone_output(self, input: Tensor) -> Tensor:
#         r"""Aggregated backbone output for unlearning metrics (DD/AD).

#         Finetuning keeps a single backbone, so we use its feature directly.
#         """
#         feature, _ = self.backbone(input, stage="unlearning_test", task_id=self.task_id)
#         return feature
