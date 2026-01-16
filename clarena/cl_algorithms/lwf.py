r"""
The submodule in `cl_algorithms` for [LwF (Learning without Forgetting) algorithm](https://ieeexplore.ieee.org/abstract/document/8107520).
"""

__all__ = ["LwF", "AmnesiacLwF"]

import logging
from copy import deepcopy
from typing import Any

import torch
from torch import Tensor

from clarena.backbones import CLBackbone
from clarena.cl_algorithms import Finetuning
from clarena.cl_algorithms.base import AmnesiacCLAlgorithm
from clarena.cl_algorithms.regularizers import DistillationReg
from clarena.heads import HeadDIL, HeadsCIL, HeadsTIL

# always get logger for built-in logging in each module
pylogger = logging.getLogger(__name__)


class LwF(Finetuning):
    r"""[LwF (Learning without Forgetting)](https://ieeexplore.ieee.org/abstract/document/8107520) algorithm.

    A regularization-based continual learning approach that constrains the feature output of the model to be similar to that of the previous tasks. From the perspective of knowledge distillation, it distills previous tasks models into the training process for new task in the regularization term. It is a simple yet effective method for continual learning.

    We implement LwF as a subclass of Finetuning algorithm, as LwF has the same `forward()`, `validation_step()` and `test_step()` method as `Finetuning` class.
    """

    def __init__(
        self,
        backbone: CLBackbone,
        heads: HeadsTIL | HeadsCIL | HeadDIL,
        distillation_reg_factor: float,
        distillation_reg_temperature: float,
        non_algorithmic_hparams: dict[str, Any] = {},
        **kwargs,
    ) -> None:
        r"""Initialize the LwF algorithm with the network.

        **Args:**
        - **backbone** (`CLBackbone`): backbone network.
        - **heads** (`HeadsTIL` | `HeadsCIL` | `HeadDIL`): output heads.
        - **distillation_reg_factor** (`float`): hyperparameter, the distillation regularization factor. It controls the strength of preventing forgetting.
        - **distillation_reg_temperature** (`float`): hyperparameter, the temperature in the distillation regularization. It controls the softness of the labels that the student model (here is the current model) learns from the teacher models (here are the previous models), thereby controlling the strength of the distillation. It controls the strength of preventing forgetting.
        - **non_algorithmic_hparams** (`dict[str, Any]`): non-algorithmic hyperparameters that are not related to the algorithm itself are passed to this `LightningModule` object from the config, such as optimizer and learning rate scheduler configurations. They are saved for Lightning APIs from `save_hyperparameters()` method. This is useful for the experiment configuration and reproducibility.
        - **kwargs**: Reserved for multiple inheritance.

        """
        super().__init__(
            backbone=backbone,
            heads=heads,
            non_algorithmic_hparams=non_algorithmic_hparams,
            **kwargs,
        )

        self.previous_task_backbones: dict[int, CLBackbone] = {}
        r"""The backbone models of the previous tasks. Keys are task IDs (int) and values are the corresponding models. Each model is a `CLBackbone` after the corresponding previous task was trained.
        
        Some would argue that since we could store the model of the previous tasks, why don't we test the task directly with the stored model, instead of doing the less easier LwF thing? The thing is, LwF only uses the model of the previous tasks to train current and future tasks, which aggregate them into a single model. Once the training of the task is done, the storage for those parameters can be released. However, this make the future tasks not able to use LwF anymore, which is a disadvantage for LwF.
        """

        self.distillation_reg_factor: float = distillation_reg_factor
        r"""The distillation regularization factor."""
        self.distillation_reg_temperature: float = distillation_reg_temperature
        r"""The distillation regularization temperature."""
        self.distillation_reg = DistillationReg(
            factor=distillation_reg_factor,
            temperature=distillation_reg_temperature,
            distance="cross_entropy",
        )
        r"""Initialize and store the distillation regularizer."""

        # save additional algorithmic hyperparameters
        self.save_hyperparameters(
            "distillation_reg_factor",
            "distillation_reg_temperature",
        )

        LwF.sanity_check(self)

    def sanity_check(self) -> None:
        r"""Sanity check."""

        if self.distillation_reg_factor <= 0:
            raise ValueError(
                f"The distillation regularization factor should be positive, but got {self.distillation_reg_factor}."
            )

        if self.distillation_reg_temperature <= 0:
            raise ValueError(
                f"The distillation regularization temperature should be positive, but got {self.distillation_reg_temperature}."
            )

    def training_step(self, batch: Any) -> dict[str, Tensor]:
        r"""Training step for current task `self.task_id`.

        **Args:**
        - **batch** (`Any`): a batch of training data.

        **Returns:**
        - **outputs** (`dict[str, Tensor]`): a dictionary contains loss and other metrics from this training step. Keys (`str`) are the metrics names, and values (`Tensor`) are the metrics. Must include the key 'loss' which is total loss in the case of automatic optimization, according to PyTorch Lightning docs.
        """
        x, y = batch

        # classification loss. See equation (1) in the [LwF paper](https://ieeexplore.ieee.org/abstract/document/8107520).
        logits, activations = self.forward(x, stage="train", task_id=self.task_id)
        loss_cls = self.criterion(logits, y)

        # regularization loss. See equation (2) (3) in the [LwF paper](https://ieeexplore.ieee.org/abstract/document/8107520).
        loss_reg = 0.0
        for previous_task_id in self.processed_task_ids:
            if previous_task_id == self.task_id:
                continue
            # sum over all previous models, because [LwF paper](https://ieeexplore.ieee.org/abstract/document/8107520) says: "If there are multiple old tasks, or if an old task is multi-label classification, we take the sum of the loss for each old task and label."

            # get the student logits for this batch using the current model (to previous output head)
            student_feature, _ = self.backbone(
                x, stage="train", task_id=previous_task_id
            )
            with torch.no_grad():  # stop updating the previous heads
                student_logits = self.heads(student_feature, task_id=previous_task_id)

            # get the teacher logits for this batch, which is from the previous model
            previous_backbone = self.previous_task_backbones[previous_task_id]
            with torch.no_grad():  # stop updating the previous backbones and heads
                teacher_feature, _ = previous_backbone(
                    x, stage="test", task_id=previous_task_id
                )
                teacher_logits = self.heads(teacher_feature, task_id=previous_task_id)

            loss_reg += self.distillation_reg(
                student_logits=student_logits,
                teacher_logits=teacher_logits,
            )

        # do not average over tasks to avoid linear increase of the regularization loss. LwF paper doesn't mention this!

        # total loss
        loss = loss_cls + loss_reg

        # predicted labels
        preds = logits.argmax(dim=1)

        # accuracy of the batch
        acc = (preds == y).float().mean()

        return {
            "preds": preds,
            "loss": loss,  # return loss is essential for training step, or backpropagation will fail
            "loss_cls": loss_cls,
            "loss_reg": loss_reg,
            "acc": acc,
            "activations": activations,
        }

    def on_train_end(self) -> None:
        r"""The backbone model after the training of a task.

        The model is stored in `self.previous_task_backbones` for constructing the regularization loss in the future tasks.
        """
        previous_backbone = deepcopy(self.backbone)
        previous_backbone.eval()  # set the stored model to evaluation mode to prevent updating
        head = (
            self.heads.heads[f"{self.task_id}"]
            if hasattr(self.heads, "heads")
            else self.heads.head
        )
        head.eval()  # set the head to evaluation mode
        self.previous_task_backbones[self.task_id] = previous_backbone


class AmnesiacLwF(AmnesiacCLAlgorithm, LwF):
    r"""Amnesiac LwF algorithm."""

    def __init__(
        self,
        backbone: CLBackbone,
        heads: HeadsTIL | HeadsCIL | HeadDIL,
        non_algorithmic_hparams: dict[str, Any] = {},
        disable_unlearning: bool = False,
        **kwargs,
    ) -> None:
        r"""Initialize the Amnesiac LwF algorithm with the network.


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
        LwF.on_train_start(self)
        AmnesiacCLAlgorithm.on_train_start(self)

    def on_train_end(self) -> None:
        """Record backbone parameters before training current task."""
        LwF.on_train_end(self)
        AmnesiacCLAlgorithm.on_train_end(self)


# class UnlearnableLwF(UnlearnableCLAlgorithm, LwF):
#     r"""Unlearnable LwF algorithm (Amnesiac-HAT / delta rollback style).

#     Key idea:
#     - For each task t, record parameter delta: Δ_t = θ_post - θ_pre
#     - When forgetting task k, rollback: θ ← θ - Δ_k
#     - Also remove task k from teacher pool so future distillation ignores it.

#     Notes:
#     - We do NOT modify the original `LwF.training_step` above (per your requirement).
#       Instead, we override `training_step` here to implement the "correct" distillation:
#         * student logits SHOULD have gradients
#         * teacher logits SHOULD be under torch.no_grad()
#     """

#     def __init__(
#         self,
#         backbone: CLBackbone,
#         heads: HeadsTIL | HeadsCIL | HeadDIL,
#         distillation_reg_factor: float,
#         distillation_reg_temperature: float,
#         non_algorithmic_hparams: dict[str, Any] = {},
#         disable_unlearning: bool = False,
#         **kwargs,
#     ) -> None:
#         super().__init__(
#             backbone=backbone,
#             heads=heads,
#             distillation_reg_factor=distillation_reg_factor,
#             distillation_reg_temperature=distillation_reg_temperature,
#             non_algorithmic_hparams=non_algorithmic_hparams,
#             disable_unlearning=disable_unlearning,
#             **kwargs,
#         )

#         # Teacher pool control: tasks in this set are eligible to be distilled from.
#         self.valid_task_ids: set[int] = set()
#         r"""Task IDs whose teacher snapshots are kept & used for distillation."""

#         # --- delta rollback buffers (minimal additions) ---
#         self._pre_task_params: dict[str, Tensor] | None = None
#         r"""Backbone parameter snapshot before training current task (θ_pre)."""

#         self.task_deltas: dict[int, dict[str, Tensor]] = {}
#         r"""Per-task parameter delta Δ_t = θ_post - θ_pre, keyed by task_id."""

#     def on_train_start(self) -> None:
#         """Record θ_pre before training current task (only trainable params)."""
#         # No special hook needed from base, but safe to keep Lightning hook chain.
#         super().on_train_start()

#         if self.disable_unlearning:
#             return

#         self._pre_task_params = {
#             n: p.detach().clone()
#             for n, p in self.backbone.named_parameters()
#             if p.requires_grad
#         }

#     def on_train_end(self) -> None:
#         """After training: store teacher snapshot, record Δ_t, and mark current task as valid."""
#         super().on_train_end()  # this calls LwF.on_train_end() which stores previous_task_backbones[task_id]

#         # Mark current task as eligible teacher
#         self.valid_task_ids.add(self.task_id)

#         if self.disable_unlearning:
#             return

#         # Record Δ_t = θ_post - θ_pre
#         if self._pre_task_params is not None:
#             delta_t: dict[str, Tensor] = {}
#             for n, p in self.backbone.named_parameters():
#                 if not p.requires_grad:
#                     continue
#                 delta_t[n] = (p.detach() - self._pre_task_params[n]).clone()
#             self.task_deltas[self.task_id] = delta_t
#             self._pre_task_params = None

#     def training_step(self, batch: Any) -> dict[str, Tensor]:
#         r"""Training step (correct distillation version for UnlearnableLwF).

#         Important fix (only here, not touching original LwF):
#         - student path MUST have gradients (no torch.no_grad on student logits)
#         - teacher path MUST be torch.no_grad

#         Distillation is computed only over `valid_task_ids` (i.e., not-unlearned teachers).
#         """
#         x, y = batch

#         # classification loss on current task head
#         logits, activations = self.forward(x, stage="train", task_id=self.task_id)
#         loss_cls = self.criterion(logits, y)

#         # distillation regularization to prevent forgetting (only from valid teachers)
#         loss_reg = 0.0
#         for previous_task_id in sorted(self.valid_task_ids):
#             if previous_task_id >= self.task_id:
#                 continue
#             if previous_task_id not in self.previous_task_backbones:
#                 continue

#             # --- student logits (WITH grad) ---
#             # Student uses current backbone, but routed to the old task head
#             student_feature, _ = self.backbone(x, stage="train", task_id=previous_task_id)
#             student_logits = self.heads(student_feature, task_id=previous_task_id)

#             # --- teacher logits (NO grad) ---
#             previous_backbone = self.previous_task_backbones[previous_task_id]
#             with torch.no_grad():
#                 teacher_feature, _ = previous_backbone(
#                     x, stage="test", task_id=previous_task_id
#                 )
#                 teacher_logits = self.heads(teacher_feature, task_id=previous_task_id)

#             loss_reg += self.distillation_reg(
#                 student_logits=student_logits,
#                 teacher_logits=teacher_logits,
#             )

#         loss = loss_cls + loss_reg

#         preds = logits.argmax(dim=1)
#         acc = (preds == y).float().mean()

#         return {
#             "preds": preds,
#             "loss": loss,
#             "loss_cls": loss_cls,
#             "loss_reg": loss_reg,
#             "acc": acc,
#             "activations": activations,
#         }

#     @torch.no_grad()
#     def unlearn_task(self, task_id: int) -> None:
#         r"""Forget task `task_id` by delta rollback: θ ← θ - Δ_task_id.

#         Also:
#         - remove from valid_task_ids so future distillation ignores it
#         - delete teacher snapshot for that task (previous_task_backbones[task_id])
#         - optional: reset task head (best-effort)
#         """
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

#         # Remove from teacher pool
#         self.valid_task_ids.discard(task_id)

#         # Delete teacher snapshot so it's not accidentally used
#         if task_id in self.previous_task_backbones:
#             del self.previous_task_backbones[task_id]

#         # Best-effort reset head (TIL/DIL)
#         self._reset_head_if_possible(task_id)

#     def _reset_head_if_possible(self, task_id: int) -> None:
#         """Best-effort reset for task-specific head (TIL/DIL)."""
#         try:
#             head = self.heads.get_head(task_id)
#             if hasattr(head, "reset_parameters"):
#                 head.reset_parameters()
#             return
#         except Exception:
#             pass

#         # fallback for HeadsTIL-like dict storage: heads.heads[str(task_id)]
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
#         r"""Aggregated backbone output for unlearning metrics.

#         LwF keeps a single backbone, so we use its feature directly.
#         """
#         feature, _ = self.backbone(input, stage="unlearning_test", task_id=self.task_id)
#         return feature


# class UnlearnableLwF(UnlearnableCLAlgorithm, LwF):
#     r"""Unlearnable LwF algorithm.

#     This is a variant of LwF that supports unlearning. It has the same functionality as LwF,
#     but it also supports unlearning requests and permanent tasks.

#     In unlearnable setting, unlearning may remove some previous tasks from being distilled.
#     Therefore, we maintain a set of valid task IDs for distillation regularization.
#     """

#     def __init__(
#         self,
#         backbone: CLBackbone,
#         heads: HeadsTIL | HeadsCIL | HeadDIL,
#         distillation_reg_factor: float,
#         distillation_reg_temperature: float,
#         non_algorithmic_hparams: dict[str, Any] = {},
#         disable_unlearning: bool = False,
#         **kwargs,
#     ) -> None:
#         r"""Initialize the Unlearnable LwF algorithm with the network.

#         **Args:**
#         - **backbone** (`CLBackbone`): backbone network.
#         - **heads** (`HeadsTIL` | `HeadsCIL` | `HeadDIL`): output heads.
#         - **distillation_reg_factor** (`float`): hyperparameter, the distillation regularization factor.
#         - **distillation_reg_temperature** (`float`): hyperparameter, the temperature in the distillation regularization.
#         - **non_algorithmic_hparams** (`dict[str, Any]`): non-algorithmic hyperparameters (optimizer, lr scheduler, etc.).
#         - **disable_unlearning** (`bool`): whether to disable unlearning. This is used in reference experiments following continual learning pipeline. Default is `False`.
#         - **kwargs**: Reserved for multiple inheritance.
#         """
#         super().__init__(
#             backbone=backbone,
#             heads=heads,
#             distillation_reg_factor=distillation_reg_factor,
#             distillation_reg_temperature=distillation_reg_temperature,
#             non_algorithmic_hparams=non_algorithmic_hparams,
#             disable_unlearning=disable_unlearning,
#             **kwargs,
#         )

#         self.valid_task_ids: set[int] = set()
#         r"""The list of task IDs that are valid for distillation regularization."""

#     def on_train_end(self) -> None:
#         r"""Store the previous backbone and mark current task as valid."""
#         super().on_train_end()
#         self.valid_task_ids.add(self.task_id)

#     def training_step(self, batch: Any) -> dict[str, Tensor]:
#         r"""Training step for current task `self.task_id`.

#         This is the same as `LwF.training_step()` except that the distillation regularization
#         only sums over valid previous tasks, because some tasks may have been unlearned.
#         """
#         x, y = batch

#         # classification loss. See equation (1) in the [LwF paper](https://ieeexplore.ieee.org/abstract/document/8107520).
#         logits, activations = self.forward(x, stage="train", task_id=self.task_id)
#         loss_cls = self.criterion(logits, y)

#         # regularization loss. See equation (2) (3) in the [LwF paper](https://ieeexplore.ieee.org/abstract/document/8107520).
#         loss_reg = 0.0

#         for previous_task_id in sorted(self.valid_task_ids):
#             if previous_task_id >= self.task_id:
#                 continue
#             # defensive: skip if teacher backbone is missing (may be removed by unlearning)
#             if previous_task_id not in self.previous_task_backbones:
#                 continue

#             # get the student logits for this batch using the current model (to previous output head)
#             student_feature, _ = self.backbone(
#                 x, stage="train", task_id=previous_task_id
#             )
#             with torch.no_grad():  # stop updating the previous heads
#                 student_logits = self.heads(student_feature, task_id=previous_task_id)

#             # get the teacher logits for this batch, which is from the previous model
#             previous_backbone = self.previous_task_backbones[previous_task_id]
#             with torch.no_grad():  # stop updating the previous backbones and heads
#                 teacher_feature, _ = previous_backbone(
#                     x, stage="test", task_id=previous_task_id
#                 )
#                 teacher_logits = self.heads(teacher_feature, task_id=previous_task_id)

#             loss_reg += self.distillation_reg(
#                 student_logits=student_logits,
#                 teacher_logits=teacher_logits,
#             )

#         # do not average over tasks to avoid linear increase of the regularization loss. LwF paper doesn't mention this!

#         # total loss
#         loss = loss_cls + loss_reg

#         # predicted labels
#         preds = logits.argmax(dim=1)

#         # accuracy of the batch
#         acc = (preds == y).float().mean()

#         return {
#             "preds": preds,
#             "loss": loss,  # return loss is essential for training step, or backpropagation will fail
#             "loss_cls": loss_cls,
#             "loss_reg": loss_reg,
#             "acc": acc,
#             "activations": activations,
#         }

#     def aggregated_backbone_output(self, input: Tensor) -> Tensor:
#         r"""Get the aggregated backbone output for the input data.

#         This output feature is used for measuring unlearning metrics, such as Distribution Distance (DD).
#         An aggregated output involving every part of the backbone is needed to ensure the fairness of the metric.

#         For LwF, we have a single backbone shared across tasks, so we directly use the current backbone feature.
#         """
#         feature, _ = self.backbone(input, stage="unlearning_test", task_id=self.task_id)
#         return feature
