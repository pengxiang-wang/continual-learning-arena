r"""
The submodule in `cl_algorithms` for [EWC (Elastic Weight Consolidation) algorithm](https://www.pnas.org/doi/10.1073/pnas.1611835114).
"""

__all__ = ["EWC", "UnlearnableEWC"]

import logging
from copy import deepcopy
from typing import Any

import torch
from torch import Tensor, nn

from clarena.backbones import CLBackbone
from clarena.cl_algorithms import Finetuning
from clarena.cl_algorithms.base import UnlearnableCLAlgorithm
from clarena.cl_algorithms.regularizers import ParameterChangeReg
from clarena.heads import HeadDIL, HeadsCIL, HeadsTIL

# always get logger for built-in logging in each module
pylogger = logging.getLogger(__name__)


class EWC(Finetuning):
    r"""[EWC (Elastic Weight Consolidation)](https://www.pnas.org/doi/10.1073/pnas.1611835114) algorithm.

    A regularization-based approach that calculates the fisher information as parameter importance for the previous tasks and penalizes the current task loss with the importance of the parameters.

    We implement EWC as a subclass of Finetuning algorithm, as EWC has the same `forward()`, `validation_step()` and `test_step()` method as `Finetuning` class.
    """

    def __init__(
        self,
        backbone: CLBackbone,
        heads: HeadsTIL | HeadsCIL | HeadDIL,
        parameter_change_reg_factor: float,
        when_calculate_fisher_information: str,
        non_algorithmic_hparams: dict[str, Any] = {},
        **kwargs,
    ) -> None:
        r"""Initialize the HAT algorithm with the network.

        **Args:**
        - **backbone** (`CLBackbone`): backbone network.
        - **heads** (`HeadsTIL` | `HeadsCIL` | `HeadDIL`): output heads.
        - **parameter_change_reg_factor** (`float`): the parameter change regularization factor. It controls the strength of preventing forgetting.
        - **when_calculate_fisher_information** (`str`): when to calculate the fisher information. It should be one of the following:
            1. 'train_end': calculate the fisher information at the end of training of the task.
            2. 'train': accumulate the fisher information in the training step of the task.
        - **non_algorithmic_hparams** (`dict[str, Any]`): non-algorithmic hyperparameters that are not related to the algorithm itself are passed to this `LightningModule` object from the config, such as optimizer and learning rate scheduler configurations. They are saved for Lightning APIs from `save_hyperparameters()` method. This is useful for the experiment configuration and reproducibility.
        - **kwargs**: Reserved for multiple inheritance.

        """
        super().__init__(
            backbone=backbone,
            heads=heads,
            non_algorithmic_hparams=non_algorithmic_hparams,
            **kwargs,
        )

        # save additional algorithmic hyperparameters
        self.save_hyperparameters(
            "parameter_change_reg_factor",
            "when_calculate_fisher_information",
        )

        # NOTE: task_id is int everywhere in the codebase; keep keys consistent
        self.parameter_importance: dict[int, dict[str, Tensor]] = {}
        r"""The parameter importance of each previous task. Keys are task IDs and values are the corresponding importance. Each importance entity is a dict where keys are parameter names (named by `named_parameters()` of the `nn.Module`) and values are the importance tensor for the layer. It has the same shape as the parameters of the layer.
        """

        # NOTE: keys are task IDs (int), not str
        self.previous_task_backbones: dict[int, nn.Module] = {}
        r"""The backbone models of the previous tasks. Keys are task IDs and values are the corresponding models. Each model is a `nn.Module` backbone after the corresponding previous task was trained.

        Some would argue that since we could store the model of the previous tasks, why don't we test the task directly with the stored model, instead of doing the less easier EWC thing? The thing is, EWC only uses the model of the previous tasks to train current and future tasks, which aggregate them into a single model. Once the training of the task is done, the storage for those parameters can be released. However, this make the future tasks not able to use EWC anymore, which is a disadvantage for EWC.
        """

        self.parameter_change_reg_factor = parameter_change_reg_factor
        r"""The parameter change regularization factor."""
        self.parameter_change_reg = ParameterChangeReg(
            factor=parameter_change_reg_factor,
        )
        r"""Initialize and store the parameter change regularizer."""

        self.when_calculate_fisher_information: str = when_calculate_fisher_information
        r"""When to calculate the fisher information."""
        self.num_data: int
        r"""The number of data used to calculate the fisher information. It is used to average the fisher information over the data."""

        # set manual optimization because we need to access gradients to calculate the fisher information in the training step
        self.automatic_optimization = False

        EWC.sanity_check(self)

    def sanity_check(self) -> None:
        r"""Sanity check."""
        if self.parameter_change_reg_factor <= 0:
            raise ValueError(
                f"The parameter change regularization factor should be positive, but got {self.parameter_change_reg_factor}."
            )

    def on_train_start(self) -> None:
        r"""Initialize the parameter importance and num of data counter."""
        self.parameter_importance[self.task_id] = {}
        for param_name, param in self.backbone.named_parameters():
            self.parameter_importance[self.task_id][param_name] = 0 * param.data
        self.num_data = 0

    def training_step(self, batch: Any) -> dict[str, Tensor]:
        r"""Training step for current task `self.task_id`."""
        x, y = batch

        opt = self.optimizers()
        opt.zero_grad()

        logits, activations = self.forward(x, stage="train", task_id=self.task_id)
        loss_cls = self.criterion(logits, y)

        batch_size = len(y)
        self.num_data += batch_size

        if self.when_calculate_fisher_information == "train":
            loss_cls.backward(retain_graph=True)
            for param_name, param in self.backbone.named_parameters():
                self.parameter_importance[self.task_id][param_name] += (
                    batch_size * param.grad**2
                )

        loss_reg = 0.0
        for previous_task_id in range(1, self.task_id):
            loss_reg += 0.5 * self.parameter_change_reg(
                target_model=self.backbone,
                ref_model=self.previous_task_backbones[previous_task_id],
                weights=self.parameter_importance[previous_task_id],
            )

        loss = loss_cls + loss_reg

        self.manual_backward(loss)
        opt.step()

        acc = (logits.argmax(dim=1) == y).float().mean()

        return {
            "loss": loss,
            "loss_cls": loss_cls,
            "loss_reg": loss_reg,
            "acc": acc,
            "activations": activations,
        }

    def on_train_end(self) -> None:
        r"""Calculate the fisher information as parameter importance and store the backbone model after the training of a task."""
        if self.when_calculate_fisher_information == "train_end":
            self.parameter_importance[self.task_id] = (
                self.accumulate_fisher_information_on_train_end()
            )

        for param_name, param in self.backbone.named_parameters():
            self.parameter_importance[self.task_id][param_name] /= self.num_data

        previous_backbone = deepcopy(self.backbone)
        previous_backbone.eval()
        self.previous_task_backbones[self.task_id] = previous_backbone

    # FIX: return type was incorrect (this function returns a dict)
    def accumulate_fisher_information_on_train_end(self) -> dict[str, Tensor]:
        r"""Accumulate the fisher information as the parameter importance for the learned task `self.task_id` at the end of its training."""
        fisher_information_t = {}

        self.eval()
        last_task_train_dataloaders = self.trainer.datamodule.train_dataloader()

        for param_name, param in self.backbone.named_parameters():
            fisher_information_t[param_name] = torch.zeros_like(param)

        for x, y in last_task_train_dataloaders:
            x = x.to(self.device)
            y = y.to(self.device)
            batch_size = len(y)

            self.backbone.zero_grad()
            logits, _ = self.forward(x, stage="train", task_id=self.task_id)
            loss_cls = self.criterion(logits, y)
            loss_cls.backward()

            for param_name, param in self.backbone.named_parameters():
                fisher_information_t[param_name] += batch_size * param.grad**2

        return fisher_information_t


# class UnlearnableEWC(UnlearnableCLAlgorithm, EWC):
#     r"""Unlearnable EWC algorithm.

#     Variant of EWC that supports unlearning requests and permanent tasks.
#     """

#     def __init__(
#         self,
#         backbone: CLBackbone,
#         heads: HeadsTIL | HeadsCIL | HeadDIL,
#         parameter_change_reg_factor: float,
#         when_calculate_fisher_information: str,
#         non_algorithmic_hparams: dict[str, Any] = {},
#         disable_unlearning: bool = False,
#         **kwargs,
#     ) -> None:
#         super().__init__(
#             backbone=backbone,
#             heads=heads,
#             parameter_change_reg_factor=parameter_change_reg_factor,
#             when_calculate_fisher_information=when_calculate_fisher_information,
#             non_algorithmic_hparams=non_algorithmic_hparams,
#             disable_unlearning=disable_unlearning,
#             **kwargs,
#         )

#         self.valid_task_ids: set[int] = set()
#         r"""Task IDs whose Fisher/importances & ref backbones are kept for regularization."""

#     # --- minimal fix: make sure EWC hooks run (avoid MRO skipping EWC.on_train_start/end) ---
#     def on_train_start(self) -> None:
#         r"""Make sure EWC initializes fisher buffers for the current task."""
#         EWC.on_train_start(self)

#     def on_train_end(self) -> None:
#         r"""Make sure EWC stores fisher and previous backbone snapshot, then mark task as valid."""
#         EWC.on_train_end(self)
#         self.valid_task_ids.add(self.task_id)

#     def training_step(self, batch: Any) -> dict[str, Tensor]:
#         r"""Same as EWC.training_step but regularization sums over valid previous tasks only."""
#         x, y = batch

#         opt = self.optimizers()
#         opt.zero_grad()

#         logits, activations = self.forward(x, stage="train", task_id=self.task_id)
#         loss_cls = self.criterion(logits, y)

#         batch_size = len(y)
#         self.num_data += batch_size

#         if self.when_calculate_fisher_information == "train":
#             loss_cls.backward(retain_graph=True)
#             for param_name, param in self.backbone.named_parameters():
#                 self.parameter_importance[self.task_id][param_name] += (
#                     batch_size * param.grad**2
#                 )

#         # FIX: only regularize towards valid (non-unlearned) previous tasks
#         loss_reg = 0.0
#         for previous_task_id in sorted(self.valid_task_ids):
#             if previous_task_id >= self.task_id:
#                 continue
#             if previous_task_id not in self.previous_task_backbones:
#                 continue
#             if previous_task_id not in self.parameter_importance:
#                 continue

#             loss_reg += 0.5 * self.parameter_change_reg(
#                 target_model=self.backbone,
#                 ref_model=self.previous_task_backbones[previous_task_id],
#                 weights=self.parameter_importance[previous_task_id],
#             )

#         loss = loss_cls + loss_reg

#         self.manual_backward(loss)
#         opt.step()

#         acc = (logits.argmax(dim=1) == y).float().mean()

#         return {
#             "loss": loss,
#             "loss_cls": loss_cls,
#             "loss_reg": loss_reg,
#             "acc": acc,
#             "activations": activations,
#         }

#     def aggregated_backbone_output(self, input: Tensor) -> Tensor:
#         r"""Aggregated backbone output for unlearning metrics.

#         EWC keeps a single backbone, so we use its feature directly.
#         """
#         feature, _ = self.backbone(input, stage="unlearning_test", task_id=self.task_id)
#         return feature


# # class UnlearnableEWC(UnlearnableCLAlgorithm, EWC):
# #     r"""Unlearnable EWC algorithm.

# #     Variant of EWC that supports unlearning requests and permanent tasks.
# #     """

# #     def __init__(
# #         self,
# #         backbone: CLBackbone,
# #         heads: HeadsTIL | HeadsCIL | HeadDIL,
# #         parameter_change_reg_factor: float,
# #         when_calculate_fisher_information: str,
# #         non_algorithmic_hparams: dict[str, Any] = {},
# #         disable_unlearning: bool = False,
# #         **kwargs,
# #     ) -> None:
# #         super().__init__(
# #             backbone=backbone,
# #             heads=heads,
# #             parameter_change_reg_factor=parameter_change_reg_factor,
# #             when_calculate_fisher_information=when_calculate_fisher_information,
# #             non_algorithmic_hparams=non_algorithmic_hparams,
# #             disable_unlearning=disable_unlearning,
# #             **kwargs,
# #         )

# #         self.valid_task_ids: set[int] = set()
# #         r"""Task IDs whose Fisher/importances & ref backbones are kept for regularization."""

# #     def on_train_end(self) -> None:
# #         super().on_train_end()
# #         self.valid_task_ids.add(self.task_id)

# #     def training_step(self, batch: Any) -> dict[str, Tensor]:
# #         r"""Same as EWC.training_step but regularization sums over valid previous tasks only."""
# #         x, y = batch

# #         opt = self.optimizers()
# #         opt.zero_grad()

# #         logits, activations = self.forward(x, stage="train", task_id=self.task_id)
# #         loss_cls = self.criterion(logits, y)

# #         batch_size = len(y)
# #         self.num_data += batch_size

# #         if self.when_calculate_fisher_information == "train":
# #             loss_cls.backward(retain_graph=True)
# #             for param_name, param in self.backbone.named_parameters():
# #                 self.parameter_importance[self.task_id][param_name] += (
# #                     batch_size * param.grad**2
# #                 )

# #         # FIX: only regularize towards valid (non-unlearned) previous tasks
# #         loss_reg = 0.0
# #         for previous_task_id in sorted(self.valid_task_ids):
# #             if previous_task_id >= self.task_id:
# #                 continue
# #             if previous_task_id not in self.previous_task_backbones:
# #                 continue
# #             if previous_task_id not in self.parameter_importance:
# #                 continue

# #             loss_reg += 0.5 * self.parameter_change_reg(
# #                 target_model=self.backbone,
# #                 ref_model=self.previous_task_backbones[previous_task_id],
# #                 weights=self.parameter_importance[previous_task_id],
# #             )

# #         loss = loss_cls + loss_reg

# #         self.manual_backward(loss)
# #         opt.step()

# #         acc = (logits.argmax(dim=1) == y).float().mean()

# #         return {
# #             "loss": loss,
# #             "loss_cls": loss_cls,
# #             "loss_reg": loss_reg,
# #             "acc": acc,
# #             "activations": activations,
# #         }

# #     def aggregated_backbone_output(self, input: Tensor) -> Tensor:
# #         r"""Aggregated backbone output for unlearning metrics.

# #         EWC keeps a single backbone, so we use its feature directly.
# #         """
# #         feature, _ = self.backbone(input, stage="unlearning_test", task_id=self.task_id)
# #         return feature

class UnlearnableEWC(UnlearnableCLAlgorithm, EWC):
    r"""Unlearnable EWC algorithm.

    Variant of EWC that supports unlearning requests and permanent tasks.

    Requirement (Amnesiac Hat style):
    - For each task t, record parameter delta: Δ_t = θ_post - θ_pre
    - When forgetting task k, rollback: θ ← θ - Δ_k
    """

    def __init__(
        self,
        backbone: CLBackbone,
        heads: HeadsTIL | HeadsCIL | HeadDIL,
        parameter_change_reg_factor: float,
        when_calculate_fisher_information: str,
        non_algorithmic_hparams: dict[str, Any] = {},
        disable_unlearning: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(
            backbone=backbone,
            heads=heads,
            parameter_change_reg_factor=parameter_change_reg_factor,
            when_calculate_fisher_information=when_calculate_fisher_information,
            non_algorithmic_hparams=non_algorithmic_hparams,
            disable_unlearning=disable_unlearning,
            **kwargs,
        )

        self.valid_task_ids: set[int] = set()
        r"""Task IDs whose Fisher/importances & ref backbones are kept for regularization."""

        # --- Amnesiac Hat buffers (minimal additions) ---
        self._pre_task_params: dict[str, Tensor] | None = None
        r"""Backbone parameter snapshot before training current task (θ_pre)."""

        self.task_deltas: dict[int, dict[str, Tensor]] = {}
        r"""Per-task parameter delta Δ_t = θ_post - θ_pre, keyed by task_id."""

    # --- minimal fix: make sure EWC hooks run (avoid MRO skipping EWC.on_train_start/end) ---
    def on_train_start(self) -> None:
        r"""Make sure EWC initializes fisher buffers and record θ_pre for current task."""
        EWC.on_train_start(self)

        # Record θ_pre (only trainable params)
        self._pre_task_params = {
            n: p.detach().clone()
            for n, p in self.backbone.named_parameters()
            if p.requires_grad
        }

    def on_train_end(self) -> None:
        r"""Make sure EWC stores fisher and previous backbone snapshot; record Δ_t and mark task as valid."""
        EWC.on_train_end(self)

        # Record Δ_t = θ_post - θ_pre
        if self._pre_task_params is not None:
            delta_t: dict[str, Tensor] = {}
            for n, p in self.backbone.named_parameters():
                if not p.requires_grad:
                    continue
                delta_t[n] = (p.detach() - self._pre_task_params[n]).clone()
            self.task_deltas[self.task_id] = delta_t
            self._pre_task_params = None

        self.valid_task_ids.add(self.task_id)

    def training_step(self, batch: Any) -> dict[str, Tensor]:
        r"""Same as EWC.training_step but regularization sums over valid previous tasks only."""
        x, y = batch

        opt = self.optimizers()
        opt.zero_grad()

        logits, activations = self.forward(x, stage="train", task_id=self.task_id)
        loss_cls = self.criterion(logits, y)

        batch_size = len(y)
        self.num_data += batch_size

        if self.when_calculate_fisher_information == "train":
            loss_cls.backward(retain_graph=True)
            for param_name, param in self.backbone.named_parameters():
                self.parameter_importance[self.task_id][param_name] += (
                    batch_size * param.grad**2
                )

        # only regularize towards valid (non-unlearned) previous tasks
        loss_reg = 0.0
        for previous_task_id in sorted(self.valid_task_ids):
            if previous_task_id >= self.task_id:
                continue
            if previous_task_id not in self.previous_task_backbones:
                continue
            if previous_task_id not in self.parameter_importance:
                continue

            loss_reg += 0.5 * self.parameter_change_reg(
                target_model=self.backbone,
                ref_model=self.previous_task_backbones[previous_task_id],
                weights=self.parameter_importance[previous_task_id],
            )

        loss = loss_cls + loss_reg

        self.manual_backward(loss)
        opt.step()

        acc = (logits.argmax(dim=1) == y).float().mean()

        return {
            "loss": loss,
            "loss_cls": loss_cls,
            "loss_reg": loss_reg,
            "acc": acc,
            "activations": activations,
        }

    @torch.no_grad()
    def unlearn_task(self, task_id: int) -> None:
        r"""Forget task `task_id` by Amnesiac Hat rollback: θ ← θ - Δ_task_id."""
        if self.disable_unlearning:
            return

        if task_id not in self.task_deltas:
            raise ValueError(f"No stored delta for task {task_id}.")

        delta = self.task_deltas[task_id]
        for n, p in self.backbone.named_parameters():
            if not p.requires_grad:
                continue
            if n in delta:
                p.sub_(delta[n])

        # Mark as no longer valid for future EWC regularization
        if task_id in self.valid_task_ids:
            self.valid_task_ids.remove(task_id)

    def aggregated_backbone_output(self, input: Tensor) -> Tensor:
        r"""Aggregated backbone output for unlearning metrics.

        EWC keeps a single backbone, so we use its feature directly.
        """
        feature, _ = self.backbone(input, stage="unlearning_test", task_id=self.task_id)
        return feature
