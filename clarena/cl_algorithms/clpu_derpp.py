r"""
The submodule in `cl_algorithms` for [CLPU-DER++](https://arxiv.org/abs/2203.12817) algorithm.
"""

__all__ = ["CLPUDERpp"]

import logging
from copy import deepcopy
from typing import Any, Callable

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader
from torchvision.transforms import transforms

from clarena.backbones import CLBackbone
from clarena.cl_algorithms import DERpp, UnlearnableCLAlgorithm
from clarena.heads import HeadDIL, HeadsCIL, HeadsTIL

# always get logger for built-in logging in each module
pylogger = logging.getLogger(__name__)


class CLPUDERpp(DERpp, UnlearnableCLAlgorithm):
    r"""[CLPU-DER++](https://arxiv.org/abs/2203.12817) algorithm.

    [CLPU-DER++](https://arxiv.org/abs/2203.12817) is an unlearnable continual learning algorithm based on Independent and DER++.

    We implement AdaHAT as a subclass of DER++, as it shares the same memory buffer as the `DERpp` class.
    """

    def __init__(
        self,
        backbone: CLBackbone,
        heads: HeadsTIL | HeadsCIL | HeadDIL,
        buffer_size: int,
        distillation_reg_factor: float,
        replay_ce_factor: float,
        temporary_backbone_init_mode: str,
        merge_iters: int,
        merge_batch_size: int,
        augmentation_transforms: Callable | transforms.Compose | None = None,
        non_algorithmic_hparams: dict[str, Any] = {},
        disable_unlearning: bool = False,
        **kwargs,
    ) -> None:
        r"""Initialize the CLPU-DER++ algorithm with the network.

        **Args:**
        - **backbone** (`CLBackbone`): backbone network.
        - **heads** (`HeadsTIL` | `HeadsCIL` | `HeadDIL`): output heads.
        - **buffer_size** (`int`): the size of the memory buffer. For now we only support fixed size buffer.
        - **distillation_reg_factor** (`float`): hyperparameter, the distillation regularization factor ($\alpha$ in the [DER++ paper](https://arxiv.org/abs/2004.07211)). It controls the strength of preventing forgetting.
        - **replay_ce_factor** (`float`): hyperparameter, the classification loss factor for replayed samples, ($\beta$ in the [DER++ paper](https://arxiv.org/abs/2004.07211)). It also controls the strength of preventing forgetting.
        - **merge_iters** (`int`): number of optimization steps to merge a temporary task back into the main backbone.
        - **merge_batch_size** (`int`): batch size used for merge.
        - **augmentation_transforms** (`transform` or `transforms.Compose` or `None`): the transforms to apply for augmentation after replay sampling. Not to confuse with the data transforms applied to the input of training data. Can be a single transform, composed transforms, or no transform.
        - **temporary_backbone_init_mode** (`str`): method to initialize temporary task backbone networks, must be one of:
            1. 'from_main': initialize from the current main backbone weights.
            2. 'from_scratch': initialize from scratch.
        - **non_algorithmic_hparams** (`dict[str, Any]`): non-algorithmic hyperparameters that are not related to the algorithm itself are passed to this `LightningModule` object from the config, such as optimizer and learning rate scheduler configurations. They are saved for Lightning APIs from `save_hyperparameters()` method. This is useful for the experiment configuration and reproducibility.
        - **disable_unlearning** (`bool`): whether to disable unlearning. This is used in reference experiments following continual learning pipeline. Default is `False`.
        - **kwargs**: Reserved for multiple inheritance.
        """
        super().__init__(
            backbone=backbone,
            heads=heads,
            buffer_size=buffer_size,
            distillation_reg_factor=distillation_reg_factor,
            replay_ce_factor=replay_ce_factor,
            augmentation_transforms=augmentation_transforms,
            non_algorithmic_hparams=non_algorithmic_hparams,
            disable_unlearning=disable_unlearning,
            **kwargs,
        )

        self.temporary_backbone_init_mode: str = temporary_backbone_init_mode
        r"""Method to initialize temporary task backbone networks."""

        self.merge_iters: int = merge_iters
        r"""Number of optimization steps used to merge a temporary task."""

        self.merge_batch_size: int = merge_batch_size
        r"""Batch size used during merge optimization."""

        self.save_hyperparameters(
            "temporary_backbone_init_mode", "merge_iters", "merge_batch_size"
        )

        if not disable_unlearning:
            self.original_backbone_state_dict: dict = deepcopy(backbone.state_dict())
            r"""The original backbone state dict before training on any task. Used to initialize new independent temporary backbones for new tasks."""

            self.temporary_backbones: nn.ModuleDict = nn.ModuleDict()
            r"""Independent temporary backbones for temporary tasks. Keys are task IDs  (in string format coz they have to) and values are the corresponding temporary backbones. Only temporary task corresponds
            to a temporary backbone.
            """

    def setup_task_id(
        self,
        task_id: int,
        num_classes: int,
        optimizer: torch.optim.Optimizer,
        lr_scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    ) -> None:
        """Set up task ID and create temporary backbones if the task is temporary."""
        super().setup_task_id(
            task_id=task_id,
            num_classes=num_classes,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
        )
        print("no_longer", self.task_ids_just_no_longer_unlearnable)

        if not self.disable_unlearning:
            # initialise a temporary backbone for the current task (if it is temporary). This has to be done before configure_optimizers(), because optimizer needs to include parameters from temporary backbone if any.
            if self.task_id not in self.task_ids_just_no_longer_unlearnable:
                # if the current task (`self.task_id`) is permanent, it just turns no longer unlearnable
                self.temporary_backbones[f"{self.task_id}"] = deepcopy(self.backbone)
                if self.temporary_backbone_init_mode == "from_scratch":
                    self.temporary_backbones[f"{self.task_id}"].load_state_dict(
                        self.original_backbone_state_dict
                    )
                pylogger.info(
                    "Temporary backbone created for temporary task %d.", self.task_id
                )

    def on_train_start(self) -> None:
        r"""Merge and delete temporary backbones that just become no longer unlearnable."""

        if not self.disable_unlearning:
            # merge and delete temporary backbones that just becomes no longer unlearnable. It corresponds to Case III in the algorithm description of the [CLPU paper](https://arxiv.org/abs/2203.12817).
            for tid in self.task_ids_just_no_longer_unlearnable:
                if tid not in self.unlearned_task_ids and tid != self.task_id:
                    pylogger.info(
                        "Merging temporary backbone for task %d into main backbone...",
                        tid,
                    )
                    self.merge_temporary_backbone_to_main(tid)

                    pylogger.info(
                        "Merging completed! Deleting temporary backbone for task %d.",
                        tid,
                    )
                    del self.temporary_backbones[f"{tid}"]

    def merge_temporary_backbone_to_main(self, task_id: int) -> None:
        r"""Merge the temporary backbone for `task_id` into the main backbone using replay."""
        if str(task_id) not in self.temporary_backbones:
            pylogger.warning("No temporary backbone found for task %d.", task_id)
            return

        for param in self.backbone.parameters():
            param.requires_grad = True
        for param in self.temporary_backbones[str(task_id)].parameters():
            param.requires_grad = False

        optimizer = self.optimizer_t(
            params=[p for p in self.parameters() if p.requires_grad]
        )

        temporary_task_ids = {int(tid) for tid in self.temporary_backbones.keys()}
        unlearned_task_ids = getattr(self, "unlearned_task_ids", set())

        replay_task_ids: list[int] = []
        if self.memory_buffer.task_labels.numel() > 0:
            buffer_task_ids = torch.unique(self.memory_buffer.task_labels).tolist()
            replay_task_ids = [
                int(tid)
                for tid in buffer_task_ids
                if int(tid) != task_id
                and int(tid) not in temporary_task_ids
                and int(tid) not in unlearned_task_ids
            ]

        self.train()
        for s in range(self.merge_iters):
            loss = 0.0

            if replay_task_ids:
                x_replay, labels_replay, logits_replay, task_labels_replay = (
                    self.memory_buffer.get_data(
                        size=self.merge_batch_size,
                        included_tasks=replay_task_ids,
                    )
                )
                if x_replay.numel() > 0:
                    if self.augmentation_transforms:
                        x_replay = self.augmentation_transforms(x_replay)

                    student_feature_replay, _ = self.backbone(x_replay, stage="test")
                    student_logits_replay = torch.cat(
                        [
                            self.heads(
                                student_feature_replay[i].unsqueeze(0),
                                task_id=int(task_labels_replay[i].item()),
                            )
                            for i in range(task_labels_replay.numel())
                        ]
                    )

                    loss += self.distillation_reg(
                        student_logits=student_logits_replay,
                        teacher_logits=logits_replay,
                    )
                    loss += self.replay_ce_factor * self.criterion(
                        student_logits_replay, labels_replay.long()
                    )

            x_t, y_t, logits_t, _ = self.memory_buffer.get_data(
                size=self.merge_batch_size, included_tasks=[task_id]
            )
            if x_t.numel() == 0:
                continue

            if self.augmentation_transforms:
                x_t = self.augmentation_transforms(x_t)

            student_feature_t, _ = self.backbone(x_t, stage="test", task_id=task_id)
            student_logits_t = self.heads(student_feature_t, task_id)
            loss += self.replay_ce_factor * self.criterion(student_logits_t, y_t.long())
            loss += self.distillation_reg(
                student_logits=student_logits_t, teacher_logits=logits_t
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    def forward(self, input: Tensor, stage: str, task_id: int | None = None) -> Tensor:
        r"""Forward pass that routes temporary tasks to their independent backbones.

        The forward pass for data from task `task_id`. Note that it is nothing to do with `forward()` method in `nn.Module`. This definition provides a template that many CL algorithm including the vanilla Finetuning algorithm use. It works both for TIL and CIL.

        **Args:**
        - **input** (`Tensor`): the input tensor from data.
        - **stage** (`str`): the stage of the forward pass; one of:
            1. 'train': training stage.
            2. 'validation': validation stage.
            3. 'test': testing stage.
        - **task_id** (`int`): the task ID where the data are from. If stage is 'train' or `validation`, it is usually from the current task `self.task_id`. If stage is 'test', it could be from any seen task. In TIL, the task IDs of test data are provided thus this argument can be used. In CIL, they are not provided, so it is just a placeholder for API consistence but never used, and best practices are not to provide this argument and leave it as the default value.

        **Returns:**
        - **logits** (`Tensor`): the output logits tensor.
        - **activations** (`dict[str, Tensor]`): the hidden features (after activation) in each weighted layer. Key (`str`) is the weighted layer name, value (`Tensor`) is the hidden feature tensor. This is used for the continual learning algorithms that need to use the hidden features for various purposes.
        """
        if str(task_id) in self.temporary_backbones.keys():
            # It corresponds to Case II in the algorithm description of the [CLPU paper](https://arxiv.org/abs/2203.12817).
            routed_backbone = self.temporary_backbones[str(task_id)]
        else:
            # It corresponds to Case I in the algorithm description of the [CLPU paper](https://arxiv.org/abs/2203.12817).
            routed_backbone = self.backbone

        feature, activations = routed_backbone(input, stage=stage, task_id=task_id)
        logits = self.heads(feature, task_id)
        return (
            logits if self.if_forward_func_return_logits_only else (logits, activations)
        )

    def training_step(self, batch: Any) -> dict[str, Tensor]:
        """Training step for current task `self.task_id`.

        **Args:**
        - **batch** (`Any`): a batch of training data.

        **Returns:**
        - **outputs** (`dict[str, Tensor]`): a dictionary contains loss and other metrics from this training step. Keys (`str`) are the metrics names, and values (`Tensor`) are the metrics. Must include the key 'loss' which is total loss in the case of automatic optimization, according to PyTorch Lightning docs.
        """
        x, y = batch
        batch_size = len(y)

        # apply augmentation transforms if any
        if self.augmentation_transforms:
            x = self.augmentation_transforms(x)

        # classification loss
        logits, activations = self.forward(x, stage="train", task_id=self.task_id)
        loss_cls = self.criterion(logits, y)

        # regularization loss. DER++ adds replay classification loss on top of DER's logit distillation
        derpp_reg = 0.0

        if str(self.task_id) in self.temporary_backbones.keys():
            # It corresponds to Case II in the algorithm description of the [CLPU paper](https://arxiv.org/abs/2203.12817).
            routed_backbone = self.temporary_backbones[str(self.task_id)]
        else:
            # It corresponds to Case I in the algorithm description of the [CLPU paper](https://arxiv.org/abs/2203.12817).
            routed_backbone = self.backbone

        derpp_reg = self.compute_distillation_and_replay_ce_reg(
            backbone=routed_backbone, batch_size=batch_size
        )

        # total loss
        loss = loss_cls + derpp_reg

        # predicted labels
        preds = logits.argmax(dim=1)

        # accuracy of the batch
        acc = (preds == y).float().mean()

        # add current batch to memory buffer
        self.memory_buffer.add_data(
            x, y, logits.detach(), torch.full_like(y, self.task_id)
        )

        return {
            "preds": preds,
            "loss": loss,  # return loss is essential for training step, or backpropagation will fail
            "loss_cls": loss_cls,
            "derpp_reg": derpp_reg,
            "acc": acc,  # Return other metrics for lightning loggers callback to handle at `on_train_batch_end()`
            "activations": activations,
        }

    def test_step(
        self,
        batch: DataLoader,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> dict[str, Tensor]:
        r"""Test step for CLPU-DER++.

        Tests all seen tasks indexed by `dataloader_idx`.

        **Args:**
        - **batch** (`Any`): a batch of test data.
        - **dataloader_idx** (`int`): task ID being tested.

        **Returns:**
        - **outputs** (`dict[str, Tensor]`): loss and accuracy.
        """
        test_task_id = self.get_test_task_id_from_dataloader_idx(dataloader_idx)

        x, y = batch

        if str(test_task_id) in self.temporary_backbones.keys():
            # use the temporary task's temporary backbone if it wasn't merged yet
            routed_backbone = self.temporary_backbones[str(test_task_id)]
        else:
            # use the main backbone if the task is permanent and the temporary backbone has been merged
            routed_backbone = self.backbone

        feature, _ = routed_backbone(x, stage="test", task_id=test_task_id)
        logits = self.heads(feature, test_task_id)

        loss_cls = self.criterion(logits, y)
        acc = (logits.argmax(dim=1) == y).float().mean()

        return {
            "loss_cls": loss_cls,
            "acc": acc,
        }


# class CLPUDERpp(DERpp):
#     r"""CLPU-DER++ algorithm.

#     This is a continual learning variant of DER++ that supports temporary (private)
#     task learning via side backbones. Temporary tasks are excluded from replay
#     and can later be consolidated into the main backbone via `finally_learn()`.
#     """

#     def __init__(
#         self,
#         backbone: CLBackbone,
#         heads: HeadsTIL | HeadsCIL | HeadDIL,
#         buffer_size: int,
#         distillation_reg_factor: float,
#         replay_ce_factor: float,
#         use_pretrain: bool = False,
#         merge_iters: int | None = None,
#         merge_batch_size: int | None = None,
#         augmentation_transforms: Callable | transforms.Compose | None = None,
#         non_algorithmic_hparams: dict[str, Any] = {},
#         **kwargs,
#     ) -> None:
#         r"""Initialize the CLPU-DER++ algorithm with the network.

#         **Args:**
#         - **backbone** (`CLBackbone`): backbone network.
#         - **heads** (`HeadsTIL` | `HeadsCIL` | `HeadDIL`): output heads.
#         - **buffer_size** (`int`): the size of the memory buffer.
#         - **distillation_reg_factor** (`float`): distillation regularization factor.
#         - **replay_ce_factor** (`float`): classification loss factor for replayed samples.
#         - **use_pretrain** (`bool`): whether to initialize side backbones from the
#           current backbone weights when temporarily learning.
#         - **merge_iters** (`int` | `None`): number of optimization steps to merge a
#           temporary task back into the main backbone.
#         - **merge_batch_size** (`int` | `None`): batch size used for merge replay.
#         - **augmentation_transforms** (`transform` or `transforms.Compose` or `None`):
#           transforms to apply for augmentation after replay sampling.
#         - **non_algorithmic_hparams** (`dict[str, Any]`): non-algorithmic hyperparameters
#           passed to `save_hyperparameters()`.
#         - **kwargs**: Reserved for multiple inheritance.
#         """
#         super().__init__(
#             backbone=backbone,
#             heads=heads,
#             buffer_size=buffer_size,
#             distillation_reg_factor=distillation_reg_factor,
#             replay_ce_factor=replay_ce_factor,
#             augmentation_transforms=augmentation_transforms,
#             non_algorithmic_hparams=non_algorithmic_hparams,
#             **kwargs,
#         )

#         self.use_pretrain: bool = use_pretrain
#         r"""Whether to initialize side backbones from the current backbone."""

#         self.merge_iters: int | None = merge_iters
#         r"""Number of optimization steps used to merge a temporary task."""

#         self.merge_batch_size: int | None = merge_batch_size
#         r"""Batch size used during merge optimization."""

#         self.side_backbones: nn.ModuleDict = nn.ModuleDict()
#         r"""Side backbones for temporarily learned tasks, keyed by task id."""

#         self.task_status: dict[int, str] = {}
#         r"""Task status map: 'R' for remembered, 'T' for temporary, 'F' for forgotten."""

#         self.remembered_task_ids: set[int] = set()
#         r"""Tasks consolidated into the main backbone."""

#         self.temporary_task_ids: set[int] = set()
#         r"""Tasks stored in side backbones and excluded from replay."""

#         self._task_train_iters: dict[int, int] = {}
#         r"""Number of training iterations recorded per task."""

#         self._last_batch_size: int | None = None
#         r"""Last observed batch size, used as fallback in merge."""

#         self.save_hyperparameters("use_pretrain", "merge_iters", "merge_batch_size")

#     def setup_task_id(
#         self,
#         task_id: int,
#         num_classes: int,
#         optimizer: torch.optim.Optimizer,
#         lr_scheduler: torch.optim.lr_scheduler.LRScheduler | None,
#         unlearnable_task_ids: list[int] | None = None,
#     ) -> None:
#         r"""Set up which task the CL experiment is on.

#         Accepts an optional `unlearnable_task_ids` argument for compatibility with
#         continual unlearning pipelines.
#         """
#         super().setup_task_id(
#             task_id=task_id,
#             num_classes=num_classes,
#             optimizer=optimizer,
#             lr_scheduler=lr_scheduler,
#         )

#         # Default to permanent if the pipeline does not provide this flag.
#         if_permanent = getattr(self, "if_permanent_t", True)

#         if if_permanent:
#             self.task_status[task_id] = "R"
#         else:
#             self.task_status[task_id] = "T"
#             self._ensure_side_backbone(task_id)

#     def on_train_start(self) -> None:
#         super().on_train_start()

#         is_temporary = self._is_temporary_task(self.task_id)
#         if is_temporary:
#             self._set_requires_grad(self.backbone, False)
#             self._set_requires_grad(self._get_side_backbone(self.task_id), True)
#         else:
#             self._set_requires_grad(self.backbone, True)
#             for backbone in self.side_backbones.values():
#                 self._set_requires_grad(backbone, False)

#         if self.trainer is not None:
#             self._task_train_iters[self.task_id] = int(
#                 self.trainer.num_training_batches * self.trainer.max_epochs
#             )

#     def on_train_end(self) -> None:
#         super().on_train_end()

#         if self._is_temporary_task(self.task_id):
#             self.temporary_task_ids.add(self.task_id)
#         else:
#             self.remembered_task_ids.add(self.task_id)

#     def forward(self, input: Tensor, stage: str, task_id: int | None = None) -> Tensor:
#         r"""Forward pass that routes temporary tasks to side backbones."""
#         backbone = self._select_backbone(task_id)
#         feature, activations = backbone(input, stage=stage, task_id=task_id)
#         logits = self.heads(feature, task_id)
#         return (
#             logits if self.if_forward_func_return_logits_only else (logits, activations)
#         )

#     def training_step(self, batch: Any) -> dict[str, Tensor]:
#         """Training step for current task `self.task_id`."""
#         x, y = batch
#         self._last_batch_size = len(y)

#         if self.augmentation_transforms:
#             x = self.augmentation_transforms(x)

#         logits, activations = self.forward(x, stage="train", task_id=self.task_id)
#         loss_cls = self.criterion(logits, y)

#         loss_reg = 0.0
#         loss_replay = 0.0

#         if (
#             not self._is_temporary_task(self.task_id)
#             and not self.memory_buffer.is_empty()
#         ):
#             replay_task_ids = self._replay_task_ids(exclude_task_id=self.task_id)
#             if replay_task_ids:
#                 x_replay, labels_replay, logits_replay, task_labels_replay = (
#                     self.memory_buffer.get_data(
#                         size=self._last_batch_size,
#                         included_tasks=replay_task_ids,
#                     )
#                 )

#                 if x_replay.numel() > 0:
#                     if self.augmentation_transforms:
#                         x_replay = self.augmentation_transforms(x_replay)

#                     student_feature_replay, _ = self.backbone(x_replay, stage="test")
#                     student_logits_replay = torch.cat(
#                         [
#                             self.heads(
#                                 student_feature_replay[i].unsqueeze(0),
#                                 task_id=int(task_labels_replay[i].item()),
#                             )
#                             for i in range(task_labels_replay.numel())
#                         ]
#                     )
#                     with torch.no_grad():
#                         teacher_logits_replay = logits_replay

#                     loss_reg = self.distillation_reg(
#                         student_logits=student_logits_replay,
#                         teacher_logits=teacher_logits_replay,
#                     )
#                     loss_replay = self.replay_ce_factor * self.criterion(
#                         student_logits_replay, labels_replay.long()
#                     )

#         loss = loss_cls + loss_reg + loss_replay
#         preds = logits.argmax(dim=1)
#         acc = (preds == y).float().mean()

#         # store current batch into replay buffer (teacher logits from current backbone)
#         self.memory_buffer.add_data(
#             x, y, logits.detach(), torch.full_like(y, self.task_id)
#         )

#         return {
#             "preds": preds,
#             "loss": loss,
#             "loss_cls": loss_cls,
#             "loss_reg": loss_reg,
#             "loss_replay": loss_replay,
#             "acc": acc,
#             "activations": activations,
#         }

#     def merge_temporary_backbone_to_main(
#         self,
#         task_id: int,
#         num_steps: int | None = None,
#         batch_size: int | None = None,
#     ) -> None:
#         r"""Merge a temporarily learned task into the main backbone using replay."""
#         side_key = str(task_id)
#         if side_key not in self.side_backbones:
#             pylogger.warning("No side backbone found for task %d.", task_id)
#             return

#         if num_steps is None:
#             num_steps = (
#                 self.merge_iters
#                 if self.merge_iters is not None
#                 else self._task_train_iters.get(task_id, 0)
#             )
#         if num_steps <= 0:
#             pylogger.warning("No merge steps configured for task %d.", task_id)
#             return

#         if batch_size is None:
#             batch_size = (
#                 self.merge_batch_size
#                 if self.merge_batch_size is not None
#                 else (self._last_batch_size or 0)
#             )
#             if batch_size <= 0:
#                 available = int(self.memory_buffer.labels.size(0))
#                 batch_size = min(32, max(1, available))

#         self._set_requires_grad(self.backbone, True)
#         self._set_requires_grad(self.side_backbones[side_key], False)

#         optimizer = self.optimizer_t(
#             params=[p for p in self.parameters() if p.requires_grad]
#         )

#         self.train()
#         for _ in range(num_steps):
#             loss = 0.0

#             replay_task_ids = self._replay_task_ids(exclude_task_id=task_id)
#             if replay_task_ids:
#                 x_replay, labels_replay, logits_replay, task_labels_replay = (
#                     self.memory_buffer.get_data(
#                         size=batch_size, included_tasks=replay_task_ids
#                     )
#                 )
#                 if x_replay.numel() > 0:
#                     if self.augmentation_transforms:
#                         x_replay = self.augmentation_transforms(x_replay)

#                     student_feature_replay, _ = self.backbone(x_replay, stage="test")
#                     student_logits_replay = torch.cat(
#                         [
#                             self.heads(
#                                 student_feature_replay[i].unsqueeze(0),
#                                 task_id=int(task_labels_replay[i].item()),
#                             )
#                             for i in range(task_labels_replay.numel())
#                         ]
#                     )

#                     loss += self.distillation_reg(
#                         student_logits=student_logits_replay,
#                         teacher_logits=logits_replay,
#                     )
#                     loss += self.replay_ce_factor * self.criterion(
#                         student_logits_replay, labels_replay.long()
#                     )

#             x_t, y_t, logits_t, _ = self.memory_buffer.get_data(
#                 size=batch_size, included_tasks=[task_id]
#             )
#             if x_t.numel() == 0:
#                 continue

#             if self.augmentation_transforms:
#                 x_t = self.augmentation_transforms(x_t)

#             student_feature_t, _ = self.backbone(x_t, stage="test", task_id=task_id)
#             student_logits_t = self.heads(student_feature_t, task_id)
#             loss += self.replay_ce_factor * self.criterion(student_logits_t, y_t.long())
#             loss += self.distillation_reg(
#                 student_logits=student_logits_t, teacher_logits=logits_t
#             )

#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()

#         del self.side_backbones[side_key]
#         self.temporary_task_ids.discard(task_id)
#         self.remembered_task_ids.add(task_id)
#         self.task_status[task_id] = "R"

#     def finally_learn(
#         self,
#         task_id: int,
#         num_steps: int | None = None,
#         batch_size: int | None = None,
#     ) -> None:
#         r"""Merge a temporarily learned task into the main backbone using replay."""
#         self.merge_temporary_backbone_to_main(
#             task_id=task_id, num_steps=num_steps, batch_size=batch_size
#         )

#     def forget(self, task_id: int) -> None:
#         r"""Forget a temporarily learned task by removing its replay data and side backbone."""
#         self.memory_buffer.delete_task(task_id)
#         side_key = str(task_id)
#         if side_key in self.side_backbones:
#             del self.side_backbones[side_key]

#         self.task_status[task_id] = "F"
#         self.temporary_task_ids.discard(task_id)
#         self.remembered_task_ids.discard(task_id)
#         self._reset_task_head(task_id)

#     def _is_temporary_task(self, task_id: int | None) -> bool:
#         return task_id is not None and self.task_status.get(task_id) == "T"

#     def _replay_task_ids(self, exclude_task_id: int | None = None) -> list[int]:
#         replay_tasks = [
#             task_id for task_id, status in self.task_status.items() if status == "R"
#         ]
#         if exclude_task_id is not None:
#             replay_tasks = [t for t in replay_tasks if t != exclude_task_id]
#         return replay_tasks

#     def _ensure_side_backbone(self, task_id: int) -> None:
#         side_key = str(task_id)
#         if side_key in self.side_backbones:
#             return

#         side_backbone = deepcopy(self.backbone)
#         if not self.use_pretrain:
#             self._reset_module_parameters(side_backbone)
#         side_backbone.setup_task_id(task_id=task_id)
#         self.side_backbones[side_key] = side_backbone

#     def _select_backbone(self, task_id: int | None) -> CLBackbone:
#         if task_id is None:
#             return self.backbone
#         side_key = str(task_id)
#         if side_key in self.side_backbones and self._is_temporary_task(task_id):
#             return self.side_backbones[side_key]
#         return self.backbone

#     def _get_side_backbone(self, task_id: int) -> CLBackbone:
#         return self.side_backbones[str(task_id)]

#     @staticmethod
#     def _set_requires_grad(module: nn.Module, requires_grad: bool) -> None:
#         for param in module.parameters():
#             param.requires_grad = requires_grad

#     @staticmethod
#     def _reset_module_parameters(module: nn.Module) -> None:
#         for submodule in module.modules():
#             if hasattr(submodule, "reset_parameters"):
#                 submodule.reset_parameters()

#     def _reset_task_head(self, task_id: int) -> None:
#         if isinstance(self.heads, HeadsTIL):
#             key = f"{task_id}"
#             if key in self.heads.heads:
#                 self.heads.heads[key].reset_parameters()
#         elif isinstance(self.heads, HeadsCIL):
#             key = f"{task_id}"
#             if key in self.heads.heads:
#                 self.heads.heads[key].reset_parameters()
