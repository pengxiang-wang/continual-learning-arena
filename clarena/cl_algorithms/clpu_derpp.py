r"""
The submodule in `cl_algorithms` for [CLPU-DER++](https://arxiv.org/abs/2203.12817) algorithm.
"""

__all__ = ["CLPUDERpp"]

import logging
import math
from copy import deepcopy
from typing import Any, Callable

import torch
from rich.progress import track
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
        merge_num_steps: int,
        merge_batch_size: int,
        augmentation_transforms: Callable | transforms.Compose | None = None,
        non_algorithmic_hparams: dict[str, Any] = {},
        disable_unlearning: bool = False,
        **kwargs,
    ) -> None:
        r"""Initialize the CLPU-DER++ algorithm with the network.

        **Args:**
        - **backbone** (`CLBackbone`): backbone network.
        - **heads** (`HeadsTIL` | `HeadDIL`): output heads.
        - **buffer_size** (`int`): the size of the memory buffer. For now we only support fixed size buffer.
        - **distillation_reg_factor** (`float`): hyperparameter, the distillation regularization factor ($\alpha$ in the [DER++ paper](https://arxiv.org/abs/2004.07211)). It controls the strength of preventing forgetting.
        - **replay_ce_factor** (`float`): hyperparameter, the classification loss factor for replayed samples, ($\beta$ in the [DER++ paper](https://arxiv.org/abs/2004.07211)). It also controls the strength of preventing forgetting.
        - **temporary_backbone_init_mode** (`str`): method to initialize temporary task backbone networks, must be one of:
            1. 'from_main': initialize from the current main backbone weights.
            2. 'from_scratch': initialize from scratch.
        - **merge_num_steps** (`int`): fallback number of optimization steps for merge when training schedule cannot be inferred. By default, merge steps follow official CLPU-DER++ behavior: `n_epochs * len(train_loader)`.
        - **merge_batch_size** (`int`): batch size used for merge.
        - **augmentation_transforms** (`transform` or `transforms.Compose` or `None`): the transforms to apply for augmentation after replay sampling. Not to confuse with the data transforms applied to the input of training data. Can be a single transform, composed transforms, or no transform.
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

        self.merge_num_steps: int = merge_num_steps
        r"""Fallback number of optimization steps used to merge a temporary task."""

        self.merge_batch_size: int = merge_batch_size
        r"""Batch size used during merge optimization."""

        self.save_hyperparameters(
            "temporary_backbone_init_mode", "merge_num_steps", "merge_batch_size"
        )

        self.original_backbone_state_dict: dict = deepcopy(backbone.state_dict())
        r"""The original backbone state dict before training on any task. Used to initialize new independent temporary backbones for new tasks."""

        self.temporary_backbones: nn.ModuleDict = nn.ModuleDict()
        r"""Independent temporary backbones for temporary tasks. Keys are task IDs (in string format) and values are the corresponding temporary backbones. Only temporary task corresponds to a temporary backbone. """

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

        if not self.disable_unlearning:
            # initialise a temporary backbone for the current task (if it is temporary). This has to be done before configure_optimizers(), because optimizer needs to include parameters from temporary backbone if any.
            if self.task_id not in self.task_ids_just_no_longer_unlearnable:
                # if the current task (`self.task_id`) is permanent, it just turns no longer unlearnable
                temporary_backbone = deepcopy(self.backbone)
                if self.temporary_backbone_init_mode == "from_scratch":
                    temporary_backbone.load_state_dict(
                        self.original_backbone_state_dict
                    )

                self.temporary_backbones[f"{self.task_id}"] = temporary_backbone
                pylogger.info(
                    "The new task %d is temporary. Temporary backbone created for this temporary task.",
                    self.task_id,
                )
            else:
                pylogger.info(
                    "The new task %d is permanent. No temporary backbone created for this permanent task. It will use the main backbone directly.",
                    self.task_id,
                )

    def on_train_start(self) -> None:
        r"""Merge and delete temporary backbones that just become no longer unlearnable."""

        if not self.disable_unlearning:
            merge_num_steps = self.resolve_merge_num_steps()
            # merge and delete temporary backbones that just becomes no longer unlearnable. It corresponds to Case III in the algorithm description of the [CLPU paper](https://arxiv.org/abs/2203.12817).
            for tid in self.task_ids_just_no_longer_unlearnable:
                if tid not in self.unlearned_task_ids and tid != self.task_id:
                    self.merge_temporary_backbone_to_main(
                        tid, merge_num_steps=merge_num_steps
                    )
                    del self.temporary_backbones[f"{tid}"]

    def resolve_merge_num_steps(self) -> int:
        r"""Resolve merge optimization steps using current training schedule.

        Official CLPU-DER++ uses `n_epochs * len(train_loader)` for merge. We
        derive that from the active trainer; if unavailable, we fall back to the
        configured `self.merge_num_steps`.
        """

        trainer = getattr(self, "trainer", None)
        if trainer is None:
            return self.merge_num_steps

        max_epochs = int(getattr(trainer, "max_epochs", 1))
        if max_epochs <= 0:
            max_epochs = 1

        steps_per_epoch = getattr(trainer, "num_training_batches", None)
        if isinstance(steps_per_epoch, list):
            steps_per_epoch = sum(steps_per_epoch)
        if steps_per_epoch is None or steps_per_epoch <= 0 or not math.isfinite(
            float(steps_per_epoch)
        ):
            datamodule = getattr(trainer, "datamodule", None)
            if datamodule is None:
                return self.merge_num_steps
            try:
                steps_per_epoch = len(datamodule.train_dataloader())
            except Exception:
                return self.merge_num_steps

        resolved_steps = max(1, int(max_epochs * steps_per_epoch))
        pylogger.info(
            "Resolved merge_num_steps from training schedule: %d (max_epochs=%d, steps_per_epoch=%d).",
            resolved_steps,
            max_epochs,
            int(steps_per_epoch),
        )
        return resolved_steps

    def merge_temporary_backbone_to_main(
        self, task_id_to_be_merged: int, merge_num_steps: int | None = None
    ) -> None:
        r"""Merge the temporary backbone for `task_id_to_be_merged` into the main backbone using DER++ replay."""

        pylogger.info(
            "Merging temporary backbone for task %d into main backbone...",
            task_id_to_be_merged,
        )
        if str(task_id_to_be_merged) not in self.temporary_backbones:
            pylogger.warning(
                "No temporary backbone found for task %d to be merged. Skip merging!",
                task_id_to_be_merged,
            )
            return

        # fix temporary backbone, train main backbone
        for param in self.backbone.parameters():
            param.requires_grad = True
        for param in self.temporary_backbones[str(task_id_to_be_merged)].parameters():
            param.requires_grad = False

        # get optimizer
        optimizer = self.optimizer_t(
            params=[p for p in self.parameters() if p.requires_grad]
        )

        merge_num_steps = (
            self.resolve_merge_num_steps()
            if merge_num_steps is None
            else max(1, int(merge_num_steps))
        )
        temporary_task_ids = {int(tid) for tid in self.temporary_backbones.keys()}
        remaining_replay_task_ids = [
            tid
            for tid in self.memory_buffer.stored_tasks()
            if tid != task_id_to_be_merged
            and tid not in temporary_task_ids
            and tid not in self.unlearned_task_ids
        ]

        # merge process with replay
        self.train()
        for s in track(
            range(merge_num_steps),
            description=f"Merging temporary backbone for task {task_id_to_be_merged}",
            transient=True,
        ):  # tqdm loop for progress bar
            loss = 0.0

            # the first loss term of Case III in the algorithm description of the [CLPU paper](https://arxiv.org/abs/2203.12817). For current temporary task
            x_t, y_t, logits_t, _ = self.memory_buffer.get_data(
                size=self.merge_batch_size, included_tasks=[task_id_to_be_merged]
            )

            if self.augmentation_transforms:
                x_t = self.augmentation_transforms(x_t)

            student_feature_t, _ = self.backbone(
                x_t, stage="train", task_id=task_id_to_be_merged
            )
            student_logits_t = self.heads(student_feature_t, task_id_to_be_merged)
            loss += self.replay_ce_factor * self.criterion(student_logits_t, y_t.long())
            loss += self.distillation_reg(
                student_logits=student_logits_t, teacher_logits=logits_t
            )

            # the second loss term of Case III in the algorithm description of the [CLPU paper](https://arxiv.org/abs/2203.12817). For replay from other remaining tasks
            if not self.memory_buffer.is_empty() and remaining_replay_task_ids:
                x_replay, labels_replay, logits_replay, task_labels_replay = (
                    self.memory_buffer.get_data(
                        size=self.merge_batch_size,
                        included_tasks=remaining_replay_task_ids,
                    )
                )
                if self.augmentation_transforms:
                    x_replay = self.augmentation_transforms(x_replay)

                student_feature_replay, _ = self.backbone(x_replay, stage="train")
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

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        pylogger.info(
            "Merging completed! Deleting temporary backbone for task %d.",
            task_id_to_be_merged,
        )

    def forward(self, input: Tensor, stage: str, task_id: int | None = None) -> Tensor:
        r"""Forward pass that routes temporary tasks to their independent backbones. If the task is permanent, it uses the main backbone.

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
        if str(task_id) in self.temporary_backbones:
            # It corresponds to Case II in the algorithm description of the [CLPU paper](https://arxiv.org/abs/2203.12817).
            routed_backbone = self.temporary_backbones[str(task_id)]
        else:
            # use the main backbone if the task is permanent and the temporary backbone has been merged
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

        if str(self.task_id) in self.temporary_backbones:
            # use the temporary task's temporary backbone if it wasn't merged yet
            # It corresponds to Case II in the algorithm description of the [CLPU paper](https://arxiv.org/abs/2203.12817).
            routed_backbone = self.temporary_backbones[str(self.task_id)]
            # Official CLPU-DER++ does not apply DER++ replay regularization while
            # a task is in temporary-learning status.
            derpp_reg = 0.0
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
        r"""Test step for CLPU-DER++. Routes temporary tasks to their independent backbones. If the task is permanent, it uses the main backbone.

        Tests all seen tasks indexed by `dataloader_idx`.

        **Args:**
        - **batch** (`Any`): a batch of test data.
        - **dataloader_idx** (`int`): task ID being tested.

        **Returns:**
        - **outputs** (`dict[str, Tensor]`): loss and accuracy.
        """
        test_task_id = self.get_test_task_id_from_dataloader_idx(dataloader_idx)

        x, y = batch

        if str(test_task_id) in self.temporary_backbones:
            # use the temporary task's temporary backbone if it wasn't merged yet
            routed_backbone = self.temporary_backbones[str(test_task_id)]
        else:
            # use the main backbone if the task is permanent and the temporary backbone has been merged. Also routed here for unlearned tasks that have no temporary backbone.
            routed_backbone = self.backbone

        feature, _ = routed_backbone(x, stage="test", task_id=test_task_id)

        logits = self.heads(
            feature, test_task_id
        )  # use the corresponding head to test (instead of the current task `self.task_id`)

        loss_cls = self.criterion(logits, y)
        preds = logits.argmax(dim=1)
        acc = (preds == y).float().mean()

        return {
            "loss_cls": loss_cls,
            "acc": acc,
            "preds": preds,
        }
