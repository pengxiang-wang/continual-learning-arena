r"""
The submodule in `cl_algorithms` for [DER (Dark Experience Replay) algorithm](https://arxiv.org/abs/2004.07211)
and its DER++ variant.
"""

__all__ = ["DER", "DERpp"]

import logging
from collections import Counter
from typing import Any, Callable

import torch
from matplotlib import transforms
from torch import Tensor
from torchvision.transforms import transforms

from clarena.backbones import CLBackbone
from clarena.cl_algorithms import Finetuning
from clarena.cl_algorithms.regularizers import DistillationReg
from clarena.heads import HeadDIL, HeadsCIL, HeadsTIL

# always get logger for built-in logging in each module
pylogger = logging.getLogger(__name__)


class DER(Finetuning):
    r"""[DER (Dark Experience Replay) algorithm](https://arxiv.org/abs/2004.07211) algorithm."""

    def __init__(
        self,
        backbone: CLBackbone,
        heads: HeadsTIL | HeadsCIL | HeadDIL,
        buffer_size: int,
        distillation_reg_factor: float,
        augmentation_transforms: Callable | transforms.Compose | None = None,
        non_algorithmic_hparams: dict[str, Any] = {},
        **kwargs,
    ) -> None:
        r"""Initialize the DER algorithm with the network.

        **Args:**
        - **backbone** (`CLBackbone`): backbone network.
        - **heads** (`HeadsTIL` | `HeadsCIL` | `HeadDIL`): output heads.
        - **buffer_size** (`int`): the size of the memory buffer. For now we only support fixed size buffer.
        - **distillation_reg_factor** (`float`): hyperparameter, the distillation regularization factor. It controls the strength of preventing forgetting.
        - **augmentation_transforms** (`transform` or `transforms.Compose` or `None`): the transforms to apply for augmentation after replay sampling. Not to confuse with the data transforms applied to the input of training data. Can be a single transform, composed transforms, or no transform.
        - **non_algorithmic_hparams** (`dict[str, Any]`): non-algorithmic hyperparameters that are not related to the algorithm itself are passed to this `LightningModule` object from the config, such as optimizer and learning rate scheduler configurations. They are saved for Lightning APIs from `save_hyperparameters()` method. This is useful for the experiment configuration and reproducibility.
        - **kwargs**: Reserved for multiple inheritance.
        """
        super().__init__(
            backbone=backbone,
            heads=heads,
            non_algorithmic_hparams=non_algorithmic_hparams,
            **kwargs,
        )

        self.memory_buffer: Buffer = Buffer(size=buffer_size)
        r"""The memory buffer for replay."""

        self.distillation_reg_factor: float = distillation_reg_factor
        r"""The distillation regularization factor."""
        self.distillation_reg = DistillationReg(
            factor=distillation_reg_factor,
            temperature=1,
            distance="MSE",
        )
        r"""Initialize and store the distillation regularizer."""

        self.augmentation_transforms: Callable | transforms.Compose | None = (
            augmentation_transforms
        )
        r"""The transforms to apply for augmentation after replay sampling."""

        # save additional algorithmic hyperparameters
        self.save_hyperparameters("buffer_size", "distillation_reg_factor")

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

        # regularization loss. In DER, this is the knowledge distillation loss from replay data
        loss_reg = 0.0

        if not self.memory_buffer.is_empty():

            # sample from memory buffer with the same batch size as current batch
            x_replay, _, logits_replay, task_labels_replay = (
                self.memory_buffer.get_data(size=batch_size)
            )

            # apply augmentation transforms if any
            if self.augmentation_transforms:
                x_replay = self.augmentation_transforms(x_replay)

            # get the student logits for this batch using the current model
            student_feature_replay, _ = self.backbone(x_replay, stage="test")
            student_logits_replay = torch.cat(
                [
                    self.heads(student_feature_replay[i].unsqueeze(0), task_id=tid)
                    for i, tid in enumerate(task_labels_replay)
                ]
            )
            with torch.no_grad():  # stop updating the previous heads

                teacher_logits_replay = logits_replay

            loss_reg = self.distillation_reg(
                student_logits=student_logits_replay,
                teacher_logits=teacher_logits_replay,
            )

        # total loss
        loss = loss_cls + loss_reg

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
            "loss_reg": loss_reg,
            "acc": acc,  # Return other metrics for lightning loggers callback to handle at `on_train_batch_end()`
            "activations": activations,
        }


class DERpp(DER):
    r"""[DER++ (Dark Experience Replay++) algorithm](https://arxiv.org/abs/2004.07211) algorithm.

    DER++ adds a replay classification loss on buffered samples on top of DER's logit distillation.
    """

    def __init__(
        self,
        backbone: CLBackbone,
        heads: HeadsTIL | HeadsCIL | HeadDIL,
        buffer_size: int,
        distillation_reg_factor: float,
        replay_ce_factor: float,
        augmentation_transforms: Callable | transforms.Compose | None = None,
        non_algorithmic_hparams: dict[str, Any] = {},
        **kwargs,
    ) -> None:
        r"""Initialize the DER++ algorithm with the network.

        **Args:**
        - **backbone** (`CLBackbone`): backbone network.
        - **heads** (`HeadsTIL` | `HeadsCIL` | `HeadDIL`): output heads.
        - **buffer_size** (`int`): the size of the memory buffer. For now we only support fixed size buffer.
        - **distillation_reg_factor** (`float`): hyperparameter, the distillation regularization factor. It controls the strength of preventing forgetting.
        - **replay_ce_factor** (`float`): hyperparameter, the classification loss factor for replayed samples.
        - **augmentation_transforms** (`transform` or `transforms.Compose` or `None`): the transforms to apply for augmentation after replay sampling. Not to confuse with the data transforms applied to the input of training data. Can be a single transform, composed transforms, or no transform.
        - **non_algorithmic_hparams** (`dict[str, Any]`): non-algorithmic hyperparameters that are not related to the algorithm itself are passed to this `LightningModule` object from the config, such as optimizer and learning rate scheduler configurations. They are saved for Lightning APIs from `save_hyperparameters()` method. This is useful for the experiment configuration and reproducibility.
        - **kwargs**: Reserved for multiple inheritance.
        """
        super().__init__(
            backbone=backbone,
            heads=heads,
            buffer_size=buffer_size,
            distillation_reg_factor=distillation_reg_factor,
            augmentation_transforms=augmentation_transforms,
            non_algorithmic_hparams=non_algorithmic_hparams,
            **kwargs,
        )

        self.replay_ce_factor: float = replay_ce_factor
        r"""The classification loss factor for replayed samples."""

        # save additional algorithmic hyperparameters
        self.save_hyperparameters("replay_ce_factor")

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

        # DER++ adds replay classification loss on top of DER's logit distillation
        loss_reg = 0.0
        loss_replay = 0.0

        if not self.memory_buffer.is_empty():

            # sample from memory buffer with the same batch size as current batch
            x_replay, labels_replay, logits_replay, task_labels_replay = (
                self.memory_buffer.get_data(size=batch_size)
            )

            # apply augmentation transforms if any
            if self.augmentation_transforms:
                x_replay = self.augmentation_transforms(x_replay)

            # get the student logits for this batch using the current model
            student_feature_replay, _ = self.backbone(x_replay, stage="test")
            student_logits_replay = torch.cat(
                [
                    self.heads(student_feature_replay[i].unsqueeze(0), task_id=tid)
                    for i, tid in enumerate(task_labels_replay)
                ]
            )
            with torch.no_grad():  # stop updating the previous heads
                teacher_logits_replay = logits_replay

            loss_reg = self.distillation_reg(
                student_logits=student_logits_replay,
                teacher_logits=teacher_logits_replay,
            )
            loss_replay = self.replay_ce_factor * self.criterion(
                student_logits_replay, labels_replay.long()
            )

        # total loss
        loss = loss_cls + loss_reg + loss_replay

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
            "loss_reg": loss_reg,
            "loss_replay": loss_replay,
            "acc": acc,  # Return other metrics for lightning loggers callback to handle at `on_train_batch_end()`
            "activations": activations,
        }


class Buffer:
    r"""The memory buffer for replay."""

    def __init__(self, size: int) -> None:
        r"""Initialize the memory buffer.

        **Args:**
        - **size** (`int`): the size of the memory buffer.
        """

        self.examples: Tensor = torch.empty((0,))
        r"""The stored examples in the buffer. The first dimension is the index."""

        self.labels: Tensor = torch.empty((0,))
        r"""The stored labels in the buffer. The first dimension is the index."""

        self.logits: Tensor = torch.empty((0,))
        r"""The stored logits predicted by previous model in the buffer. The first dimension is the index."""

        self.task_labels: Tensor = torch.empty((0,), dtype=torch.long)
        r"""The stored task labels in the buffer. The first dimension is the index."""

        self.num_seen_examples: int = 0
        r"""The number of seen examples so far."""

        self.size: int = size
        r"""The size of the memory buffer."""

    def add_data(
        self,
        examples: Tensor,
        labels: Tensor,
        logits: Tensor,
        task_labels: Tensor,
    ) -> None:
        """Add new batch of data, using der + incremental tensor concatenation.

        **Args:**
        - **examples** (`Tensor`): a batch of examples to add to the buffer.
        - **labels** (`Tensor`): a batch of labels to add to the buffer.
        - **logits** (`Tensor`): a batch of logits to add to the buffer.
        - **task_labels** (`Tensor`): a batch of task labels to add to the buffer.
        """
        batch_size = examples.size(0)

        for i in range(batch_size):
            self.num_seen_examples += 1

            if self.labels.numel() < self.size:
                # buffer not full: append
                idx = self.labels.numel()
            else:
                # buffer is full: reservoir replace
                j = torch.randint(0, self.num_seen_examples, (1,)).item()
                if j >= self.size:
                    continue
                idx = j

            x_i = examples[i].unsqueeze(0)
            y_i = labels[i].unsqueeze(0)
            logit_i = logits[i].unsqueeze(0)
            t_i = task_labels[i].unsqueeze(0)

            if idx == self.labels.numel():
                # append case
                # examples might be multi-dimensional, so need to pad/reshape
                if self.examples.numel() == 0:
                    # first time: set shape based on x_i
                    self.examples = x_i
                else:
                    self.examples = torch.cat((self.examples, x_i), dim=0)
                self.labels = torch.cat((self.labels, y_i), dim=0)
                self.logits = torch.cat((self.logits, logit_i), dim=0)
                self.task_labels = torch.cat((self.task_labels, t_i), dim=0)
            else:
                # replace at idx
                self.examples[idx].copy_(x_i.squeeze(0))
                self.labels[idx].copy_(y_i.squeeze(0))
                self.logits[idx].copy_(logit_i.squeeze(0))
                self.task_labels[idx].copy_(t_i.squeeze(0))

        # print(f"Task labels percentage: {Counter(self.task_labels.tolist())}")

    def get_data(
        self,
        size: int,
        included_tasks: list[int] | None = None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Sample a batch from buffer.

        **Args:**
        - **size** (`int`): the size of the batch to sample.
        - **included_tasks** (`list[int]` | None): the list of task IDs to include in sampling.
        If `None`, samples from all tasks.

        **Returns:**
        - **examples** (`Tensor`): sampled examples from the buffer.
        - **labels** (`Tensor`): sampled labels from the buffer.
        - **logits** (`Tensor`): sampled logits from the buffer.
        - **task_labels** (`Tensor`): sampled task labels from the buffer.
        """
        available = self.labels.size(0)
        if available == 0:
            # empty buffer
            return (
                torch.empty((0,)),
                torch.empty((0,)),
                torch.empty((0,)),
                torch.empty((0,)),
            )

        # filter to only included tasks (if specified)
        if included_tasks:
            included = torch.tensor(included_tasks, device=self.task_labels.device)
            mask = torch.isin(self.task_labels, included)
            valid_indices = torch.nonzero(mask, as_tuple=True)[0]
        else:
            valid_indices = torch.arange(available, device=self.task_labels.device)

        # handle case where no valid samples remain
        if valid_indices.numel() == 0:
            return (
                torch.empty((0,)),
                torch.empty((0,)),
                torch.empty((0,)),
                torch.empty((0,)),
            )

        # limit batch size to available samples
        size = min(size, valid_indices.numel())

        # sample indices randomly
        chosen = valid_indices[torch.randint(0, valid_indices.numel(), (size,))]

        return (
            self.examples[chosen],
            self.labels[chosen],
            self.logits[chosen],
            self.task_labels[chosen],
        )

    def is_empty(self) -> bool:
        r"""Check if the buffer is empty.

        **Returns:**
        - **is_empty** (`bool`): `True` if the buffer is empty, `False` otherwise.
        """
        return len(self.labels) == 0

    def delete_task(self, task_id: int) -> None:
        r"""Delete all data in the buffer belonging to a specific task.

        **Args:**
        - **task_id** (`int`): the task ID to delete from the buffer.
        """
        mask = self.task_labels != task_id
        self.examples = self.examples[mask]
        self.labels = self.labels[mask]
        self.logits = self.logits[mask]
        self.task_labels = self.task_labels[mask]
