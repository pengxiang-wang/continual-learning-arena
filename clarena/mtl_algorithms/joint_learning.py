r"""
The submodule in `mtl_algorithms` for joint learning algorithm.
"""

__all__ = ["JointLearning"]

import logging
from typing import Any

import torch
from torch import Tensor
from torch.utils.data import DataLoader

from clarena.backbones import Backbone
from clarena.heads import HeadsMTL
from clarena.mtl_algorithms import MTLAlgorithm

# always get logger for built-in logging in each module
pylogger = logging.getLogger(__name__)


class JointLearning(MTLAlgorithm):
    r"""Joint learning algorithm.

    The most naive way for multi-task learning. It directly trains all tasks.
    """

    def __init__(
        self,
        backbone: Backbone,
        heads: HeadsMTL,
        non_algorithmic_hparams: dict[str, Any] = {},
    ) -> None:
        r"""Initialize the JointLearning algorithm with the network. It has no additional hyperparameters.

        **Args:**
        - **backbone** (`Backbone`): backbone network.
        - **heads** (`HeadsMTL`): output heads.
        - **non_algorithmic_hparams** (`dict[str, Any]`): non-algorithmic hyperparameters that are not related to the algorithm itself are passed to this `LightningModule` object from the config, such as optimizer and learning rate scheduler configurations. They are saved for Lightning APIs from `save_hyperparameters()` method. This is useful for the experiment configuration and reproducibility.
        """
        super().__init__(
            backbone=backbone,
            heads=heads,
            non_algorithmic_hparams=non_algorithmic_hparams,
        )

    def training_step(self, batch: Any) -> dict[str, Tensor]:
        r"""Training step.

        **Args:**
        - **batch** (`Any`): a batch of training data, which can be from any mixed tasks. Must include task IDs in the batch.

        **Returns:**
        - **outputs** (`dict[str, Tensor]`): a dictionary contains loss and accuracy from this training step. Keys (`str`) are the metrics names, and values (`Tensor`) are the metrics. Must include the key 'loss' which is total loss in the case of automatic optimization, according to PyTorch Lightning docs.
        """
        x, y, task_ids = batch  # train data are provided task ID in case of MTL
        logits, activations = self.forward(x, stage="train", task_ids=task_ids)

        # the data are from different tasks, so we need to calculate the loss and accuracy for each task separately
        preds = torch.zeros_like(y)
        loss_cls = 0.0
        acc = 0.0

        for task_id in torch.unique(task_ids):  # for each unique task in the batch
            idx = (task_ids == task_id).nonzero(as_tuple=True)[
                0
            ]  # indices of the current task in the batch
            logits_t = logits[idx]  # get the logits for the current task
            y_t = y[idx]  # class labels for the current task

            # classification loss
            loss_cls_t = self.criterion(logits_t, y_t)
            loss_cls = loss_cls + loss_cls_t

            # predicted labels of this task
            preds_t = logits_t.argmax(dim=1)
            preds[idx] = preds_t

            # accuracy of this task
            acc_task = (preds_t == y_t).float().mean()
            acc = acc + acc_task

        loss_cls = loss_cls / len(torch.unique(task_ids))  # average loss over tasks
        acc = acc / len(torch.unique(task_ids))  # average accuracy over tasks

        # total loss
        loss = loss_cls

        return {
            "preds": preds,
            "loss": loss,  # return loss is essential for training step, or backpropagation will fail
            "loss_cls": loss_cls,
            "acc": acc,  # return other metrics for lightning loggers callback to handle at `on_train_batch_end()`
            "activations": activations,
        }

    def validation_step(
        self, batch: DataLoader, batch_idx: int, dataloader_idx: int = 0
    ) -> dict[str, Tensor]:
        r"""Validation step. This is done task by task rather than mixing the tasks in batches.

        **Args:**
        - **batch** (`Any`): a batch of validation data.
        - **dataloader_idx** (`int`): the task ID of seen tasks to be validated. A default value of 0 is given otherwise the LightningModule will raise a `RuntimeError`.

        **Returns:**
        - **outputs** (`dict[str, Tensor]`): a dictionary contains loss and accuracy from this validation step. Keys (`str`) are the metrics names, and values (`Tensor`) are the metrics.
        """
        val_task_id = self.get_val_task_id_from_dataloader_idx(dataloader_idx)

        x, y, _ = batch  # validation data are not provided task ID

        # the batch is from the same task, so no need to divide the input batch by tasks
        logits, _ = self.forward(
            x, stage="validation", task_ids=val_task_id
        )  # use the corresponding head to get the logits
        loss_cls = self.criterion(logits, y)
        preds = logits.argmax(dim=1)
        acc = (preds == y).float().mean()

        # return metrics for lightning loggers callback to handle at `on_validation_batch_end()`
        return {
            "preds": preds,
            "loss_cls": loss_cls,
            "acc": acc,
        }

    def test_step(
        self, batch: DataLoader, batch_idx: int, dataloader_idx: int = 0
    ) -> dict[str, Tensor]:
        r"""Test step. This is done task by task rather than mixing the tasks in batches.

        **Args:**
        - **batch** (`Any`): a batch of test data.
        - **dataloader_idx** (`int`): the task ID of seen tasks to be tested. A default value of 0 is given otherwise the LightningModule will raise a `RuntimeError`.

        **Returns:**
        - **outputs** (`dict[str, Tensor]`): a dictionary contains loss and accuracy from this test step. Keys (`str`) are the metrics names, and values (`Tensor`) are the metrics.
        """
        test_task_id = self.get_test_task_id_from_dataloader_idx(dataloader_idx)

        x, y, _ = batch

        # the batch is from the same task, so no need to divide the input batch by tasks
        logits, _ = self.forward(
            x, stage="test", task_ids=test_task_id
        )  # use the corresponding head to get the logits
        loss_cls = self.criterion(logits, y)
        preds = logits.argmax(dim=1)
        acc = (preds == y).float().mean()

        # return metrics for lightning loggers callback to handle at `on_test_batch_end()`
        return {
            "preds": preds,
            "loss_cls": loss_cls,
            "acc": acc,
        }
