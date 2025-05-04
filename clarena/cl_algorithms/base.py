r"""
The submodule in `cl_algorithms` for CL algorithm bases.
"""

__all__ = ["CLAlgorithm", "JointLearning"]

import logging
from typing import Any

import torch
from lightning import LightningModule
from omegaconf import DictConfig
from torch import Tensor, nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader

from clarena.backbones import CLBackbone
from clarena.cl_heads import HeadsCIL, HeadsTIL

# always get logger for built-in logging in each module
pylogger = logging.getLogger(__name__)


class CLAlgorithm(LightningModule):
    r"""The base class of continual learning algorithms, inherited from `LightningModule`."""

    def __init__(
        self,
        backbone: CLBackbone,
        heads: HeadsTIL | HeadsCIL,
    ) -> None:
        r"""Initialise the CL algorithm with the network.

        **Args:**
        - **backbone** (`CLBackbone`): backbone network.
        - **heads** (`HeadsTIL` | `HeadsCIL`): output heads.
        """
        LightningModule.__init__(self)

        self.backbone: CLBackbone = backbone
        r"""Store the backbone network."""
        self.heads: HeadsTIL | HeadsCIL = heads
        r"""Store the output heads."""
        self.optimizer_t: Optimizer
        r"""Store the optimizer object (partially initialised) for the backpropagation of task `self.task_id`. Will be equipped with parameters in `configure_optimizers()`."""
        self.lr_scheduler_t: LRScheduler | None
        r"""Store the learning rate scheduler for the optimizer. If `None`, no scheduler is used."""
        self.criterion = nn.CrossEntropyLoss()
        r"""The loss function bewteen the output logits and the target labels. Default is cross-entropy loss."""

        self.if_forward_func_return_logits_only: bool = False
        r"""Whether the `forward()` method returns logits only. If `False`, it returns a dictionary containing logits and other information. Default is `False`."""

        self.task_id: int
        r"""Task ID counter indicating which task is being processed. Self updated during the task loop. Starting from 1. """
        self.seen_task_ids: list[int] = []
        r"""The list of task IDs that have been seen in the experiment."""
        self.unlearning_task_ids: list[int]
        r"""The list of task IDs to be unlearned after `self.task_id`. Only used for unlearning algorithms. """

        self.unlearned_task_ids: set[int] = set()
        r"""Store the list of task IDs that have been unlearned in the experiment."""
        self.cfg_unlearning_test_reference: DictConfig
        r"""The reference experiment configuration for unlearning test. Only used for unlearning algorithms. """

        CLAlgorithm.sanity_check(self)

    def sanity_check(self) -> None:
        r"""Check the sanity of the arguments.

        **Raises:**
        - **ValueError**: if the `output_dim` of backbone network is not equal to the `input_dim` of CL heads.
        """
        if self.backbone.output_dim != self.heads.input_dim:
            raise ValueError(
                "The output_dim of backbone network should be equal to the input_dim of CL heads!"
            )

    def setup_task_id(
        self,
        task_id: int,
        num_classes_t: int,
        optimizer: Optimizer,
        lr_scheduler: LRScheduler | None,
    ) -> None:
        r"""Set up which task the CL experiment is on. This must be done before `forward()` method is called.

        **Args:**
        - **task_id** (`int`): the target task ID.
        - **num_classes_t** (`int`): the number of classes in the task.
        - **optimizer** (`Optimizer`): the optimizer object (partially initialised) for the task `self.task_id`.
        - **lr_scheduler** (`LRScheduler` | `None`): the learning rate scheduler for the optimizer. If `None`, no scheduler is used.
        """
        self.task_id = task_id
        self.seen_task_ids.append(task_id)
        self.heads.setup_task_id(task_id, num_classes_t)
        self.optimizer_t = optimizer
        self.lr_scheduler_t = lr_scheduler

    def get_test_task_id_from_dataloader_idx(self, dataloader_idx: int) -> int:
        r"""Get the test task ID from the dataloader index.

        **Args:**
        - **dataloader_idx** (`int`): the dataloader index.

        **Returns:**
        - **test_task_id** (`str`): the test task ID.
        """
        dataset_test = self.trainer.datamodule.dataset_test
        test_task_id = list(dataset_test.keys())[dataloader_idx]
        return test_task_id

    def set_forward_func_return_logits_only(
        self, forward_func_return_logits_only: bool
    ) -> None:
        r"""Set whether the `forward()` method returns logits only.

        **Args:**
        - **forward_func_return_logits_only** (`bool`): whether the `forward()` method returns logits only. If `False`, it returns a dictionary containing logits and other information.
        """
        self.if_forward_func_return_logits_only = forward_func_return_logits_only

    def preceding_layer(self, layer_name: str) -> nn.Module | None:
        r"""Get the preceding layer of the given layer. If the given layer is the first layer, return `None`.

        **Args:**
        - **layer_name** (`str`): the name of the layer.

        **Returns:**
        - **preceding_layer** (`nn.Module` | `None`): the preceding layer.
        """
        if layer_name == "heads":
            backbone_last_layer_name = self.backbone.weighted_layer_names[-1]
            backbone_last_layer = self.backbone.get_layer_by_name(
                backbone_last_layer_name
            )
            return backbone_last_layer
        else:
            preceding_layer_name = self.backbone.preceding_layer_name(layer_name)
            preceding_layer = self.backbone.get_layer_by_name(preceding_layer_name)
        return preceding_layer

    def next_layer(self, layer_name: str) -> nn.Module | None:
        r"""Get the next layer of the given layer. If the given layer is the first layer, return `None`.

        **Args:**
        - **layer_name** (`str`): the name of the layer.

        **Returns:**
        - **preceding_layer** (`nn.Module` | `None`): the next layer.
        """
        if layer_name == "heads":
            return None
        else:
            next_layer_name = self.backbone.next_layer_name(layer_name)
            if next_layer_name is not None:
                next_layer = self.backbone.get_layer_by_name(next_layer_name)
            else:
                next_layer = self.heads.get_head(self.task_id)
        return next_layer

    def forward(self, input: Tensor, stage: str, task_id: int | None = None) -> Tensor:
        r"""The forward pass for data from task `task_id`. Note that it is nothing to do with `forward()` method in `nn.Module`. This definition provides a template that many CL algorithm including the vanilla Finetuning algorithm use. It works both for TIL and CIL.

        **Args:**
        - **input** (`Tensor`): The input tensor from data.
        - **stage** (`str`): the stage of the forward pass, should be one of the following:
            1. 'train': training stage.
            2. 'validation': validation stage.
            3. 'test': testing stage.
        - **task_id** (`int`): the task ID where the data are from. If stage is 'train' or `validation`, it is usually from the current task `self.task_id`. If stage is 'test', it could be from any seen task. In TIL, the task IDs of test data are provided thus this argument can be used. In CIL, they are not provided, so it is just a placeholder for API consistence but never used, and best practices are not to provide this argument and leave it as the default value.

        **Returns:**
        - **logits** (`Tensor`): the output logits tensor.
        - **activations** (`dict[str, Tensor]`): the hidden features (after activation) in each weighted layer. Key (`str`) is the weighted layer name, value (`Tensor`) is the hidden feature tensor. This is used for the continual learning algorithms that need to use the hidden features for various purposes. Although Finetuning algorithm does not need this, it is still provided for API consistence for other algorithms.
        """
        feature, activations = self.backbone(input, stage=stage, task_id=task_id)
        logits = self.heads(feature, task_id)
        return (
            logits if self.if_forward_func_return_logits_only else (logits, activations)
        )

    def aggregated_backbone_output(self, input: Tensor) -> Tensor:
        r"""Get the aggregated backbone output for the input data. All parts of backbones should be aggregated together.

        This output feature is used for measuring unlearning effectiveness. An aggregated output involving every corner of the backbone is needed to ensure the fairness of the metric.

        **Args:**
        - **input** (`Tensor`): The input tensor from data.

        **Returns:**
        - **output** (`Tensor`): the aggregated backbone output tensor.
        """
        feature = 0

        for i in self.seen_task_ids:
            feature_i = self.backbone(input, stage="train", task_id=i)[0]
            feature += feature_i
        feature = feature / len(self.seen_task_ids)

        return feature

    def configure_optimizers(self) -> Optimizer:
        r"""
        Configure optimizer hooks by Lightning.
        See [Lightning docs](https://lightning.ai/docs/pytorch/stable/common/lightning_module.html#configure-optimizers) for more details.
        """
        # finish partially initialised optimizer by specifying model parameters. The `parameters()` method of this `CLAlrogithm` (inherited from `LightningModule`) returns both backbone and heads parameters
        fully_initialised_optimizer = self.optimizer_t(params=self.parameters())

        if self.lr_scheduler_t:
            fully_initialised_lr_scheduler = self.lr_scheduler_t(
                optimizer=fully_initialised_optimizer
            )

            return {
                "optimizer": fully_initialised_optimizer,
                "lr_scheduler": {
                    "scheduler": fully_initialised_lr_scheduler,
                    "monitor": f"task_{self.task_id}/learning_curve/val/loss_cls",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }

        return {"optimizer": fully_initialised_optimizer}


class JointLearning(LightningModule):
    r"""The class of joint learning, inherited from `LightningModule`."""

    def __init__(
        self,
        backbone: CLBackbone,
        heads: HeadsTIL | HeadsCIL,
        optimizer: Optimizer,
        lr_scheduler: LRScheduler | None,
    ) -> None:
        r"""Initialise the joint learning algorithm with the network.

        **Args:**
        - **backbone** (`CLBackbone`): backbone network.
        - **heads** (`HeadsTIL` | `HeadsCIL`): output heads.
        - **optimizer** (`Optimizer`): the optimizer object (partially initialised).
        - **lr_scheduler** (`LRScheduler` | `None`): the learning rate scheduler for the optimizer. If `None`, no scheduler is used.
        """
        LightningModule.__init__(self)

        self.backbone: CLBackbone = backbone
        r"""Store the backbone network."""
        self.heads: HeadsTIL | HeadsCIL = heads
        r"""Store the output heads."""
        self.optimizer: Optimizer = optimizer
        r"""Store the optimizer object (partially initialised) for the backpropagation of task `self.task_id`. Will be equipped with parameters in `configure_optimizers()`."""
        self.lr_scheduler: LRScheduler | None = lr_scheduler
        r"""Store the learning rate scheduler for the optimizer. If `None`, no scheduler is used."""
        self.criterion = nn.CrossEntropyLoss()
        r"""The loss function bewteen the output logits and the target labels. Default is cross-entropy loss."""

        JointLearning.sanity_check(self)

    def sanity_check(self) -> None:
        r"""Check the sanity of the arguments.

        **Raises:**
        - **ValueError**: if the `output_dim` of backbone network is not equal to the `input_dim` of CL heads.
        """
        if self.backbone.output_dim != self.heads.input_dim:
            raise ValueError(
                "The output_dim of backbone network should be equal to the input_dim of CL heads!"
            )

    def get_val_task_id_from_dataloader_idx(self, dataloader_idx: int) -> int:
        r"""Get the validation task ID from the dataloader index.

        **Args:**
        - **dataloader_idx** (`int`): the dataloader index.

        **Returns:**
        - **test_task_id** (`str`): the test task ID.
        """
        dataset_val = self.trainer.datamodule.dataset_val
        test_task_id = list(dataset_val.keys())[dataloader_idx]
        return test_task_id

    def get_test_task_id_from_dataloader_idx(self, dataloader_idx: int) -> int:
        r"""Get the test task ID from the dataloader index.

        **Args:**
        - **dataloader_idx** (`int`): the dataloader index.

        **Returns:**
        - **test_task_id** (`str`): the test task ID.
        """
        dataset_test = self.trainer.datamodule.dataset_test
        test_task_id = list(dataset_test.keys())[dataloader_idx]
        return test_task_id

    def forward(self, input: Tensor, stage: str, task_id: int | None = None) -> Tensor:
        r"""The forward pass for data from task `task_id`. Note that it is nothing to do with `forward()` method in `nn.Module`. It works both for TIL and CIL.

        This forward pass does not accept input batch in different tasks. Please make sure the input batch is from the same task. If you want to use this forward pass for different tasks, please divide the input batch by tasks and call this forward pass for each task separately.

        **Args:**
        - **input** (`Tensor`): The input tensor from data.
        - **stage** (`str`): the stage of the forward pass, should be one of the following:
            1. 'train': training stage.
            2. 'validation': validation stage.
            3. 'test': testing stage.
        - **task_id** (`int`): the task ID where the data are from. If stage is 'train' or `validation`, it is usually from the current task `self.task_id`. If stage is 'test', it could be from any seen task. In TIL, the task IDs of test data are provided thus this argument can be used. In CIL, they are not provided, so it is just a placeholder for API consistence but never used, and best practices are not to provide this argument and leave it as the default value.

        **Returns:**
        - **logits** (`Tensor`): the output logits tensor.
        """
        feature, _ = self.backbone(input, stage=stage, task_id=task_id)
        logits = self.heads(feature, task_id)
        return logits

    def training_step(self, batch: Any) -> dict[str, Tensor]:
        r"""Training step for joint learning.

        **Args:**
        - **batch** (`Any`): a batch of training data.

        **Returns:**
        - **outputs** (`dict[str, Tensor]`): a dictionary contains loss and accuracy from this training step. Key (`str`) is the metrics name, value (`Tensor`) is the metrics. Must include the key 'loss' which is total loss in the case of automatic optimization, according to PyTorch Lightning docs.
        """
        x, y, t = batch  # train data are provided task ID in case of TIL

        loss_cls = 0.0
        acc = 0.0

        # classification loss
        for task_id in torch.unique(t):
            # divide the input batch by tasks
            idx_task = t == task_id
            x_task, y_task = x[idx_task], y[idx_task]

            # do the forward pass for each task separately
            logits_task = self.forward(
                x_task, stage="train", task_id=task_id.item()
            )  # use the corresponding head to get the logits
            loss_cls_task = self.criterion(logits_task, y_task)
            loss_cls = loss_cls + loss_cls_task

            acc_task = (logits_task.argmax(dim=1) == y_task).float().mean()
            acc = acc + acc_task

        # total loss
        loss = loss_cls

        return {
            "loss": loss,  # Return loss is essential for training step, or backpropagation will fail
            "loss_cls": loss_cls,
            "acc": acc,  # Return other metrics for lightning loggers callback to handle at `on_train_batch_end()`
        }

    def validation_step(
        self, batch: DataLoader, batch_idx: int, dataloader_idx: int = 0
    ) -> dict[str, Tensor]:
        r"""Validation step for joint learning. This is done task by task rather than mixing the tasks in batches.

        **Args:**
        - **batch** (`Any`): a batch of validation data.

        **Returns:**
        - **outputs** (`dict[str, Tensor]`): a dictionary contains loss and accuracy from this validation step. Key (`str`) is the metrics name, value (`Tensor`) is the metrics.
        """
        val_task_id = self.get_val_task_id_from_dataloader_idx(dataloader_idx)

        x, y = batch  # validation data are not provided task ID

        # the batch is from the same task, so no need to divide the input batch by tasks
        logits = self.forward(
            x, stage="validation", task_id=val_task_id
        )  # use the corresponding head to get the logits
        loss_cls = self.criterion(logits, y)
        acc = (logits.argmax(dim=1) == y).float().mean()

        # Return metrics for lightning loggers callback to handle at `on_validation_batch_end()`
        return {
            "loss_cls": loss_cls,
            "acc": acc,
        }

    def test_step(
        self, batch: DataLoader, batch_idx: int, dataloader_idx: int = 0
    ) -> dict[str, Tensor]:
        r"""Test step for joint learning. This is done task by task rather than mixing the tasks in batches.

        **Args:**
        - **batch** (`Any`): a batch of test data.

        **Returns:**
        - **outputs** (`dict[str, Tensor]`): a dictionary contains loss and accuracy from this test step. Key (`str`) is the metrics name, value (`Tensor`) is the metrics.
        """
        test_task_id = self.get_test_task_id_from_dataloader_idx(dataloader_idx)

        x, y = batch

        # the batch is from the same task, so no need to divide the input batch by tasks
        logits = self.forward(
            x, stage="test", task_id=test_task_id
        )  # use the corresponding head to get the logits
        loss_cls = self.criterion(logits, y)
        acc = (logits.argmax(dim=1) == y).float().mean()

        # Return metrics for lightning loggers callback to handle at `on_validation_batch_end()`
        return {
            "loss_cls": loss_cls,
            "acc": acc,
        }

    def configure_optimizers(self) -> Optimizer:
        r"""
        Configure optimizer hooks by Lightning.
        See [Lightning docs](https://lightning.ai/docs/pytorch/stable/common/lightning_module.html#configure-optimizers) for more details.
        """
        # finish partially initialised optimizer by specifying model parameters. The `parameters()` method of this `CLAlrogithm` (inherited from `LightningModule`) returns both backbone and heads parameters
        fully_initialised_optimizer = self.optimizer(params=self.parameters())

        if self.lr_scheduler:
            fully_initialised_lr_scheduler = self.lr_scheduler(
                optimizer=fully_initialised_optimizer
            )

            return {
                "optimizer": fully_initialised_optimizer,
                "lr_scheduler": {
                    "scheduler": fully_initialised_lr_scheduler,
                    "monitor": "learning_curve/val/loss_cls",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }

        return {"optimizer": fully_initialised_optimizer}
