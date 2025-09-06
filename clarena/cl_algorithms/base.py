r"""
The submodule in `cl_algorithms` for continual learning algorithm bases.
"""

__all__ = ["CLAlgorithm", "UnlearnableCLAlgorithm"]

import logging
from typing import Any

from lightning import LightningModule
from torch import Tensor, nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from clarena.backbones import CLBackbone
from clarena.heads import HeadsCIL, HeadsTIL

# always get logger for built-in logging in each module
pylogger = logging.getLogger(__name__)


class CLAlgorithm(LightningModule):
    r"""The base class of continual learning algorithms."""

    def __init__(
        self,
        backbone: CLBackbone,
        heads: HeadsTIL | HeadsCIL,
        non_algorithmic_hparams: dict[str, Any] = {},
    ) -> None:
        r"""
        **Args:**
        - **backbone** (`CLBackbone`): backbone network.
        - **heads** (`HeadsTIL` | `HeadsCIL`): output heads.
        - **non_algorithmic_hparams** (`dict[str, Any]`): non-algorithmic hyperparameters that are not related to the algorithm itself are passed to this `LightningModule` object from the config, such as optimizer and learning rate scheduler configurations. They are saved for Lightning APIs from `save_hyperparameters()` method. This is useful for the experiment configuration and reproducibility.
        """
        super().__init__()
        self.save_hyperparameters(non_algorithmic_hparams)

        # components
        self.backbone: CLBackbone = backbone
        r"""The backbone network."""
        self.heads: HeadsTIL | HeadsCIL = heads
        r"""The output heads."""
        self.optimizer_t: Optimizer
        r"""Optimizer (partially initialized) for the current task `self.task_id`. Will be equipped with parameters in `configure_optimizers()`."""
        self.lr_scheduler_t: LRScheduler | None
        r"""Learning rate scheduler for the optimizer of the current task `self.task_id`. If `None`, no scheduler is used."""
        self.criterion = nn.CrossEntropyLoss()
        r"""Loss function between the output logits and the target labels. Default is cross-entropy loss."""

        self.if_forward_func_return_logits_only: bool = False
        r"""Whether the `forward()` method returns logits only. If `False`, it returns a dictionary containing logits and other information. Default is `False`."""

        # task ID control
        self.task_id: int
        r"""Task ID counter indicating which task is being processed. Self updated during the task loop. Valid from 1 to `cl_dataset.num_tasks`."""
        self.processed_task_ids: list[int] = []
        r"""Task IDs that have been processed."""

        CLAlgorithm.sanity_check(self)

    def sanity_check(self) -> None:
        r"""Sanity check."""

        # check backbone and heads compatibility
        if self.backbone.output_dim != self.heads.input_dim:
            raise ValueError(
                "The output_dim of the backbone must equal the input_dim of the CL heads."
            )

    def setup_task_id(
        self,
        task_id: int,
        num_classes: int,
        optimizer: Optimizer,
        lr_scheduler: LRScheduler | None,
    ) -> None:
        r"""Set up which task the CL experiment is on. This must be done before `forward()` method is called.

        **Args:**
        - **task_id** (`int`): the target task ID.
        - **num_classes** (`int`): the number of classes in the task.
        - **optimizer** (`Optimizer`): the optimizer object (partially initialized) for the task.
        - **lr_scheduler** (`LRScheduler` | `None`): the learning rate scheduler for the optimizer. If `None`, no scheduler is used.
        """
        self.task_id = task_id
        self.processed_task_ids.append(task_id)
        self.backbone.setup_task_id(task_id=task_id)
        self.heads.setup_task_id(task_id, num_classes)
        self.optimizer_t = optimizer
        self.lr_scheduler_t = lr_scheduler

    def get_test_task_id_from_dataloader_idx(self, dataloader_idx: int) -> int:
        r"""Get the test task ID from the dataloader index.

        **Args:**
        - **dataloader_idx** (`int`): the dataloader index.

        **Returns:**
        - **test_task_id** (`int`): the test task ID.
        """
        dataset_test = self.trainer.datamodule.dataset_test
        test_task_id = list(dataset_test.keys())[dataloader_idx]
        return test_task_id

    def set_forward_func_return_logits_only(
        self, forward_func_return_logits_only: bool
    ) -> None:
        r"""Set whether the `forward()` method returns logits only. This is useful for some CL algorithms that require the forward function to return logits only, such as FG-AdaHAT.

        **Args:**
        - **forward_func_return_logits_only** (`bool`): whether the `forward()` method returns logits only. If `False`, it returns a dictionary containing logits and other information.
        """
        self.if_forward_func_return_logits_only = forward_func_return_logits_only

    def preceding_layer(self, layer_name: str) -> nn.Module | None:
        r"""Get the preceding layer of the given layer (including backbone and output heads). If the given layer is the first layer, return `None`.

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
        r"""Get the next layer of the given layer (including backbone and output heads). If the given layer is the last layer, return `None`.

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
        feature, activations = self.backbone(input, stage=stage, task_id=task_id)
        logits = self.heads(feature, task_id)
        return (
            logits if self.if_forward_func_return_logits_only else (logits, activations)
        )

    def configure_optimizers(self) -> Optimizer:
        r"""Configure optimizer hooks by Lightning. See [Lightning docs](https://lightning.ai/docs/pytorch/stable/common/lightning_module.html#configure-optimizers) for more details."""
        # finish partially initialized optimizer by specifying model parameters. The `parameters()` method of this `CLAlgorithm` (inherited from `LightningModule`) returns both backbone and heads parameters
        fully_initialized_optimizer = self.optimizer_t(params=self.parameters())

        if self.lr_scheduler_t:
            fully_initialized_lr_scheduler = self.lr_scheduler_t(
                optimizer=fully_initialized_optimizer
            )

            return {
                "optimizer": fully_initialized_optimizer,
                "lr_scheduler": {
                    "scheduler": fully_initialized_lr_scheduler,
                    "monitor": f"task_{self.task_id}/learning_curve/val/loss_cls",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }

        return {"optimizer": fully_initialized_optimizer}


class UnlearnableCLAlgorithm(CLAlgorithm):
    r"""The base class of unlearnable continual learning algorithms."""

    def __init__(
        self,
        backbone: CLBackbone,
        heads: HeadsTIL | HeadsCIL,
        non_algorithmic_hparams: dict[str, Any] = {},
    ) -> None:
        r"""
        **Args:**
        - **backbone** (`CLBackbone`): backbone network.
        - **heads** (`HeadsTIL` | `HeadsCIL`): output heads.
        - **non_algorithmic_hparams** (`dict[str, Any]`): non-algorithmic hyperparameters that are not related to the algorithm itself are passed to this `LightningModule` object from the config, such as optimizer and learning rate scheduler configurations. They are saved for Lightning APIs from `save_hyperparameters()` method. This is useful for the experiment configuration and reproducibility.
        """
        super().__init__(
            backbone=backbone,
            heads=heads,
            non_algorithmic_hparams=non_algorithmic_hparams,
        )

        self.unlearning_task_ids: list[int]
        r"""The list of task IDs that are requested to be unlearned after training `self.task_id`."""

        self.unlearned_task_ids: set[int] = set()
        r"""The list of task IDs that have been unlearned in the experiment."""

        UnlearnableCLAlgorithm.sanity_check(self)

    def sanity_check(self) -> None:
        r"""Sanity check."""

    def aggregated_backbone_output(self, input: Tensor) -> Tensor:
        r"""Get the aggregated backbone output for the input data. All parts of backbones should be aggregated together.

        This output feature is used for measuring unlearning metrics, such as Distribution Distance (DD). An aggregated output involving every part of the backbone is needed to ensure the fairness of the metric.

        **Args:**
        - **input** (`Tensor`): the input tensor from data.

        **Returns:**
        - **output** (`Tensor`): the aggregated backbone output tensor.
        """
        feature = 0

        for i in self.processed_task_ids:
            feature_i = self.backbone(input, stage="train", task_id=i)[0]
            feature += feature_i
        feature = feature / len(self.processed_task_ids)

        return feature
