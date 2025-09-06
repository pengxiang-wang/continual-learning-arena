r"""
The submodule in `mtl_algorithms` for multi-task learning algorithm bases.
"""

__all__ = ["MTLAlgorithm"]

import logging
from typing import Any

from lightning import LightningModule
from torch import Tensor, nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from clarena.backbones import Backbone
from clarena.heads import HeadsMTL

# always get logger for built-in logging in each module
pylogger = logging.getLogger(__name__)


class MTLAlgorithm(LightningModule):
    r"""The base class of multi-task learning algorithms."""

    def __init__(
        self,
        backbone: Backbone,
        heads: HeadsMTL,
        non_algorithmic_hparams: dict[str, Any] = {},
    ) -> None:
        r"""
        **Args:**
        - **backbone** (`Backbone`): backbone network.
        - **heads** (`HeadsMTL`): output heads.
        - **non_algorithmic_hparams** (`dict[str, Any]`): non-algorithmic hyperparameters that are not related to the algorithm itself are passed to this `LightningModule` object from the config, such as optimizer and learning rate scheduler configurations. They are saved for Lightning APIs from `save_hyperparameters()` method. This is useful for the experiment configuration and reproducibility.
        """
        super().__init__()
        self.save_hyperparameters(non_algorithmic_hparams)

        # components
        self.backbone: Backbone = backbone
        r"""The backbone network."""
        self.heads: HeadsMTL = heads
        r"""The output heads."""
        self.optimizer: Optimizer
        r"""Optimizer (partially initialized) for the backpropagation. Will be equipped with parameters in `configure_optimizers()`."""
        self.lr_scheduler: LRScheduler | None
        r"""The learning rate scheduler for the optimizer. If `None`, no scheduler is used."""
        self.criterion = nn.CrossEntropyLoss()
        r"""The loss function bewteen the output logits and the target labels. Default is cross-entropy loss."""

        self.if_forward_func_return_logits_only: bool = False
        r"""Whether the `forward()` method returns logits only. If `False`, it returns a dictionary containing logits and other information. Default is `False`."""

        MTLAlgorithm.sanity_check(self)

    def sanity_check(self) -> None:
        r"""Sanity check."""

        # check backbone and heads compatibility
        if self.backbone.output_dim != self.heads.input_dim:
            raise ValueError(
                "The output_dim of backbone network should be equal to the input_dim of MTL heads!"
            )

    def setup_tasks(
        self,
        task_ids: list[int],
        num_classes: dict[int, int],
        optimizer: Optimizer,
        lr_scheduler: LRScheduler | None,
    ) -> None:
        r"""Set up tasks for the MTL algorithm. This must be done before `forward()` method is called.

        **Args:**
        - **task_ids** (`list[int]`): the list of task IDs.
        - **num_classes** (`dict[int, int]`): a dictionary mapping each task ID to its number of classes.
        - **optimizer** (`Optimizer`): the optimizer object (partially initialized).
        - **lr_scheduler** (`LRScheduler` | None): the learning rate scheduler for the optimizer. If `None`, no scheduler is used.
        """
        self.heads.setup_tasks(task_ids=task_ids, num_classes=num_classes)
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler

    def get_val_task_id_from_dataloader_idx(self, dataloader_idx: int) -> int:
        r"""Get the validation task ID from the dataloader index.

        **Args:**
        - **dataloader_idx** (`int`): the dataloader index.

        **Returns:**
        - **val_task_id** (`int`): the validation task ID.
        """
        dataset_val = self.trainer.datamodule.dataset_val
        val_task_id = list(dataset_val.keys())[dataloader_idx]
        return val_task_id

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

    def forward(self, input: Tensor, task_ids: int | Tensor, stage: str) -> Tensor:
        r"""The forward pass for data. Note that it is nothing to do with `forward()` method in `nn.Module`. This definition provides a template that many MTL algorithm including the vanilla JointLearning algorithm use.

        This forward pass does not accept input batch in different tasks. Please make sure the input batch is from the same task. If you want to use this forward pass for different tasks, please divide the input batch by tasks and call this forward pass for each task separately.

        **Args:**
        - **input** (`Tensor`): The input tensor from data.
        - **task_ids** (`int` | `Tensor`): the task ID(s) for the input data. If the input batch is from the same task, this can be a single integer.
        - **stage** (`str`): the stage of the forward pass; one of:
            1. 'train': training stage.
            2. 'validation': validation stage.
            3. 'test': testing stage.

        **Returns:**
        - **logits** (`Tensor`): the output logits tensor.
        - **activations** (`dict[str, Tensor]`): the hidden features (after activation) in each weighted layer. Key (`str`) is the weighted layer name, value (`Tensor`) is the hidden feature tensor. This is used for the continual learning algorithms that need to use the hidden features for various purposes.
        """
        feature, activations = self.backbone(input, stage=stage)
        logits = self.heads(feature, task_ids)
        return (
            logits if self.if_forward_func_return_logits_only else (logits, activations)
        )

    def configure_optimizers(self) -> Optimizer:
        r"""Configure optimizer hooks by Lightning. See [Lightning docs](https://lightning.ai/docs/pytorch/stable/common/lightning_module.html#configure-optimizers) for more details."""
        # finish partially initialized optimizer by specifying model parameters. The `parameters()` method of this `MTLAlgorithm` (inherited from `LightningModule`) returns both backbone and heads parameters
        fully_initialized_optimizer = self.optimizer(params=self.parameters())

        if self.lr_scheduler:
            fully_initialized_lr_scheduler = self.lr_scheduler(
                optimizer=fully_initialized_optimizer
            )

            return {
                "optimizer": fully_initialized_optimizer,
                "lr_scheduler": {
                    "scheduler": fully_initialized_lr_scheduler,
                    "monitor": "learning_curve/val/loss_cls",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }

        return {"optimizer": fully_initialized_optimizer}
