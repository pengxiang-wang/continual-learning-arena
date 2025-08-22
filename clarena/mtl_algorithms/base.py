r"""
The submodule in `mtl_algorithms` for MTL algorithm bases.
"""

__all__ = ["MTLAlgorithm"]

import logging

from lightning import LightningModule
from torch import Tensor, nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from clarena.backbones import Backbone
from clarena.heads import HeadsMTL

# always get logger for built-in logging in each module
pylogger = logging.getLogger(__name__)


class MTLAlgorithm(LightningModule):
    r"""The class of multi-task learning, inherited from `LightningModule`."""

    def __init__(
        self,
        backbone: Backbone,
        heads: HeadsMTL,
    ) -> None:
        r"""Initialize the MTL algorithm with the network.

        **Args:**
        - **backbone** (`Backbone`): backbone network.
        - **heads** (`HeadsMTL`): output heads.
        """
        super().__init__()

        self.backbone: Backbone = backbone
        r"""Store the backbone network."""
        self.heads: HeadsMTL = heads
        r"""Store the output heads."""
        self.optimizer: Optimizer
        r"""Store the optimizer object (partially initialized) for the backpropagation of task `self.task_id`. Will be equipped with parameters in `configure_optimizers()`."""
        self.lr_scheduler: LRScheduler | None
        r"""Store the learning rate scheduler for the optimizer. If `None`, no scheduler is used."""
        self.criterion = nn.CrossEntropyLoss()
        r"""The loss function bewteen the output logits and the target labels. Default is cross-entropy loss."""

        self.if_forward_func_return_logits_only: bool = False
        r"""Whether the `forward()` method returns logits only. If `False`, it returns a dictionary containing logits and other information. Default is `False`."""

        MTLAlgorithm.sanity_check(self)

    def sanity_check(self) -> None:
        r"""Check the sanity of the arguments."""

        # check backbone and heads compatibility
        if self.backbone.output_dim != self.heads.input_dim:
            raise ValueError(
                "The output_dim of backbone network should be equal to the input_dim of CL heads!"
            )

    def setup_tasks(
        self,
        optimizer: Optimizer,
        lr_scheduler: LRScheduler | None,
    ) -> None:
        r"""Setup tasks for the MTL algorithm. This must be done before `forward()` method is called.

        **Args:**
        - **optimizer** (`Optimizer`): the optimizer object (partially initialized).
        - **lr_scheduler** (`LRScheduler` | None): the learning rate scheduler for the optimizer. If `None`, no scheduler is used.
        """
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler

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

    def forward(self, input: Tensor, task_ids: int | Tensor, stage: str) -> Tensor:
        r"""The forward pass for data. Note that it is nothing to do with `forward()` method in `nn.Module`.

        This forward pass does not accept input batch in different tasks. Please make sure the input batch is from the same task. If you want to use this forward pass for different tasks, please divide the input batch by tasks and call this forward pass for each task separately.

        **Args:**
        - **input** (`Tensor`): The input tensor from data.
        - **task_ids** (`int` | `Tensor`): the task ID(s) for the input data. If the input batch is from the same task, this can be a single integer.
        - **stage** (`str`): the stage of the forward pass, should be one of the following:
            1. 'train': training stage.
            2. 'validation': validation stage.
            3. 'test': testing stage.

        **Returns:**
        - **logits** (`Tensor`): the output logits tensor.
        """
        feature, activations = self.backbone(input, stage=stage)
        logits = self.heads(feature, task_ids)
        return (
            logits if self.if_forward_func_return_logits_only else (logits, activations)
        )

    def configure_optimizers(self) -> Optimizer:
        r"""
        Configure optimizer hooks by Lightning.
        See [Lightning docs](https://lightning.ai/docs/pytorch/stable/common/lightning_module.html#configure-optimizers) for more details.
        """
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
