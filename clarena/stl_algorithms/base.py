r"""
The submodule in `stl_algorithms` for single-task learning algorithm bases.
"""

__all__ = ["STLAlgorithm"]

import logging

from lightning import LightningModule
from torch import Tensor, nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from clarena.backbones import Backbone
from clarena.heads import HeadsMTL

# always get logger for built-in logging in each module
pylogger = logging.getLogger(__name__)


class STLAlgorithm(LightningModule):
    r"""The class of single-task learning."""

    def __init__(
        self,
        backbone: Backbone,
        head: HeadsMTL,
    ) -> None:
        r"""Initialize the multi-task learning algorithm with the network.

        **Args:**
        - **backbone** (`Backbone`): backbone network.
        - **head** (`HeadsTIL` | `HeadsCIL`): output head.
        """
        super().__init__()

        self.backbone: Backbone = backbone
        r"""Store the backbone network."""
        self.head: HeadsMTL = head
        r"""Store the output head."""
        self.optimizer: Optimizer
        r"""Store the optimizer object (partially initialized) for the backpropagation of task `self.task_id`. Will be equipped with parameters in `configure_optimizers()`."""
        self.lr_scheduler: LRScheduler | None
        r"""Store the learning rate scheduler for the optimizer. If `None`, no scheduler is used."""
        self.criterion = nn.CrossEntropyLoss()
        r"""The loss function bewteen the output logits and the target labels. Default is cross-entropy loss."""

        self.if_forward_func_return_logits_only: bool = False
        r"""Whether the `forward()` method returns logits only. If `False`, it returns a dictionary containing logits and other information. Default is `False`."""

        STLAlgorithm.sanity_check(self)

    def sanity_check(self) -> None:
        r"""Check the sanity of the arguments."""
        if self.backbone.output_dim != self.head.input_dim:
            raise ValueError(
                "The output_dim of backbone network should be equal to the input_dim of CL head!"
            )

    def setup_task(
        self,
        optimizer: Optimizer,
        lr_scheduler: LRScheduler | None,
    ) -> None:
        r"""Setup the components for the STL algorithm. This must be done before `forward()` method is called.

        **Args:**
        - **optimizer** (`Optimizer`): the optimizer object (partially initialized).
        - **lr_scheduler** (`LRScheduler` | None): the learning rate scheduler for the optimizer. If `None`, no scheduler is used.
        """
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler

    def forward(self, input: Tensor, stage: str) -> Tensor:
        r"""The forward pass for data from task `task_id`. Note that it is nothing to do with `forward()` method in `nn.Module`.

        **Args:**
        - **input** (`Tensor`): The input tensor from data.
        - **stage** (`str`): the stage of the forward pass, should be one of the following:
            1. 'train': training stage.
            2. 'validation': validation stage.
            3. 'test': testing stage.

        **Returns:**
        - **logits** (`Tensor`): the output logits tensor.
        """
        feature, activations = self.backbone(input, stage=stage)
        logits = self.head(feature)
        return (
            logits if self.if_forward_func_return_logits_only else (logits, activations)
        )

    def configure_optimizers(self) -> Optimizer:
        r"""
        Configure optimizer hooks by Lightning.
        See [Lightning docs](https://lightning.ai/docs/pytorch/stable/common/lightning_module.html#configure-optimizers) for more details.
        """
        # finish partially initialized optimizer by specifying model parameters. The `parameters()` method of this `MTLAlgorithm` (inherited from `LightningModule`) returns both backbone and head parameters
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
