r"""
The submodule in `stl_algorithms` for single-task learning algorithm bases.
"""

__all__ = ["STLAlgorithm"]

import logging
from typing import Any

from lightning import LightningModule
from torch import Tensor, nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from clarena.backbones import Backbone
from clarena.heads import HeadSTL

# always get logger for built-in logging in each module
pylogger = logging.getLogger(__name__)


class STLAlgorithm(LightningModule):
    r"""The base class of single-task learning algorithms."""

    def __init__(
        self,
        backbone: Backbone,
        head: HeadSTL,
        non_algorithmic_hparams: dict[str, Any] = {},
    ) -> None:
        r"""
        **Args:**
        - **backbone** (`Backbone`): backbone network.
        - **head** (`HeadsSTL`): output head.
        - **non_algorithmic_hparams** (`dict[str, Any]`): non-algorithmic hyperparameters that are not related to the algorithm itself are passed to this `LightningModule` object from the config, such as optimizer and learning rate scheduler configurations. They are saved for Lightning APIs from `save_hyperparameters()` method. This is useful for the experiment configuration and reproducibility.
        """
        super().__init__()
        self.save_hyperparameters(non_algorithmic_hparams)

        self.backbone: Backbone = backbone
        r"""The backbone network."""
        self.head: HeadSTL = head
        r"""The output head."""
        self.optimizer: Optimizer
        r"""Optimizer (partially initialized). Will be equipped with parameters in `configure_optimizers()`."""
        self.lr_scheduler: LRScheduler | None
        r"""The learning rate scheduler for the optimizer. If `None`, no scheduler is used."""
        self.criterion = nn.CrossEntropyLoss()
        r"""The loss function bewteen the output logits and the target labels. Default is cross-entropy loss."""

        self.if_forward_func_return_logits_only: bool = False
        r"""Whether the `forward()` method returns logits only. If `False`, it returns a dictionary containing logits and other information. Default is `False`."""

        STLAlgorithm.sanity_check(self)

    def sanity_check(self) -> None:
        r"""Sanity check."""

        # check backbone and heads compatibility
        if self.backbone.output_dim != self.head.input_dim:
            raise ValueError(
                "The output_dim of backbone network should be equal to the input_dim of STL head!"
            )

    def setup_task(
        self,
        num_classes: int,
        optimizer: Optimizer,
        lr_scheduler: LRScheduler | None,
    ) -> None:
        r"""Setup the components for the STL algorithm. This must be done before `forward()` method is called.

        **Args:**
        - **num_classes** (`int`): the number of classes for the single-task learning.
        - **optimizer** (`Optimizer`): the optimizer object (partially initialized).
        - **lr_scheduler** (`LRScheduler` | None): the learning rate scheduler for the optimizer. If `None`, no scheduler is used.
        """
        self.head.setup_task(num_classes=num_classes)

        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler

    def forward(self, input: Tensor, stage: str) -> Tensor:
        r"""The forward pass for data from task `task_id`. Note that it is nothing to do with `forward()` method in `nn.Module`. This definition provides a template that many STL algorithm including the vanilla SingleLearning algorithm use.

        **Args:**
        - **input** (`Tensor`): The input tensor from data.
        - **stage** (`str`): the stage of the forward pass; one of:
            1. 'train': training stage.
            2. 'validation': validation stage.
            3. 'test': testing stage.

        **Returns:**
        - **logits** (`Tensor`): the output logits tensor.
        - **activations** (`dict[str, Tensor]`): the hidden features (after activation) in each weighted layer. Key (`str`) is the weighted layer name, value (`Tensor`) is the hidden feature tensor. This is used for the continual learning algorithms that need to use the hidden features for various purposes.
        """
        feature, activations = self.backbone(input, stage=stage)
        logits = self.head(feature)
        return (
            logits if self.if_forward_func_return_logits_only else (logits, activations)
        )

    def configure_optimizers(self) -> Optimizer:
        r"""Configure optimizer hooks by Lightning. See [Lightning docs](https://lightning.ai/docs/pytorch/stable/common/lightning_module.html#configure-optimizers) for more details."""
        # finish partially initialized optimizer by specifying model parameters. The `parameters()` method of this `STLAlgorithm` (inherited from `LightningModule`) returns both backbone and head parameters
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
