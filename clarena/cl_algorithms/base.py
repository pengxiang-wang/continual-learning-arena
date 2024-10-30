"""
The submodule in `cl_algorithms` for CL algorithm bases.
"""

__all__ = ["CLAlgorithm"]

import logging

from lightning import LightningModule
from torch import Tensor, nn
from torch.optim import Optimizer
from typing_extensions import override

from clarena.backbones import CLBackbone
from clarena.cl_heads import HeadsCIL, HeadsTIL

# always get logger for built-in logging in each module
pylogger = logging.getLogger(__name__)


class CLAlgorithm(LightningModule):
    """
    The base class of continual learning algorithms, inherited from `LightningModule`.
    """

    def __init__(
        self,
        backbone: CLBackbone,
        heads: HeadsTIL | HeadsCIL,
    ) -> None:
        """Initialise the CL algorithm with the network.

        Args:
        - **backbone** (`CLBackbone`): backbone network.
        - **heads** (`HeadsTIL` | `HeadsCIL`): output heads.
        """
        super().__init__()

        self.backbone: CLBackbone = backbone
        """Store the backbone network."""
        self.heads: HeadsTIL | HeadsCIL = heads
        """Store the output heads."""
        self.optimizer: Optimizer
        """Store the optimizer object (partially initialised) for the backpropagation of task `self.task_id`. Will be equipped with parameters in `configure_optimizers()`."""
        self.criterion = nn.CrossEntropyLoss()
        """The loss function bewteen the output logits and the target labels. Default is cross-entropy loss."""

        self.task_id: int
        """Task ID counter indicating which task is being processed. Self updated during the task loop."""

        self.sanity_check()

    def sanity_check(self) -> None:
        """Check the sanity of the arguments.

        **Raises:**
        - **ValueError**: if the `output_dim` of backbone network is not equal to the `input_dim` of CL heads.
        """
        if self.backbone.output_dim != self.heads.input_dim:
            raise ValueError(
                "The output_dim of backbone network should be equal to the input_dim of CL heads!"
            )

    def setup_task_id(
        self, task_id: int, num_classes_t: int, optimizer: Optimizer
    ) -> None:
        """Set up which task's dataset the CL experiment is on. This must be done before `forward()` method is called.

        **Args:**
        - **task_id** (`int`): the target task ID.
        - **num_classes_t** (`int`): the number of classes in the task.
        - **optimizer** (`Optimizer`): the optimizer object (partially initialised) for the task `self.task_id`.
        """
        self.task_id = task_id
        self.heads.setup_task_id(task_id, num_classes_t)
        self.optimizer = optimizer

    @override
    def forward(self, input: Tensor, task_id: int):
        """The default forward pass for data from task `task_id`. Note that it is nothing to do with `forward()` method in `nn.Module`.

        **Args:**
        - **input** (`Tensor`): The input tensor from data.
        - **task_id** (`int`): the task ID where the data are from.

        Returns:
        - The output logits tensor.
        """
        feature = self.backbone(input, task_id)
        logits = self.heads(feature, task_id)
        return logits

    def configure_optimizers(self) -> Optimizer:
        """
        Configure optimizer hooks by Lightning.
        See [Lightning docs](https://lightning.ai/docs/pytorch/stable/common/lightning_module.html#configure-optimizers) for more details.
        """
        # finish partially initialised optimizer by specifying model parameters. The `parameters()` method of this `CLAlrogithm` (inherited from `LightningModule`) returns both backbone and heads parameters
        return self.optimizer(params=self.parameters())
