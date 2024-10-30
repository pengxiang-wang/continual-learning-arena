"""
The submodule in `backbones` for CL backbone network bases.
"""

__all__ = ["CLBackbone"]

import logging

from torch import Tensor, nn
from typing_extensions import override

# always get logger for built-in logging in each module
pylogger = logging.getLogger(__name__)


class CLBackbone(nn.Module):
    """The base class of continual learning backbone networks, inherited from `torch.nn.Module`."""

    def __init__(self, output_dim: int) -> None:
        """Initialise the CL backbone network.

        **Args:**
        - **output_dim** (`int`): The output dimension which connects to CL output heads. The `input_dim` of output heads are expected to be the same as this `output_dim`.
        """
        super().__init__()

        self.output_dim = output_dim
        """Store the output dimension of the backbone network."""

        self.task_id: int
        """Task ID counter indicating which task is being processed. Self updated during the task loop."""

    def setup_task_id(self, task_id: int) -> None:
        """Set up which task's dataset the CL experiment is on. This must be done before `forward()` method is called.

        **Args:**
        - **task_id** (`int`): the target task ID.
        """
        self.task_id = task_id

    @override
    def forward(self, input: Tensor, task_id: int | None = None) -> Tensor:
        """The forward pass for data from task `task_id`. In some backbones, the forward pass might be different for different tasks.

        **Args:**
        - **input** (`Tensor`): The input tensor from data.
        - **task_id** (`int`): the task ID where the data are from. In TIL, the task IDs of test data are provided thus this argument can be used. In CIL, they are not provided, so it is just a placeholder for API consistence but never used, and best practices are not to provide this argument and leave it as the default value.

        **Returns:**
        - The output feature tensor to be passed into heads.
        """
