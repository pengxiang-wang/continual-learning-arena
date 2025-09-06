r"""The submodule in `heads` for STL head."""

__all__ = ["HeadSTL"]

import logging

from torch import Tensor, nn

# always get logger for built-in logging in each module
pylogger = logging.getLogger(__name__)


class HeadSTL(nn.Module):
    r"""The output head for Single-Task Learning (STL)."""

    def __init__(self, input_dim: int) -> None:
        r"""Initializes STL head object.

        **Args:**
        - **input_dim** (`int`): the input dimension of the head. Must be equal to the `output_dim` of the connected backbone.
        """
        super().__init__()

        self.head: nn.Linear = None
        r"""STL output head. """

        self.input_dim: int = input_dim
        r"""Store the input dimension of the head. Used when creating new head."""

    def setup_task(self, num_classes: dict[int, int]) -> None:
        r"""Create the output head. This must be done before `forward()` is called.

        **Args:**
        - **num_classes** (`int`): the number of classes in the task.
        """
        self.head = nn.Linear(self.input_dim, num_classes)

    def forward(self, feature: Tensor) -> Tensor:
        r"""The forward pass for data.

        **Args:**
        - **feature** (`Tensor`): the feature tensor from the backbone network.

        **Returns:**
        - **logits** (`Tensor`): the output logits tensor.
        """
        logits = self.head(feature)

        return logits
        return logits
