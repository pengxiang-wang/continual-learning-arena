r"""The submodule in `heads` for DIL head."""

__all__ = ["HeadDIL"]

import logging

from torch import Tensor, nn

# always get logger for built-in logging in each module
pylogger = logging.getLogger(__name__)


class HeadDIL(nn.Module):
    r"""The output head for Domain-Incremental Learning (DIL)."""

    def __init__(self, input_dim: int) -> None:
        r"""Initializes DIL head object.

        **Args:**
        - **input_dim** (`int`): the input dimension of the head. Must be equal to the `output_dim` of the connected backbone.
        """
        super().__init__()

        self.head: nn.Linear = None
        r"""DIL output head. """

        self.input_dim: int = input_dim
        r"""Store the input dimension of the head. Used when creating new head."""

        self._if_head_setup: bool = False
        r"""Flag indicating whether the head has been set up."""

    def if_head_setup(self) -> bool:
        r"""Check whether the head has been set up.

        **Returns:**
        - **if_head_setup** (`bool`): whether the head has been set up.
        """
        return self._if_head_setup

    def get_head(self, task_id: int | None = None) -> nn.Linear:
        r"""Get the output head for DIL.

        **Args:**
        - **task_id** (`int` or `None`): the task ID where the data are from. This does not matter at all for DIL head as there is only one head for all tasks, so it is just a placeholder for API consistence with the TIL heads but never used. Best practices are not to provide this argument and leave it as the default value.

        **Returns:**
        - **head** (`nn.Linear`): the output head for DIL.
        """
        return self.head

    def setup_task(self, num_classes: dict[int, int]) -> None:
        r"""Create the output head. This must be done before `forward()` is called.

        **Args:**
        - **num_classes** (`int`): the number of classes in the task.
        """
        self.head = nn.Linear(self.input_dim, num_classes)
        self._if_head_setup = True

    def forward(self, feature: Tensor, task_id: int | None = None) -> Tensor:
        r"""The forward pass for data. The information of which `task_id` the data are from is not provided.

        **Args:**
        - **feature** (`Tensor`): the feature tensor from the backbone network.
        - **task_id** (`int` or `None`): the task ID where the data are from. In DIL, it is just a placeholder for API consistence with the TIL heads but never used. Best practices are not to provide this argument and leave it as the default value.

        **Returns:**
        - **logits** (`Tensor`): the output logits tensor.
        """
        logits = self.head(feature)

        return logits
