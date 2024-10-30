"""The submodule in `cl_heads` for CIL heads."""

__all__ = ["HeadsCIL"]

import torch
from torch import Tensor, nn


class HeadsCIL(nn.Module):
    """The output heads for Class-Incremental Learning (CIL). Head of all classes from CIL tasks takes the output from backbone network and forwards it into logits for predicting classes of all tasks."""

    def __init__(self, input_dim: int) -> None:
        """Initializes a CIL heads object with no heads.

        **Args:**
        - **input_dim** (`int`): the input dimension of the heads. Must be equal to the `output_dim` of the connected backbone.
        """
        super().__init__()

        self.heads: nn.ModuleDict = {}  # initially no heads
        """The TIL output heads are stored independently in a ModuleDict (rather than dict just to make sure the parameters can be recorded in model summaries). Keys are task IDs and values are the corresponding `nn.Linear` heads."""
        self.input_dim: int = input_dim
        """The input dimension of the heads. Used when creating new heads."""

        self.task_id: int
        """Task ID counter indicating which task is being processed. Self updated during the task loop."""

    def setup_task_id(self, task_id: int, num_classes_t: int) -> None:
        """Create the output head when task `task_id` arrives if there's no. This must be done before `forward()` is called.

        **Args:**
        - **task_id** (`int`): the target task ID.
        - **num_classes_t** (`int`): the number of classes in the task.
        """
        self.task_id = task_id
        if self.task_id not in self.heads.keys():
            self.heads[self.task_id] = nn.Linear(self.input_dim, num_classes_t)

    def forward(self, feature: Tensor, task_id: int | None = None) -> Tensor:
        """The forward pass for data. The information of which `task_id` the data are from is not provided. The head for all classes is selected and the feature is passed.

        **Args:**
        - **feature** (`Tensor`): the feature tensor from the backbone network.
        - **task_id** (`int` or `None`): the task ID where the data are from. In CIL, it is just a placeholder for API consistence with the TIL heads but never used. Best practices are not to provide this argument and leave it as the default value.

        **Returns:**
        - The output logits tensor.
        """
        logits = torch.cat(
            [self.heads[t](feature) for t in range(1, self.task_id + 1)], dim=-1
        )  # concatenate logits of classes from all heads

        return logits
