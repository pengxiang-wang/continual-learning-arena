"""The submodule in `cl_heads` for TIL heads."""

__all__ = ["HeadsTIL"]

from torch import Tensor, nn


class HeadsTIL(nn.Module):
    """The output heads for Task-Incremental Learning (TIL). Independent head assigned to each TIL task takes the output from backbone network and forwards it into logits for predicting classes of the task."""

    def __init__(self, input_dim: int) -> None:
        """Initializes TIL heads object with no heads.

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

    def forward(self, feature: Tensor, task_id: int) -> Tensor:
        """The forward pass for data from task `task_id`. A head is selected according to the task_id and the feature is passed through the head.

        **Args:**
        - **feature** (`Tensor`): the feature tensor from the backbone network.
        - **task_id** (`int`): the task ID where the data are from, which is provided by task-incremental setting.

        **Returns:**
        - The output logits tensor.
        """

        head_t = self.heads[task_id]
        logits = head_t(feature)

        return logits
