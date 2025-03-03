r"""The submodule in `cl_heads` for TIL heads."""

__all__ = ["HeadsTIL"]

from torch import Tensor, nn


class HeadsTIL(nn.Module):
    r"""The output heads for Task-Incremental Learning (TIL). Independent head assigned to each TIL task takes the output from backbone network and forwards it into logits for predicting classes of the task."""

    def __init__(self, input_dim: int) -> None:
        r"""Initializes TIL heads object with no heads.

        **Args:**
        - **input_dim** (`int`): the input dimension of the heads. Must be equal to the `output_dim` of the connected backbone.
        """
        nn.Module.__init__(self)

        self.heads: nn.ModuleDict = nn.ModuleDict()  # initially no heads
        r"""TIL output heads are stored independently in a `ModuleDict`. Keys are task IDs (string type) and values are the corresponding `nn.Linear` heads. We use `ModuleDict` rather than `dict` to make sure `LightningModule` can track these model parameters for the purpose of, such as automatically to device, recorded in model summaries.
        
        Note that the task IDs must be string type in order to let `LightningModule` identify this part of the model. """
        self.input_dim: int = input_dim
        r"""Store the input dimension of the heads. Used when creating new heads."""

        self.task_id: int
        r"""Task ID counter indicating which task is being processed. Self updated during the task loop."""

    def setup_task_id(self, task_id: int, num_classes_t: int) -> None:
        r"""Create the output head when task `task_id` arrives if there's no. This must be done before `forward()` is called.

        **Args:**
        - **task_id** (`int`): the target task ID.
        - **num_classes_t** (`int`): the number of classes in the task.
        """
        self.task_id = task_id
        if self.task_id not in self.heads.keys():
            self.head_t = nn.Linear(self.input_dim, num_classes_t)
            self.heads[f"{self.task_id}"] = self.head_t

    def forward(self, feature: Tensor, task_id: int) -> Tensor:
        r"""The forward pass for data from task `task_id`. A head is selected according to the task_id and the feature is passed through the head.

        **Args:**
        - **feature** (`Tensor`): the feature tensor from the backbone network.
        - **task_id** (`int`): the task ID where the data are from, which is provided by task-incremental setting.

        **Returns:**
        - **logits** (`Tensor`): the output logits tensor.
        """

        head_t = self.heads[f"{task_id}"]
        logits = head_t(feature)

        return logits
