r"""The submodule in `heads` for MTL heads."""

__all__ = ["HeadsMTL"]

import logging

import torch
from torch import Tensor, nn

# always get logger for built-in logging in each module
pylogger = logging.getLogger(__name__)


class HeadsMTL(nn.Module):
    r"""The output heads for Multi-Task Learning (MTL). Independent head assigned to each task takes the output from backbone network and forwards it into logits for predicting classes of the task."""

    def __init__(self, input_dim: int) -> None:
        r"""Initializes MTL heads object.

        **Args:**
        - **input_dim** (`int`): the input dimension of the heads. Must be equal to the `output_dim` of the connected backbone.
        """
        super().__init__()

        self.heads: nn.ModuleDict = nn.ModuleDict()
        r"""MTL output heads are stored independently in a `ModuleDict`. Keys are task IDs and values are the corresponding `nn.Linear` heads. We use `ModuleDict` rather than `dict` to make sure `LightningModule` can track these model parameters for the purpose of, such as automatically to device, recorded in model summaries.
        
        Note that the task IDs must be string type in order to let `LightningModule` identify this part of the model. """

        self.input_dim: int = input_dim
        r"""Store the input dimension of the heads. Used when creating new heads."""

    def setup_tasks(self, task_ids: list[int], num_classes: dict[int, int]) -> None:
        r"""Create the output heads. This must be done before `forward()` is called.

        **Args:**
        - **task_id** (`list[int]`): the target task IDs.
        - **num_classes** (`dict[int, int]`): the number of classes in each task. Keys are task IDs and values are the number of classes for the corresponding task.
        """
        for task_id in task_ids:
            self.heads[f"{task_id}"] = nn.Linear(self.input_dim, num_classes[task_id])

    def get_head(self, task_id: int) -> nn.Linear:
        r"""Get the output head for task `task_id`.

        **Args:**
        - **task_id** (`int`): the target task ID.

        **Returns:**
        - **head_t** (`nn.Linear`): the output head for task `task_id`.
        """
        return self.heads[f"{task_id}"]

    def forward(self, feature: Tensor, task_ids: int | Tensor) -> Tensor:
        r"""The forward pass for data from task `task_id`. A head is selected according to the task_id and the feature is passed through the head.

        **Args:**
        - **feature** (`Tensor`): the feature tensor from the backbone network.
        - **task_ids** (`int` | `Tensor`): the task ID(s) for the input data. If the input batch is from the same task, this can be a single integer.

        **Returns:**
        - **logits** (`Tensor`): the output logits tensor.
        """

        if isinstance(task_ids, int):
            head_t = self.get_head(task_ids)
            logits = head_t(feature)

        elif isinstance(task_ids, Tensor):
            logits_list = []
            for task_id in torch.unique(task_ids):  # for each unique task in the batch
                idx = (task_ids == task_id).nonzero(as_tuple=True)[
                    0
                ]  # indices of the current task in the batch
                features_t = feature[idx]  # get the features for the current task
                head_t = self.get_head(task_id.item())
                logits_t = head_t(features_t)
                logits_list.append((idx, logits_t))

            # reconstruct logits tensor in the order of task_ids
            logits = torch.zeros(len(task_ids), logits_t.size(1), device=feature.device)
            for idx, logits_t in logits_list:
                logits[idx] = logits_t

        return logits
