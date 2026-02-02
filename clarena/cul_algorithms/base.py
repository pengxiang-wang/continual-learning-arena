r"""
The submoduule in `cul_algorithms` for continual unlearning alogrithm bases.
"""

__all__ = ["CULAlgorithm", "AmnesiacCULAlgorithm"]

import logging
from abc import abstractmethod
from copy import deepcopy

from clarena.cl_algorithms import AmnesiacCLAlgorithm, UnlearnableCLAlgorithm

# always get logger for built-in logging in each module
pylogger = logging.getLogger(__name__)


class CULAlgorithm:
    r"""The base class of continual unlearning algorithms."""

    def __init__(self, model: UnlearnableCLAlgorithm) -> None:
        r"""
        **Args:**
        - **model** (`UnlearnableCLAlgorithm`): the continual learning model.
        """

        # components
        self.model: UnlearnableCLAlgorithm = model
        r"""The continual learning model."""

        # task ID control
        self.task_id: int
        r"""Task ID counter indicating which task is being processed. Self updated during the task loop. Valid from 1 to `cl_dataset.num_tasks`."""
        self.processed_task_ids: list[int] = []
        r"""Task IDs that have been processed."""
        self.unlearning_task_ids: list[int] = []
        r"""The list of task IDs that are requested to be unlearned after training `self.task_id`."""
        self.unlearned_task_ids: set[int] = set()
        r"""The list of task IDs that have been unlearned in the experiment. """
        self.unlearnable_task_ids: list[int] = []
        r"""The list of task IDs that are unlearnable at the current `self.task_id`."""
        self.task_ids_just_no_longer_unlearnable: list[int] = []
        r"""The list of task IDs that are just no longer unlearnable at the current `self.task_id`."""

    def setup_task_id(
        self,
        task_id: int,
        unlearning_requests: dict[int, list[int]],
        unlearnable_task_ids: list[int],
        task_ids_just_no_longer_unlearnable: list[int],
    ) -> None:
        r"""Set up which task the CUL experiment is on. This must be done before `unlearn()` method is called.

        **Args:**
        - **task_id** (`int`): the target task ID to be set up.
        - **unlearning_requests** (`dict[int, list[int]]`): the entire unlearning requests. Keys are IDs of the tasks that request unlearning after their learning, and values are the list of the previous tasks to be unlearned.
        - **unlearnable_task_ids** (`list[int]`): the list of unlearnable task IDs at the current `self.task_id`.
        - **task_ids_just_no_longer_unlearnable** (`list[int]`): the list of task IDs that are just no longer unlearnable at the current `self.task_id`.
        """
        self.task_id = task_id
        self.processed_task_ids.append(task_id)

        unlearning_task_ids = (
            unlearning_requests[task_id] if task_id in unlearning_requests else []
        )
        self.unlearning_task_ids = unlearning_task_ids
        self.model.unlearning_task_ids = unlearning_task_ids

        self.unlearnable_task_ids = unlearnable_task_ids
        self.model.unlearnable_task_ids = unlearnable_task_ids

        self.task_ids_just_no_longer_unlearnable = task_ids_just_no_longer_unlearnable
        self.model.task_ids_just_no_longer_unlearnable = (
            task_ids_just_no_longer_unlearnable
        )

    def setup_test_task_id(self) -> None:
        r"""Set up before testing `self.task_id`. This must be done after `unlearn()` method is called."""

        self.unlearned_task_ids.update(
            self.unlearning_task_ids
        )  # update the maintained set of unlearned task IDs
        self.model.unlearned_task_ids = (
            self.unlearned_task_ids
        )  # let model know the unlearned task IDs

    @abstractmethod
    def unlearn(self) -> None:
        r"""Unlearn the requested unlearning tasks (`self.unlearning_task_ids`) after training `self.task_id`. **It must be implemented in subclasses.**"""


class AmnesiacCULAlgorithm(CULAlgorithm):
    r"""The base class of Amnesiac continual unlearning algorithm.

    The Amnesiac continual unlearning algorithm refers to update deletion operation that directly delete the parameter updates during a task's training. This is inspired by [AmnesiacML](https://arxiv.org/abs/2010.10981) in machine unlearning. In detail, the task-wise parameter updates are stored:

    $$\theta_{l,ij}^{(t)} = \theta_{l,ij}^{(0)} + \sum_{\tau=1}^{t} \Delta \theta_{l,ij}^{(\tau)}$$

    To unlearn $u(t)$, delete these updates:

    $$\theta_{l,ij}^{(t-u(t))} = \theta_{l,ij}^{(t)} - \sum_{\tau\in u(t)}\Delta \theta_{l,ij}^{(\tau)}$$

    It is mainly used in AmnesaicHAT, but can also be used in constructing other vanilla baseline continual unlearning algorithms based on different continual learning algorithms.
    """

    def __init__(
        self,
        model: AmnesiacCLAlgorithm,
    ) -> None:
        r"""Initialize the unlearning algorithm with the continual learning model.

        **Args:**
        - **model** (`AmnesiacHAT`): the continual learning model. It must be `AmnesicCLAlgorithm`.
        """
        super().__init__(model=model)

    def delete_update(self):
        r"""Delete the updates for unlearning tasks from the current parameters."""

        # substract the corresponding parameter update from backbone
        updated_state_dict = deepcopy(self.model.backbone.state_dict())
        for task_id in self.unlearning_task_ids:
            param_update = self.model.parameters_task_update.get(task_id)
            if param_update is None:
                pylogger.warning(
                    "Attempted to delete backbone update for task %d, but it was not found.",
                    task_id,
                )
                continue
            for layer_name, param_tensor in param_update.items():
                if layer_name in updated_state_dict:
                    target_tensor = updated_state_dict[layer_name]
                    if (
                        param_tensor.device != target_tensor.device
                        or param_tensor.dtype != target_tensor.dtype
                    ):
                        param_tensor = param_tensor.to(
                            device=target_tensor.device, dtype=target_tensor.dtype
                        )
                    updated_state_dict[layer_name] -= param_tensor

            del self.model.parameters_task_update[task_id]  # delete the record

        self.model.backbone.load_state_dict(updated_state_dict, strict=False)

        # substract the corresponding parameter update from heads
        updated_heads_state_dict = deepcopy(self.model.heads.state_dict())
        for task_id in self.unlearning_task_ids:
            param_update = self.model.parameters_task_update_heads.get(task_id)
            if param_update is None:
                pylogger.warning(
                    "Attempted to delete head update for task %d, but it was not found.",
                    task_id,
                )
                continue
            for param_name, param_tensor in param_update.items():
                if param_name in updated_heads_state_dict:
                    target_tensor = updated_heads_state_dict[param_name]
                    if (
                        param_tensor.device != target_tensor.device
                        or param_tensor.dtype != target_tensor.dtype
                    ):
                        param_tensor = param_tensor.to(
                            device=target_tensor.device, dtype=target_tensor.dtype
                        )
                    updated_heads_state_dict[param_name] -= param_tensor

            del self.model.parameters_task_update_heads[task_id]  # delete the record

        self.model.heads.load_state_dict(updated_heads_state_dict, strict=False)

    def unlearn(self) -> None:
        r"""Unlearn the requested unlearning tasks in the current task `self.task_id`.

        This is the default implementation of `unlearn()` method for Amnesiac continual unlearning algorithms. Please override it in subclasses if necessary.
        """

        # delete the corresponding parameter update records
        self.delete_update()
