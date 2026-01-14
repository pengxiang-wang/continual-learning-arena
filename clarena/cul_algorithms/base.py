r"""
The submoduule in `cul_algorithms` for continual unlearning alogrithm bases.
"""

__all__ = ["CULAlgorithm", "AmnesiacCULAlgorithm"]

import logging
from abc import abstractmethod

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
        self.task_ids_no_longer_unlearnable: list[int] = []
        r"""The list of task IDs that are just no longer unlearnable at the current `self.task_id`."""

    def setup_task_id(
        self,
        task_id: int,
        unlearning_requests: dict[int, list[int]],
        unlearnable_task_ids: list[int],
        task_ids_no_longer_unlearnable: list[int],
    ) -> None:
        r"""Set up which task the CUL experiment is on. This must be done before `unlearn()` method is called.

        **Args:**
        - **task_id** (`int`): the target task ID to be set up.
        - **unlearning_requests** (`dict[int, list[int]]`): the entire unlearning requests. Keys are IDs of the tasks that request unlearning after their learning, and values are the list of the previous tasks to be unlearned.
        - **unlearnable_task_ids** (`list[int]`): the list of unlearnable task IDs at the current `self.task_id`.
        - **task_ids_no_longer_unlearnable** (`list[int]`): the list of task IDs that are just no longer unlearnable at the current `self.task_id`.
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

        self.task_ids_no_longer_unlearnable = task_ids_no_longer_unlearnable
        self.model.task_ids_no_longer_unlearnable = task_ids_no_longer_unlearnable

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

    def delete_update(self, unlearning_task_ids: list[int]) -> None:
        r"""Delete the update of the specified unlearning task.

        **Args:**
        - **unlearning_task_id** (`list[int]`): the ID of the unlearning task to delete the update.
        """

        for unlearning_task_id in unlearning_task_ids:
            if unlearning_task_id not in self.model.parameters_task_update:
                pylogger.warning(
                    "Attempted to delete update for task %d, but it was not found.",
                    unlearning_task_id,
                )
                continue

            # delete the parameter update for the unlearning task so that it won't be used in future parameter constructions
            del self.model.parameters_task_update[unlearning_task_id]

        pylogger.info(
            "Deleted parameter update for unlearning task %s.", unlearning_task_ids
        )

    def unlearn(self) -> None:
        r"""Unlearn the requested unlearning tasks in the current task `self.task_id`."""

        # delete the corresponding parameter update records
        self.delete_update(self.unlearning_task_ids)

        # reconstruct the model parameters
        self.model.construct_parameters_from_updates()
