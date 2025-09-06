r"""
The submoduule in `cul_algorithms` for continual unlearning alogrithm bases.
"""

__all__ = ["CULAlgorithm"]

import logging
from abc import abstractmethod

from clarena.cl_algorithms import UnlearnableCLAlgorithm

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
        self.if_permanent_t: bool
        r"""Whether the task is permanent or not. If `True`, the task will not be unlearned i.e. not shown in future unlearning requests."""

    def setup_task_id(
        self,
        task_id: int,
        unlearning_requests: dict[int, list[int]],
        if_permanent: bool,
    ) -> None:
        r"""Set up which task the CUL experiment is on. This must be done before `unlearn()` method is called.

        **Args:**
        - **task_id** (`int`): the target task ID to be set up.
        - **unlearning_requests** (`dict[int, list[int]]`): the entire unlearning requests. Keys are IDs of the tasks that request unlearning after their learning, and values are the list of the previous tasks to be unlearned.
        - **if_permanent** (`bool`): whether the task is permanent or not. If `True`, the task will not be unlearned i.e. not shown in future unlearning requests.
        """
        self.task_id = task_id

        unlearning_task_ids = (
            unlearning_requests[task_id] if task_id in unlearning_requests else []
        )
        self.unlearning_task_ids = unlearning_task_ids
        self.model.unlearning_task_ids = unlearning_task_ids

        self.if_permanent_t = if_permanent

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
        r"""Unlearn the requested unlearning tasks after training `self.task_id`. **It must be implemented in subclasses.**"""
