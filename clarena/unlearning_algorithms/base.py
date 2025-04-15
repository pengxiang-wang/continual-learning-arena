r"""
The submoduule in `unlearning_algorithms` for unlearning alogrithm bases.
"""

__all__ = ["UnlearningAlgorithm"]

import logging
from abc import abstractmethod

from omegaconf import DictConfig

from clarena.cl_algorithms import CLAlgorithm

# always get logger for built-in logging in each module
pylogger = logging.getLogger(__name__)


class UnlearningAlgorithm:
    r"""The base class of unlearning algorithms."""

    def __init__(self, model: CLAlgorithm) -> None:
        r"""Initialise the unlearning algorithm with the continual learning model.

        **Args:**
        - **model** (`CLAlgorithm`): the continual learning model (`CLAlgorithm` object which already contains the backbone and heads).
        """
        self.model: CLAlgorithm = model
        r"""Store the continual learning model."""

        self.task_id: int
        r"""Task ID counter indicating which task is being processed. Self updated during the task loop. Starting from 1. """
        self.unlearning_task_ids: list[int]
        r"""The list of task IDs to be unlearned after `self.task_id`. If none of the tasks need to be unlearned, it will be an empty list."""
        self.unlearned_task_ids: set[int] = set()
        r"""Store the list of task IDs that have been unlearned in the experiment. """
        self.if_permanent_task_id: bool
        r"""Whether the task is permanent or not. If `True`, the task will not be unlearned i.e. not shown in future unlearning requests."""
        self.cfg_unlearning_test_reference: DictConfig
        r"""The reference experiment configuration for unlearning test. """

    def setup_task_id(
        self,
        task_id: int,
        unlearning_requests: dict[int, list[int]],
        cfg_unlearning_test_reference: DictConfig,
        if_permanent: bool,
    ) -> None:
        r"""Set up which task the CUL experiment is on. This must be done before `unlearn()` method is called.

        **Args:**
        - **task_id** (`int`): the target task ID to be set up.
        - **unlearning_requests** (`dict[int, list[int]]`): the entire unlearning requests. Keys are IDs of the tasks that request unlearning after their learning, and values are the list of the previous tasks to be unlearned.
        - **cfg_unlearning_test_reference** (`DictConfig`): the reference experiment configuration for unlearning test.
        - **if_permanent** (`bool`): whether the task is permanent or not. If `True`, the task will not be unlearned i.e. not shown in future unlearning requests.
        """
        self.task_id = task_id

        unlearning_task_ids = (
            unlearning_requests[task_id] if task_id in unlearning_requests else []
        )
        self.unlearning_task_ids = unlearning_task_ids
        self.model.unlearning_task_ids = unlearning_task_ids

        self.cfg_unlearning_test_reference = cfg_unlearning_test_reference
        self.model.cfg_unlearning_test_reference = cfg_unlearning_test_reference

        self.if_permanent_task_id = if_permanent

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
        r"""Unlearn the requested unlearning tasks in current task `self.task_id`."""
