r"""
The submoduule in `cul_algorithms` for the [CLPU-DER++](https://arxiv.org/abs/2203.12817) unlearning algorithm.
"""

__all__ = ["CLPUDERppUnlearn"]

import logging

from clarena.cl_algorithms import CLPUDERpp
from clarena.cul_algorithms import CULAlgorithm

# always get logger for built-in logging in each module
pylogger = logging.getLogger(__name__)


class CLPUDERppUnlearn(CULAlgorithm):
    r"""CLPU-DER++ unlearning algorithm."""

    def __init__(self, model: CLPUDERpp) -> None:
        r"""
        **Args:**
        - **model** (`CLPUDERpp`): the continual learning model (`CLAlgorithm` object which already contains the backbone and heads). It must be `CLPUDERpp` algorithm.
        """
        super().__init__(model=model)

    def unlearn(self) -> None:
        r"""Unlearn the requested unlearning tasks after training `self.task_id`."""
        # It corresponds to Case IV in the algorithm description of the [CLPU paper](https://arxiv.org/abs/2203.12817).

        for unlearning_task_id in self.unlearning_task_ids:

            # delete task from the memory buffer
            self.model.memory_buffer.delete_task(unlearning_task_id)

            if str(unlearning_task_id) in self.model.temporary_backbones:
                del self.model.temporary_backbones[str(unlearning_task_id)]
            else:
                pylogger.warning(
                    "Temporary backbone for task %d not found in CLPUDERppUnlearn unlearning.",
                    unlearning_task_id,
                )
