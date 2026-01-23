r"""
The submoduule in `cul_algorithms` for the CLPU-DER++ unlearning algorithm.
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

        print("unlearning_task_id", self.unlearning_task_ids)
        for unlearning_task_id in self.unlearning_task_ids:
            temp_key = str(unlearning_task_id)
            if (
                temp_key not in self.model.temporary_backbones
                and unlearning_task_id not in self.model.temporary_backbones
            ):
                pylogger.warning(
                    "No temporary backbone found for task %d; skipping.",
                    unlearning_task_id,
                )
                continue

            self.model.memory_buffer.delete_task(unlearning_task_id)

            if temp_key in self.model.temporary_backbones:
                del self.model.temporary_backbones[temp_key]
            elif unlearning_task_id in self.model.temporary_backbones:
                del self.model.temporary_backbones[unlearning_task_id]
