r"""
The submodule in `cul_algorithms` for [LwF (Learning without Forgetting) algorithm](https://ieeexplore.ieee.org/abstract/document/8107520) unlearning.
"""

__all__ = ["AmnesiacLwFUnlearn"]

import logging

from clarena.cl_algorithms import AmnesiacLwF
from clarena.cul_algorithms import AmnesiacCULAlgorithm
from clarena.heads import HeadDIL

pylogger = logging.getLogger(__name__)


class AmnesiacLwFUnlearn(AmnesiacCULAlgorithm):
    r"""Amnesiac continual unlearning algorithm for [LwF (Learning without Forgetting)](https://ieeexplore.ieee.org/abstract/document/8107520)."""

    def __init__(
        self,
        model: AmnesiacLwF,
    ) -> None:
        r"""Initialize the unlearning algorithm with the continual learning model.

        **Args:**
        - **model** (`AmnesiacLwF`): the continual learning model. It must be `AmnesiacLwF`.
        """
        super().__init__(model=model)

    def unlearn(self) -> None:
        r"""Unlearn the requested unlearning tasks after training `self.task_id`."""

        # delete the corresponding parameter update records
        self.delete_update()

        # remove the teacher network snapshot for the unlearning tasks
        for unlearning_task_id in self.unlearning_task_ids:

            del self.model.previous_task_backbones[unlearning_task_id]

            if isinstance(self.model.heads, HeadDIL):
                del self.model.previous_task_heads[unlearning_task_id]
