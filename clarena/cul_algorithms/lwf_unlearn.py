r"""
The submodule in `cul_algorithms` for the vanilla unlearning algorithm for LwF (Learning without Forgetting).
"""

__all__ = ["LwFUnlearn"]

import logging

from clarena.cl_algorithms import UnlearnableLwF
from clarena.cul_algorithms import CULAlgorithm

# always get logger for built-in logging in each module
pylogger = logging.getLogger(__name__)


class LwFUnlearn(CULAlgorithm):
    r"""Vanilla unlearning algorithm for LwF (Learning without Forgetting)."""

    def __init__(self, model: UnlearnableLwF) -> None:
        r"""
        **Args:**
        - **model** (`UnlearnableLwF`): the continual learning model (`CLAlgorithm` object which already contains the backbone and heads). It must be `UnlearnableLwF` algorithm.
        """
        super().__init__(model=model)

    def unlearn(self) -> None:
        r"""Unlearn the requested unlearning tasks after training `self.task_id`.

        For LwF, unlearning means:
        - Remove the corresponding teacher backbone snapshot for the unlearning task,
          so it will no longer be used for distillation in future tasks.
        - Reset the corresponding head parameters (default behavior consistent with other unlearning algorithms).
        - Remove the task id from `valid_task_ids` so that future training won't regularize toward it.
        """

        for unlearning_task_id in self.unlearning_task_ids:

            # remove the teacher backbone snapshot for this task
            if unlearning_task_id in self.model.previous_task_backbones:
                del self.model.previous_task_backbones[unlearning_task_id]

            # reset the corresponding head (consistent with Independent/EWC style)
            self.model.heads.get_head(unlearning_task_id).reset_parameters()

            # remove from valid task ids (for distillation regularization)
            if hasattr(self.model, "valid_task_ids") and unlearning_task_id in self.model.valid_task_ids:
                self.model.valid_task_ids.remove(unlearning_task_id)
