r"""
The submodule in `cul_algorithms` for the vanilla unlearning algorithm for LwF.
"""

__all__ = ["LwFUnlearn"]

import logging

from clarena.cl_algorithms import UnlearnableLwF
from clarena.cul_algorithms import CULAlgorithm

pylogger = logging.getLogger(__name__)


class LwFUnlearn(CULAlgorithm):
    r"""Vanilla unlearning algorithm for LwF.

    For each requested unlearning task u:
    - Remove u from valid_task_ids (so future distillation ignores it)
    - Delete stored teacher backbone snapshot of u
    - Reset the corresponding head parameters if the head exists (TIL/DIL case)
    """

    def __init__(self, model: UnlearnableLwF) -> None:
        super().__init__(model=model)

    def _reset_head_if_possible(self, task_id: int) -> None:
        """Best-effort reset for task-specific head (TIL/DIL)."""
        try:
            # Some heads implementations provide get_head(task_id)
            head = self.model.heads.get_head(task_id)
            if hasattr(head, "reset_parameters"):
                head.reset_parameters()
        except Exception:
            # Fallback for common HeadsTIL implementation: heads.heads is a dict-like
            try:
                if hasattr(self.model.heads, "heads"):
                    key = f"{task_id}"
                    if key in self.model.heads.heads:
                        head = self.model.heads.heads[key]
                        if hasattr(head, "reset_parameters"):
                            head.reset_parameters()
            except Exception:
                pass

    def unlearn(self) -> None:
        if not self.unlearning_task_ids:
            return

        pylogger.info(
            "Starting unlearning process for tasks: %s...", self.unlearning_task_ids
        )

        for unlearning_task_id in self.unlearning_task_ids:
            # 1) remove from valid set
            if hasattr(self.model, "valid_task_ids"):
                self.model.valid_task_ids.discard(unlearning_task_id)

            # 2) delete stored teacher snapshot for that task if present
            if hasattr(self.model, "previous_task_backbones") and (
                unlearning_task_id in self.model.previous_task_backbones
            ):
                del self.model.previous_task_backbones[unlearning_task_id]

            # 3) reset the task head if applicable (TIL/DIL)
            self._reset_head_if_possible(unlearning_task_id)

        pylogger.info("Unlearning process finished.")

# r"""
# The submodule in `cul_algorithms` for the vanilla unlearning algorithm for LwF (Learning without Forgetting).
# """

# __all__ = ["LwFUnlearn"]

# import logging

# from clarena.cl_algorithms import UnlearnableLwF
# from clarena.cul_algorithms import CULAlgorithm

# # always get logger for built-in logging in each module
# pylogger = logging.getLogger(__name__)


# class LwFUnlearn(CULAlgorithm):
#     r"""Vanilla unlearning algorithm for LwF (Learning without Forgetting)."""

#     def __init__(self, model: UnlearnableLwF) -> None:
#         r"""
#         **Args:**
#         - **model** (`UnlearnableLwF`): the continual learning model (`CLAlgorithm` object which already contains the backbone and heads). It must be `UnlearnableLwF` algorithm.
#         """
#         super().__init__(model=model)

#     def unlearn(self) -> None:
#         r"""Unlearn the requested unlearning tasks after training `self.task_id`.

#         For LwF, unlearning means:
#         - Remove the corresponding teacher backbone snapshot for the unlearning task,
#           so it will no longer be used for distillation in future tasks.
#         - Reset the corresponding head parameters (default behavior consistent with other unlearning algorithms).
#         - Remove the task id from `valid_task_ids` so that future training won't regularize toward it.
#         """

#         for unlearning_task_id in self.unlearning_task_ids:

#             # remove the teacher backbone snapshot for this task
#             if unlearning_task_id in self.model.previous_task_backbones:
#                 del self.model.previous_task_backbones[unlearning_task_id]

#             # reset the corresponding head (consistent with Independent/EWC style)
#             self.model.heads.get_head(unlearning_task_id).reset_parameters()

#             # remove from valid task ids (for distillation regularization)
#             if hasattr(self.model, "valid_task_ids") and unlearning_task_id in self.model.valid_task_ids:
#                 self.model.valid_task_ids.remove(unlearning_task_id)
