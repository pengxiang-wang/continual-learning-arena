r"""
The submodule in `cul_algorithms` for LwF unlearning algorithm.
"""

__all__ = ["AmnesiacLwFUnlearn"]

import logging

from clarena.cl_algorithms import AmnesiacLwF
from clarena.cul_algorithms import AmnesiacCULAlgorithm

pylogger = logging.getLogger(__name__)


class AmnesiacLwFUnlearn(AmnesiacCULAlgorithm):
    r"""Amnesiac continual unlearning algorithm for EWC."""

    def __init__(
        self,
        model: AmnesiacLwF,
    ) -> None:
        r"""Initialize the unlearning algorithm with the continual learning model.

        **Args:**
        - **model** (`AmnesiacLwF`): the continual learning model. It must be `AmnesiacLwF`.
        """
        super().__init__(model=model)


# class LwFUnlearn(CULAlgorithm):
#     r"""Vanilla unlearning algorithm for LwF (Amnesiac-HAT / delta rollback style).

#     For each requested unlearning task u:
#     - Roll back parameters by subtracting recorded delta: θ ← θ - Δ_u
#       (implemented by `UnlearnableLwF.unlearn_task(u)`)
#     - Remove u from valid_task_ids (so future distillation ignores it)
#     - Delete stored teacher snapshot (previous_task_backbones[u])
#     - Reset the corresponding head parameters if head exists (TIL/DIL case)
#     """

#     def __init__(self, model: UnlearnableLwF) -> None:
#         super().__init__(model=model)

#     def unlearn(self) -> None:
#         if not self.unlearning_task_ids:
#             return

#         pylogger.info(
#             "Starting unlearning process for tasks: %s...", self.unlearning_task_ids
#         )

#         for unlearning_task_id in self.unlearning_task_ids:
#             # delegate rollback + cleanup to model
#             if hasattr(self.model, "unlearn_task"):
#                 self.model.unlearn_task(unlearning_task_id)
#             else:
#                 raise AttributeError(
#                     "Model does not implement unlearn_task(). "
#                     "Please use UnlearnableLwF with delta rollback support."
#                 )

#         pylogger.info("Unlearning process finished.")


# class LwFUnlearnHeadOnly(CULAlgorithm):
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
#             if (
#                 hasattr(self.model, "valid_task_ids")
#                 and unlearning_task_id in self.model.valid_task_ids
#             ):
#                 self.model.valid_task_ids.remove(unlearning_task_id)
