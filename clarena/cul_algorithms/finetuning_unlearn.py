r"""
The submodule in `cul_algorithms` for the vanilla unlearning algorithm for Finetuning (delta rollback style).
"""

__all__ = ["AmnesiacFinetuningUnlearn"]

import logging

from clarena.cl_algorithms import AmnesiacFinetuning
from clarena.cul_algorithms import AmnesiacCULAlgorithm

pylogger = logging.getLogger(__name__)


class AmnesiacFinetuningUnlearn(AmnesiacCULAlgorithm):
    r"""Amnesiac continual unlearning algorithm for Finetuning."""

    def __init__(
        self,
        model: AmnesiacFinetuning,
    ) -> None:
        r"""Initialize the unlearning algorithm with the continual learning model.

        **Args:**
        - **model** (`AmnesiacHAT`): the continual learning model. It must be `AmnesicCLAlgorithm`.
        """
        super().__init__(model=model)


# class FinetuningUnlearn(CULAlgorithm):
#     r"""Vanilla unlearning algorithm for Finetuning (delta rollback style).

#     For each requested unlearning task u:
#     - Roll back parameters by subtracting the recorded delta: θ ← θ - Δ_u
#       (implemented by `UnlearnableFinetuning.unlearn_task(u)`)
#     - Reset the corresponding head parameters if possible (TIL/DIL)
#     """

#     def __init__(self, model: UnlearnableFinetuning) -> None:
#         super().__init__(model=model)

#     def unlearn(self) -> None:
#         if not self.unlearning_task_ids:
#             return

#         pylogger.info(
#             "Starting unlearning process for tasks: %s...", self.unlearning_task_ids
#         )

#         for unlearning_task_id in self.unlearning_task_ids:
#             if hasattr(self.model, "unlearn_task"):
#                 self.model.unlearn_task(unlearning_task_id)
#             else:
#                 raise AttributeError(
#                     "Model does not implement unlearn_task(). "
#                     "Please use UnlearnableFinetuning with delta rollback support."
#                 )

#         pylogger.info("Unlearning process finished.")
