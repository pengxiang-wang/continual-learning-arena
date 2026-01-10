r"""
The submodule in `cul_algorithms` for the vanilla unlearning algorithm for LwF (delta rollback style).
"""

__all__ = ["LwFUnlearn"]

import logging

from clarena.cl_algorithms import UnlearnableLwF
from clarena.cul_algorithms import CULAlgorithm

pylogger = logging.getLogger(__name__)


class LwFUnlearn(CULAlgorithm):
    r"""Vanilla unlearning algorithm for LwF (Amnesiac-HAT / delta rollback style).

    For each requested unlearning task u:
    - Roll back parameters by subtracting recorded delta: θ ← θ - Δ_u
      (implemented by `UnlearnableLwF.unlearn_task(u)`)
    - Remove u from valid_task_ids (so future distillation ignores it)
    - Delete stored teacher snapshot (previous_task_backbones[u])
    - Reset the corresponding head parameters if head exists (TIL/DIL case)
    """

    def __init__(self, model: UnlearnableLwF) -> None:
        super().__init__(model=model)

    def unlearn(self) -> None:
        if not self.unlearning_task_ids:
            return

        pylogger.info(
            "Starting unlearning process for tasks: %s...", self.unlearning_task_ids
        )

        for unlearning_task_id in self.unlearning_task_ids:
            # delegate rollback + cleanup to model
            if hasattr(self.model, "unlearn_task"):
                self.model.unlearn_task(unlearning_task_id)
            else:
                raise AttributeError(
                    "Model does not implement unlearn_task(). "
                    "Please use UnlearnableLwF with delta rollback support."
                )

        pylogger.info("Unlearning process finished.")
