r"""
The submodule in `cul_algorithms` for the vanilla unlearning algorithm for Finetuning (delta rollback style).
"""

__all__ = ["FinetuningUnlearn"]

import logging

from clarena.cl_algorithms import UnlearnableFinetuning
from clarena.cul_algorithms import CULAlgorithm

pylogger = logging.getLogger(__name__)


class FinetuningUnlearn(CULAlgorithm):
    r"""Vanilla unlearning algorithm for Finetuning (delta rollback style).

    For each requested unlearning task u:
    - Roll back parameters by subtracting the recorded delta: θ ← θ - Δ_u
      (implemented by `UnlearnableFinetuning.unlearn_task(u)`)
    - Reset the corresponding head parameters if possible (TIL/DIL)
    """

    def __init__(self, model: UnlearnableFinetuning) -> None:
        super().__init__(model=model)

    def unlearn(self) -> None:
        if not self.unlearning_task_ids:
            return

        pylogger.info(
            "Starting unlearning process for tasks: %s...", self.unlearning_task_ids
        )

        for unlearning_task_id in self.unlearning_task_ids:
            if hasattr(self.model, "unlearn_task"):
                self.model.unlearn_task(unlearning_task_id)
            else:
                raise AttributeError(
                    "Model does not implement unlearn_task(). "
                    "Please use UnlearnableFinetuning with delta rollback support."
                )

        pylogger.info("Unlearning process finished.")
