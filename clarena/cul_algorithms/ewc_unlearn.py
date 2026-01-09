r"""
The submodule in `cul_algorithms` for the vanilla unlearning algorithm for EWC.
"""

__all__ = ["EWCUnlearn"]

import logging

from clarena.cl_algorithms import UnlearnableEWC
from clarena.cul_algorithms import CULAlgorithm

pylogger = logging.getLogger(__name__)


class EWCUnlearn(CULAlgorithm):
    r"""Vanilla unlearning algorithm for EWC (Amnesiac Hat style).

    For each requested unlearning task u:
    - Roll back parameters by subtracting the recorded delta: θ ← θ - Δ_u
      (implemented by `UnlearnableEWC.unlearn_task(u)`)
    - Remove u from valid_task_ids (so future regularization ignores it)
    - Delete stored fisher/importances & previous backbone snapshot of u
    - Reset the corresponding head parameters if the head exists (TIL/DIL case)
    """

    def __init__(self, model: UnlearnableEWC) -> None:
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
            # 0) Amnesiac Hat rollback: θ ← θ - Δ_u
            #    (minimal change: delegate to model's rollback method)
            if hasattr(self.model, "unlearn_task"):
                self.model.unlearn_task(unlearning_task_id)

            # 1) remove from valid set (so future EWC regularization ignores it)
            if hasattr(self.model, "valid_task_ids"):
                self.model.valid_task_ids.discard(unlearning_task_id)

            # 2) delete stored EWC memory for that task if present
            if hasattr(self.model, "parameter_importance") and (
                unlearning_task_id in self.model.parameter_importance
            ):
                del self.model.parameter_importance[unlearning_task_id]

            if hasattr(self.model, "previous_task_backbones") and (
                unlearning_task_id in self.model.previous_task_backbones
            ):
                del self.model.previous_task_backbones[unlearning_task_id]

            # 3) reset the task head if applicable (TIL/DIL)
            self._reset_head_if_possible(unlearning_task_id)

        pylogger.info("Unlearning process finished.")
