r"""
The submodule in `cul_algorithms` for the vanilla unlearning algorithm for EWC.
"""

__all__ = ["EWCUnlearn"]

import logging

from clarena.cl_algorithms import UnlearnableEWC
from clarena.cul_algorithms import CULAlgorithm

pylogger = logging.getLogger(__name__)



class EWCUnlearn(CULAlgorithm):
    r"""Vanilla unlearning algorithm for EWC.

    For each requested unlearning task u:
    - Remove u from the valid_task_ids (so future regularization ignores it)
    - Delete stored fisher/importances & previous backbone snapshot of u (optional but recommended)
    - Reset the corresponding head parameters if the head exists (TIL/DIL case)
    """

    def __init__(self, model: UnlearnableEWC) -> None:
        super().__init__(model=model)

    def unlearn(self) -> None:
        for unlearning_task_id in self.unlearning_task_ids:
            # 1) remove from valid set (use discard to avoid KeyError)
            self.model.valid_task_ids.discard(unlearning_task_id)

            # 2) delete stored EWC memory for that task if present
            if unlearning_task_id in self.model.parameter_importance:
                del self.model.parameter_importance[unlearning_task_id]
            if unlearning_task_id in self.model.previous_task_backbones:
                del self.model.previous_task_backbones[unlearning_task_id]

            # 3) reset the task head if applicable
            #    (safe-guard because HeadsCIL may not have per-task heads)
            try:
                head = self.model.heads.get_head(unlearning_task_id)
                if hasattr(head, "reset_parameters"):
                    head.reset_parameters()
            except Exception:
                # If heads does not support get_head or task_id not applicable, ignore.
                pass
