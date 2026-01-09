# # r"""
# # The submodule in `cul_algorithms` for the vanilla unlearning algorithm for EWC.
# # """

# # __all__ = ["EWCUnlearn"]

# # import logging

# # from clarena.cl_algorithms import UnlearnableEWC
# # from clarena.cul_algorithms import CULAlgorithm

# # pylogger = logging.getLogger(__name__)



# # class EWCUnlearn(CULAlgorithm):
# #     r"""Vanilla unlearning algorithm for EWC.

# #     For each requested unlearning task u:
# #     - Remove u from the valid_task_ids (so future regularization ignores it)
# #     - Delete stored fisher/importances & previous backbone snapshot of u (optional but recommended)
# #     - Reset the corresponding head parameters if the head exists (TIL/DIL case)
# #     """

# #     def __init__(self, model: UnlearnableEWC) -> None:
# #         super().__init__(model=model)

# #     def unlearn(self) -> None:
# #         for unlearning_task_id in self.unlearning_task_ids:
# #             # 1) remove from valid set (use discard to avoid KeyError)
# #             self.model.valid_task_ids.discard(unlearning_task_id)

# #             # 2) delete stored EWC memory for that task if present
# #             if unlearning_task_id in self.model.parameter_importance:
# #                 del self.model.parameter_importance[unlearning_task_id]
# #             if unlearning_task_id in self.model.previous_task_backbones:
# #                 del self.model.previous_task_backbones[unlearning_task_id]

# #             # 3) reset the task head if applicable
# #             #    (safe-guard because HeadsCIL may not have per-task heads)
# #             try:
# #                 head = self.model.heads.get_head(unlearning_task_id)
# #                 if hasattr(head, "reset_parameters"):
# #                     head.reset_parameters()
# #             except Exception:
# #                 # If heads does not support get_head or task_id not applicable, ignore.
# #                 pass

# r"""
# The submodule in `cul_algorithms` for the vanilla unlearning algorithm for EWC.
# """

# __all__ = ["EWCUnlearn"]

# import logging

# from clarena.cl_algorithms import UnlearnableEWC
# from clarena.cul_algorithms import CULAlgorithm

# pylogger = logging.getLogger(__name__)


# class EWCUnlearn(CULAlgorithm):
#     r"""Vanilla unlearning algorithm for EWC.

#     We follow the same unlearning logic as AmnesiacHAT:
#     - Delete the stored parameter update of the unlearning task(s)
#     - Reconstruct the backbone parameters from the remaining updates

#     Note:
#     EWC itself is a single aggregated model; we cannot "remove" a task perfectly like Independent.
#     This unlearning implementation removes the parameter contribution (Δθ_u) of the requested task(s).
#     """

#     def __init__(self, model: UnlearnableEWC) -> None:
#         super().__init__(model=model)

#     def delete_update(self, unlearning_task_ids: list[int]) -> None:
#         r"""Delete the update of the specified unlearning task.

#         **Args:**
#         - **unlearning_task_id** (`list[int]`): the ID of the unlearning task to delete the update.
#         """
#         for unlearning_task_id in unlearning_task_ids:
#             if unlearning_task_id not in self.model.parameters_task_update:
#                 pylogger.warning(
#                     "Attempted to delete update for task %d, but it was not found.",
#                     unlearning_task_id,
#                 )
#                 continue

#             # delete the parameter update for the unlearning task so that it won't be used in future parameter constructions
#             del self.model.parameters_task_update[unlearning_task_id]

#         pylogger.info(
#             "Deleted parameter update for unlearning task %s.", unlearning_task_ids
#         )

#     def unlearn(self) -> None:
#         r"""Unlearn the requested unlearning tasks after training `self.task_id`."""
#         if self.unlearning_task_ids == []:
#             return

#         pylogger.info(
#             "Starting unlearning process for tasks: %s...", self.unlearning_task_ids
#         )

#         # Step 1: delete the update(s)
#         self.delete_update(self.unlearning_task_ids)

#         # Step 2: reconstruct backbone parameters from remaining updates
#         self.model.construct_parameters_from_updates()

#         # Optional: reset head parameters for unlearned tasks (TIL/DIL case)
#         # Keeping this consistent with your "default reset head" preference.
#         for unlearning_task_id in self.unlearning_task_ids:
#             try:
#                 head = self.model.heads.get_head(unlearning_task_id)
#                 if hasattr(head, "reset_parameters"):
#                     head.reset_parameters()
#             except Exception:
#                 pass

#         pylogger.info("Unlearning process finished.")

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
