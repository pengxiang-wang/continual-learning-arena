r"""
The submodule in `cul_algorithms` for [EWC (Elastic Weight Consolidation) algorithm](https://www.pnas.org/doi/10.1073/pnas.1611835114) unlearning.
"""

__all__ = ["AmnesiacEWCUnlearn"]

import logging
from copy import deepcopy

from clarena.cl_algorithms import AmnesiacEWC
from clarena.cul_algorithms import AmnesiacCULAlgorithm
from clarena.heads import HeadDIL

pylogger = logging.getLogger(__name__)


class AmnesiacEWCUnlearn(AmnesiacCULAlgorithm):
    r"""Amnesiac continual unlearning algorithm for [EWC (Elastic Weight Consolidation) algorithm](https://www.pnas.org/doi/10.1073/pnas.1611835114)."""

    def __init__(
        self,
        model: AmnesiacEWC,
        if_unlearn_previous_network: bool = False,
    ) -> None:
        r"""Initialize the unlearning algorithm with the continual learning model.

        **Args:**
        - **model** (`AmnesiacEWC`): the continual learning model. It must be `AmnesiacEWC`.
        - **if_unlearn_previous_network** (`bool`): whether to adjust previous task snapshots after unlearning. Default is `False`.
        """
        super().__init__(model=model)

        self.if_unlearn_previous_network: bool = if_unlearn_previous_network
        r"""Whether to adjust previous task snapshots after unlearning."""

    def unlearn(self) -> None:
        r"""Unlearn the requested unlearning tasks after training `self.task_id`."""

        # delete the corresponding parameter update records
        self.delete_update()

        # remove the memory related to unlearning tasks
        for unlearning_task_id in self.unlearning_task_ids:

            del self.model.parameter_importance[unlearning_task_id]
            del self.model.previous_task_backbones[unlearning_task_id]

            if (
                unlearning_task_id in self.model.parameter_importance_heads
                and isinstance(self.model.heads, HeadDIL)
            ):
                del self.model.parameter_importance_heads[unlearning_task_id]
            if unlearning_task_id in self.model.previous_task_heads and isinstance(
                self.model.heads, HeadDIL
            ):
                del self.model.previous_task_heads[unlearning_task_id]

    def delete_update(self) -> None:
        r"""Delete the updates for unlearning tasks from the current parameters, and optionally from previous network snapshots."""

        # delete updates from previous network snapshots
        if self.if_unlearn_previous_network:
            self.delete_previous_network_update()

        # delete updates from current parameters and the records
        super().delete_update()

    def delete_previous_network_update(self) -> None:
        r"""Delete the updates for unlearning tasks from the previous network snapshots."""

        for unlearning_task_id in self.unlearning_task_ids:
            param_update = self.model.parameters_task_update.get(unlearning_task_id)
            if param_update is None:
                pylogger.warning(
                    "Attempted to delete update from previous network snapshot for task %d, but the update was not found.",
                    unlearning_task_id,
                )
                continue
            for (
                previous_task_id,
                previous_backbone,
            ) in self.model.previous_task_backbones.items():
                # only delete updates for tasks after the unlearning task
                if previous_task_id <= unlearning_task_id:
                    continue

                updated_state_dict = deepcopy(previous_backbone.state_dict())
                for layer_name, param_tensor in param_update.items():
                    if layer_name in updated_state_dict:
                        target_tensor = updated_state_dict[layer_name]
                        if (
                            param_tensor.device != target_tensor.device
                            or param_tensor.dtype != target_tensor.dtype
                        ):
                            param_tensor = param_tensor.to(
                                device=target_tensor.device,
                                dtype=target_tensor.dtype,
                            )
                        updated_state_dict[layer_name] -= param_tensor
                previous_backbone.load_state_dict(updated_state_dict, strict=False)

        if isinstance(self.model.heads, HeadDIL):
            for unlearning_task_id in self.unlearning_task_ids:
                param_update = self.model.parameters_task_update_heads.get(
                    unlearning_task_id
                )
                if param_update is None:
                    pylogger.warning(
                        "Attempted to delete update from previous network snapshot for task %d, but the update was not found.",
                        unlearning_task_id,
                    )
                    continue
                for (
                    previous_task_id,
                    previous_heads,
                ) in self.model.previous_task_heads.items():
                    # only delete updates for tasks after the unlearning task
                    if previous_task_id <= unlearning_task_id:
                        continue
                    updated_state_dict = deepcopy(previous_heads.state_dict())
                    for param_name, param_tensor in param_update.items():
                        if param_name in updated_state_dict:
                            target_tensor = updated_state_dict[param_name]
                            if (
                                param_tensor.device != target_tensor.device
                                or param_tensor.dtype != target_tensor.dtype
                            ):
                                param_tensor = param_tensor.to(
                                    device=target_tensor.device,
                                    dtype=target_tensor.dtype,
                                )
                            updated_state_dict[param_name] -= param_tensor
                    previous_heads.load_state_dict(updated_state_dict, strict=False)
