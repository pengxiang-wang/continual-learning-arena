r"""
The submoduule in `cul_algorithms` for AmnesiacHAT unlearning algorithm.
"""

__all__ = ["AmnesiacHATUnlearn"]

import logging

import torch

from clarena.cl_algorithms import HAT
from clarena.cul_algorithms import CULAlgorithm

# always get logger for built-in logging in each module
pylogger = logging.getLogger(__name__)


class AmnesiacHATUnlearn(CULAlgorithm):
    r"""The base class of the AmnesiacHAT unlearning algorithm."""

    def __init__(self, model: HAT) -> None:
        r"""Initialize the unlearning algorithm with the continual learning model.

        **Args:**
        - **model** (`Independent`): the continual learning model (`CLAlgorithm` object which already contains the backbone and heads). It must be `HAT` algorithm.
        """
        super().__init__(model=model)

    def delete_update(self, unlearning_task_id: str) -> None:
        r"""Delete the update of the specified unlearning task.
        **Args:**
        - **unlearning_task_id** (`str`): the ID of the unlearning task to delete the update.
        """
        if unlearning_task_id not in self.model.parameters_task_update:
            raise ValueError(
                f"Unlearning task ID {unlearning_task_id} is not in the model's parameters_task_update."
            )
        # delete the update of the specified unlearning task
        del self.model.parameters_task_update[unlearning_task_id]

    def compensate_layer_if_first_task(
        self,
        layer_name: str,
        unlearning_task_id: str,
        if_first_task_layer: tuple[bool, str],
        next_masked_task_layer: tuple[bool, str],
    ) -> None:
        r"""Compensate if the first task."""

        # compensate the layer if it is the first task layer

        for l in range(
            len(self.model.backbone.masks[f"{unlearning_task_id}"][layer_name])
        ):
            # if the layer is the first task layer, then we need to compensate it
            if if_first_task_layer[l] and next_masked_task_layer[l] is None:
                self.model.parameters_task_update[layer_name] *= (
                    self.model.backbone.masks[f"{next_masked_task_layer[l]}"][
                        layer_name
                    ][l]
                    / self.model.adjustment_intensity
                )

    def unlearn(self) -> None:
        r"""Unlearn the requested unlearning tasks in current task `self.task_id`."""

        current_state_dict = self.model.backbone.state_dict()

        # substract the update from the model
        for unlearning_task_id in self.unlearning_task_ids:

            # substract the update from the model
            for layer_name, _ in current_state_dict.items():

                current_state_dict[layer_name] -= self.model.parameters_task_update[
                    f"{unlearning_task_id}"
                ][layer_name]

                self.model.backbone.masks[f"{unlearning_task_id}"][layer_name] = 0

                # compensate
                compensation = torch.zeros_like(current_state_dict[layer_name])

                # decide if
                if_first_task_layer = self.model.if_first_task_layer(
                    layer_name,
                    unlearning_task_id,
                    self.model.processed_task_ids,
                    self.model.backbone.masks,
                )
                next_masked_task_layer = self.model.next_masked_task_layer(
                    layer_name,
                    unlearning_task_id,
                    self.model.processed_task_ids,
                    self.model.backbone.masks,
                )

                # delete the update of the unlearning task
                self.delete_update(unlearning_task_id)

                # compensate if the first task
                self.compensate_layer_if_first_task(
                    layer_name,
                    unlearning_task_id,
                    if_first_task_layer,
                    next_masked_task_layer,
                )

            # update the model's state dict
            self.model.construct_parameters_from_updates()

        # for unlearning_task_id in self.unlearning_task_ids:
        #     self.model.backbones[f"{unlearning_task_id}"].load_state_dict(
        #         self.model.original_backbone_state_dict
        #     )

        # substract the update from the model
        # self.model.state_dict = self.
