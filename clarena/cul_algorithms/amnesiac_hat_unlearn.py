r"""
The submoduule in `cul_algorithms` for AmnesiacHAT unlearning algorithm.
"""

__all__ = ["AmnesiacHATUnlearn"]

import logging

import torch

from clarena.cl_algorithms.amnesiac_hat import AmnesiacHAT
from clarena.cul_algorithms import CULAlgorithm

# always get logger for built-in logging in each module
pylogger = logging.getLogger(__name__)


class AmnesiacHATUnlearn(CULAlgorithm):
    r"""The base class of the AmnesiacHAT unlearning algorithm."""

    def __init__(self, model: AmnesiacHAT, fix_batch_size: int) -> None:
        r"""Initialize the unlearning algorithm with the continual learning model.

        **Args:**
        - **model** (`AmnesiacHAT`): the continual learning model (`CLAlgorithm` object which already contains the backbone and heads). It must be an `AmnesiacHAT` algorithm.
        - **fix_batch_size** (`int`): the batch size used during the fixing with replay after unlearning.
        """
        super().__init__(model=model)

        self.fix_batch_size: int = fix_batch_size
        r"""The batch size used during the fixing with replay after unlearning."""

    def delete_update(self, unlearning_task_ids: list[int]) -> None:
        r"""Delete the update of the specified unlearning task.

        **Args:**
        - **unlearning_task_id** (`list[int]`): the ID of the unlearning task to delete the update.
        """

        for unlearning_task_id in self.unlearning_task_ids:
            if unlearning_task_id not in self.model.parameters_task_update:
                pylogger.warning(
                    "Attempted to delete update for task %d, but it was not found.",
                    unlearning_task_id,
                )
                continue

            # delete the parameter update for the unlearning task so that it won't be used in future parameter constructions
            del self.model.parameters_task_update[unlearning_task_id]

            # delete the data of the unlearning task from the memory buffer
            self.model.memory_buffer.delete_task(unlearning_task_id)

        pylogger.info(
            "Deleted parameter update for unlearning task %s.", unlearning_task_ids
        )

    def fixing_with_replay(self, unlearning_task_ids: list[int]) -> None:
        r"""Fixing the model with replay after unlearning.

        **Args:**
        - **unlearning_task_id** (`list[int]`): the ID of the unlearning task to delete the update.
        """

        unlearning_mask = {}
        for layer_name in self.model.backbone.weighted_layer_names:
            mask_tensors = torch.stack(
                [
                    self.model.backbone.masks[unlearning_task_id][layer_name]
                    for unlearning_task_id in unlearning_task_ids
                ],
                dim=0,
            )
            # take element-wise maximum across all unlearning tasks to build a single mask
            unlearning_mask[layer_name] = torch.max(mask_tensors, dim=0).values

        for s in range(10):

            # get replay data for fixing from memory buffer
            x_replay, _, logits_replay, task_labels_replay = (
                self.model.memory_buffer.get_data(
                    self.fix_batch_size,
                    included_tasks=self.model.affected_tasks_upon_unlearning(),
                )
            )

            # zero the gradients before forward pass in manual optimization mode
            opt = self.model.optimizers()
            opt.zero_grad()

            student_feature_replay = torch.cat(
                [
                    self.model.backbone(
                        x_replay[i].unsqueeze(0), stage="test", test_task_id=tid.item()
                    )[0]
                    for i, tid in enumerate(task_labels_replay)
                ]
            )

            student_logits_replay = torch.cat(
                [
                    self.model.heads(
                        student_feature_replay[i].unsqueeze(0), task_id=tid
                    )
                    for i, tid in enumerate(task_labels_replay)
                ]
            )

            with torch.no_grad():  # stop updating the previous heads

                teacher_logits_replay = logits_replay

            loss = self.model.distillation_reg(
                student_logits=student_logits_replay,
                teacher_logits=teacher_logits_replay,
            )

            self.model.manual_backward(loss)  # calculate the gradients

            self.model.clip_grad_by_unlearning_mask(unlearning_mask=unlearning_mask)

            # update parameters with the modified gradients
            opt.step()

    def unlearn(self) -> None:
        r"""Unlearn the requested unlearning tasks in the current task `self.task_id`."""
        if not self.unlearning_task_ids:
            return

        pylogger.info(
            "Starting unlearning process for tasks: %s...", self.unlearning_task_ids
        )

        self.delete_update(self.unlearning_task_ids)

        # # nullify the stored mask for the unlearned task by setting it to zeros
        # if unlearning_task_id in self.model.backbone.masks:
        #     pylogger.info(
        #         f"Nullifying stored mask for unlearned task '{unlearning_task_id}' by setting it to zeros."
        #     )
        #     # Iterate through all layer masks for the task and set them to zero
        #     for module_name, mask_tensor in self.model.backbone.masks[
        #         unlearning_task_id
        #     ].items():
        #         self.model.backbone.masks[unlearning_task_id][module_name] = (
        #             torch.zeros_like(mask_tensor)
        #         )

        # after all compensations and deletions are done, reconstruct the model parameters once
        self.model.construct_parameters_from_updates()

        # fixing with replay must be done after parameter reconstruction
        self.fixing_with_replay(self.unlearning_task_ids)

        pylogger.info("Unlearning process finished.")
