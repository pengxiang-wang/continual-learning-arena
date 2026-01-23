r"""
The submoduule in `cul_algorithms` for AmnesiacHAT unlearning algorithm.
"""

__all__ = ["AmnesiacHATUnlearn"]

import logging

import torch

from clarena.cl_algorithms.amnesiac_hat import AmnesiacHAT
from clarena.cul_algorithms import AmnesiacCULAlgorithm

# always get logger for built-in logging in each module
pylogger = logging.getLogger(__name__)


class AmnesiacHATUnlearn(AmnesiacCULAlgorithm):
    r"""The base class of the AmnesiacHAT unlearning algorithm."""

    def __init__(
        self,
        model: AmnesiacHAT,
        if_backup_compensation: bool,
        compensate_order: str | None,
        if_replay_fixing: bool,
        fix_batch_size: int | None,
        fix_num_steps: int | None,
        fix_strategy: str | None,
    ) -> None:
        r"""Initialize the unlearning algorithm with the continual learning model.

        **Args:**
        - **model** (`AmnesiacHAT`): the continual learning model (`CLAlgorithm` object which already contains the backbone and heads). It must be an `AmnesiacHAT` algorithm.
        - **if_backup_compensation** (`bool`): whether to perform compensation using the backup model before unlearning.
        - **compensate_order** (`str`): the order to compensate the affected tasks upon unlearning. It can be either 'normal' (from oldest to newest) or 'reverse' (from newest to oldest).
        - **if_replay_fixing** (`bool`): whether to perform fixing with replay after unlearning.
        - **fix_batch_size** (`int`): the batch size used during the fixing with replay after unlearning.
        - **fix_num_steps** (`int`): the number of steps to perform fixing with replay after unlearning.
        - **fix_strategy** (`str`): the strategy to perform fixing with replay after unlearning. It can be:
            - **joint**: use joint replay data from all affected tasks for fixing.
            - **sequential**: use replay data from each affected task one by one for fixing.
        """
        super().__init__(model=model)

        self.if_backup_compensation: bool = if_backup_compensation
        r"""Whether to perform compensation using the backup model before unlearning."""
        self.compensate_order: str = compensate_order
        r"""The order to compensate the affected tasks upon unlearning. It can be either 'normal' (from oldest to newest) or 'reverse' (from newest to oldest)."""

        self.if_replay_fixing: bool = if_replay_fixing
        r"""Whether to perform fixing with replay after unlearning."""
        self.fix_batch_size: int = fix_batch_size
        r"""The batch size used during the fixing with replay after unlearning."""
        self.fix_num_steps: int = fix_num_steps
        r"""The number of steps to perform fixing with replay after unlearning."""
        self.fix_strategy: str = fix_strategy
        r"""The strategy to perform fixing with replay after unlearning."""

    def compensate_by_backup(self, unlearning_task_ids: list[int]) -> None:
        r"""Compensate the model before unlearning using the backup model.

        **Args:**
        - **unlearning_task_id** (`list[int]`): the ID of the unlearning task to delete the update.
        """

        unlearning_task_id = unlearning_task_ids[
            0
        ]  # only one unlearning task is supported for now

        task_ids_to_compensate = self.model.affected_tasks_upon_unlearning()

        if self.compensate_order == "reverse":
            task_ids_to_compensate.reverse()  # compensate in reverse order

        pylogger.info(f"Tasks to compensate order: {task_ids_to_compensate}")

        for affected_task_id in task_ids_to_compensate:

            print(affected_task_id, unlearning_task_id)

            print(self.model.backbone.backup_state_dicts.keys())
            backup_state_dict = self.model.backbone.backup_state_dicts[
                (unlearning_task_id, affected_task_id)
            ]
            compensate_mask = self.model.backbone.masks_intersection(
                [
                    self.model.backbone.masks[affected_task_id],
                    self.model.backbone.masks[unlearning_task_id],
                ]
            )

            for layer_name in self.model.backbone.weighted_layer_names:
                layer = self.model.backbone.get_layer_by_name(layer_name)

                weight_mask, bias_mask = (
                    self.model.backbone.get_layer_measure_parameter_wise(
                        neuron_wise_measure=compensate_mask,
                        layer_name=layer_name,
                        aggregation_mode="min",
                    )
                )

                layer.weight.data = torch.where(
                    weight_mask.bool(),
                    backup_state_dict[layer_name.replace("/", ".") + ".weight"],
                    layer.weight.data,
                )
                if layer.bias is not None:

                    layer.bias.data = torch.where(
                        bias_mask.bool(),
                        backup_state_dict[layer_name.replace("/", ".") + ".bias"],
                        layer.bias.data,
                    )

    # def compensate_by_backup(self, unlearning_task_ids: list[int]) -> None:
    #     r"""Compensate the model before unlearning using the backup model.

    #     **Args:**
    #     - **unlearning_task_id** (`list[int]`): the ID of the unlearning task to delete the update.
    #     """

    #     unlearning_task_id = unlearning_task_ids[
    #         0
    #     ]  # only one unlearning task is supported for now

    #     task_ids_to_compensate = self.model.affected_tasks_upon_unlearning()

    #     if self.compensate_order == "reverse":
    #         task_ids_to_compensate.reverse()  # compensate in reverse order

    #     pylogger.info(f"Tasks to compensate order: {task_ids_to_compensate}")

    #     compensate_masks = {}
    #     backup_state_dicts = {}

    #     for affected_task_id in task_ids_to_compensate:
    #         print(affected_task_id, unlearning_task_id)

    #         print(self.model.backbone.backup_state_dicts.keys())
    #         backup_state_dicts[affected_task_id] = self.model.backbone.backup_state_dicts[
    #             (unlearning_task_id, affected_task_id)
    #         ]
    #         compensate_masks[affected_task_id] = self.model.backbone.masks_intersection(
    #             [
    #                 self.model.backbone.masks[affected_task_id],
    #                 self.model.backbone.masks[unlearning_task_id],
    #             ]
    #         )

    #     def apply_random_compensation(
    #         current_tensor: torch.Tensor,
    #         candidate_tensors: list[torch.Tensor],
    #         mask_tensors: list[torch.Tensor],
    #     ) -> torch.Tensor:
    #         if not candidate_tensors:
    #             return current_tensor

    #         mask_stack = torch.stack(mask_tensors, dim=0).bool()
    #         union_mask = mask_stack.any(dim=0)
    #         if not union_mask.any().item():
    #             return current_tensor

    #         candidate_stack = torch.stack(candidate_tensors, dim=0)
    #         rand = torch.rand(mask_stack.shape, device=mask_stack.device)
    #         rand = rand.masked_fill(~mask_stack, -1.0)
    #         selected_idx = rand.argmax(dim=0)
    #         chosen = candidate_stack.gather(0, selected_idx.unsqueeze(0)).squeeze(0)

    #         return torch.where(union_mask, chosen, current_tensor)

    #     for layer_name in self.model.backbone.weighted_layer_names:
    #         layer = self.model.backbone.get_layer_by_name(layer_name)

    #         weight_candidates = []
    #         weight_masks = []
    #         bias_candidates = []
    #         bias_masks = []

    #         for affected_task_id in task_ids_to_compensate:
    #             weight_mask, bias_mask = (
    #                 self.model.backbone.get_layer_measure_parameter_wise(
    #                     neuron_wise_measure=compensate_masks[affected_task_id],
    #                     layer_name=layer_name,
    #                     aggregation_mode="min",
    #                 )
    #             )
    #             weight_candidates.append(
    #                 backup_state_dicts[affected_task_id][
    #                     layer_name.replace("/", ".") + ".weight"
    #                 ]
    #             )
    #             weight_masks.append(weight_mask)
    #             if layer.bias is not None:
    #                 bias_candidates.append(
    #                     backup_state_dicts[affected_task_id][
    #                         layer_name.replace("/", ".") + ".bias"
    #                     ]
    #                 )
    #                 bias_masks.append(bias_mask)

    #         layer.weight.data = apply_random_compensation(
    #             layer.weight.data,
    #             weight_candidates,
    #             weight_masks,
    #         )
    #         if layer.bias is not None:

    #             layer.bias.data = apply_random_compensation(
    #                 layer.bias.data,
    #                 bias_candidates,
    #                 bias_masks,
    #             )

    def fixing_with_replay(self, unlearning_task_ids: list[int]) -> None:
        r"""Fixing the model with replay after unlearning.

        **Args:**
        - **unlearning_task_id** (`list[int]`): the ID of the unlearning task to delete the update.
        """

        task_ids_to_fix = self.model.affected_tasks_upon_unlearning()

        pylogger.info(f"Tasks to fix after unlearning: {task_ids_to_fix}")

        if len(task_ids_to_fix) == 0:
            pylogger.info("No tasks to fix after unlearning. Skipping fixing step.")
            return

        print("flag")
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

        if self.fix_strategy == "sequential":

            summative_mask_for_previous_tasks_in_unlearning_fix = {
                layer_name: torch.zeros(
                    self.model.backbone.get_layer_by_name(layer_name).weight.shape[0]
                )
                for layer_name in self.model.backbone.weighted_layer_names
            }

            for task_id_to_fix in task_ids_to_fix:

                for s in range(self.fix_num_steps):
                    # get replay data for fixing from memory buffer
                    x_replay, _, logits_replay, _ = self.model.memory_buffer.get_data(
                        self.fix_batch_size,
                        included_tasks=[task_id_to_fix],
                    )

                    # zero the gradients before forward pass in manual optimization mode
                    opt = self.model.optimizers()
                    opt.zero_grad()

                    student_feature_replay = self.model.backbone(
                        x_replay,
                        stage="test",
                        test_task_id=task_id_to_fix,
                    )[0]

                    student_logits_replay = self.model.heads(
                        student_feature_replay, task_id=task_id_to_fix
                    )

                    with torch.no_grad():  # stop updating the previous heads

                        teacher_logits_replay = logits_replay

                    loss = self.model.distillation_reg(
                        student_logits=student_logits_replay,
                        teacher_logits=teacher_logits_replay,
                    )

                    self.model.manual_backward(loss)  # calculate the gradients

                    self.model.clip_grad_by_unlearning_mask(
                        unlearning_mask=unlearning_mask
                    )

                    # self.model.clip_grad_by_adjustment_in_unlearning_fix(
                    #     summative_mask_for_previous_tasks_in_unlearning_fix=summative_mask_for_previous_tasks_in_unlearning_fix
                    # )

                    # update parameters with the modified gradients
                    opt.step()

                summative_mask_for_previous_tasks_in_unlearning_fix = {
                    layer_name: summative_mask_for_previous_tasks_in_unlearning_fix[
                        layer_name
                    ]
                    + self.model.backbone.masks[task_id_to_fix][layer_name]
                    for layer_name in self.model.backbone.weighted_layer_names
                }

        elif self.fix_strategy == "joint":
            pylogger.info("Using joint replay fixing strategy.")
            for s in range(self.fix_num_steps):

                # get replay data for fixing from memory buffer
                x_replay, _, logits_replay, task_labels_replay = (
                    self.model.memory_buffer.get_data(
                        self.fix_batch_size,
                        included_tasks=task_ids_to_fix,
                    )
                )

                # zero the gradients before forward pass in manual optimization mode
                opt = self.model.optimizers()
                opt.zero_grad()

                student_feature_replay = torch.cat(
                    [
                        self.model.backbone(
                            x_replay[i].unsqueeze(0),
                            stage="test",
                            test_task_id=tid.item(),
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
        r"""Unlearn the requested unlearning tasks (`self.unlearning_task_ids`) in the current task `self.task_id`."""

        # delete updates from current parameters before removing update records
        self.model.construct_parameters_from_updates()

        # delete the corresponding parameter update records
        self.delete_update(self.unlearning_task_ids)

        for unlearning_task_id in self.unlearning_task_ids:

            # delete the data of the unlearning task from the memory buffer
            self.model.memory_buffer.delete_task(unlearning_task_id)

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

        if self.if_backup_compensation:
            self.compensate_by_backup(self.unlearning_task_ids)

        if self.if_replay_fixing:
            self.fixing_with_replay(self.unlearning_task_ids)
