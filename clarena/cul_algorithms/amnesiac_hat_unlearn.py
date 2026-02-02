r"""
The submoduule in `cul_algorithms` for AmnesiacHAT unlearning algorithm.
"""

__all__ = ["AmnesiacHATUnlearn"]

import logging

import torch
from rich.progress import track

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
        if_replay_repairing: bool,
        repair_batch_size: int | None,
        repair_num_steps: int | None,
        repair_strategy: str | None,
    ) -> None:
        r"""Initialize the unlearning algorithm with the continual learning model.

        **Args:**
        - **model** (`AmnesiacHAT`): the continual learning model (`CLAlgorithm` object which already contains the backbone and heads). It must be an `AmnesiacHAT` algorithm.
        - **if_backup_compensation** (`bool`): whether to perform compensation using the backup backbones after unlearning.
        - **compensate_order** (`str` | `None`): the order to compensate the affected tasks after unlearning (used when `if_backup_compensation` is 'True'), must be:
            - 'forward': from oldest to newest.
            - 'reverse': from newest to oldest.
        - **if_replay_repairing** (`bool`): whether to perform replay after unlearning.
        - **repair_batch_size** (`int` | `None`): the batch size used during the replay repairing after unlearning (used when `if_replay_repairing` is 'True').
        - **repair_num_steps** (`int` | `None`): the number of steps to perform replay repairing after unlearning (used when `if_replay_repairing` is 'True').
        - **repair_strategy** (`str` | `None`): the strategy to perform replay repairing after unlearning (used when `if_replay_repairing` is 'True'). must be:
            - 'joint': use joint replay data from all affected tasks for repairing.
            - 'sequential_finetuning': use replay data from each affected task one by one (from oldest to newest) for repairing, with only one epoch per task. This forms a mini continual learning process during repairing, where we use Finetuning (no additional operation) to learn each affected task sequentially.
            - 'sequential_adahat': use replay data from each affected task one by one (from oldest to newest) for repairing. This forms a mini continual learning process during repairing, where we use AdaHAT (no mask sparsity reg) to learn each affected task sequentially.
        """
        super().__init__(model=model)

        self.if_backup_compensation: bool = if_backup_compensation
        r"""Whether to perform compensation using the backup backbones after unlearning."""
        if self.if_backup_compensation:
            self.compensate_order: str = compensate_order
            r"""The order to compensate the affected tasks after unlearning."""

        self.if_replay_repairing: bool = if_replay_repairing
        r"""Whether to perform replay repairing after unlearning."""
        if self.if_replay_repairing:
            self.repair_batch_size: int = repair_batch_size
            r"""The batch size used during the replay repairing after unlearning."""
            self.repair_num_steps: int = repair_num_steps
            r"""The number of steps to perform replay repairing after unlearning."""
            self.repair_strategy: str = repair_strategy
            r"""The strategy to perform replay repairing after unlearning."""

    def compensate_by_backup(self) -> None:
        r"""Compensate the model using the backup backbones after unlearning."""

        unlearning_task_id = self.unlearning_task_ids[
            0
        ]  # only one unlearning task is supported for now

        task_ids_to_compensate = self.model.affected_tasks_after_unlearning()

        if len(task_ids_to_compensate) == 0:
            pylogger.info(
                "No tasks to compensate after unlearning. Skipping compensation phase."
            )
            return

        if self.compensate_order == "reverse":
            task_ids_to_compensate.reverse()  # compensate in reverse order

        pylogger.debug(
            "Affected tasks by unlearning task %s is %s, will be compensated in this order.",
            unlearning_task_id,
            task_ids_to_compensate,
        )

        for task_id_to_compensate in task_ids_to_compensate:

            # get the backup state dict
            backup_state_dict = self.model.backbone.backup_state_dicts[
                (unlearning_task_id, task_id_to_compensate)
            ]

            # only compensate the intersected neurons between the unlearning task and the affected task
            compensate_mask = self.model.backbone.combine_masks(
                [
                    self.model.backbone.masks[task_id_to_compensate],
                    self.model.backbone.masks[unlearning_task_id],
                ],
                mode="intersection",
            )

            for layer_name in self.model.backbone.weighted_layer_names:
                layer = self.model.backbone.get_layer_by_name(layer_name)

                # construct parameter-wise mask for the layer
                weight_mask, bias_mask = (
                    self.model.backbone.get_layer_measure_parameter_wise(
                        neuron_wise_measure=compensate_mask,
                        layer_name=layer_name,
                        aggregation_mode="min",
                    )
                )

                # compensate the parameters using the backup state dict
                target_device = layer.weight.device
                target_dtype = layer.weight.dtype
                if weight_mask.device != target_device:
                    weight_mask = weight_mask.to(device=target_device)
                backup_weight = backup_state_dict[
                    layer_name.replace("/", ".") + ".weight"
                ].to(device=target_device, dtype=target_dtype)
                layer.weight.data = torch.where(
                    weight_mask.bool(),
                    backup_weight,
                    layer.weight.data,
                )
                if layer.bias is not None:
                    if bias_mask.device != target_device:
                        bias_mask = bias_mask.to(device=target_device)
                    backup_bias = backup_state_dict[
                        layer_name.replace("/", ".") + ".bias"
                    ].to(device=target_device, dtype=layer.bias.dtype)
                    layer.bias.data = torch.where(
                        bias_mask.bool(),
                        backup_bias,
                        layer.bias.data,
                    )

            pylogger.debug(
                "Compensated affected task %s using backup from unlearning task %s.",
                task_id_to_compensate,
                unlearning_task_id,
            )

    def replay_repairing(self) -> None:
        r"""Repairing the model with replay after unlearning."""

        task_ids_to_repair = self.model.affected_tasks_after_unlearning()

        if len(task_ids_to_repair) == 0:
            pylogger.info(
                "No tasks to repair after unlearning. Skipping repairing phase."
            )
            return
        else:
            pylogger.info(
                "Starting replay repairing tasks %s, after unlearning: %s. Repair strategy: %s.",
                task_ids_to_repair,
                self.unlearning_task_ids,
                self.repair_strategy,
            )

        # align model device with replay buffer if needed (trainer may move model to CPU after fit)
        buffer_device = (
            self.model.memory_buffer.examples.device
            if self.model.memory_buffer.examples.numel() > 0
            else next(self.model.parameters()).device
        )
        if next(self.model.parameters()).device != buffer_device:
            self.model.to(buffer_device)
        model_device = next(self.model.parameters()).device

        def _move_optimizer_state_to_device(optimizer, device: torch.device) -> None:
            if isinstance(optimizer, (list, tuple)):
                for opt in optimizer:
                    _move_optimizer_state_to_device(opt, device)
                return
            opt = optimizer.optimizer if hasattr(optimizer, "optimizer") else optimizer
            for state in opt.state.values():
                for key, value in state.items():
                    if torch.is_tensor(value) and value.device != device:
                        state[key] = value.to(device)

        # build the unlearning mask that aggregates all unlearning tasks. This mask is used to clip the gradients during replay repairing to prevent changing unaffected parameters
        union_unlearning_mask = self.model.backbone.combine_masks(
            [
                self.model.backbone.masks[unlearning_task_id]
                for unlearning_task_id in self.unlearning_task_ids
            ],
            mode="union",
        )
        union_unlearning_mask = {
            layer_name: mask_tensor.to(model_device)
            if mask_tensor.device != model_device
            else mask_tensor
            for layer_name, mask_tensor in union_unlearning_mask.items()
        }

        if (
            self.repair_strategy == "sequential_finetuning"
            or self.repair_strategy == "sequential_adahat"
        ):

            summative_mask_for_previous_tasks_in_replay_repairing = {
                layer_name: torch.zeros(
                    self.model.backbone.get_layer_by_name(layer_name).weight.shape[0],
                    device=model_device,
                )
                for layer_name in self.model.backbone.weighted_layer_names
            }

            opt = self.model.optimizers()
            _move_optimizer_state_to_device(opt, model_device)

            total_tasks = len(task_ids_to_repair)
            for task_index, task_id_to_repair in enumerate(task_ids_to_repair, start=1):

                for s in track(
                    range(self.repair_num_steps),
                    description=f"Replay repairing task {task_id_to_repair} ({task_index}/{total_tasks})",
                    transient=True,
                ):
                    # get replay data for repairing from memory buffer
                    x_replay, _, logits_replay, _ = self.model.memory_buffer.get_data(
                        self.repair_batch_size,
                        included_tasks=[task_id_to_repair],
                    )
                    if x_replay.device != model_device:
                        x_replay = x_replay.to(model_device)
                    if logits_replay.device != model_device:
                        logits_replay = logits_replay.to(model_device)

                    # zero the gradients before forward pass in manual optimization mode
                    opt.zero_grad()

                    student_feature_replay = self.model.backbone(
                        x_replay,
                        stage="test",
                        test_task_id=task_id_to_repair,
                    )[0]

                    student_logits_replay = self.model.heads(
                        student_feature_replay, task_id=task_id_to_repair
                    )

                    with torch.no_grad():  # stop updating the previous heads
                        teacher_logits_replay = logits_replay

                    loss = self.model.distillation_reg(
                        student_logits=student_logits_replay,
                        teacher_logits=teacher_logits_replay,
                    )

                    self.model.manual_backward(loss)  # calculate the gradients

                    # Clip the gradients that are not masked by unlearning tasks. This is used in the unlearning replay repairing phase to make sure repairing only happens on the parameters affected by unlearning.

                    self.model.clip_grad_by_mask(
                        mask=union_unlearning_mask, aggregation_mode="min"
                    )

                    if self.repair_strategy == "sequential_adahat":
                        self.model.clip_grad_by_adjustment_in_replay_repairing(
                            summative_mask_for_previous_tasks_in_replay_repairing=summative_mask_for_previous_tasks_in_replay_repairing
                        )

                    # update parameters with the modified gradients
                    opt.step()

                summative_mask_for_previous_tasks_in_replay_repairing = {
                    layer_name: summative_mask_for_previous_tasks_in_replay_repairing[
                        layer_name
                    ]
                    + (
                        self.model.backbone.masks[task_id_to_repair][layer_name].to(
                            model_device
                        )
                        if self.model.backbone.masks[task_id_to_repair][
                            layer_name
                        ].device
                        != model_device
                        else self.model.backbone.masks[task_id_to_repair][layer_name]
                    )
                    for layer_name in self.model.backbone.weighted_layer_names
                }

        elif self.repair_strategy == "joint":

            opt = self.model.optimizers()
            _move_optimizer_state_to_device(opt, model_device)

            for s in track(
                range(self.repair_num_steps),
                description="Replay repairing (joint)",
                transient=True,
            ):

                # get replay data for repairing from memory buffer
                x_replay, _, logits_replay, task_labels_replay = (
                    self.model.memory_buffer.get_data(
                        self.repair_batch_size,
                        included_tasks=task_ids_to_repair,
                    )
                )
                if x_replay.device != model_device:
                    x_replay = x_replay.to(model_device)
                if logits_replay.device != model_device:
                    logits_replay = logits_replay.to(model_device)
                if task_labels_replay.device != model_device:
                    task_labels_replay = task_labels_replay.to(model_device)

                # zero the gradients before forward pass in manual optimization mode
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

                self.model.clip_grad_by_mask(
                    mask=union_unlearning_mask, aggregation_mode="min"
                )

                # update parameters with the modified gradients
                opt.step()

    def unlearn(self) -> None:
        r"""Unlearn the requested unlearning tasks (`self.unlearning_task_ids`) in the current task `self.task_id`."""

        # delete the corresponding parameter update records
        self.delete_update()

        for unlearning_task_id in self.unlearning_task_ids:

            # delete the data of the unlearning task from the memory buffer
            self.model.memory_buffer.delete_task(unlearning_task_id)

        if self.if_backup_compensation:
            self.compensate_by_backup()

        if self.if_replay_repairing:
            self.replay_repairing()

        # do not delete the masks and other related info of the unlearning tasks, as they may be needed in testing
