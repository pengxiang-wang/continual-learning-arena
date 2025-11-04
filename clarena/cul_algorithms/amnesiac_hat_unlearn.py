r"""
The submoduule in `cul_algorithms` for AmnesiacHAT unlearning algorithm.
"""

__all__ = ["AmnesiacHATUnlearn"]

import logging

import torch
from torch import Tensor

from clarena.cl_algorithms.amnesiac_hat import AmnesiacHAT
from clarena.cul_algorithms import CULAlgorithm

# always get logger for built-in logging in each module
pylogger = logging.getLogger(__name__)


class AmnesiacHATUnlearn(CULAlgorithm):
    r"""The base class of the AmnesiacHAT unlearning algorithm."""

    def __init__(self, model: AmnesiacHAT) -> None:
        r"""Initialize the unlearning algorithm with the continual learning model.

        **Args:**
        - **model** (`AmnesiacHAT`): the continual learning model (`CLAlgorithm` object which already contains the backbone and heads). It must be an `AmnesiacHAT` algorithm.
        """
        super().__init__(model=model)

    def delete_update(self, unlearning_task_id: str) -> None:
        r"""Delete the update of the specified unlearning task.
        **Args:**
        - **unlearning_task_id** (`str`): the ID of the unlearning task to delete the update.
        """
        if unlearning_task_id not in self.model.parameters_task_update:
            pylogger.warning(
                f"Attempted to delete update for task {unlearning_task_id}, but it was not found."
            )
            return
        del self.model.parameters_task_update[unlearning_task_id]
        pylogger.info(
            f"Deleted parameter update for unlearning task '{unlearning_task_id}'."
        )

    def compensate_update_with_innovation(
        self,
        unlearning_task_id: int,
        if_first_task_neurons: dict[str, Tensor],
        next_task_for_neurons: dict[str, Tensor],
    ) -> None:
        r"""
        Compensate updates for tasks that reused neurons first used by the unlearning task.

        **Args:**
        - **unlearning_task_id** (int): ID of the task being unlearned.
        - **if_first_task_neurons** (dict[str, Tensor]): for each layer, tensor (1 if neuron first used by this task).
        - **next_task_for_neurons** (dict[str, Tensor]): for each layer, tensor with next task ID using each neuron (or -1 if none).

        For each neuron that was first used by the unlearning task, the update of the next
        task that uses this neuron will be compensated (divided by adjustment_intensity)
        in that neuron's region.
        """
        adjustment_intensity = self.model.adjustment_intensity
        pylogger.info(f"Start compensation for unlearning task {unlearning_task_id}")

        torch.set_printoptions(threshold=1000000)

        # Step 1: Collect all unique next task IDs across layers
        unique_next_tasks = set()
        for layer_name, first_task_mask in if_first_task_neurons.items():
            next_task_map = next_task_for_neurons[layer_name]
            first_used_neurons = first_task_mask.bool()
            next_tasks_in_layer = next_task_map[first_used_neurons].unique()
            unique_next_tasks.update(
                next_tasks_in_layer[next_tasks_in_layer != -1].tolist()
            )

        # Step 2: For each affected next task
        for next_task_id in unique_next_tasks:
            pylogger.info(f"Compensating updates for next task {next_task_id}")

            neuron_mask = {}
            for layer_name, first_task_mask in if_first_task_neurons.items():
                next_task_map = next_task_for_neurons[layer_name]

                # print("DXXXX", next_task_map, first_used_neurons)
                if not isinstance(first_task_mask, torch.Tensor) or not isinstance(
                    next_task_map, torch.Tensor
                ):
                    continue

                first_used_neurons = first_task_mask.bool()
                neuron_mask[layer_name] = (
                    next_task_map == next_task_id
                ) & first_used_neurons

                print("layer_name", layer_name)
                print("nueron mask", neuron_mask[layer_name])
                print("task mask 1", self.model.backbone.masks[1][layer_name])
                print("task mask 2", self.model.backbone.masks[2][layer_name])

            for layer_name in self.model.backbone.weighted_layer_names:

                # Step 3: Parameter-wise mask
                weight_mask_expanded, bias_mask_expanded = (
                    self.model.backbone.get_layer_measure_parameter_wise(
                        neuron_wise_measure=neuron_mask,
                        layer_name=layer_name,
                        aggregation_mode="min",
                    )
                )

                # print(
                #     layer_name,
                #     weight_mask_expanded.sum(),
                #     bias_mask_expanded.sum(),
                # )

                # Step 4: Apply compensation
                weight_key = layer_name.replace("/", ".") + ".weight"
                bias_key = layer_name.replace("/", ".") + ".bias"
                weight_update = self.model.parameters_task_update[next_task_id][
                    weight_key
                ]

                bias_update = self.model.parameters_task_update[next_task_id][bias_key]

                print("bias_update", bias_update)

                weight_update = weight_update.clone()
                bias_update = bias_update.clone()

                weight_update[weight_mask_expanded] /= adjustment_intensity
                bias_update[bias_mask_expanded] /= adjustment_intensity

                self.model.parameters_task_update[next_task_id][
                    weight_key
                ] = weight_update
                self.model.parameters_task_update[next_task_id][bias_key] = bias_update

            pylogger.info(f"Finished compensation for next task {next_task_id}")

        pylogger.info(
            f"All compensations done for unlearning task {unlearning_task_id}"
        )

    def unlearn(self) -> None:
        r"""Unlearn the requested unlearning tasks in the current task `self.task_id`."""
        if not self.unlearning_task_ids:
            return

        pylogger.info(
            f"Starting unlearning process for tasks: {self.unlearning_task_ids}..."
        )

        for unlearning_task_id in self.unlearning_task_ids:
            pylogger.info(f"Processing unlearning for task '{unlearning_task_id}'.")

            if unlearning_task_id not in self.model.parameters_task_update:
                pylogger.warning(
                    f"Task '{unlearning_task_id}' has no parameter update to unlearn. Skipping."
                )
                continue

            if_first_task_neurons = {}
            next_task_for_neurons = {}

            # Step 1: Gather all necessary information for all relevant layers
            for layer_name in self.model.backbone.weighted_layer_names:
                if_first_task_neurons[layer_name] = self.model.if_first_task_layer(
                    layer_name,
                    unlearning_task_id,
                    self.model.processed_task_ids,
                    self.model.backbone.masks,
                )
                next_task_for_neurons[layer_name] = self.model.next_masked_task_layer(
                    layer_name,
                    unlearning_task_id,
                    self.model.processed_task_ids,
                    self.model.backbone.masks,
                )

            # Step 2: Perform compensation by transferring the scaled update (YOUR INNOVATION)
            # self.compensate_update_with_innovation(
            #     unlearning_task_id, if_first_task_neurons, next_task_for_neurons
            # )

            # Step 3: Delete the update of the unlearning task AFTER compensation
            self.delete_update(unlearning_task_id)

            # # Step 4: Nullify the stored mask for the unlearned task by setting it to zeros
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
        # Step 5: After all compensations and deletions are done, reconstruct the model parameters once
        pylogger.info(
            "Reconstructing model parameters from all remaining task updates..."
        )
        self.model.construct_parameters_from_updates()

        pylogger.info("Unlearning process finished.")
