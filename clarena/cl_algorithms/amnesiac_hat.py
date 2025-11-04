r"""
The submodule in `cl_algorithms` for [AmnesiacHAT (Amnesiac Hard Attention to the Task)]() algorithm.
"""

__all__ = ["AmnesiacHAT"]

import logging
from copy import deepcopy
from typing import Any

import torch
from torch import Tensor

from clarena.backbones import HATMaskBackbone
from clarena.cl_algorithms import AdaHAT, UnlearnableCLAlgorithm
from clarena.heads import HeadDIL, HeadsCIL, HeadsTIL

# always get logger for built-in logging in each module
pylogger = logging.getLogger(__name__)


class AmnesiacHAT(AdaHAT, UnlearnableCLAlgorithm):
    r"""AmnesiacHAT (Amnesiac Hard Attention to the Task) algorithm.

    A variant of [HAT (Hard Attention to the Task)](http://proceedings.mlr.press/v80/serra18a) enabling HAT with unlearning ability, based on the [AdaHAT (Adaptive Hard Attention to the Task)](https://link.springer.com/chapter/10.1007/978-3-031-70352-2_9) algorithm.

    This algorithm is paired with the `AmnesiacHATUnlearn` unlearning algorithm.

    We implement AmnesiacHAT as a subclass of `AdaHAT` algorithm.
    """

    def __init__(
        self,
        backbone: HATMaskBackbone,
        heads: HeadsTIL,
        adjustment_mode: str,
        adjustment_intensity: float,
        s_max: float,
        clamp_threshold: float,
        mask_sparsity_reg_factor: float,
        mask_sparsity_reg_mode: str = "original",
        task_embedding_init_mode: str = "N01",
        epsilon: float = 0.1,
        non_algorithmic_hparams: dict[str, Any] = {},
    ) -> None:
        r"""Initialize the AmnesiacHAT algorithm with the network.

        **Args:**
        - **backbone** (`HATMaskBackbone`): must be a backbone network with HAT mask mechanism.
        - **heads** (`HeadsTIL`): output heads. AmnesiacHAT algorithm only supports TIL (Task-Incremental Learning).
        - **adjustment_mode** (`str`): the strategy of adjustment i.e. the mode of gradient clipping; one of:
            1. 'adahat': set the gradients of parameters linking to masked units to a soft adjustment rate in the original AdaHAT approach. This is the way that AdaHAT does, which allowes the part of network for previous tasks to be updated slightly. See equation (8) and (9) chapter 3.1 in the [AdaHAT paper](https://link.springer.com/chapter/10.1007/978-3-031-70352-2_9).
            2. 'adahat_no_sum': set the gradients of parameters linking to masked units to a soft adjustment rate in the original AdaHAT approach, but without considering the information of parameter importance i.e. summative mask. This is the way that one of the AdaHAT ablation study does. See chapter 4.3 in the [AdaHAT paper](https://link.springer.com/chapter/10.1007/978-3-031-70352-2_9).
            3. 'adahat_no_reg': set the gradients of parameters linking to masked units to a soft adjustment rate in the original AdaHAT approach, but without considering the information of network sparsity i.e. mask sparsity regularization value. This is the way that one of the AdaHAT ablation study does. See chapter 4.3 in the [AdaHAT paper](https://link.springer.com/chapter/10.1007/978-3-031-70352-2_9).
        - **adjustment_intensity** (`float`): hyperparameter, control the overall intensity of gradient adjustment. It's the $\alpha$ in equation (9) in the [AdaHAT paper](https://link.springer.com/chapter/10.1007/978-3-031-70352-2_9).
        - **s_max** (`float`): hyperparameter, the maximum scaling factor in the gate function. See chapter 2.4 "Hard Attention Training" in the [HAT paper](http://proceedings.mlr.press/v80/serra18a).
        - **clamp_threshold** (`float`): the threshold for task embedding gradient compensation. See chapter 2.5 "Embedding Gradient compensation" in the [HAT paper](http://proceedings.mlr.press/v80/serra18a).
        - **mask_sparsity_reg_factor** (`float`): hyperparameter, the regularization factor for mask sparsity.
        - **mask_sparsity_reg_mode** (`str`): the mode of mask sparsity regularization; one of:
            1. 'original' (default): the original mask sparsity regularization in HAT paper.
            2. 'cross': the cross version mask sparsity regularization.
        - **task_embedding_init_mode** (`str`): the initialization method for task embeddings; one of:
            1. 'N01' (default): standard normal distribution $N(0, 1)$.
            2. 'U-11': uniform distribution $U(-1, 1)$.
            3. 'U01': uniform distribution $U(0, 1)$.
            4. 'U-10': uniform distribution $U(-1, 0)$.
            5. 'last': inherit task embedding from last task.
        - **epsilon** (`float`): the value added to network sparsity to avoid division by zero appeared in equation (9) in the [AdaHAT paper](https://link.springer.com/chapter/10.1007/978-3-031-70352-2_9).
        - **non_algorithmic_hparams** (`dict[str, Any]`): non-algorithmic hyperparameters that are not related to the algorithm itself are passed to this `LightningModule` object from the config, such as optimizer and learning rate scheduler configurations. They are saved for Lightning APIs from `save_hyperparameters()` method. This is useful for the experiment configuration and reproducibility.
        """
        super().__init__(
            backbone=backbone,
            heads=heads,
            adjustment_mode=adjustment_mode,
            adjustment_intensity=adjustment_intensity,
            s_max=s_max,
            clamp_threshold=clamp_threshold,
            mask_sparsity_reg_factor=mask_sparsity_reg_factor,
            mask_sparsity_reg_mode=mask_sparsity_reg_mode,
            task_embedding_init_mode=task_embedding_init_mode,
            epsilon=epsilon,
            non_algorithmic_hparams=non_algorithmic_hparams,
        )

        # save additional hyperparameters
        self.save_hyperparameters("adjustment_intensity", "epsilon")

        self.original_backbone_state_dict: dict = deepcopy(backbone.state_dict())
        r"""Store the original backbone network state dict. """

        self.parameters_task_update: dict[int, dict[str, Tensor]] = {}
        r"""Store the parameters update in each task. Keys are task IDs and values are the corresponding parameters update tensor. Each tensor is a dict where keys are parameter names and values are the corresponding parameter update tensor for the layer. """

    def construct_parameters_from_updates(self):
        r"""Construct the parameters of the model from parameters task updates."""
        # FIXED: `self.original_backbone_state_dict` is an attribute, not a method. Removed the parentheses.
        updated_state_dict = deepcopy(self.original_backbone_state_dict)
        for task_id, param_update in self.parameters_task_update.items():
            for layer_name, param_tensor in param_update.items():
                if layer_name in updated_state_dict:
                    updated_state_dict[layer_name] += param_tensor
                else:
                    pylogger.warning(
                        f"Layer {layer_name} from task {task_id} update not found in original state dict."
                    )

        self.backbone.load_state_dict(updated_state_dict)

    def if_first_task_layer(
        self,
        layer_name: str,
        task_id: int,
        processed_task_ids: list[int],
        masks: dict[int, dict[str, Tensor]],
    ) -> Tensor:
        r"""Check if the task is the first task in each neuron of a layer.

        **Args:**
        - **layer_name** (`str`): the name of the **parameter** to check (e.g., 'fc.0.weight').
        - **task_id** (`int`): the task ID to check.
        - **processed_task_ids** (`list[int]`): the list of seen task IDs.
        - **masks** (`dict[str, dict[str, Tensor]]`): the masks for each task.

        **Returns:**
        - **is_first_task_layer** (`Tensor`): a tensor indicating if the task is the first task in each neuron of the layer. The tensor is of the same shape as the mask for the layer, with values of 1 for the first task and 0 for others.
        """
        # --- MODIFICATION START ---
        # Convert parameter name (e.g., 'fc.0.weight') to module name (e.g., 'fc.0')
        # layer_name = layer_name.replace('.weight', '').replace('.bias', '')

        # # Robustly create the initial zero tensor based on the module's architecture.
        # try:
        #     layer = self.backbone.get_layer_by_name(layer_name)
        #     num_units = layer.weight.shape[0]
        #     device = layer.weight.device
        #     previous_tasks_mask_sum = torch.zeros(num_units, device=device)
        # except Exception as e:
        #     pylogger.error(f"Could not create initial mask for module '{layer_name}'. Error: {e}")
        #     return torch.empty(0)
        # --- MODIFICATION END ---
        previous_tasks_mask_sum = torch.zeros_like(masks[task_id][layer_name])

        for seen_task_id in processed_task_ids:
            if seen_task_id == task_id:
                break

            # --- MODIFICATION START ---
            # Use layer_name for lookups and add a check for tensor type.
            mask_item = masks[seen_task_id][layer_name]
            if isinstance(mask_item, Tensor):
                previous_tasks_mask_sum += mask_item
            else:
                pylogger.warning(
                    f"Skipping non-tensor mask for task {seen_task_id}, module {layer_name}. Found type: {type(mask_item)}"
                )
            # --- MODIFICATION END ---

        # FIXED: The original logic was incomplete.
        # A neuron is used for the first time by the current task if and only if
        # the sum of masks from all previous tasks is 0 at that neuron's position.
        # We convert the resulting boolean tensor to float (0.0 or 1.0).
        is_first_task_layer = (previous_tasks_mask_sum == 0) & (
            masks[task_id][layer_name] == 1
        )

        return is_first_task_layer

    # REWRITTEN: This method was fundamentally broken with NameError and flawed logic.
    # The following is a plausible implementation based on the method's name, assuming
    # it's intended to find the ID of the *next* task that uses each neuron.
    def next_masked_task_layer(
        self,
        layer_name: str,
        task_id: int,
        processed_task_ids: list[int],
        masks: dict[int, dict[str, Tensor]],
    ) -> Tensor:
        r"""Get the next masked task ID of a designated task ID for each neuron of a layer.

        For each neuron in the specified layer, this method finds the ID of the first task
        that comes after `task_id` in `processed_task_ids` and uses that neuron.

        **Args:**
        - **layer_name** (`str`): The name of the **parameter** to check (e.g., 'fc.0.weight').
        - **task_id** (`int`): the task ID to start the search from.
        - **processed_task_ids** (`list[int]`): the sorted list of seen task IDs.
        - **masks** (`dict[str, dict[str, Tensor]]`): the masks for each task.

        **Returns:**
        - **next_task_ids** (`Tensor`): A tensor of the same shape as the layer's mask. Each element
          contains the ID of the next task using that neuron, or a default value (e.g., -1) if no subsequent
          task uses it.
        """
        # --- MODIFICATION START ---
        # Convert parameter name (e.g., 'fc.0.weight') to module name (e.g., 'fc.0')
        # layer_name = layer_name.replace('.weight', '').replace('.bias', '')

        # Use layer_name for lookups and add a check for tensor type.
        current_mask = masks[task_id][layer_name]
        # if not isinstance(current_mask, Tensor):
        #     pylogger.error(
        #         f"Could not find a valid mask tensor for task {task_id}, module {layer_name}."
        #     )
        #     # Attempt to get a shape from the architecture as a fallback
        #     try:
        #         layer = self.backbone.get_layer_by_name(layer_name)
        #         num_units = layer.weight.shape[0]
        #         device = layer.weight.device
        #         return torch.full((num_units,), -1, dtype=torch.long, device=device)
        #     except Exception:
        #         return torch.empty(
        #             0, dtype=torch.long
        #         )  # Return empty if all else fails
        # --- MODIFICATION END ---

        # Initialize with a default value indicating no next task is found
        next_task_ids = torch.full_like(current_mask, -1, dtype=torch.long)

        # Find the index of the current task to slice the list of subsequent tasks
        try:
            current_task_index = processed_task_ids.index(task_id)
            future_task_ids = processed_task_ids[current_task_index + 1 :]
        except ValueError:
            # Current task_id not in the list, cannot find next tasks
            return next_task_ids

        # Iterate backwards from the last future task to the first
        # This ensures we find the *earliest* next task for each neuron
        for next_task_id in reversed(future_task_ids):
            # --- MODIFICATION START ---
            # Use layer_name for lookups and check for tensor type
            next_mask = masks[next_task_id][layer_name]
            if isinstance(next_mask, Tensor):
                # Where the next mask is active (1), update the neuron's next task ID
                next_task_ids[next_mask == 1] = next_task_id
            # --- MODIFICATION END ---

        return next_task_ids

    def on_train_start(self):
        r"""Store the current state dict at the start of training."""
        super().on_train_start()
        self.state_dict_task_start = deepcopy(self.backbone.state_dict())

    # REWRITTEN: This method had multiple NameErrors, an AttributeError, and incorrect loop logic.
    # It has been rewritten to correctly isolate and store the parameter updates for the current task.
    def on_train_end(self):
        r"""Store the parameters update of a task at the end of its training."""
        # First, call the parent method
        super().on_train_end()

        # The goal is to calculate: Update(t) = FinalState(t) - OriginalState - sum(Update(i) for i < t)

        current_state_dict = self.backbone.state_dict()
        parameters_task_t_update = {}

        # Iterate over each layer in the current model's state
        for layer_name, current_param_tensor in current_state_dict.items():
            # Ensure the layer exists in the original state dict
            if layer_name in self.state_dict_task_start:
                # # Start with the current parameters and subtract the original parameters
                # # This gives the total update from task 0 to the current task t.
                # total_update_so_far = (
                #     current_param_tensor - self.original_backbone_state_dict[layer_name]
                # )

                # # Now, subtract the updates from all previously completed tasks
                # # to isolate the update for the current task.
                # update_from_previous_tasks = torch.zeros_like(total_update_so_far)
                # for task_id, prev_update_dict in self.parameters_task_update.items():
                #     if layer_name in prev_update_dict:
                #         update_from_previous_tasks += prev_update_dict[layer_name]

                # # The update for the current task is the total update minus previous updates
                # update_for_current_task = (
                #     total_update_so_far - update_from_previous_tasks
                # )
                # parameters_task_t_update[layer_name] = (
                #     update_for_current_task.cpu()
                # )  # use .cpu() to optimize GPU memory
                parameters_task_t_update[layer_name] = (
                    current_param_tensor - self.state_dict_task_start[layer_name]
                ).cpu()  # use .cpu() to optimize GPU memory

        # Store the isolated parameters update for the current task
        self.parameters_task_update[self.task_id] = parameters_task_t_update

        # print("parameters_task_t_update", parameters_task_t_update)
