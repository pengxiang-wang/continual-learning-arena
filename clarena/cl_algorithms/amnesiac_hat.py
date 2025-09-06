r"""
The submodule in `cl_algorithms` for [AmnesiacHAT (Amnesiac Hard Attention to the Task)]() algorithm.
"""

__all__ = ["AmnesiacHAT"]

import logging
from copy import deepcopy

import torch
from torch import Tensor
from typing import Any

from clarena.backbones import HATMaskBackbone
from clarena.cl_algorithms.adahat import AdaHAT
from clarena.heads import HeadsTIL

# always get logger for built-in logging in each module
pylogger = logging.getLogger(__name__)


class AmnesiacHAT(AdaHAT):
    r"""AmnesiacHAT (Amnesiacard Attention to the Task) algorithm.

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
   、non_algorih
        )


save additional hyperparameters
       self.save_hyperparameters("adjustment_intensity", "epsilon")

        self.original_backbone_state_dict: dict = deepcopy(backbone.state_dict())
        r"""Store the original backbone network state dict. """

        self.parameters_task_update: dict[str, dict[str, Tensor]] = {}
        r"""Store the parameters update in each task. Keys are task IDs and values are the corresponding parameters update tensor. Each tensor is a dict where keys are parameter names and values are the corresponding parameter update tensor for the layer. """

    def construct_parameters_from_updates(self):
        r"""Construct the parameters of the model from parameters task updates."""
        updated_state_dict = deepcopy(self.original_backbone_state_dict())
        for task_id, param_update in self.parameters_task_update.items():
            for layer_name, param_tensor in param_update.items():
                updated_state_dict[layer_name] = (
                    updated_state_dict[layer_name] + param_tensor
                )

        self.backbone.load_state_dict(updated_state_dict)

    def if_first_task_layer(
        self,
        layer_name: str,
        task_id: int,
        processed_task_ids: list[int],
        masks: dict[str, dict[str, Tensor]],
    ) -> Tensor:
        r"""Check if the task is the first task in each neuron of a layer.

        **Args:**
        - **layer_name** (`str`): the name of the layer to check.
        - **task_id** (`int`): the task ID to check.
        - **processed_task_ids** (`list[int]`): the list of seen task IDs.
        - **masks** (`dict[str, dict[str, Tensor]]`): the masks for each task.

        **Returns:**
        - **is_first_task_layer** (`Tensor`): a tensor indicating if the task is the first task in each neuron of the layer. The tensor is of the same shape as the mask for the layer, with values of 1 for the first task and 0 for others.
        """
        if_first_task_layer = torch.zeros_like(masks[f"{task_id}"][f"{layer_name}"])

        for seen_task_id in processed_task_ids:
            if seen_task_id == task_id:
                break

            # if the mask is not zero, then it is not the first task
            if_first_task_layer += masks[f"{seen_task_id}"][f"{layer_name}"]

        return if_first_task_layer

    def next_masked_task_layer(
        self,
        layer_name: str,
        task_id: int,
        processed_task_ids: list[int],
        masks: dict[str, dict[str, Tensor]],
    ) -> dict[str, Tensor]:
        r"""Get the next masked task ID of a designated task ID for each neuron of a layer.

        **Args:**
        - **task_id** (`int`): the task ID to start.
        - **processed_task_ids** (`list[int]`): the list of seen task IDs.
        - **masks** (`dict[str, dict[str, Tensor]]`): the masks for each task.

        **Returns:**
        - **next_masked_task_layer** (`bool`): `True` if the task is the first task, otherwise `False`.
        """
        next_masked_task_layer = [None] * len(s)

        for seen_task_id in processed_task_ids:
            if seen_task_id == task_id:
                # if the mask is not zero, then it is not the first task
                next_masked_task_layer = [
                    (
                        seen_task_id
                        if masks[f"{seen_task_id}"][f"{layer_name}"][l] == 1
                        else l
                    )
                    for l in next_masked_task_layer
                ]

        return next_masked_task_layer

    def on_train_end(self):
        r"""Store the parameters update of a task at the end of its training."""
        AdaHAT.on_train_end(self)

        # start from the current parameters
        parameters_task_t_update = deepcopy(self.backbone.state_dict())

        for layer_name, param_tensor in param_update.items():

            # substract the update from the parameters
            for task_id, param_update in self.parameters_task_update.items():
                parameters_task_t_update[layer_name] -= param_tensor

            # substract the original backbone state dict
            parameters_task_t_update[
                layer_name
            ] -= self.backbone.original_backbone_state_dict[layer_name]

            # now the rest is the update for the task

        # store the parameters update for the task in the dict
        self.parameters_task_update[f"{self.task_id}"] = parameters_task_t_update

        # optimize storage
