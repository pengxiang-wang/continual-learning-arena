r"""
The submodule in `cl_algorithms` for [CBP (Continual Backpropagation)](https://www.nature.com/articles/s41586-024-07711-7) algorithm.
"""

__all__ = ["CBP"]

import logging
from typing import Any

import torch
from torch import Tensor

from clarena.backbones import CLBackbone
from clarena.cl_algorithms import Finetuning
from clarena.cl_heads import HeadsCIL, HeadsTIL
from clarena.utils.transforms import min_max_normalise

# always get logger for built-in logging in each module
pylogger = logging.getLogger(__name__)


class CBP(Finetuning):
    r"""CBP (Continual Backpropagation) algorithm.

    [CBP (Continual Backpropagation, 2024)](https://www.nature.com/articles/s41586-024-07711-7) is a continual learning approach that reinitialises a small number of units during training, using an utility measures to determine which units to reinitialise. It aims to address loss of plasticity problem for learning new tasks, yet not very well solve the catastrophic forgetting problem in continual learning.

    We implement CBP as a subclass of Finetuning algorithm, as CBP has the same `forward()`, `training_step()`, `validation_step()` and `test_step()` method as `Finetuning` class.
    """

    def __init__(
        self,
        backbone: CLBackbone,
        heads: HeadsTIL | HeadsCIL,
        replacement_rate: float,
        maturity_threshold: int,
        utility_decay_rate: float,
    ) -> None:
        r"""Initialise the Finetuning algorithm with the network. It has no additional hyperparamaters.

        **Args:**
        - **backbone** (`CLBackbone`): backbone network.
        - **heads** (`HeadsTIL` | `HeadsCIL`): output heads.
        - **replacement_rate** (`float`): the replacement rate of units. It is the precentage of units to be reinitialised during training.
        - **maturity_threshold** (`int`): the maturity threshold of units. It is the number of training steps before a unit can be reinitialised.
        - **utility_decay_rate** (`float`): the utility decay rate of units. It is the rate at which the utility of a unit decays over time.
        """
        Finetuning.__init__(self, backbone=backbone, heads=heads)

        self.replacement_rate: float = replacement_rate
        r"""Store the replacement rate of units. """
        self.maturity_threshold: int = maturity_threshold
        r"""Store the maturity threshold of units. """
        self.utility_decay_rate: float = utility_decay_rate
        r"""Store the utility decay rate of units. """

        self.contribution_utility: dict[str, Tensor] = {}
        r"""Store the contribution utility of units. See equation (1) in the [continual backpropagation paper](https://www.nature.com/articles/s41586-024-07711-7). Keys are layer names and values are the utility tensor for the layer. The utility tensor is the same size as the feature tensor with size (number of units). """
        self.num_replacements: dict[str, int] = {}
        r"""Store the number of replacements of units in each layer. Keys are layer names and values are the number of replacements for the layer. """
        self.age: dict[str, Tensor] = {}
        r"""Store the age of units. Keys are layer names and values are the age tensor for the layer. The age tensor is the same size as the feature tensor with size (1, number of units). """

    def on_train_start(self) -> None:
        r"""Initialise the utility, number of replacements and age for each layer as zeros."""

        # initialise the utility, number of replacements and age as zeros at the beginning of first task. This should not be called in `__init__()` method as the `self.device` is not available at that time.
        if self.task_id == 1:
            for layer_name in self.backbone.weighted_layer_names:
                layer = self.backbone.get_layer_by_name(
                    layer_name
                )  # get the layer by its name
                num_units = layer.weight.shape[0]

                self.contribution_utility[layer_name] = torch.zeros(num_units).to(
                    self.device
                )
                self.num_replacements[layer_name] = 0
                self.age[layer_name] = torch.zeros(num_units).to(self.device)

    def on_train_batch_end(
        self, outputs: dict[str, Any], batch: Any, batch_idx: int
    ) -> None:
        r"""Update the contribution utility and age of units after each training step, and conduct reinitialisation of units based on utility measures. This is the core of the CBP algorithm.

        **Args:**
        - **outputs** (`dict[str, Any]`): the outputs of the training step, which is the returns of the `training_step()` method in the `CLAlgorithm`.
        - **batch** (`Any`): the training data batch.
        - **batch_idx** (`int`): the index of the current batch. This is for the file name of mask figures.
        """

        hidden_features = outputs["hidden_features"]

        for layer_name in self.backbone.weighted_layer_names:
            # layer-wise operation

            layer = self.backbone.get_layer_by_name(
                layer_name
            )  # get the layer by its name

            # update age
            self.age[layer_name] += 1

            # calculate current contribution utility
            current_contribution_utility = (
                torch.mean(
                    torch.abs(hidden_features[layer_name]),
                    dim=0,  # average the features over batch samples
                )
                * torch.sum(
                    torch.abs(layer.weight),
                    dim=1,  # sum over the output dimension
                )
            ).detach()
            current_contribution_utility = min_max_normalise(
                current_contribution_utility
            )  # normalise the utility to [0,1] to avoid linearly increasing utility

            # update utility
            self.contribution_utility[layer_name] = (
                self.utility_decay_rate * self.contribution_utility[layer_name]
                + (1 - self.utility_decay_rate) * current_contribution_utility
            )

            # find eligible units
            eligible_mask = self.age[layer_name] > self.maturity_threshold
            eligible_indices = torch.where(eligible_mask)[0]

            # update the number of replacements
            num_eligible_units = eligible_indices.numel()
            self.num_replacements[layer_name] += int(
                self.replacement_rate * num_eligible_units
            )

            # if the number of replacements is greater than 1, execute the replacement
            if self.num_replacements[layer_name] > 1:

                # find the unit with smallest utility among eligible units
                replaced_unit_idx = eligible_indices[
                    torch.argmin(
                        self.contribution_utility[layer_name][eligible_indices]
                        / self.age[layer_name][eligible_indices]
                    ).item()
                ]

                # reinitialise the input weights of the unit
                preceding_layer_name = self.backbone.preceding_layer_name(layer_name)
                if preceding_layer_name is not None:
                    preceding_layer = self.backbone.get_layer_by_name(
                        preceding_layer_name
                    )
                    with torch.no_grad():

                        preceding_layer.weight[:, replaced_unit_idx] = torch.rand_like(
                            preceding_layer.weight[:, replaced_unit_idx]
                        )

                # reinitalise the output weights of the unit
                with torch.no_grad():
                    layer.weight[replaced_unit_idx] = torch.rand_like(
                        layer.weight[replaced_unit_idx]
                    )

                # reinitialise utility
                self.contribution_utility[layer_name][replaced_unit_idx] = 0.0

                # reintialise age
                self.age[layer_name][replaced_unit_idx] = 0

                # update the number of replacements
                self.num_replacements[layer_name] -= 1
