r"""
The submodule in `cl_algorithms` for [NISPA (Neuro-Inspired Stability-Plasticity Adaptation)](https://proceedings.mlr.press/v162/gurbuz22a/gurbuz22a.pdf) algorithm.
"""

__all__ = ["NISPA"]

import logging
import math
from copy import deepcopy
from typing import Any

import torch
from torch import Tensor
from torch.utils.data import DataLoader

from clarena.backbones import NISPAMaskBackbone
from clarena.cl_algorithms import CLAlgorithm
from clarena.cl_heads import HeadsTIL

# always get logger for built-in logging in each module
pylogger = logging.getLogger(__name__)


class NISPA(CLAlgorithm):
    r"""NISPA (Neuro-Inspired Stability-Plasticity Adaptation) algorithm.

    [NISPA (Neuro-Inspired Stability-Plasticity Adaptation), 2022](https://proceedings.mlr.press/v162/gurbuz22a/gurbuz22a.pdf) is an architecture-based continual learning algorithm. It
    """

    def __init__(
        self,
        backbone: NISPAMaskBackbone,
        heads: HeadsTIL,
        num_epochs_per_phase: int,
        accuracy_fall_threshold: float,
        k: float,
    ) -> None:
        r"""Initialise the NISPA algorithm with the network.

        **Args:**
        - **backbone** (`NISPAMaskBackbone`): must be a backbone network with NISPA mask mechanism.
        - **heads** (`HeadsTIL`): output heads. NISPA algorithm only supports TIL (Task-Incremental Learning).
        - **num_epochs_per_phase** (`int`): the number of epochs per phase. One phase consists of several epochs. At the end of each phase, unit selection and rewire is performed.
        - **accuracy_fall_threshold** (`float`): the accuracy fall threshold to stop the phases. If the accuracy of the current task is below this (best accuracy of phases - this threshold), the phase will stop, or else it will continue next phase.
        - **k** (`float`): hyperparameter for scheduling activation fraction $\tau$. See equation (3) in [NISPA paper](https://proceedings.mlr.press/v162/gurbuz22a/gurbuz22a.pdf).

        """
        CLAlgorithm.__init__(self, backbone=backbone, heads=heads)

        self.num_epochs_per_phase: int = num_epochs_per_phase
        r"""Store the number of epochs per phase."""
        self.accuracy_fall_threshold: float = accuracy_fall_threshold
        r"""Store the accuracy fall threshold to stop the phases."""
        self.k: float = k
        r"""Store the hyperparameter for scheduling activation fraction $\tau$. See equation (3) in [NISPA paper](https://proceedings.mlr.press/v162/gurbuz22a/gurbuz22a.pdf)."""

        self.candidate_stable_unit_mask_t: dict[str, Tensor] = {}
        r"""Store the candidate stable unit mask for each layer. Key (`str`) is layer name, value (`Tensor`) is the mask tensor. The mask tensor has size (number of units, )."""
        self.stable_unit_mask_t: dict[str, Tensor] = {}
        r"""Store the stable unit mask for each layer. Key (`str`) is layer name, value (`Tensor`) is the mask tensor. The mask tensor has size (number of units, )."""
        self.plastic_unit_mask_t: dict[str, Tensor] = {}
        r"""Store the plastic unit mask for each layer. Key (`str`) is layer name, value (`Tensor`) is the mask tensor. The mask tensor has size (number of units, )."""

        self.best_phase_acc: float
        r"""Store the best accuracy of the current task in the current phase."""
        self.phase_idx: int
        r"""Store the index of the current phase."""

    def on_train_start(self) -> None:
        r"""Initialise the masks at the beginning of first task."""

        self.best_phase_acc = 0.0
        self.phase_idx = 0

        # initialise the masks at the beginning of first task. This should not be called in `__init__()` method as the `self.device` is not available at that time.
        if self.task_id == 1:

            # initialise NISPA backbone weight mask
            self.backbone.initialise_parameter_mask()

            for layer_name in self.backbone.weighted_layer_names:
                layer = self.backbone.get_layer_by_name(
                    layer_name
                )  # get the layer by its name
                num_units = layer.weight.shape[0]

                # initialise unit masks in NISPA algorithm
                self.candidate_stable_unit_mask_t[layer_name] = torch.zeros(
                    num_units
                ).to(self.device)

                self.stable_unit_mask_t[layer_name] = torch.zeros(num_units).to(
                    self.device
                )
                self.plastic_unit_mask_t[layer_name] = torch.ones(num_units).to(
                    self.device
                )

    def clip_grad_by_frozen_mask(
        self,
    ) -> None:
        r"""Clip the gradient by the frozen parameter mask. The gradient is multiplied by (1 - the frozen parameter mask) making masked parameters fixed. See "frozen connections" in [NISPA paper](https://proceedings.mlr.press/v162/gurbuz22a/gurbuz22a.pdf)."""

        for layer_name in self.backbone.weighted_layer_names:
            layer = self.backbone.get_layer_by_name(layer_name)

            layer.weight.grad.data *= 1 - self.frozen_weight_mask_t[layer_name]
            if layer.bias is not None:
                layer.bias.grad.data *= 1 - self.frozen_bias_mask_t[layer_name]

    def forward(
        self,
        input: torch.Tensor,
        stage: str,
        task_id: int | None = None,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        r"""The forward pass for data from task `task_id`. Note that it is nothing to do with `forward()` method in `nn.Module`.

        **Args:**
        - **input** (`Tensor`): The input tensor from data.
        - **stage** (`str`): the stage of the forward pass, should be one of the following:
            1. 'train': training stage.
            2. 'validation': validation stage.
            3. 'test': testing stage.Applies only to training stage. For other stages, it is default `None`.
        - **task_id** (`int`| `None`): the task ID where the data are from. If the stage is 'train' or 'validation', it should be the current task `self.task_id`. If stage is 'test', it could be from any seen task. In TIL, the task IDs of test data are provided thus this argument can be used. NISPA algorithm works only for TIL.

        **Returns:**
        - **logits** (`Tensor`): the output logits tensor.
        - **weight_mask** (`dict[str, Tensor]`): the weight mask for the current task. Key (`str`) is layer name, value (`Tensor`) is the mask tensor. The mask tensor has same (output features, input features) as weight.
        - **bias_mask** (`dict[str, Tensor]`): the bias mask for the current task. Key (`str`) is layer name, value (`Tensor`) is the mask tensor. The mask tensor has same (output features, ) as bias. If the layer doesn't have bias, it is `None`.
        - **activations** (`dict[str, Tensor]`): the hidden features (after activation) in each weighted layer. Key (`str`) is the weighted layer name, value (`Tensor`) is the hidden feature tensor. This is used for the continual learning algorithms that need to use the hidden features for various purposes.
        """
        feature, weight_mask, bias_mask, activations = self.backbone(
            input,
            stage=stage,
        )
        logits = self.heads(feature, task_id)

        return (
            logits
            if self.if_forward_func_return_logits_only
            else (logits, weight_mask, bias_mask, activations)
        )

    def training_step(self, batch: Any) -> dict[str, Tensor]:
        r"""Training step for current task `self.task_id`.

        **Args:**
        - **batch** (`Any`): a batch of training data.

        **Returns:**
        - **outputs** (`dict[str, Tensor]`): a dictionary contains loss and other metrics from this training step. Key (`str`) is the metrics name, value (`Tensor`) is the metrics. Must include the key 'loss' which is total loss in the case of automatic optimization, according to PyTorch Lightning docs. For WSN, it includes 'weight_mask' and 'bias_mask' for logging.
        """
        x, y = batch

        # zero the gradients before forward pass in manual optimisation mode
        opt = self.optimizers()
        opt.zero_grad()

        # classification loss
        logits, weight_mask, bias_mask, activations = self.forward(
            x, stage="train", task_id=self.task_id
        )
        loss_cls = self.criterion(logits, y)

        # total loss
        loss = loss_cls

        # backward step (manually)
        self.manual_backward(loss)  # calculate the gradients
        # WSN hard clip gradients by the cumulative masks. See equation (4) in [WSN paper](https://proceedings.mlr.press/v162/kang22b/kang22b.pdf).
        self.clip_grad_by_frozen_mask()

        # update parameters with the modified gradients
        opt.step()

        # accuracy of the batch
        acc = (logits.argmax(dim=1) == y).float().mean()

        return {
            "loss": loss,  # Return loss is essential for training step, or backpropagation will fail
            "loss_cls": loss_cls,
            "acc": acc,
            "activations": activations,
            "weight_mask": weight_mask,  # Return other metrics for lightning loggers callback to handle at `on_train_batch_end()`
            "bias_mask": bias_mask,
        }

    def on_train_end(self) -> None:
        r"""."""
        # freeze and reinit
        pass

    def validation_step(self, batch: Any) -> dict[str, Tensor]:
        r"""Validation step for current task `self.task_id`.

        **Args:**
        - **batch** (`Any`): a batch of validation data.

        **Returns:**
        - **outputs** (`dict[str, Tensor]`): a dictionary contains loss and other metrics from this validation step. Key (`str`) is the metrics name, value (`Tensor`) is the metrics.
        """
        x, y = batch
        logits, weight_mask, bias_mask, activations = self.forward(
            x, stage="validation", task_id=self.task_id
        )
        loss_cls = self.criterion(logits, y)
        acc = (logits.argmax(dim=1) == y).float().mean()

        return {
            "loss_cls": loss_cls,
            "acc": acc,  # Return metrics for lightning loggers callback to handle at `on_validation_batch_end()`
        }

    def on_validation_epoch_end(self, outputs: dict[str, Any]):
        r"""

        **Args:**
        - **outputs** (`dict[str, Any]`): the outputs of the training step, which is the returns of the `training_step()` method in the `CLAlgorithm`.

        """

        if self.current_epoch % self.num_epochs_per_phase == 0:

            cached_state = {
                "model": deepcopy(self.state_dict()),
                "optimizer": self.trainer.optimizers[0].state_dict(),
            }

            val_acc = outputs["acc"]  # accuracy of current epoch
            if val_acc > self.best_phase_acc:
                self.best_phase_acc = val_acc  # update the best accuracy

            if (
                val_acc >= self.best_phase_acc - self.accuracy_fall_threshold
            ):  # stoppping criterion. See 3.8 "Stopping Criterion" in [NISPA paper](https://proceedings.mlr.press/v162/gurbuz22a/gurbuz22a.pdf)

                tau = 0.5 * (
                    1 + math.cos(self.phase_idx * math.pi / self.k)
                )  # calculate the fraction of activation to select the candidate stable units. See equation (3) in [NISPA paper](https://proceedings.mlr.press/v162/gurbuz22a/gurbuz22a.pdf)

                self.select_candidate_stable_units(
                    activation_fraction=tau,
                )

                # cached_stable_unit_mask = union(
                #     deepcopy(self.stable_unit_mask_t), self.candidate_stable_unit_mask_t
                # )

                # rewire the connections
                # num_connections_dropped = self.drop_connections_plastic_to_stable()
                # self.grow_new_connections(num_connections_dropped)

            else:
                pass

    def select_candidate_stable_units(self, activation_fraction: float) -> None:
        r"""Select candidate stable units that have highest summed activations in each layer. Thresholded by the fraction of summed activations of all units in the layer. See chapter 3.3 "Selecting Candidate Stable Units" in [NISPA paper](https://proceedings.mlr.press/v162/gurbuz22a/gurbuz22a.pdf).

        **Args:**
        - **activation_fraction** (`float`): the activation fraction threshold. See equation (3) in [NISPA paper](https://proceedings.mlr.press/v162/gurbuz22a/gurbuz22a.pdf). The value should be between 0 and 1.
        """
        train_dataloader = self.trainer.datamodule.train_dataloader()

        summed_activations = {
            layer_name: torch.zeros_like(self.stable_unit_mask_t[layer_name])
            for layer_name in self.weighted_layer_names
        }

        for x, y in train_dataloader:

            # move data to device manually
            x = x.to(self.device)
            y = y.to(self.device)

            batch_size = len(y)
            num_data += batch_size
            _, activations = self.forward(x, stage="validation")

            for layer_name in self.weighted_layer_names:
                summed_activations[layer_name] = (
                    summed_activations[layer_name] + activations[layer_name]
                )

        for layer_name in self.backbone.weighted_layer_names:
            summed_activations_layer = summed_activations[layer_name]
            summed_activations_layer_threshold = (
                sum(summed_activations_layer) * activation_fraction
            )

            cumulative_sum = 0.0
            while cumulative_sum < summed_activations_layer_threshold:
                max_idx = torch.argmax(summed_activations_layer, dim=0)
                max_value = summed_activations_layer[max_idx]

                cumulative_sum = cumulative_sum + max_value

                self.candidate_stable_unit_mask_t[layer_name][max_idx] = 1.0

    def drop_connections_plastic_to_stable(self):
        r"""Drop the connections from plastic units to stable units."""
        for layer_name in self.weighted_layer_names:
            self.plastic_unit_mask_t[layer_name] = {
                "weight": torch.ones(
                    self.weight_mask_t[layer_name]["weight"].size()
                ).to(self.device),
                "bias": torch.ones(self.weight_mask_t[layer_name]["bias"].size()).to(
                    self.device
                ),
            }

    def grow_new_connections(self, num):
        r"""Grow new connections for the plastic units.

        **Args:**
        - **num** (`int`): the number of new connections to be grown.
        """
        for layer_name in self.weighted_layer_names:
            self.plastic_unit_mask_t[layer_name] = {
                "weight": torch.ones(
                    self.weight_mask_t[layer_name]["weight"].size()
                ).to(self.device),
                "bias": torch.ones(self.weight_mask_t[layer_name]["bias"].size()).to(
                    self.device
                ),
            }

    def test_step(
        self, batch: DataLoader, batch_idx: int, dataloader_idx: int = 0
    ) -> dict[str, Tensor]:
        r"""Test step for current task `self.task_id`, which tests for all seen tasks indexed by `dataloader_idx`.

        **Args:**
        - **batch** (`Any`): a batch of test data.
        - **dataloader_idx** (`int`): the task ID of seen tasks to be tested. A default value of 0 is given otherwise the LightningModule will raise a `RuntimeError`.

        **Returns:**
        - **outputs** (`dict[str, Tensor]`): a dictionary contains loss and other metrics from this test step. Key (`str`) is the metrics name, value (`Tensor`) is the metrics.
        """

        x, y = batch
        logits, weight_mask, bias_mask, activations = self.forward(
            x,
            stage="test",
        )  # use the corresponding head and mask to test (instead of the current task `self.task_id`)
        loss_cls = self.criterion(logits, y)
        acc = (logits.argmax(dim=1) == y).float().mean()

        return {
            "loss_cls": loss_cls,
            "acc": acc,  # Return metrics for lightning loggers callback to handle at `on_test_batch_end()`
        }
