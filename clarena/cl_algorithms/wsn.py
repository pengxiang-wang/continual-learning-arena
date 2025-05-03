r"""
The submodule in `cl_algorithms` for [WSN (Winning Subnetworks)](https://proceedings.mlr.press/v162/kang22b/kang22b.pdf) algorithm.
"""

__all__ = ["WSN"]

import logging
from typing import Any

import torch
from torch import Tensor
from torch.utils.data import DataLoader

from clarena.backbones import WSNMaskBackbone
from clarena.cl_algorithms import CLAlgorithm
from clarena.cl_heads import HeadsCIL, HeadsTIL

# always get logger for built-in logging in each module
pylogger = logging.getLogger(__name__)


class WSN(CLAlgorithm):
    r"""WSN (Winning Subnetworks) algorithm.

    [WSN (Winning Subnetworks, 2022)](https://proceedings.mlr.press/v162/kang22b/kang22b.pdf) is an architecture-based continual learning algorithm. It trains learnable parameter-wise score and select the most scored $c\%$ of the network parameters to be used for each task.
    """

    def __init__(
        self,
        backbone: WSNMaskBackbone,
        heads: HeadsTIL | HeadsCIL,
        mask_percentage: float,
        parameter_score_init_mode: str = "default",
    ) -> None:
        r"""Initialise the WSN algorithm with the network.

        **Args:**
        - **backbone** (`WSNMaskBackbone`): must be a backbone network with WSN mask mechanism.
        - **heads** (`HeadsTIL` | `HeadsCIL`): output heads.
        - **mask_percentage** (`float`): the percentage of parameters to be used for each task. See $c\%$ in [WSN paper](https://proceedings.mlr.press/v162/kang22b/kang22b.pdf).
        - **parameter_score_init_mode** (`str`): the initialisation mode for parameter scores, should be one of the following:
            1. 'default': the default initialisation mode in original WSN codes.
            2. 'N01': standard normal distribution $N(0, 1)$.
            3. 'U01': uniform distribution $U(0, 1)$.

        """
        CLAlgorithm.__init__(self, backbone=backbone, heads=heads)

        self.mask_percentage: float = mask_percentage
        r"""Store the percentage of parameters to be used for each task."""
        self.parameter_score_init_mode: str = parameter_score_init_mode
        r"""Store the parameter score initialisation mode."""

        self.weight_masks: dict[str, dict[str, Tensor]] = {}
        r"""Store the binary weight mask of each previous task percentile gated from the weight score. Keys are task IDs (string type) and values are the corresponding mask. Each mask is a dict where keys are layer names and values are the binary mask tensor for the layer. The mask tensor has the same size (output features, input features) as weight. """
        self.bias_masks: dict[str, dict[str, Tensor]] = {}
        r"""Store the binary bias mask of each previous task percentile gated from the bias score. Keys are task IDs (string type) and values are the corresponding mask. Each mask is a dict where keys are layer names and values are the binary mask tensor for the layer. The mask tensor has the same size (output features, ) as bias. If the layer doesn't have bias, it is `None`."""

        self.cumulative_weight_mask_for_previous_tasks: dict[str, dict[str, Tensor]] = (
            {}
        )
        r"""Store the cumulative binary weight mask $\mathbf{M}_{t-1}$  of previous tasks $1, \cdots, t-1$, percentile gated from the weight score. Keys are task IDs (string type) and values are the corresponding mask. Each mask is a dict where keys are layer names and values are the binary mask tensor for the layer. The mask tensor has the same size (output features, input features) as weight. """
        self.cumulative_bias_mask_for_previous_tasks: dict[str, dict[str, Tensor]] = {}
        r"""Store the cumulative binary bias mask $\mathbf{M}_{t-1}$ of previous tasks $1, \cdots, t-1$, percentile gated from the bias score. Keys are task IDs (string type) and values are the corresponding mask. Each mask is a dict where keys are layer names and values are the binary mask tensor for the layer. The mask tensor has the same size (output features, ) as bias. If the layer doesn't have bias, it is `None`."""

        # set manual optimisation
        self.automatic_optimization = False

        WSN.sanity_check(self)

    def sanity_check(self) -> None:
        r"""Check the algorithm is correctly initialised.

        **Raises:**
        - **ValueError**: when backbone is not designed for WSN, or the mask percentage is not in (0, 1].
        """
        if not isinstance(self.backbone, WSNMaskBackbone):
            raise ValueError("The backbone should be an instance of WSNMaskBackbone.")

        if not (0 < self.mask_percentage <= 1):
            raise ValueError(
                f"Mask percentage should be in (0, 1], but got {self.mask_percentage}."
            )

    def on_train_start(self) -> None:
        r"""Initialise the parameter score before training the next task and initialise the cumulative mask at the beginning of first task."""

        self.backbone.initialise_parameter_score(mode=self.parameter_score_init_mode)

        # initialise the cumulative mask at the beginning of first task. This should not be called in `__init__()` method as the `self.device` is not available at that time.
        if self.task_id == 1:
            for layer_name in self.backbone.weighted_layer_names:
                layer = self.backbone.get_layer_by_name(
                    layer_name
                )  # get the layer by its name

                self.cumulative_weight_mask_for_previous_tasks[layer_name] = (
                    torch.zeros_like(layer.weight).to(self.device)
                )
                if layer.bias is not None:
                    self.cumulative_bias_mask_for_previous_tasks[layer_name] = (
                        torch.zeros_like(layer.bias).to(self.device)
                    )
                else:
                    self.cumulative_bias_mask_for_previous_tasks[layer_name] = None
                # the cumulative mask $\mathrm{M}_{t-1}$ is initialised as zeros mask ($t = 1$)

    def clip_grad_by_mask(
        self,
    ) -> None:
        r"""Clip the gradient by the cumulative mask. The gradient is multiplied by (1 - the cumulative previous mask) making previous masked parameters fixed. See equation (4) in [WSN paper](https://proceedings.mlr.press/v162/kang22b/kang22b.pdf)."""

        for layer_name in self.backbone.weighted_layer_names:
            layer = self.backbone.get_layer_by_name(layer_name)

            layer.weight.grad.data *= (
                1 - self.cumulative_weight_mask_for_previous_tasks[layer_name]
            )
            if layer.bias is not None:
                layer.bias.grad.data *= (
                    1 - self.cumulative_bias_mask_for_previous_tasks[layer_name]
                )

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
        - **task_id** (`int`| `None`): the task ID where the data are from. If the stage is 'train' or 'validation', it should be the current task `self.task_id`. If stage is 'test', it could be from any seen task. In TIL, the task IDs of test data are provided thus this argument can be used. WSN algorithm works only for TIL.

        **Returns:**
        - **logits** (`Tensor`): the output logits tensor.
        - **weight_mask** (`dict[str, Tensor]`): the weight mask for the current task. Key (`str`) is layer name, value (`Tensor`) is the mask tensor. The mask tensor has same (output features, input features) as weight.
        - **bias_mask** (`dict[str, Tensor]`): the bias mask for the current task. Key (`str`) is layer name, value (`Tensor`) is the mask tensor. The mask tensor has same (output features, ) as bias. If the layer doesn't have bias, it is `None`.
        - **activations** (`dict[str, Tensor]`): the hidden features (after activation) in each weighted layer. Key (`str`) is the weighted layer name, value (`Tensor`) is the hidden feature tensor. This is used for the continual learning algorithms that need to use the hidden features for various purposes.
        """
        feature, weight_mask, bias_mask, activations = self.backbone(
            input,
            stage=stage,
            mask_percentage=self.mask_percentage,
            test_mask=(
                (self.weight_masks[f"{task_id}"], self.bias_masks[f"{task_id}"])
                if stage == "test"
                else None
            ),
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
        self.clip_grad_by_mask()

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
        r"""Store the weight and bias masks and update the cumulative mask after training the task."""

        # get the weight and bias mask for the current task
        weight_mask_t = {}
        bias_mask_t = {}
        for layer_name in self.backbone.weighted_layer_names:
            layer = self.backbone.get_layer_by_name(layer_name)

            weight_mask_t[layer_name] = self.backbone.gate_fn.apply(
                self.backbone.weight_score_t[layer_name].weight, self.mask_percentage
            )
            if layer.bias is not None:
                bias_mask_t[layer_name] = self.backbone.gate_fn.apply(
                    self.backbone.bias_score_t[layer_name].weight.squeeze(
                        0
                    ),  # from (1, output_dim) to (output_dim, )
                    self.mask_percentage,
                )
            else:
                bias_mask_t[layer_name] = None

        # store the weight and bias mask for the current task
        self.weight_masks[f"{self.task_id}"] = weight_mask_t
        self.bias_masks[f"{self.task_id}"] = bias_mask_t

        # update the cumulative mask
        for layer_name in self.backbone.weighted_layer_names:
            layer = self.backbone.get_layer_by_name(layer_name)

            self.cumulative_weight_mask_for_previous_tasks[layer_name] = torch.max(
                self.cumulative_weight_mask_for_previous_tasks[layer_name],
                weight_mask_t[layer_name],
            )
            if layer.bias is not None:
                print(
                    self.cumulative_bias_mask_for_previous_tasks[layer_name].shape,
                    bias_mask_t[layer_name].shape,
                )
                self.cumulative_bias_mask_for_previous_tasks[layer_name] = torch.max(
                    self.cumulative_bias_mask_for_previous_tasks[layer_name],
                    bias_mask_t[layer_name],
                )
            else:
                self.cumulative_bias_mask_for_previous_tasks[layer_name] = None

        print(self.cumulative_bias_mask_for_previous_tasks)

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
        test_task_id = self.get_test_task_id_from_dataloader_idx(dataloader_idx)

        x, y = batch
        logits, weight_mask, bias_mask, activations = self.forward(
            x,
            stage="test",
            task_id=test_task_id,
        )  # use the corresponding head and mask to test (instead of the current task `self.task_id`)
        loss_cls = self.criterion(logits, y)
        acc = (logits.argmax(dim=1) == y).float().mean()

        return {
            "loss_cls": loss_cls,
            "acc": acc,  # Return metrics for lightning loggers callback to handle at `on_test_batch_end()`
        }
