r"""
The submodule in `cl_algorithms` for [EWC (Elastic Weight Consolidation) algorithm](https://www.pnas.org/doi/10.1073/pnas.1611835114).
"""

__all__ = ["EWC", "AmnesiacEWC"]

import logging
from copy import deepcopy
from typing import Any

import torch
from torch import Tensor, nn

from clarena.backbones import CLBackbone
from clarena.cl_algorithms import AmnesiacCLAlgorithm, Finetuning
from clarena.cl_algorithms.regularizers import ParameterChangeReg
from clarena.heads import HeadDIL, HeadsCIL, HeadsTIL

# always get logger for built-in logging in each module
pylogger = logging.getLogger(__name__)


class EWC(Finetuning):
    r"""[EWC (Elastic Weight Consolidation)](https://www.pnas.org/doi/10.1073/pnas.1611835114) algorithm.

    A regularization-based approach that calculates the fisher information as parameter importance for the previous tasks and penalizes the current task loss with the importance of the parameters.

    We implement EWC as a subclass of Finetuning algorithm, as EWC has the same `forward()`, `validation_step()` and `test_step()` method as `Finetuning` class.
    """

    def __init__(
        self,
        backbone: CLBackbone,
        heads: HeadsTIL | HeadsCIL | HeadDIL,
        parameter_change_reg_factor: float,
        when_calculate_fisher_information: str,
        non_algorithmic_hparams: dict[str, Any] = {},
        **kwargs,
    ) -> None:
        r"""Initialize the EWC algorithm with the network.

        **Args:**
        - **backbone** (`CLBackbone`): backbone network.
        - **heads** (`HeadsTIL` | `HeadsCIL` | `HeadDIL`): output heads.
        - **parameter_change_reg_factor** (`float`): the parameter change regularization factor. It controls the strength of preventing forgetting.
        - **when_calculate_fisher_information** (`str`): when to calculate the fisher information. It should be one of the following:
            1. 'train_end': calculate the fisher information at the end of training of the task.
            2. 'train': accumulate the fisher information in the training step of the task.
        - **non_algorithmic_hparams** (`dict[str, Any]`): non-algorithmic hyperparameters that are not related to the algorithm itself are passed to this `LightningModule` object from the config, such as optimizer and learning rate scheduler configurations. They are saved for Lightning APIs from `save_hyperparameters()` method. This is useful for the experiment configuration and reproducibility.
        - **kwargs**: Reserved for multiple inheritance.

        """
        super().__init__(
            backbone=backbone,
            heads=heads,
            non_algorithmic_hparams=non_algorithmic_hparams,
            **kwargs,
        )

        # save additional algorithmic hyperparameters
        self.save_hyperparameters(
            "parameter_change_reg_factor",
            "when_calculate_fisher_information",
        )

        self.parameter_importance: dict[int, dict[str, Tensor]] = {}
        r"""The parameter importance of each previous task. Keys are task IDs and values are the corresponding importance. Each importance entity is a dict where keys are parameter names (named by `named_parameters()` of the `nn.Module`) and values are the importance tensor for the layer. It has the same shape as the parameters of the layer.
        """

        self.previous_task_backbones: dict[int, nn.Module] = {}
        r"""The backbone models of the previous tasks. Keys are task IDs and values are the corresponding models. Each model is a `nn.Module` backbone after the corresponding previous task was trained.

        Some would argue that since we could store the model of the previous tasks, why don't we test the task directly with the stored model, instead of doing the less easier EWC thing? The thing is, EWC only uses the model of the previous tasks to train current and future tasks, which aggregate them into a single model. Once the training of the task is done, the storage for those parameters can be released. However, this make the future tasks not able to use EWC anymore, which is a disadvantage for EWC.
        """
        self.parameter_importance_heads: dict[int, dict[str, Tensor]] = {}
        r"""The head parameter importance of each previous task (DIL only)."""
        self.previous_task_heads: dict[int, nn.Module] = {}
        r"""The head models of the previous tasks (DIL only)."""

        self.parameter_change_reg_factor = parameter_change_reg_factor
        r"""The parameter change regularization factor."""
        self.parameter_change_reg = ParameterChangeReg(
            factor=parameter_change_reg_factor,
        )
        r"""Initialize and store the parameter change regularizer."""

        self.when_calculate_fisher_information: str = when_calculate_fisher_information
        r"""When to calculate the fisher information."""
        self.num_data: int
        r"""The number of data used to calculate the fisher information. It is used to average the fisher information over the data."""

        # set manual optimization because we need to access gradients to calculate the fisher information in the training step
        self.automatic_optimization = False

        EWC.sanity_check(self)

    def sanity_check(self) -> None:
        r"""Sanity check."""
        if self.parameter_change_reg_factor <= 0:
            raise ValueError(
                f"The parameter change regularization factor should be positive, but got {self.parameter_change_reg_factor}."
            )

    def on_train_start(self) -> None:
        r"""Initialize the parameter importance and num of data counter."""

        self.parameter_importance[self.task_id] = {}
        for param_name, param in self.backbone.named_parameters():
            self.parameter_importance[self.task_id][param_name] = 0 * param.data
        if isinstance(self.heads, HeadDIL):
            self.parameter_importance_heads[self.task_id] = {}
            for param_name, param in self.heads.named_parameters():
                self.parameter_importance_heads[self.task_id][param_name] = (
                    0 * param.data
                )
        self.num_data = 0

    def training_step(self, batch: Any) -> dict[str, Tensor]:
        r"""Training step for current task `self.task_id`."""
        x, y = batch

        opt = self.optimizers()
        opt.zero_grad()

        # classification loss
        logits, activations = self.forward(x, stage="train", task_id=self.task_id)
        loss_cls = self.criterion(logits, y)

        batch_size = len(y)
        self.num_data += batch_size

        # accumulate fisher information during training step if specified
        if self.when_calculate_fisher_information == "train":
            # Use autograd.grad to get explicit gradients for Fisher accumulation without
            # relying on global .backward() state in manual optimization.
            backbone_params: list[tuple[str, Tensor]] = []
            for param_name, param in self.backbone.named_parameters():
                if not param.requires_grad:
                    continue
                backbone_params.append((param_name, param))
            if isinstance(self.heads, HeadDIL):
                head_params: list[tuple[str, Tensor]] = []
                for param_name, param in self.heads.named_parameters():
                    if not param.requires_grad:
                        continue
                    head_params.append((param_name, param))
            else:
                head_params = []

            if backbone_params:
                grads = torch.autograd.grad(
                    loss_cls,
                    [param for _, param in backbone_params],
                    retain_graph=True,
                    allow_unused=True,
                )
                for (param_name, _), grad in zip(backbone_params, grads):
                    if grad is None:
                        continue
                    self.parameter_importance[self.task_id][param_name] += (
                        batch_size * grad.detach() ** 2
                    )

            if head_params:
                grads = torch.autograd.grad(
                    loss_cls,
                    [param for _, param in head_params],
                    retain_graph=True,
                    allow_unused=True,
                )
                for (param_name, _), grad in zip(head_params, grads):
                    if grad is None:
                        continue
                    self.parameter_importance_heads[self.task_id][param_name] += (
                        batch_size * grad.detach() ** 2
                    )

        # regularization loss. See equation (3) in the [EWC paper](https://www.pnas.org/doi/10.1073/pnas.1611835114)
        ewc_reg = 0.0
        for previous_task_id, previous_backbone in self.previous_task_backbones.items():
            ewc_reg += 0.5 * self.parameter_change_reg(
                target_model=self.backbone,
                ref_model=previous_backbone,
                weights=self.parameter_importance[previous_task_id],
            )
            if isinstance(self.heads, HeadDIL):
                ewc_reg += 0.5 * self.parameter_change_reg(
                    target_model=self.heads,
                    ref_model=self.previous_task_heads[previous_task_id],
                    weights=self.parameter_importance_heads[previous_task_id],
                )

        # total loss
        loss = loss_cls + ewc_reg

        self.manual_backward(loss)
        opt.step()

        # predicted labels
        preds = logits.argmax(dim=1)

        # accuracy of the batch
        acc = (preds == y).float().mean()

        return {
            "preds": preds,
            "loss": loss,  # return loss is essential for training step, or backpropagation will fail
            "loss_cls": loss_cls,
            "ewc_reg": ewc_reg,
            "acc": acc,
            "activations": activations,
        }

    def on_train_end(self) -> None:
        r"""Calculate the fisher information as parameter importance and store the backbone model after the training of a task."""

        # calculate fisher information at the end of training if specified
        if self.when_calculate_fisher_information == "train_end":
            fisher, fisher_heads, fisher_num_data = (
                self.accumulate_fisher_information_on_train_end()
            )
            self.parameter_importance[self.task_id] = fisher
            if fisher_heads is not None:
                self.parameter_importance_heads[self.task_id] = fisher_heads
            num_data = fisher_num_data
        else:
            num_data = self.num_data

        # no matter when we calculate the fisher information, we need to average it over the number of data
        for param_name, param in self.backbone.named_parameters():
            self.parameter_importance[self.task_id][param_name] /= num_data
        if isinstance(self.heads, HeadDIL):
            for param_name, param in self.heads.named_parameters():
                self.parameter_importance_heads[self.task_id][param_name] /= num_data

        # store the backbone model after training the task
        previous_backbone = deepcopy(self.backbone)
        previous_backbone.eval()
        self.previous_task_backbones[self.task_id] = previous_backbone
        if isinstance(self.heads, HeadDIL):
            previous_heads = deepcopy(self.heads)
            previous_heads.eval()
            self.previous_task_heads[self.task_id] = previous_heads

    def accumulate_fisher_information_on_train_end(
        self,
    ) -> tuple[dict[str, Tensor], dict[str, Tensor] | None, int]:
        r"""Accumulate the fisher information as the parameter importance for the learned task `self.task_id` at the end of its training."""
        fisher_information_t = {}
        fisher_information_heads: dict[str, Tensor] | None = None
        num_data = 0

        self.eval()
        last_task_train_dataloaders = self.trainer.datamodule.train_dataloader()

        for param_name, param in self.backbone.named_parameters():
            fisher_information_t[param_name] = torch.zeros_like(param)
        if isinstance(self.heads, HeadDIL):
            fisher_information_heads = {}
            for param_name, param in self.heads.named_parameters():
                fisher_information_heads[param_name] = torch.zeros_like(param)

        for x, y in last_task_train_dataloaders:
            x = x.to(self.device)
            y = y.to(self.device)
            batch_size = len(y)
            num_data += batch_size

            self.backbone.zero_grad()
            if isinstance(self.heads, HeadDIL):
                self.heads.zero_grad()
            logits, _ = self.forward(x, stage="train", task_id=self.task_id)
            loss_cls = self.criterion(logits, y)
            loss_cls.backward()

            for param_name, param in self.backbone.named_parameters():
                fisher_information_t[param_name] += batch_size * param.grad**2
            if fisher_information_heads is not None:
                for param_name, param in self.heads.named_parameters():
                    if param.grad is None:
                        continue
                    fisher_information_heads[param_name] += batch_size * param.grad**2

        return fisher_information_t, fisher_information_heads, num_data


class AmnesiacEWC(AmnesiacCLAlgorithm, EWC):
    r"""Amnesiac EWC algorithm."""

    def __init__(
        self,
        backbone: CLBackbone,
        heads: HeadsTIL | HeadsCIL | HeadDIL,
        non_algorithmic_hparams: dict[str, Any] = {},
        disable_unlearning: bool = False,
        **kwargs,
    ) -> None:
        r"""Initialize the Amnesiac EWC algorithm with the network.

        **Args:**
        - **backbone** (`CLBackbone`): backbone network.
        - **heads** (`HeadsTIL` | `HeadsCIL` | `HeadDIL`): output heads.
        - **non_algorithmic_hparams** (`dict[str, Any]`): non-algorithmic hyperparameters that are not related to the algorithm itself are passed to this `LightningModule` object from the config, such as optimizer and learning rate scheduler configurations. They are saved for Lightning APIs from `save_hyperparameters()` method. This is useful for the experiment configuration and reproducibility.
        - **disable_unlearning** (`bool`): whether to disable the unlearning functionality. Default is `False`.
        - **kwargs**: Reserved for multiple inheritance.
        """
        super().__init__(
            backbone=backbone,
            heads=heads,
            non_algorithmic_hparams=non_algorithmic_hparams,
            disable_unlearning=disable_unlearning,
            **kwargs,
        )

    def on_train_start(self) -> None:
        """Record backbone parameters before training current task."""
        EWC.on_train_start(self)
        AmnesiacCLAlgorithm.on_train_start(self)

    def on_train_end(self) -> None:
        """Record backbone parameters before training current task."""
        EWC.on_train_end(self)
        AmnesiacCLAlgorithm.on_train_end(self)
