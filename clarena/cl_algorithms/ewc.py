r"""
The submodule in `cl_algorithms` for [EWC (Elastic Weight Consolidation) algorithm](https://www.pnas.org/doi/10.1073/pnas.1611835114).
"""

__all__ = ["EWC"]

import logging
from copy import deepcopy
from typing import Any

import torch
from torch import Tensor, nn

from clarena.backbones import CLBackbone
from clarena.cl_algorithms import Finetuning
from clarena.cl_algorithms.regularizers import ParameterChangeReg
from clarena.heads import HeadsCIL, HeadsTIL

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
        heads: HeadsTIL | HeadsCIL,
        parameter_change_reg_factor: float,
        when_calculate_fisher_information: str,
        non_algorithmic_hparams: dict[str, Any] = {},
    ) -> None:
        r"""Initialize the HAT algorithm with the network.

        **Args:**
        - **backbone** (`CLBackbone`): backbone network.
        - **heads** (`HeadsTIL` | `HeadsCIL`): output heads.
        - **parameter_change_reg_factor** (`float`): the parameter change regularization factor. It controls the strength of preventing forgetting.
        - **when_calculate_fisher_information** (`str`): when to calculate the fisher information. It should be one of the following:
            1. 'train_end': calculate the fisher information at the end of training of the task.
            2. 'train': accumulate the fisher information in the training step of the task.
        - **non_algorithmic_hparams** (`dict[str, Any]`): non-algorithmic hyperparameters that are not related to the algorithm itself are passed to this `LightningModule` object from the config, such as optimizer and learning rate scheduler configurations. They are saved for Lightning APIs from `save_hyperparameters()` method. This is useful for the experiment configuration and reproducibility.

        """
        super().__init__(
            backbone=backbone,
            heads=heads,
            non_algorithmic_hparams=non_algorithmic_hparams,
        )

        # save additional algorithmic hyperparameters
        self.save_hyperparameters(
            "parameter_change_reg_factor",
            "when_calculate_fisher_information",
        )

        self.parameter_importance: dict[str, dict[str, Tensor]] = {}
        r"""The parameter importance of each previous task. Keys are task IDs and values are the corresponding importance. Each importance entity is a dict where keys are parameter names (named by `named_parameters()` of the `nn.Module`) and values are the importance tensor for the layer. It has the same shape as the parameters of the layer.
        """
        self.previous_task_backbones: dict[str, nn.Module] = {}
        r"""The backbone models of the previous tasks. Keys are task IDs and values are the corresponding models. Each model is a `nn.Module` backbone after the corresponding previous task was trained.

        Some would argue that since we could store the model of the previous tasks, why don't we test the task directly with the stored model, instead of doing the less easier EWC thing? The thing is, EWC only uses the model of the previous tasks to train current and future tasks, which aggregate them into a single model. Once the training of the task is done, the storage for those parameters can be released. However, this make the future tasks not able to use EWC anymore, which is a disadvantage for EWC.
        """

        self.parameter_change_reg_factor = parameter_change_reg_factor
        r"""The parameter change regularization factor."""
        self.parameter_change_reg = ParameterChangeReg(
            factor=parameter_change_reg_factor,
        )
        r"""Initialize and store the parameter change regulariser."""

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

        self.num_data = 0

    def training_step(self, batch: Any) -> dict[str, Tensor]:
        r"""Training step for current task `self.task_id`.

        **Args:**
        - **batch** (`Any`): a batch of training data.

        **Returns:**
        - **outputs** (`dict[str, Tensor]`): a dictionary contains loss and other metrics from this training step. Keys (`str`) are the metrics names, and values (`Tensor`) are the metrics. Must include the key 'loss' which is total loss in the case of automatic optimization, according to PyTorch Lightning docs.
        """
        x, y = batch

        # zero the gradients before forward pass in manual optimization mode
        opt = self.optimizers()
        opt.zero_grad()

        # classification loss
        logits, activations = self.forward(x, stage="train", task_id=self.task_id)
        loss_cls = self.criterion(logits, y)

        batch_size = len(y)
        self.num_data += batch_size

        if self.when_calculate_fisher_information == "train":
            # do the backpropagation only for classification loss to get the gradients
            loss_cls.backward(retain_graph=True)
            # collect and accumulate the squared gradients into fisher information
            for param_name, param in self.backbone.named_parameters():
                self.parameter_importance[self.task_id][param_name] += (
                    batch_size * param.grad**2
                )

        # regularization loss. See equation (3) in the [EWC paper](https://www.pnas.org/doi/10.1073/pnas.1611835114).
        loss_reg = 0.0
        for previous_task_id in range(1, self.task_id):
            # sum over all previous models, because [EWC paper](https://www.pnas.org/doi/10.1073/pnas.1611835114) says: "When moving to a third task, task C, EWC will try to keep the network parameters close to the learned parameters of both tasks A and B. This can be enforced either with two separate penalties or as one by noting that the sum of two quadratic penalties is itself a quadratic penalty."
            loss_reg += 0.5 * self.parameter_change_reg(
                target_model=self.backbone,
                ref_model=self.previous_task_backbones[previous_task_id],
                weights=self.parameter_importance[previous_task_id],
            )  # the factor 1/2 in equation (3) in the [EWC paper](https://www.pnas.org/doi/10.1073/pnas.1611835114) is included here instead of the parameter change regulariser.

        # do not average over tasks to avoid linear increase of the regularization loss. EWC paper doesn't mention this!

        # total loss
        loss = loss_cls + loss_reg

        # backward step (manually)
        self.manual_backward(loss)  # calculate the gradients for update
        opt.step()

        # accuracy of the batch
        acc = (logits.argmax(dim=1) == y).float().mean()

        return {
            "loss": loss,  # return loss is essential for training step, or backpropagation will fail
            "loss_cls": loss_cls,
            "loss_reg": loss_reg,
            "acc": acc,
            "activations": activations,
        }

    def on_train_end(self) -> None:
        r"""Calculate the fisher information as parameter importance and store the backbone model after the training of a task.

        The calculated importance and model are stored in `self.parameter_importance[self.task_id]` and `self.previous_task_backbones[self.task_id]` respectively for constructing the regularization loss in the future tasks.
        """
        if self.when_calculate_fisher_information == "train_end":
            self.parameter_importance[self.task_id] = (
                self.accumulate_fisher_information_on_train_end()
            )

        for param_name, param in self.backbone.named_parameters():
            self.parameter_importance[self.task_id][
                param_name
            ] /= self.num_data  # average over data, do not average over parameters

        previous_backbone = deepcopy(self.backbone)
        previous_backbone.eval()  # set the stored model to evaluation mode to prevent updating
        self.previous_task_backbones[self.task_id] = previous_backbone

    def accumulate_fisher_information_on_train_end(self) -> None:
        r"""Accumulate the fisher information as the parameter importance for the learned task `self.task_id` at the end of its training. This is only called after the training of a task, which is the last previous task $t-1$. The accumulate importance is stored in `self.parameter_importance[self.task_id]` for constructing the regularization loss in the future tasks.

        According to [the EWC paper](https://www.pnas.org/doi/10.1073/pnas.1611835114), the importance tensor is a Laplace approximation to Fisher information matrix by taking the digonal, i.e. $F_i$, where $i$ is the index of a parameter. The calculation is not following that theory but the derived formula below:

        $$\omega_i = F_i  =\frac{1}{N_{t-1}} \sum_{(\mathbf{x}, y)\in \mathcal{D}^{(t-1)}_{\text{train}}} \left[\frac{\partial l(f^{(t-1)}\left(\mathbf{x}, \theta), y\right)}{\partial \theta_i}\right]^2$$

        For a parameter $i$, its fisher information is the magnitude (square here) of gradient of the loss of model just trained over the training data just used. The $l$ is the classification loss. It shows the sensitivity of the loss to the parameter. The larger it is, the more it changed the performance (which is the loss) of the model, which indicates the importance of the parameter.

        **Returns:**
        - **fisher_information_t** (`dict[str, Tensor]`): the fisher information for the learned task. Keys are parameter names (named by `named_parameters()` of the `nn.Module`) and values are the importance tensor for the layer. It has the same shape as the parameters of the layer.
        """
        fisher_information_t = {}

        # set model to evaluation mode to prevent updating the model parameters
        self.eval()

        # get the training data
        last_task_train_dataloaders = self.trainer.datamodule.train_dataloader()

        # initialize the accumulation of the squared gradients
        for param_name, param in self.backbone.named_parameters():
            fisher_information_t[param_name] = torch.zeros_like(param)

        for x, y in last_task_train_dataloaders:

            # move data to device manually
            x = x.to(self.device)
            y = y.to(self.device)

            batch_size = len(y)

            # compute the gradients within a batch
            self.backbone.zero_grad()  # reset gradients
            logits, _ = self.forward(x, stage="train", task_id=self.task_id)
            loss_cls = self.criterion(logits, y)
            loss_cls.backward()  # compute gradients

            # collect and accumulate the squared gradients into fisher information
            for param_name, param in self.backbone.named_parameters():
                fisher_information_t[param_name] += batch_size * param.grad**2

        return fisher_information_t
