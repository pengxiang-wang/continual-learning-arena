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
from clarena.cl_algorithms.regularisers import ParameterChangeReg
from clarena.cl_heads import HeadsCIL, HeadsTIL

# always get logger for built-in logging in each module
pylogger = logging.getLogger(__name__)


class EWC(Finetuning):
    r"""EWC (Elastic Weight Consolidation) algorithm.

    [EWC (Elastic Weight Consolidation, 2017)](https://www.pnas.org/doi/10.1073/pnas.1611835114) is a regularisation-based continual learning approach that calculates parameter importance for the previous tasks and penalises the current task loss with the importance of the parameters.

    We implement EWC as a subclass of Finetuning algorithm, as EWC has the same `forward()`, `validation_step()` and `test_step()` method as `Finetuning` class.
    """

    def __init__(
        self,
        backbone: CLBackbone,
        heads: HeadsTIL | HeadsCIL,
        parameter_change_reg_factor: float,
        parameter_change_reg_p_norm: float,
    ) -> None:
        r"""Initialise the HAT algorithm with the network.

        **Args:**
        - **backbone** (`CLBackbone`): backbone network.
        - **heads** (`HeadsTIL` | `HeadsCIL`): output heads.
        - **parameter_change_reg_factor** (`float`): the parameter change regularisation factor. It controls the strength of preventing forgetting.
        - **parameter_change_reg_p_norm** (`float`): the norm of the distance of parameters between previous tasks and current task in the parameter change regularisation.

        """
        Finetuning.__init__(self, backbone=backbone, heads=heads)

        self.parameter_importance: dict[str, dict[str, Tensor]] = {}
        r"""Store the parameter importance of each previous task. Keys are task IDs (string type) and values are the corresponding importance. Each importance entity is a dict where keys are parameter names (named by `named_parameters()` of the `nn.Module`) and values are the importance tensor for the layer. It has the same shape as the parameters of the layer.
        """
        self.previous_task_backbones: dict[str, nn.Module] = {}
        r"""Store the backbone models of the previous tasks. Keys are task IDs (string type) and values are the corresponding models. Each model is a `nn.Module` backbone after the corresponding previous task was trained.
        
        Some would argue that since we could store the model of the previous tasks, why don't we test the task directly with the stored model, instead of doing the less easier EWC thing? The thing is, EWC only uses the model of the previous tasks to train current and future tasks, which aggregate them into a single model. Once the training of the task is done, the storage for those parameters can be released. However, this make the future tasks not able to use EWC anymore, which is a disadvantage for EWC.
        """

        self.parameter_change_reg_factor = parameter_change_reg_factor
        r"""Store parameter change regularisation factor."""
        self.parameter_change_reg_p_norm = parameter_change_reg_p_norm
        r"""Store norm of the distance used in parameter change regularisation."""
        self.parameter_change_reg = ParameterChangeReg(
            factor=parameter_change_reg_factor,
            p_norm=parameter_change_reg_p_norm,
        )
        r"""Initialise and store the parameter change regulariser."""

        EWC.sanity_check(self)

    def sanity_check(self) -> None:
        r"""Check the sanity of the arguments.

        **Raises:**
        - **ValueError**: If the regularisation factor is not positive.
        """

        if self.parameter_change_reg_factor <= 0:
            raise ValueError(
                f"The parameter change regularisation factor should be positive, but got {self.parameter_change_reg_factor}."
            )

    def calculate_parameter_importance(self) -> None:
        r"""Calculate the parameter importance for the learned task. This is only called after the training of a task, which is the last previous task $t-1$. The calculated importance is stored in `self.parameter_importance[self.task_id]` for constructing the regularisation loss in the future tasks.

        According to [the EWC paper](https://www.pnas.org/doi/10.1073/pnas.1611835114), the importance tensor is a Laplace approximation to Fisher information matrix by taking the digonal, i.e. $F_i$, where $i$ is the index of a parameter. The calculation is not following that theory but the derived formula below:

        $$\omega_i = F_i  =\frac{1}{N_{t-1}} \sum_{(\mathbf{x}, y)\in \mathcal{D}^{(t-1)}_{\text{train}}} \left[\frac{\partial l(f^{(t-1)}\left(\mathbf{x}, \theta), y\right)}{\partial \theta_i}\right]^2$$

        For a parameter $i$, its importance is the magnitude (square here) of gradient of the loss of model just trained over the training data just used. The $l$ is the classification loss. It shows the sensitivity of the loss to the parameter. The larger it is, the more it changed the performance (which is the loss) of the model, which indicates the importance of the parameter.
        """
        parameter_importance_t = {}

        # set model to evaluation mode to prevent updating the model parameters
        self.eval()

        # get the training data
        last_task_train_dataloaders = self.trainer.datamodule.train_dataloader()

        # initialise the accumulation of the squared gradients
        num_data = 0
        for param_name, param in self.backbone.named_parameters():
            parameter_importance_t[param_name] = torch.zeros_like(param)

        for x, y in last_task_train_dataloaders:

            # move data to device manually
            x = x.to(self.device)
            y = y.to(self.device)

            batch_size = len(y)
            num_data += batch_size

            # compute the gradients within a batch
            self.backbone.zero_grad()  # reset gradients
            logits, _ = self.forward(x, stage="train", task_id=self.task_id)
            loss_cls = self.criterion(logits, y)
            loss_cls.backward()  # compute gradients

            # collect and accumulate the squared gradients into parameter importance
            for param_name, param in self.backbone.named_parameters():
                parameter_importance_t[param_name] += batch_size * param.grad**2

        num_params = sum(p.numel() for p in self.backbone.parameters())

        for param_name, param in self.backbone.named_parameters():
            parameter_importance_t[param_name] /= (
                num_data * num_params
            )  # average over data and parameters

        self.parameter_importance[self.task_id] = parameter_importance_t

    def training_step(self, batch: Any) -> dict[str, Tensor]:
        r"""Training step for current task `self.task_id`.

        **Args:**
        - **batch** (`Any`): a batch of training data.

        **Returns:**
        - **outputs** (`dict[str, Tensor]`): a dictionary contains loss and other metrics from this training step. Key (`str`) is the metrics name, value (`Tensor`) is the metrics. Must include the key 'loss' which is total loss in the case of automatic optimization, according to PyTorch Lightning docs.
        """
        x, y = batch

        # classification loss
        logits, hidden_features = self.forward(x, stage="train", task_id=self.task_id)
        loss_cls = self.criterion(logits, y)

        # regularisation loss. See equation (3) in the [EWC paper](https://www.pnas.org/doi/10.1073/pnas.1611835114).
        loss_reg = 0.0
        for previous_task_id in range(1, self.task_id):
            # sum over all previous models, because [EWC paper](https://www.pnas.org/doi/10.1073/pnas.1611835114) says: "When moving to a third task, task C, EWC will try to keep the network parameters close to the learned parameters of both tasks A and B. This can be enforced either with two separate penalties or as one by noting that the sum of two quadratic penalties is itself a quadratic penalty."
            loss_reg += self.parameter_change_reg(
                target_model=self.backbone,
                ref_model=self.previous_task_backbones[previous_task_id],
                weights=self.parameter_importance[previous_task_id],
            )

        if self.task_id != 1:
            loss_reg /= (
                self.task_id
            )  # average over tasks to avoid linear increase of the regularisation loss

        # total loss
        loss = loss_cls + loss_reg

        # accuracy of the batch
        acc = (logits.argmax(dim=1) == y).float().mean()

        return {
            "loss": loss,  # Return loss is essential for training step, or backpropagation will fail
            "loss_cls": loss_cls,
            "loss_reg": loss_reg,
            "acc": acc,
            "hidden_features": hidden_features,
        }

    def on_train_end(self) -> None:
        r"""Calculate the parameter importance and store the backbone model after the training of a task.

        The calculated importance and model are stored in `self.parameter_importance[self.task_id]` and `self.previous_task_backbones[self.task_id]` respectively for constructing the regularisation loss in the future tasks.
        """
        self.calculate_parameter_importance()

        previous_backbone = deepcopy(self.backbone)
        previous_backbone.eval()  # set the store model to evaluation mode to prevent updating
        self.previous_task_backbones[self.task_id] = previous_backbone
