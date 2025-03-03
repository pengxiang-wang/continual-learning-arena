r"""
The submodule in `cl_algorithms` for [LwF (Learning without Forgetting) algorithm](https://ieeexplore.ieee.org/abstract/document/8107520).
"""

__all__ = ["LwF"]

import logging
from copy import deepcopy
from typing import Any

import torch
from torch import Tensor, nn

from clarena.backbones import CLBackbone
from clarena.cl_algorithms import Finetuning
from clarena.cl_algorithms.regularisers import DistillationReg
from clarena.cl_heads import HeadsCIL, HeadsTIL

# always get logger for built-in logging in each module
pylogger = logging.getLogger(__name__)


class LwF(Finetuning):
    r"""LwF (Learning without Forgetting) algorithm.

    [LwF (Learning without Forgetting, 2017)](https://ieeexplore.ieee.org/abstract/document/8107520) is a regularisation-based continual learning approach that constrains the feature output of the model to be similar to that of the previous tasks. From the perspective of knowledge distillation, it distills previous tasks models into the training process for new task in the regularisation term. It is a simple yet effective method for continual learning.

    We implement LwF as a subclass of Finetuning algorithm, as LwF has the same `forward()`, `validation_step()` and `test_step()` method as `Finetuning` class.
    """

    def __init__(
        self,
        backbone: CLBackbone,
        heads: HeadsTIL | HeadsCIL,
        distillation_reg_factor: float,
        distillation_reg_temparture: float,
    ) -> None:
        r"""Initialise the LwF algorithm with the network.

        **Args:**
        - **backbone** (`CLBackbone`): backbone network.
        - **heads** (`HeadsTIL` | `HeadsCIL`): output heads.
        - **distillation_reg_factor** (`float`): hyperparameter, the distillation regularisation factor. It controls the strength of preventing forgetting.
        - **distillation_reg_temparture** (`float`): hyperparameter, the temperature in the distillation regularisation. It controls the softness of the labels that the student model (here is the current model) learns from the teacher models (here are the previous models), thereby controlling the strength of the distillation. It controls the strength of preventing forgetting.
        """
        Finetuning.__init__(self, backbone=backbone, heads=heads)

        self.previous_task_backbones: dict[str, nn.Module] = {}
        r"""Store the backbone models of the previous tasks. Keys are task IDs (string type) and values are the corresponding models. Each model is a `nn.Module` backbone after the corresponding previous task was trained.
        
        Some would argue that since we could store the model of the previous tasks, why don't we test the task directly with the stored model, instead of doing the less easier LwF thing? The thing is, LwF only uses the model of the previous tasks to train current and future tasks, which aggregate them into a single model. Once the training of the task is done, the storage for those parameters can be released. However, this make the future tasks not able to use LwF anymore, which is a disadvantage for LwF.
        """

        self.distillation_reg_factor = distillation_reg_factor
        r"""Store distillation regularisation factor."""
        self.distillation_reg_temperature = distillation_reg_temparture
        r"""Store distillation regularisation temperature."""
        self.distillation_reg = DistillationReg(
            factor=distillation_reg_factor,
            temperature=distillation_reg_temparture,
            distance="cross_entropy",
        )
        r"""Initialise and store the distillation regulariser."""

        LwF.sanity_check(self)

    def sanity_check(self) -> None:
        r"""Check the sanity of the arguments.

        **Raises:**
        - **ValueError**: If the regularisation factor and distillation temperature is not positive.
        """

        if self.distillation_reg_factor <= 0:
            raise ValueError(
                f"The distillation regularisation factor should be positive, but got {self.distillation_reg_factor}."
            )

        if self.distillation_reg_temperature <= 0:
            raise ValueError(
                f"The distillation regularisation temperature should be positive, but got {self.distillation_reg_temperature}."
            )

    def training_step(self, batch: Any) -> dict[str, Tensor]:
        r"""Training step for current task `self.task_id`.

        **Args:**
        - **batch** (`Any`): a batch of training data.

        **Returns:**
        - **outputs** (`dict[str, Tensor]`): a dictionary contains loss and other metrics from this training step. Key (`str`) is the metrics name, value (`Tensor`) is the metrics. Must include the key 'loss' which is total loss in the case of automatic optimization, according to PyTorch Lightning docs.
        """
        x, y = batch

        # classification loss. See equation (1) in the [LwF paper](https://ieeexplore.ieee.org/abstract/document/8107520).
        logits, hidden_features = self.forward(x, stage="train", task_id=self.task_id)
        loss_cls = self.criterion(logits, y)

        # regularisation loss. See equation (2) (3) in the [LwF paper](https://ieeexplore.ieee.org/abstract/document/8107520).
        loss_reg = 0.0
        for previous_task_id in range(1, self.task_id):
            # sum over all previous models, because [LwF paper](https://ieeexplore.ieee.org/abstract/document/8107520) says: "If there are multiple old tasks, or if an old task is multi-label classification, we take the sum of the loss for each old task and label."

            # get the teacher logits for this batch, which is from the current model (to previous output head)
            student_feature, _ = self.backbone(
                x, stage="train", task_id=previous_task_id
            )
            with torch.no_grad():  # stop updating the previous heads
                student_logits = self.heads(student_feature, task_id=previous_task_id)

            # get the teacher logits for this batch, which is from the previous model
            previous_backbone = self.previous_task_backbones[previous_task_id]
            with torch.no_grad():  # stop updating the previous backbones and heads
                teacher_feature, _ = previous_backbone(
                    x, stage="test", task_id=previous_task_id
                )

                teacher_logits = self.heads(teacher_feature, task_id=previous_task_id)

            loss_reg += self.distillation_reg(
                student_logits=student_logits,
                teacher_logits=teacher_logits,
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
        r"""Store the backbone model after the training of a task.

        The model is stored in `self.previous_task_backbones` for constructing the regularisation loss in the future tasks.
        """
        previous_backbone = deepcopy(self.backbone)
        previous_backbone.eval()  # set the store model to evaluation mode to prevent updating
        self.heads.heads[
            f"{self.task_id}"
        ].eval()  # set the store model to evaluation mode to prevent updating
        self.previous_task_backbones[self.task_id] = previous_backbone
