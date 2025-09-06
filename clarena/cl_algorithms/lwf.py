r"""
The submodule in `cl_algorithms` for [LwF (Learning without Forgetting) algorithm](https://ieeexplore.ieee.org/abstract/document/8107520).
"""

__all__ = ["LwF"]

import logging
from copy import deepcopy
from typing import Any

import torch
from torch import Tensor

from clarena.backbones import CLBackbone
from clarena.cl_algorithms import Finetuning
from clarena.cl_algorithms.regularizers import DistillationReg
from clarena.heads import HeadsCIL, HeadsTIL

# always get logger for built-in logging in each module
pylogger = logging.getLogger(__name__)


class LwF(Finetuning):
    r"""[LwF (Learning without Forgetting)](https://ieeexplore.ieee.org/abstract/document/8107520) algorithm.

    A regularization-based continual learning approach that constrains the feature output of the model to be similar to that of the previous tasks. From the perspective of knowledge distillation, it distills previous tasks models into the training process for new task in the regularization term. It is a simple yet effective method for continual learning.

    We implement LwF as a subclass of Finetuning algorithm, as LwF has the same `forward()`, `validation_step()` and `test_step()` method as `Finetuning` class.
    """

    def __init__(
        self,
        backbone: CLBackbone,
        heads: HeadsTIL | HeadsCIL,
        distillation_reg_factor: float,
        distillation_reg_temperature: float,
        non_algorithmic_hparams: dict[str, Any] = {},
    ) -> None:
        r"""Initialize the LwF algorithm with the network.

        **Args:**
        - **backbone** (`CLBackbone`): backbone network.
        - **heads** (`HeadsTIL` | `HeadsCIL`): output heads.
        - **distillation_reg_factor** (`float`): hyperparameter, the distillation regularization factor. It controls the strength of preventing forgetting.
        - **distillation_reg_temperature** (`float`): hyperparameter, the temperature in the distillation regularization. It controls the softness of the labels that the student model (here is the current model) learns from the teacher models (here are the previous models), thereby controlling the strength of the distillation. It controls the strength of preventing forgetting.
        - **non_algorithmic_hparams** (`dict[str, Any]`): non-algorithmic hyperparameters that are not related to the algorithm itself are passed to this `LightningModule` object from the config, such as optimizer and learning rate scheduler configurations. They are saved for Lightning APIs from `save_hyperparameters()` method. This is useful for the experiment configuration and reproducibility.

        """
        super().__init__(
            backbone=backbone,
            heads=heads,
            non_algorithmic_hparams=non_algorithmic_hparams,
        )

        self.previous_task_backbones: dict[int, CLBackbone] = {}
        r"""The backbone models of the previous tasks. Keys are task IDs (int) and values are the corresponding models. Each model is a `CLBackbone` after the corresponding previous task was trained.
        
        Some would argue that since we could store the model of the previous tasks, why don't we test the task directly with the stored model, instead of doing the less easier LwF thing? The thing is, LwF only uses the model of the previous tasks to train current and future tasks, which aggregate them into a single model. Once the training of the task is done, the storage for those parameters can be released. However, this make the future tasks not able to use LwF anymore, which is a disadvantage for LwF.
        """

        self.distillation_reg_factor: float = distillation_reg_factor
        r"""The distillation regularization factor."""
        self.distillation_reg_temperature: float = distillation_reg_temperature
        r"""The distillation regularization temperature."""
        self.distillation_reg = DistillationReg(
            factor=distillation_reg_factor,
            temperature=distillation_reg_temperature,
            distance="cross_entropy",
        )
        r"""Initialize and store the distillation regulariser."""

        # save additional algorithmic hyperparameters
        self.save_hyperparameters(
            "distillation_reg_factor",
            "distillation_reg_temperature",
        )

        LwF.sanity_check(self)

    def sanity_check(self) -> None:
        r"""Sanity check."""

        if self.distillation_reg_factor <= 0:
            raise ValueError(
                f"The distillation regularization factor should be positive, but got {self.distillation_reg_factor}."
            )

        if self.distillation_reg_temperature <= 0:
            raise ValueError(
                f"The distillation regularization temperature should be positive, but got {self.distillation_reg_temperature}."
            )

    def training_step(self, batch: Any) -> dict[str, Tensor]:
        r"""Training step for current task `self.task_id`.

        **Args:**
        - **batch** (`Any`): a batch of training data.

        **Returns:**
        - **outputs** (`dict[str, Tensor]`): a dictionary contains loss and other metrics from this training step. Keys (`str`) are the metrics names, and values (`Tensor`) are the metrics. Must include the key 'loss' which is total loss in the case of automatic optimization, according to PyTorch Lightning docs.
        """
        x, y = batch

        # classification loss. See equation (1) in the [LwF paper](https://ieeexplore.ieee.org/abstract/document/8107520).
        logits, activations = self.forward(x, stage="train", task_id=self.task_id)
        loss_cls = self.criterion(logits, y)

        # regularization loss. See equation (2) (3) in the [LwF paper](https://ieeexplore.ieee.org/abstract/document/8107520).
        loss_reg = 0.0
        for previous_task_id in range(1, self.task_id):
            # sum over all previous models, because [LwF paper](https://ieeexplore.ieee.org/abstract/document/8107520) says: "If there are multiple old tasks, or if an old task is multi-label classification, we take the sum of the loss for each old task and label."

            # get the student logits for this batch using the current model (to previous output head)
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

        # do not average over tasks to avoid linear increase of the regularization loss. LwF paper doesn't mention this!

        # total loss
        loss = loss_cls + loss_reg

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
        r"""The backbone model after the training of a task.

        The model is stored in `self.previous_task_backbones` for constructing the regularization loss in the future tasks.
        """
        previous_backbone = deepcopy(self.backbone)
        previous_backbone.eval()  # set the stored model to evaluation mode to prevent updating
        head = self.heads.heads[f"{self.task_id}"]
        head.eval()  # set the head to evaluation mode
        self.previous_task_backbones[self.task_id] = previous_backbone
