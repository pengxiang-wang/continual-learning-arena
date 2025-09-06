r"""The submodule in `regularizers` for distillation regularization."""

__all__ = ["DistillationReg"]

import logging

import torch
import torch.nn.functional as F
from torch import Tensor, nn

# always get logger for built-in logging in each module
pylogger = logging.getLogger(__name__)


class DistillationReg(nn.Module):
    r"""Distillation regularizer. This is the core of [knowledge distillation](https://research.google/pubs/distilling-the-knowledge-in-a-neural-network/) used as a regularizer in continual learning.

    $$R(\theta^{\text{student}}) = \text{factor} * \frac1N \sum_{(\mathbf{x}, y)\in \mathcal{D}} \text{distance}\left(f(\mathbf{x};\theta^{\text{student}}),f(\mathbf{x};\theta^{\text{teacher}})\right)$$

    It promotes the target (student) model output logits $f(\mathbf{x};\theta^{\text{student}})$ not changing too much from the reference (teacher) model output logits $f(\mathbf{x};\theta^{\text{teacher}})$. The loss is averaged over the dataset $\mathcal{D}$.

    It is used in:
    - [LwF (Learning without Forgetting) algorithm](https://ieeexplore.ieee.org/abstract/document/8107520): as a distillation regularizer for the output logits by current task model to be closer to output logits by previous tasks models. It uses a modified cross entropy as the distance. See equation (2) (3) in the [LwF paper](https://ieeexplore.ieee.org/abstract/document/8107520).
    """

    def __init__(
        self,
        factor: float,
        temperature: float,
        distance: str,
    ) -> None:
        r"""
        **Args:**
        - **factor** (`float`): the regularization factor.
        - **temperature** (`float`): the temperature of the distillation, should be a positive float.
        - **distance** (`str`): the type of distance function used in the distillation; one of:
            1. "lwf_cross_entropy": the modified cross entropy loss from LwF. See equation (3) in the [LwF paper](https://ieeexplore.ieee.org/abstract/document/8107520).
        """
        super().__init__()

        self.factor = factor
        """The regularisation factor for distillation."""
        self.temperature = temperature
        """The temperature of the distillation. """
        self.distance = distance
        """The type of distance function used in the distillation."""

    def forward(
        self,
        student_logits: nn.Module,
        teacher_logits: nn.Module,
    ) -> Tensor:
        r"""Calculate the regularisation loss.

        **Args:**
        - **student_logits** (`Tensor`): the output logits of target (student) model to learn the knowledge from distillation. In LwF, it's the model of current training task.
        - **teacher_logits** (`Tensor`): the output logits of reference (teacher) model that knowledge is distilled. In LwF, it's the model of one of the previous tasks.

        **Returns:**
        - **reg** (`Tensor`): the distillation regularisation value.
        """

        if self.distance == "cross_entropy":

            # get the probabilities first (which are $y_o^{(i)}$ and $\hat{y}_o^{(i)}$ in the [LwF paper](https://ieeexplore.ieee.org/abstract/document/8107520))
            student_probs = F.softmax(
                input=student_logits,
                dim=1,
            )
            teacher_probs = F.softmax(
                input=teacher_logits,
                dim=1,
            )

            # apply temperature scaling
            student_probs = student_probs.pow(1 / self.temperature)
            teacher_probs = teacher_probs.pow(1 / self.temperature)

            # normalize the probabilities second time
            student_probs = torch.div(
                student_probs, torch.sum(student_probs, 1, keepdim=True)
            )
            teacher_probs = torch.div(
                teacher_probs, torch.sum(teacher_probs, 1, keepdim=True)
            )

            student_probs = (
                student_probs + 1e-5
            )  # simply add a small value to avoid log(0)

            return self.factor * -(teacher_probs * student_probs.log()).sum(1).mean()
