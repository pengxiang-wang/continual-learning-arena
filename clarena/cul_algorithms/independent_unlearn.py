r"""
The submoduule in `cul_algorithms` for the vanilla unlearning algorithm for independent learning.
"""

__all__ = ["IndependentUnlearn"]

import logging

from clarena.cl_algorithms import Independent
from clarena.cul_algorithms import CULAlgorithm

# always get logger for built-in logging in each module
pylogger = logging.getLogger(__name__)


class IndependentUnlearn(CULAlgorithm):
    r"""Vanilla unlearning algorithm for independent learning."""

    def __init__(self, model: Independent) -> None:
        r"""
        **Args:**
        - **model** (`Independent`): the continual learning model (`CLAlgorithm` object which already contains the backbone and heads). It must be `Independent` algorithm.
        """
        super().__init__(model=model)

    def unlearn(self) -> None:
        r"""Unlearn the requested unlearning tasks after training `self.task_id`."""
        for unlearning_task_id in self.unlearning_task_ids:
            self.model.backbones[unlearning_task_id].load_state_dict(
                self.model.original_backbone_state_dict
            )
