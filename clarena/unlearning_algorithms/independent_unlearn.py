r"""
The submoduule in `unlearning_algorithms` for unlearning algorithm of independent learning.
"""

__all__ = ["IndependentUnlearn"]

import logging

from clarena.cl_algorithms import Independent
from clarena.unlearning_algorithms import UnlearningAlgorithm

# always get logger for built-in logging in each module
pylogger = logging.getLogger(__name__)


class IndependentUnlearn(UnlearningAlgorithm):
    r"""The base class of the unlearning algorithm of independent learning."""

    def __init__(self, model: Independent) -> None:
        r"""Initialise the unlearning algorithm with the continual learning model.

        **Args:**
        - **model** (`Independent`): the continual learning model (`CLAlgorithm` object which already contains the backbone and heads). It must be `Independent` algorithm.
        """
        UnlearningAlgorithm.__init__(self, model=model)

    def unlearn(self) -> None:
        r"""Unlearn the unlearning tasks."""
        for unlearning_task_id in self.unlearning_task_ids:
            self.model.backbones[f"{unlearning_task_id}"].load_state_dict(
                self.model.original_backbone_state_dict
            )
