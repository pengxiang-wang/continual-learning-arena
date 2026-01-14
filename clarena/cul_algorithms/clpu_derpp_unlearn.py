r"""
The submoduule in `cul_algorithms` for the CLPU-DER++ unlearning algorithm.
"""

__all__ = ["CLPUDERppUnlearn"]

import logging

from clarena.cl_algorithms import CLPUDERpp
from clarena.cul_algorithms import CULAlgorithm

# always get logger for built-in logging in each module
pylogger = logging.getLogger(__name__)


class CLPUDERppUnlearn(CULAlgorithm):
    r"""CLPU-DER++ unlearning algorithm."""

    def __init__(self, model: CLPUDERpp) -> None:
        r"""
        **Args:**
        - **model** (`CLPUDERpp`): the continual learning model (`CLAlgorithm` object which already contains the backbone and heads). It must be `CLPUDERpp` algorithm.
        """
        super().__init__(model=model)

    def unlearn(self) -> None:
        r"""Unlearn the requested unlearning tasks after training `self.task_id`."""
