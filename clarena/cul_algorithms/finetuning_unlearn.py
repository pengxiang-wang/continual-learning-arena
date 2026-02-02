r"""
The submodule in `cul_algorithms` for Finetuning unlearning.
"""

__all__ = ["AmnesiacFinetuningUnlearn"]

import logging

from clarena.cl_algorithms import AmnesiacFinetuning
from clarena.cul_algorithms import AmnesiacCULAlgorithm

pylogger = logging.getLogger(__name__)


class AmnesiacFinetuningUnlearn(AmnesiacCULAlgorithm):
    r"""Amnesiac continual unlearning algorithm for Finetuning."""

    def __init__(
        self,
        model: AmnesiacFinetuning,
    ) -> None:
        r"""Initialize the unlearning algorithm with the continual learning model.

        **Args:**
        - **model** (`AmnesiacHAT`): the continual learning model. It must be `AmnesicCLAlgorithm`.
        """
        super().__init__(model=model)
