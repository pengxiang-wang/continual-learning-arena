r"""
The submodule in `cul_algorithms` for DER unlearning algorithm.
"""

__all__ = ["AmnesiacDERUnlearn", "AmnesiacDERppUnlearn"]

import logging

from clarena.cl_algorithms import AmnesiacDER
from clarena.cl_algorithms.der import AmnesiacDERpp
from clarena.cul_algorithms import AmnesiacCULAlgorithm

pylogger = logging.getLogger(__name__)


class AmnesiacDERUnlearn(AmnesiacCULAlgorithm):
    r"""Amnesiac continual unlearning algorithm for DER."""

    def __init__(
        self,
        model: AmnesiacDER,
    ) -> None:
        r"""Initialize the unlearning algorithm with the continual learning model.

        **Args:**
        - **model** (`AmnesiacDER`): the continual learning model. It must be `AmnesiacDER`.
        """
        super().__init__(model=model)


class AmnesiacDERppUnlearn(AmnesiacCULAlgorithm):
    r"""Amnesiac continual unlearning algorithm for DER++."""

    def __init__(
        self,
        model: AmnesiacDERpp,
    ) -> None:
        r"""Initialize the unlearning algorithm with the continual learning model.

        **Args:**
        - **model** (`AmnesiacDERpp`): the continual learning model. It must be `AmnesiacDERpp`.
        """
        super().__init__(model=model)
