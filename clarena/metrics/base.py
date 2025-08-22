r"""
The submodule in `metrics` for metric callback bases.
"""

__all__ = [
    "MetricCallback",
]

import logging
import os

from lightning import Callback

# always get logger for built-in logging in each module
pylogger = logging.getLogger(__name__)


class MetricCallback(Callback):
    r"""
    The base class for all metrics callbacks in CLArena.

    This class is a placeholder for future metrics callbacks that can be used in continual learning experiments.
    It is not intended to be instantiated directly, but rather to be subclassed by specific metrics callbacks.
    """

    def __init__(self, save_dir: str) -> None:
        r"""Initialize the `MetricCallback`.

        **Args:**
        - **save_dir** (`str`): The directory where data and figures of metrics will be saved. Better inside the output folder.
        """
        super().__init__()

        os.makedirs(save_dir, exist_ok=True)

        self.save_dir: str = save_dir
        r"""Store the directory where data and figures of metrics will be saved."""
