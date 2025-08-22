r"""
The submodule in `callbacks` for Pylogger callbacks.
"""

__all__ = ["CLPylogger", "CULPylogger", "MTLPylogger", "STLPylogger"]

import logging

from lightning import Callback, Trainer

from clarena.cl_algorithms import CLAlgorithm
from clarena.mtl_algorithms import MTLAlgorithm
from clarena.stl_algorithms import STLAlgorithm

# always get logger for built-in logging in each module
pylogger = logging.getLogger(__name__)


class CLPylogger(Callback):
    r"""Provides additional logging messages for during continual learning progress.

    Put logging messages here if you don't want to mess up the `CLAlgorithm` (`LightningModule`) with a huge amount of logging codes.
    """

    def on_fit_start(self, trainer: Trainer, pl_module: CLAlgorithm) -> None:
        r"""Log messages for the start of training task."""
        pylogger.info("Start training continual learning task %s!", pl_module.task_id)

    def on_train_end(self, trainer: Trainer, pl_module: CLAlgorithm) -> None:
        r"""Log messages for the end of training task."""
        pylogger.info("Finish training continual learning task %s!", pl_module.task_id)

    def on_test_start(self, trainer: Trainer, pl_module: CLAlgorithm) -> None:
        r"""Log messages for the start of testing task."""
        pylogger.info(
            "Start testing continual learning task %s on all previous and current tasks!",
            pl_module.task_id,
        )

    def on_test_end(self, trainer: Trainer, pl_module: CLAlgorithm) -> None:
        r"""Log messages for the end of testing task."""
        pylogger.info(
            "Finish testing continual learning task %s on all previous and current tasks!",
            pl_module.task_id,
        )


class CULPylogger(Callback):
    r"""Provides additional logging messages for during continual learning progress.

    Put logging messages here if you don't want to mess up the `CLAlgorithm` (`LightningModule`) with a huge amount of logging codes.
    """

    def on_fit_start(self, trainer: Trainer, pl_module: CLAlgorithm) -> None:
        r"""Log messages for the start of training task."""
        pylogger.info("Start training continual learning task %s!", pl_module.task_id)

    def on_train_end(self, trainer: Trainer, pl_module: CLAlgorithm) -> None:
        r"""Log messages for the end of training task."""
        pylogger.info("Finish training continual learning task %s!", pl_module.task_id)

    def on_test_start(self, trainer: Trainer, pl_module: CLAlgorithm) -> None:
        r"""Log messages for the start of testing task."""
        pylogger.info(
            "Start testing continual learning task %s on all previous and current tasks!",
            pl_module.task_id,
        )

    def on_test_end(self, trainer: Trainer, pl_module: CLAlgorithm) -> None:
        r"""Log messages for the end of testing task."""
        pylogger.info(
            "Finish testing continual learning task %s on all previous and current tasks!",
            pl_module.task_id,
        )


class MTLPylogger(Callback):
    r"""Pylogger Callback provides additional logging for during multi-task learning progress.

    Put logging messages here if you don't want to mess up the `MTLAlgorithm` (`LightningModule`) with a huge amount of logging codes.
    """

    def on_fit_start(self, trainer: Trainer, pl_module: MTLAlgorithm) -> None:
        r"""Log messages for the start of training."""
        pylogger.info("Start multi-task training!")

    def on_train_end(self, trainer: Trainer, pl_module: MTLAlgorithm) -> None:
        r"""Log messages for the end of training."""
        pylogger.info("Finish multi-task training!")

    def on_test_start(self, trainer: Trainer, pl_module: MTLAlgorithm) -> None:
        r"""Log messages for the start of testing."""
        pylogger.info("Start testing multi-task testing!")

    def on_test_end(self, trainer: Trainer, pl_module: MTLAlgorithm) -> None:
        r"""Log messages for the end of testing."""
        pylogger.info("Finish testing multi-task testing!")


class STLPylogger(Callback):
    r"""Pylogger Callback provides additional logging for during single-task learning progress.

    Put logging messages here if you don't want to mess up the `STLAlgorithm` (`LightningModule`) with a huge amount of logging codes.
    """

    def on_fit_start(self, trainer: Trainer, pl_module: STLAlgorithm) -> None:
        r"""Log messages for the start of training."""
        pylogger.info("Start single-task training!")

    def on_train_end(self, trainer: Trainer, pl_module: STLAlgorithm) -> None:
        r"""Log messages for the end of training."""
        pylogger.info("Finish single-task training!")

    def on_test_start(self, trainer: Trainer, pl_module: STLAlgorithm) -> None:
        r"""Log messages for the start of testing."""
        pylogger.info("Start single-task testing!")

    def on_test_end(self, trainer: Trainer, pl_module: STLAlgorithm) -> None:
        r"""Log messages for the end of testing."""
        pylogger.info("Finish single-task testing!")
