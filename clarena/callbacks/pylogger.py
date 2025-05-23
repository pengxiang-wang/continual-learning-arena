r"""
The submodule in `callbacks` for Pylogger callbacks.
"""

__all__ = ["CLPyloggerCallback", "JLPyloggerCallback"]

import logging

from lightning import Callback, Trainer

from clarena.cl_algorithms import CLAlgorithm

# always get logger for built-in logging in each module
pylogger = logging.getLogger(__name__)


class CLPyloggerCallback(Callback):
    r"""Pylogger Callback provides additional logging for during continual learning progress.

    Put logging messages here if you don't want to mess up the `CLAlgorithm` (`LightningModule`) with a huge amount of logging codes.
    """

    def on_fit_start(self, trainer: Trainer, pl_module: CLAlgorithm) -> None:
        r"""Log messages for the start of training task."""
        pylogger.info("Start training task %s!", pl_module.task_id)

    def on_train_end(self, trainer: Trainer, pl_module: CLAlgorithm) -> None:
        r"""Log messages for the end of training task."""
        pylogger.info("Finish training task %s!", pl_module.task_id)

    def on_test_start(self, trainer: Trainer, pl_module: CLAlgorithm) -> None:
        r"""Log messages for the start of testing task."""
        pylogger.info(
            "Start testing task %s on all previous and current tasks!",
            pl_module.task_id,
        )

    def on_test_end(self, trainer: Trainer, pl_module: CLAlgorithm) -> None:
        r"""Log messages for the end of testing task."""
        pylogger.info(
            "Finish testing task %s on all previous and current tasks!",
            pl_module.task_id,
        )


class JLPyloggerCallback(Callback):
    r"""Pylogger Callback provides additional logging for during joint learning progress.

    Put logging messages here if you don't want to mess up the `JointLearning` (`LightningModule`) with a huge amount of logging codes.
    """

    def on_fit_start(self, trainer: Trainer, pl_module: CLAlgorithm) -> None:
        r"""Log messages for the start of joint training."""
        pylogger.info("Start joint training task!")

    def on_train_end(self, trainer: Trainer, pl_module: CLAlgorithm) -> None:
        r"""Log messages for the end of joint training."""
        pylogger.info("Finish training task %s!")

    def on_test_start(self, trainer: Trainer, pl_module: CLAlgorithm) -> None:
        r"""Log messages for the start of testing."""
        pylogger.info("Start testing tasks!")

    def on_test_end(self, trainer: Trainer, pl_module: CLAlgorithm) -> None:
        r"""Log messages for the end of testing."""
        pylogger.info("Finish testing tasks!")
