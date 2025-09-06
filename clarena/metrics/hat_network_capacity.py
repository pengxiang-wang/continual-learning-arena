r"""
The submodule in `metrics` for `HATNetworkCapacity`.
"""

__all__ = ["HATNetworkCapacity"]

import logging
from typing import Any

from lightning import Trainer
from lightning.pytorch.utilities import rank_zero_only

from clarena.cl_algorithms import HAT
from clarena.metrics import MetricCallback

# always get logger for built-in logging in each module
pylogger = logging.getLogger(__name__)


class HATNetworkCapacity(MetricCallback):
    r"""Provides all actions that are related to network capacity of [HAT (Hard Attention to the Task)](http://proceedings.mlr.press/v80/serra18a) algorithm and its extensions, which include:

    - Logging network capacity during training. See the "Evaluation Metrics" section in Sec. 4.1 in the [AdaHAT paper](https://link.springer.com/chapter/10.1007/978-3-031-70352-2_9) for more details about network capacity.

    """

    def __init__(
        self,
        save_dir: str,
    ) -> None:
        r"""
        **Args:**
        - **save_dir** (`str`): the directory to save the mask figures. Better inside the output folder.
        """
        super().__init__(save_dir=save_dir)

        # task ID control
        self.task_id: int
        r"""Task ID counter indicating which task is being processed. Self updated during the task loop. Valid from 1 to `cl_dataset.num_tasks`."""

    def on_fit_start(self, trainer: Trainer, pl_module: HAT) -> None:
        r"""Get the current task ID in the beginning of a task's fitting (training and validation). Sanity check the `pl_module` to be `HAT`.

        **Raises:**
        -**TypeError**: when the `pl_module` is not `HAT`.
        """

        # get the current task_id from the `CLAlgorithm` object
        self.task_id = pl_module.task_id

        # sanity check
        if not isinstance(pl_module, HAT):
            raise TypeError(
                "The `CLAlgorithm` should be `HAT` to apply `HATMetricCallback`!"
            )

    @rank_zero_only
    def on_train_batch_end(
        self,
        trainer: Trainer,
        pl_module: HAT,
        outputs: dict[str, Any],
        batch: Any,
        batch_idx: int,
    ) -> None:
        r"""Plot training mask, adjustment rate and log network capacity after training batch.

        **Args:**
        - **outputs** (`dict[str, Any]`): the outputs of the training step, which is the returns of the `training_step()` method in the `HAT`.
        - **batch** (`Any`): the training data batch.
        - **batch_idx** (`int`): the index of the current batch. This is for the file name of mask figures.
        """

        # get the current network capacity
        capacity = outputs["capacity"]

        # log the network capacity to Lightning loggers
        pl_module.log(
            f"task_{self.task_id}/train/network_capacity", capacity, prog_bar=True
        )
