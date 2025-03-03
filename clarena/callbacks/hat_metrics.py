r"""
The submodule in `callbacks` for `HATMetricsCallback`.
"""

__all__ = ["HATMetricsCallback"]

import logging
import os
from typing import Any

from lightning import Callback, Trainer

from clarena.cl_algorithms import HAT, CLAlgorithm
from clarena.utils import plot

# always get logger for built-in logging in each module
pylogger = logging.getLogger(__name__)


class HATMetricsCallback(Callback):
    r"""Provides all actions that are related to metrics used for [HAT (Hard Attention to the Task)](http://proceedings.mlr.press/v80/serra18a) algorithm, which include:

    - Visualising mask and cumulative mask figures during training and testing as figures.
    - Logging network capacity during training. See the "Evaluation Metrics" section in chapter 4.1 in [AdaHAT paper](https://link.springer.com/chapter/10.1007/978-3-031-70352-2_9) for more details about network capacity.

    Lightning provides `self.log()` to log metrics in `LightningModule` where our `CLAlgorithm` based. You can put `self.log()` here if you don't want to mess up the `CLAlgorithm` with a huge amount of logging codes.

    The callback is able to produce the following outputs:

    - **Mask Figures**: both training and test, masks and cumulative masks.


    """

    def __init__(
        self,
        test_masks_plot_dir: str | None,
        test_cumulative_masks_plot_dir: str | None,
        training_masks_plot_dir: str | None,
        plot_training_mask_every_n_steps: int | None,
    ) -> None:
        r"""Initialise the `HATMetricsCallback`.

        **Args:**
        - **test_masks_plot_dir** (`str` | `None`): the directory to save the test mask figures. If `None`, no file will be saved.
        - **test_cumulative_masks_plot_dir** (`str` | `None`): the directory to save the test cumulative mask figures. If `None`, no file will be saved.
        - **training_masks_plot_dir** (`str` | `None`): the directory to save the training mask figures. If `None`, no file will be saved.
        - **plot_training_mask_every_n_steps** (`int`): the frequency of plotting training mask figures in terms of number of batches during training. Only applies when `training_masks_plot_dir` is not `None`.
        """
        Callback.__init__(self)

        # paths
        if not os.path.exists(test_masks_plot_dir):
            os.makedirs(test_masks_plot_dir, exist_ok=True)
        self.test_masks_plot_dir: str = test_masks_plot_dir
        r"""Store the directory to save the test mask figures."""
        if not os.path.exists(test_cumulative_masks_plot_dir):
            os.makedirs(test_cumulative_masks_plot_dir, exist_ok=True)
        self.test_cumulative_masks_plot_dir: str = test_cumulative_masks_plot_dir
        r"""Store the directory to save the test cumulative mask figures."""
        if not os.path.exists(training_masks_plot_dir):
            os.makedirs(training_masks_plot_dir, exist_ok=True)
        self.training_masks_plot_dir: str = training_masks_plot_dir
        r"""Store the directory to save the training mask figures."""

        # other settings
        self.plot_training_mask_every_n_steps: int = plot_training_mask_every_n_steps
        r"""Store the frequency of plotting training masks in terms of number of batches."""

        self.task_id: int
        r"""Task ID counter indicating which task is being processed. Self updated during the task loop."""

    def on_fit_start(self, trainer: Trainer, pl_module: CLAlgorithm) -> None:
        r"""Get the current task ID in the beginning of a task's fitting (training and validation). Sanity check the `pl_module` to be `HAT` or `AdaHAT`.

        **Raises:**
        -**TypeError**: when the `pl_module` is not `HAT` or `AdaHAT`.
        """

        # get the current task_id from the `CLAlgorithm` object
        self.task_id = pl_module.task_id

        # sanity check
        if not isinstance(pl_module, HAT):
            raise TypeError(
                "The `CLAlgorithm` should be `HAT` or `AdaHAT` to apply `HATMetricsCallback`!"
            )

    def on_train_batch_end(
        self,
        trainer: Trainer,
        pl_module: CLAlgorithm,
        outputs: dict[str, Any],
        batch: Any,
        batch_idx: int,
    ) -> None:
        r"""Plot training mask and log network capacity after training batch.

        **Args:**
        - **outputs** (`dict[str, Any]`): the outputs of the training step, which is the returns of the `training_step()` method in the `CLAlgorithm`.
        - **batch** (`Any`): the training data batch.
        - **batch_idx** (`int`): the index of the current batch. This is for the file name of mask figures.
        """

        # get the mask over the model after training the batch
        mask = outputs["mask"]
        # get the current network capacity
        capacity = outputs["capacity"]

        # plot the mask
        if batch_idx % self.plot_training_mask_every_n_steps == 0:
            plot.plot_hat_mask(
                mask=mask,
                plot_dir=self.training_masks_plot_dir,
                task_id=self.task_id,
                step=batch_idx,
            )

        # log the network capacity to Lightning loggers
        pl_module.log(
            f"task_{self.task_id}/train/network_capacity", capacity, prog_bar=True
        )

    def on_test_start(self, trainer: Trainer, pl_module: CLAlgorithm) -> None:
        r"""Plot test mask and cumulative mask figures."""

        # test mask
        mask = pl_module.masks[f"{self.task_id}"]
        plot.plot_hat_mask(
            mask=mask, plot_dir=self.test_masks_plot_dir, task_id=self.task_id
        )

        # cumulative mask
        cumulative_mask = pl_module.cumulative_mask_for_previous_tasks
        plot.plot_hat_mask(
            mask=cumulative_mask,
            plot_dir=self.test_cumulative_masks_plot_dir,
            task_id=self.task_id,
        )
