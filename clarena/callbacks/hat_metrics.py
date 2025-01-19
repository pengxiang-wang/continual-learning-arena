"""
The submodule in `callbacks` for `MetricsCallback`.
"""

__all__ = ["MetricsCallback"]

import logging
import os
from typing import Any

from lightning import Callback, Trainer

from clarena.cl_algorithms import HAT, AdaHAT, CLAlgorithm
from clarena.utils import plot

# always get logger for built-in logging in each module
pylogger = logging.getLogger(__name__)


class HATMetricsCallback(Callback):
    """HAT Metrics Callback provides class for logging monitored metrics to Lightning loggers, saving metrics to files, plotting metrics to figures and so on. The metrics are particularly related to [HAT (Hard Attention to the Task)](http://proceedings.mlr.press/v80/serra18a)) algorithm.

    Put `self.log()` here if you don't want to mess up the `CLAlgorithm` (`LightningModule`) with a huge amount of logging.
    """

    def __init__(
        self,
        test_mask_plot_dir: str,
        test_cumulative_mask_plot_dir: str,
        if_plot_train_mask: bool,
        train_mask_plot_dir: str,
        plot_train_mask_every_n_steps: int, 
    ) -> None:
        """Initialise the Metrics Callback.

        **Args:**
        - **test_mask_plot_dir** (`str`): the directory to save the test mask figure.
        - **test_cumulative_mask_plot_dir** (`str`): the directory to save the test cumulative mask figure.
        - **if_plot_train_mask** (`bool`): whether to plot mask figure during training. 
        - **train_mask_plot_dir** (`str`): the directory to save the train mask figure.
        - **plot_train_mask_every_n_steps** (`int`): the frequency of plotting mask figure in terms of number of batches during training.
        - **if_log_network_capacity** (`bool`): whether to log mask.
        """
        if not os.path.exists(test_mask_plot_dir):
            os.makedirs(test_mask_plot_dir, exist_ok=True)
        self.test_mask_plot_dir: str = test_mask_plot_dir
        """Store the directory to save the test mask figure."""
        if not os.path.exists(test_cumulative_mask_plot_dir):
            os.makedirs(test_cumulative_mask_plot_dir, exist_ok=True)
        self.test_cumulative_mask_plot_dir: str = test_cumulative_mask_plot_dir
        """Store the directory to save the test cumulative mask figure."""

        self.if_plot_train_mask: bool = if_plot_train_mask
        """Store whether to plot train mask."""
                
        if not os.path.exists(train_mask_plot_dir):
            os.makedirs(train_mask_plot_dir, exist_ok=True)
        self.train_mask_plot_dir: str = train_mask_plot_dir
        """Store the directory to save the train mask figure."""
        self.plot_train_mask_every_n_steps: int = plot_train_mask_every_n_steps
        """Store the frequency of plotting train mask in terms of number of batches."""

        self.task_id: int
        """Task ID counter indicating which task is being processed. Self updated during the task loop."""


    def on_fit_start(self, trainer: Trainer, pl_module: CLAlgorithm) -> None:
        """Get the current task ID in the beginning of a task's fitting (training and validation).
        
        **Raises:**
        -**TypeError**: when the `pl_module` is not HAT or AdaHAT.
        """
        
        # get the current task_id from the `CLAlgorithm` object
        self.task_id = pl_module.task_id
        
        if not isinstance(pl_module, (HAT, AdaHAT)):
            raise TypeError("The CLAlgorithm should be HAT or AdaHAT to apply HATMetricsCallback!")

    def on_train_batch_end(
        self,
        trainer: Trainer,
        pl_module: CLAlgorithm,
        outputs: dict[str, Any],
        batch: Any,
        batch_idx: int,
    ) -> None:
        """Plot mask and log HAT related metrics from training batch.

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
        if batch_idx % self.plot_train_mask_every_n_steps == 0:
            plot.plot_hat_mask(mask=mask, plot_dir=self.train_mask_plot_dir, task_id=self.task_id, batch_idx=batch_idx)

        # log the network capacity to Lightning loggers
        pl_module.log(
            f"task_{self.task_id}/train/network_capacity", capacity, prog_bar=True
        )
        

    def on_test_start(self, trainer: Trainer, pl_module: CLAlgorithm) -> None:
        """Plot mask and log HAT related metrics of testing. """

        # get the test mask
        mask = pl_module.backbone.get_mask(stage="test")
        plot.plot_hat_mask(mask=mask, plot_dir=self.test_mask_plot_dir, task_id=self.task_id)
        
        # get the cumulative mask
        cumulative_mask = pl_module.backbone.get_cumulative_mask()
        plot.plot_hat_mask(mask=cumulative_mask, plot_dir=self.test_cumulative_mask_plot_dir, task_id=self.task_id)