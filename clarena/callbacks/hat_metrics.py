"""
The submodule in `callbacks` for `MetricsCallback`.
"""

__all__ = ["MetricsCallback"]

import csv
import logging
import os
from re import L
from typing import Any

import pandas as pd
import torch
from lightning import Callback, Trainer
from matplotlib import pyplot as plt
from torch import Tensor
from torchmetrics import MeanMetric, Metric

from clarena.cl_algorithms import HAT, AdaHAT, CLAlgorithm

# always get logger for built-in logging in each module
pylogger = logging.getLogger(__name__)


class HATMetricsCallback(Callback):
    """HAT Metrics Callback provides class for logging monitored metrics to Lightning loggers, saving metrics to files, plotting metrics to figures and so on. The metrics are particularly related to [HAT (Hard Attention to the Task)](http://proceedings.mlr.press/v80/serra18a)) algorithm.

    Put `self.log()` here if you don't want to mess up the `CLAlgorithm` (`LightningModule`) with a huge amount of logging.
    """

    def __init__(
        self,
        if_plot_train_mask: bool, 
        plot_train_mask_every_n_batches: int, 
        test_results_output_dir: str,
    ) -> None:
        """Initialise the Metrics Callback.

        **Args:**
        - **if_plot_train_mask** (`bool`): whether to plot mask figure during training. 
        - **plot_train_mask_every_n_batches** (`int`): the frequency of plotting mask figure in terms of number of batches during training.
        - **if_log_network_capacity** (`bool`): whether to log mask 
        - **test_results_output_dir** (`str`): the directory to save test results related to HAT as files (mask figures, capacity plots, ...). Better same as the output directory of the experiment.
        """


        self.if_plot_train_mask: bool = if_plot_train_mask
        """Store whether to plot train mask."""
        self.plot_train_mask_every_n_batches: int = plot_train_mask_every_n_batches
        """Store the frequency of plotting train mask in terms of number of batches."""
        
        if not os.path.exists(test_results_output_dir):
            os.makedirs(test_results_output_dir, exist_ok=True)
        self.test_results_output_dir: str = test_results_output_dir
        """Store the `test_results_output_dir` argument."""

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
        """
        
        # get the mask over the model after training the batch
        mask = outputs["mask"]
        
        self._plot_mask(mask)
        
        
        
        
       