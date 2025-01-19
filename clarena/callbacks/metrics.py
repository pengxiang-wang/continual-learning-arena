"""
The submodule in `callbacks` for `MetricsCallback`.
"""

__all__ = ["MetricsCallback"]

import logging
import os
from typing import Any

import pandas as pd
import torch
from lightning import Callback, Trainer
from matplotlib import pyplot as plt
from sympy import im
from torch import Tensor

from clarena.cl_algorithms import CLAlgorithm
from clarena.utils import MeanMetricBatch, plot, save

# always get logger for built-in logging in each module
pylogger = logging.getLogger(__name__)


class MetricsCallback(Callback):
    """Metrics Callback provides class for logging monitored metrics to Lightning loggers, saving metrics to files, plotting metrics to figures and so on.

    Put `self.log()` here if you don't want to mess up the `CLAlgorithm` (`LightningModule`) with a huge amount of logging.
    """

    def __init__(
        self,
        acc_csv_path: str,
        loss_cls_csv_path: str, 
        if_plot_test_acc: bool,
        if_plot_test_loss_cls: bool,
        ave_acc_plot_path: str | None = None, 
        acc_matrix_plot_path: str | None  = None, 
        ave_loss_cls_plot_path: str | None  = None,
        loss_cls_matrix_plot_path: str | None  = None, 
    ) -> None:
        """Initialise the Metrics Callback.

        **Args:**
        - **acc_csv_path** (`str`): path to save accuracy csv file.
        - **loss_cls_csv_path**(`str`): path to save classification loss file.
        - **if_plot_test_acc** (`bool`): whether to plot accuracy results of testing.
        - **ave_acc_plot_path** (`str`): path to save average accuracy line chart plot over different training tasks.
        - **acc_matrix_plot_path** (`str`): path to save accuracy matrix plot.
        - **if_plot_test_loss_cls** (`bool`): whether to plot classification loss results of testing.
        - **ave_loss_cls_plot_path** (`str`): path to save average classification loss line chart plot.
        - **loss_cls_matrix_plot_path** (`str`): path to save classification loss matrix plot.
        """
        
        self.acc_csv_path: str = acc_csv_path
        """Store the path to save accuracy csv file."""
        self.loss_cls_csv_path: str = loss_cls_csv_path
        """Store the path to save classification loss file."""
        self.if_plot_test_acc: bool = if_plot_test_acc
        """Store whether to plot accuracy results of testing."""
        self.ave_acc_plot_path: str = ave_acc_plot_path
        """Store the path to save average accuracy line chart plot."""
        self.acc_matrix_plot_path: str = acc_matrix_plot_path
        """Store the path to save accuracy matrix plot."""
        self.if_plot_test_loss_cls: bool = if_plot_test_loss_cls
        """Store whether to plot classification loss results of testing."""
        self.ave_loss_cls_plot_path: str = ave_loss_cls_plot_path
        """Store the path to save average classification loss line chart plot."""
        self.loss_cls_matrix_plot_path: str = loss_cls_matrix_plot_path
        """Store the path to save classification loss matrix plot."""

        self.task_id: int
        """Task ID counter indicating which task is being processed. Self updated during the task loop."""

        self.loss_cls_train: MeanMetricBatch
        """Classification loss of the training data. Accumulated and calculated from the training batchs."""
        self.loss_train: MeanMetricBatch
        """Total loss of the training data. Accumulated and calculated from the training batchs."""
        self.acc_train: MeanMetricBatch
        """Classification accuracy of the training data. Accumulated and calculated from the training batchs."""

        self.loss_cls_val: MeanMetricBatch
        """Classification loss of the validation data. Accumulated and calculated from the validation batchs."""
        self.acc_val: MeanMetricBatch
        """Classification accuracy of the validation data. Accumulated and calculated from the validation batchs."""

        self.loss_cls_test: dict[int, MeanMetricBatch]
        """Classification loss of the test data of each seen task. Accumulated and calculated from the test batchs."""
        self.acc_test: dict[int, MeanMetricBatch]
        """Classification accuracy of the test data of each seen task. Accumulated and calculated from the test batchs."""

    def on_fit_start(self, trainer: Trainer, pl_module: CLAlgorithm) -> None:
        """Initialise the metrics for training and validation and get the current task ID in the beginning of a task's fitting (training and validation)."""
        # get the current task_id from the `CLAlgorithm` object
        self.task_id = pl_module.task_id

        # initialise training metrics
        self.loss_cls_train = MeanMetricBatch()
        self.loss_train = MeanMetricBatch()
        self.acc_train = MeanMetricBatch()

        # initialise validation metrics
        self.loss_cls_val = MeanMetricBatch()
        self.acc_val = MeanMetricBatch()

    def on_train_batch_end(
        self,
        trainer: Trainer,
        pl_module: CLAlgorithm,
        outputs: dict[str, Any],
        batch: Any,
        batch_idx: int,
    ) -> None:
        """Log metrics from training batch.

        **Args:**
        - **outputs** (`dict[str, Any]`): the outputs of the training step, which is the returns of the `training_step()` method in the `CLAlgorithm`.
        - **batch** (`Any`): the training data batch.
        """
        # get the batch size
        batch_size = len(batch)

        # get the metrics values of the batch from the outputs
        loss_cls_batch = outputs["loss_cls"]
        loss_batch = outputs["loss"]
        acc_batch = outputs["acc"]

        # update the metrics in this callback in order to accumulate and calculate the metrics of the epoch
        self.loss_cls_train.update(loss_cls_batch, batch_size)
        self.loss_train.update(loss_cls_batch, batch_size)
        self.acc_train.update(acc_batch, batch_size)

        # log the metrics of the batch to Lightning loggers
        pl_module.log(
            f"task_{self.task_id}/train/loss_cls_batch", loss_cls_batch, prog_bar=True
        )
        pl_module.log(
            f"task_{self.task_id}/train/loss_batch", loss_batch, prog_bar=True
        )
        pl_module.log(f"task_{self.task_id}/train/acc_batch", acc_batch, prog_bar=True)

        # log the accumulated and computed metrics till the batch to Lightning loggers
        pl_module.log(
            f"task_{self.task_id}/train/loss_cls",
            self.loss_cls_train.compute(),
            prog_bar=True,
        )
        pl_module.log(
            f"task_{self.task_id}/train/loss", self.loss_train.compute(), prog_bar=True
        )
        pl_module.log(
            f"task_{self.task_id}/train/acc", self.acc_train.compute(), prog_bar=True
        )

    def on_train_epoch_end(
        self,
        trainer: Trainer,
        pl_module: CLAlgorithm,
    ) -> None:
        """Log metrics from training epoch to plot learning curves and reset the metrics accumulation at the end of training epoch."""

        # log the accumulated and computed metrics of the epoch to Lightning loggers, specially for plotting learning curves
        pl_module.log(
            f"task_{self.task_id}/learning_curve/train/loss_cls",
            self.loss_cls_train.compute(),
            on_epoch=True,
            prog_bar=True,
        )
        pl_module.log(
            f"task_{self.task_id}/learning_curve/train/acc",
            self.acc_train.compute(),
            on_epoch=True,
            prog_bar=True,
        )

        # reset the metrics accumulation every epoch as there are more epochs to go and not only one epoch like in the validation and test
        self.loss_cls_train.reset()
        self.loss_train.reset()
        self.acc_train.reset()

    def on_validation_batch_end(
        self,
        trainer: Trainer,
        pl_module: CLAlgorithm,
        outputs: dict[str, Any],
        batch: Any,
        batch_idx: int,
    ) -> None:
        """Accumulating metrics from validation batch. We don't need to log and monitor the metrics of validation batches.

        **Args:**
        - **outputs** (`dict[str, Any]`): the outputs of the validation step, which is the returns of the `validation_step()` method in the `CLAlgorithm`.
        - **batch** (`Any`): the validation data batch.
        """

        # get the batch size
        batch_size = len(batch)

        # get the metrics values of the batch from the outputs
        loss_cls_batch = outputs["loss_cls"]
        acc_batch = outputs["acc"]

        # update the metrics in this callback in order to accumulate and calculate the metrics of the epoch
        self.loss_cls_val.update(loss_cls_batch, batch_size)
        self.acc_val.update(acc_batch, batch_size)

    def on_validation_epoch_end(
        self,
        trainer: Trainer,
        pl_module: CLAlgorithm,
    ) -> None:
        """Log metrics of validation to plot learning curves and reset the metrics accumulation at the end of validation epoch."""

        # log the accumulated and computed metrics of the epoch to Lightning loggers, specially for plotting learning curves
        pl_module.log(
            f"task_{self.task_id}/learning_curve/val/loss_cls",
            self.loss_cls_val.compute(),
            on_epoch=True,
            prog_bar=True,
        )
        pl_module.log(
            f"task_{self.task_id}/learning_curve/val/acc",
            self.acc_val.compute(),
            on_epoch=True,
            prog_bar=True,
        )

    def on_test_start(
        self,
        trainer: Trainer,
        pl_module: CLAlgorithm,
    ) -> None:
        """Initialise the metrics for testing each seen task in the beginning of a task's testing."""

        # get the current task_id again (double checking) from the `CLAlgorithm` object
        self.task_id = pl_module.task_id

        # initialise test metrics for each seen task
        self.loss_cls_test = {
            task_id: MeanMetricBatch() for task_id in range(1, self.task_id + 1)
        }
        self.acc_test = {
            task_id: MeanMetricBatch() for task_id in range(1, self.task_id + 1)
        }

    def on_test_batch_end(
        self,
        trainer: Trainer,
        pl_module: CLAlgorithm,
        outputs: dict[str, Any],
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Accumulating metrics from test batch. We don't need to log and monitor the metrics of test batches.

        **Args:**
        - **outputs** (`dict[str, Any]`): the outputs of the test step, which is the returns of the `test_step()` method in the `CLAlgorithm`.
        - **batch** (`Any`): the validation data batch.
        - **dataloader_idx** (`int`): the task ID of seen tasks to be tested. A default value of 0 is given otherwise the LightningModule will raise a `RuntimeError`.
        """

        # get the batch size
        batch_size = len(batch)

        task_id = dataloader_idx + 1

        # get the metrics values of the batch from the outputs
        loss_cls_batch = outputs["loss_cls"]
        acc_batch = outputs["acc"]

        # update the metrics in this callback in order to accumulate and calculate the metrics of the epoch
        self.loss_cls_test[task_id].update(loss_cls_batch, batch_size)
        self.acc_test[task_id].update(acc_batch, batch_size)

    def on_test_epoch_end(
        self,
        trainer: Trainer,
        pl_module: CLAlgorithm,
    ) -> None:
        """Save and plot metrics of testing to csv files. """

        save.save_acc_to_csv(acc_test_metric=self.acc_test, task_id=self.task_id, csv_path=self.acc_csv_path)
        save.save_loss_cls_to_csv(loss_cls_test_metric=self.loss_cls_test, task_id=self.task_id, csv_path=self.loss_cls_csv_path)        
        if self.if_plot_test_acc:
            plot.plot_ave_acc_from_csv(csv_path=self.acc_csv_path, task_id=self.task_id, plot_path=self.ave_acc_plot_path)
            plot.plot_acc_matrix_from_csv(csv_path=self.acc_csv_path, task_id=self.task_id, plot_path=self.acc_matrix_plot_path)

        if self.if_plot_test_loss_cls:
            plot.plot_ave_loss_cls_from_csv(csv_path=self.loss_cls_csv_path, task_id=self.task_id, plot_path=self.ave_loss_cls_plot_path)
            plot.plot_loss_cls_matrix_from_csv(csv_path=self.loss_cls_csv_path, task_id=self.task_id, plot_path=self.loss_cls_matrix_plot_path)
