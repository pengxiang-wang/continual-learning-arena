r"""
The submodule in `metrics` for `STLAccuracy`.
"""

__all__ = ["STLAccuracy"]

import csv
import logging
import os
from typing import Any

from lightning import Trainer

from clarena.metrics import MetricCallback
from clarena.stl_algorithms import STLAlgorithm
from clarena.utils.metrics import MeanMetricBatch

# always get logger for built-in logging in each module
pylogger = logging.getLogger(__name__)


class STLAccuracy(MetricCallback):
    r"""Provides all actions that are related to STL accuracy metric, which include:

    - Defining, initializing and recording accuracy metric.
    - Logging training and validation accuracy metric to Lightning loggers in real time.

    Saving test accuracy metric to files.

    - The callback is able to produce the following outputs:
    - CSV files for test accuracy.
    """

    def __init__(
        self,
        save_dir: str,
        test_acc_csv_name: str = "acc.csv",
    ) -> None:
        r"""
        **Args:**
        - **save_dir** (`str`): The directory where data and figures of metrics will be saved. Better inside the output folder.
        - **test_acc_csv_name** (`str`): file name to save test accuracy of all tasks and average accuracy as CSV file.
        """
        super().__init__(save_dir=save_dir)

        # paths
        self.test_acc_csv_path: str = os.path.join(save_dir, test_acc_csv_name)
        r"""The path to save test accuracy of all tasks and average accuracy CSV file."""

        # training accumulated metrics
        self.acc_training_epoch: MeanMetricBatch
        r"""Classification accuracy of training epoch. Accumulated and calculated from the training batches. """

        # validation accumulated metrics
        self.acc_val: MeanMetricBatch
        r"""Validation classification accuracy of the model after training epoch. Accumulated and calculated from the validation batches. """

        # test accumulated metrics
        self.acc_test: MeanMetricBatch
        r"""Test classification accuracy. Accumulated and calculated from the test batches."""

    def on_fit_start(self, trainer: Trainer, pl_module: STLAlgorithm) -> None:
        r"""Initialize training and validation metrics."""

        # initialize training metrics
        self.acc_training_epoch = MeanMetricBatch()

        # initialize validation metrics
        self.acc_val = MeanMetricBatch()

    def on_train_batch_end(
        self,
        trainer: Trainer,
        pl_module: STLAlgorithm,
        outputs: dict[str, Any],
        batch: Any,
        batch_idx: int,
    ) -> None:
        r"""Record training metrics from training batch, log metrics of training batch and accumulated metrics of the epoch to Lightning loggers.

        **Args:**
        - **outputs** (`dict[str, Any]`): the outputs of the training step, the returns of the `training_step()` method in the `STLAlgorithm`.
        - **batch** (`Any`): the training data batch.
        """
        # get the batch size
        batch_size = len(batch)

        # get training metrics values of current training batch from the outputs of the `training_step()`
        acc_batch = outputs["acc"]

        # update accumulated training metrics to calculate training metrics of the epoch
        self.acc_training_epoch.update(acc_batch, batch_size)

        # log training metrics of current training batch to Lightning loggers
        pl_module.log("train/acc_batch", acc_batch, prog_bar=True)

        # log accumulated training metrics till this training batch to Lightning loggers
        pl_module.log(
            "task/train/acc",
            self.acc_training_epoch.compute(),
            prog_bar=True,
        )

    def on_train_epoch_end(
        self,
        trainer: Trainer,
        pl_module: STLAlgorithm,
    ) -> None:
        r"""Log metrics of training epoch to plot learning curves and reset the metrics accumulation at the end of training epoch."""

        # log the accumulated and computed metrics of the epoch to Lightning loggers, specially for plotting learning curves
        pl_module.log(
            "learning_curve/train/acc",
            self.acc_training_epoch.compute(),
            on_epoch=True,
            prog_bar=True,
        )

        # reset the metrics of training epoch as there are more epochs to go and not only one epoch like in the validation and test
        self.acc_training_epoch.reset()

    def on_validation_batch_end(
        self,
        trainer: Trainer,
        pl_module: STLAlgorithm,
        outputs: dict[str, Any],
        batch: Any,
        batch_idx: int,
    ) -> None:
        r"""Accumulating metrics from validation batch. We don't need to log and monitor the metrics of validation batches.

        **Args:**
        - **outputs** (`dict[str, Any]`): the outputs of the validation step, which is the returns of the `validation_step()` method in the `STLAlgorithm`.
        - **batch** (`Any`): the validation data batch.
        """

        # get the batch size
        batch_size = len(batch)

        # get the metrics values of the batch from the outputs
        acc_batch = outputs["acc"]

        # update the accumulated metrics in order to calculate the validation metrics
        self.acc_val.update(acc_batch, batch_size)

    def on_validation_epoch_end(
        self,
        trainer: Trainer,
        pl_module: STLAlgorithm,
    ) -> None:
        r"""Log validation metrics to plot learning curves."""

        # log the accumulated and computed metrics of the epoch to Lightning loggers, specially for plotting learning curves
        pl_module.log(
            "learning_curve/val/acc",
            self.acc_val.compute(),
            on_epoch=True,
            prog_bar=True,
        )

    def on_test_start(
        self,
        trainer: Trainer,
        pl_module: STLAlgorithm,
    ) -> None:
        r"""Initialize the testing metrics."""

        # initialize test metrics for current and previous tasks
        self.acc_test = MeanMetricBatch()

    def on_test_batch_end(
        self,
        trainer: Trainer,
        pl_module: STLAlgorithm,
        outputs: dict[str, Any],
        batch: Any,
        batch_idx: int,
    ) -> None:
        r"""Accumulating metrics from test batch. We don't need to log and monitor the metrics of test batches.

        **Args:**
        - **outputs** (`dict[str, Any]`): the outputs of the test step, which is the returns of the `test_step()` method in the `STLAlgorithm`.
        - **batch** (`Any`): the test data batch.
        """

        # get the batch size
        batch_size = len(batch)

        # get the metrics values of the batch from the outputs
        acc_batch = outputs["acc"]

        # update the accumulated metrics in order to calculate the metrics of the epoch
        self.acc_test.update(acc_batch, batch_size)

    def on_test_epoch_end(
        self,
        trainer: Trainer,
        pl_module: STLAlgorithm,
    ) -> None:
        r"""Save and plot test metrics at the end of test."""

        # save (update) the test metrics to CSV files
        self.save_test_acc_to_csv(
            csv_path=self.test_acc_csv_path,
        )

    def save_test_acc_to_csv(
        self,
        csv_path: str,
    ) -> None:
        r"""Save the test accuracy metrics of all tasks in single-task learning to an CSV file.

        **Args:**
        - **csv_path** (`str`): save the test metric to path. E.g. './outputs/expr_name/1970-01-01_00-00-00/results/acc.csv'.
        """
        fieldnames = ["accuracy"]
        new_line = {}
        new_line["accuracy"] = self.acc_test.compute().item()

        # write
        with open(csv_path, "w", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerow(new_line)
