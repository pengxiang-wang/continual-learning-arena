r"""
The submodule in `metrics` for `MTLAccuracy`.
"""

__all__ = ["MTLAccuracy"]

import csv
import logging
import os
from typing import Any

import pandas as pd
from lightning import Trainer
from matplotlib import pyplot as plt
from torchmetrics import MeanMetric

from clarena.metrics import MetricCallback
from clarena.mtl_algorithms import MTLAlgorithm
from clarena.utils.metrics import MeanMetricBatch

# always get logger for built-in logging in each module
pylogger = logging.getLogger(__name__)


class MTLAccuracy(MetricCallback):
    r"""Provides all actions that are related to MTL accuracy metrics, which include:

    - Defining, initializing and recording accuracy metric.
    - Logging training and validation accuracy metric to Lightning loggers in real time.
    - Saving test accuracy metric to files.
    - Visualizing test accuracy metric as plots.

    The callback is able to produce the following outputs:

    - CSV files for test accuracy of all tasks and average accuracy.
    - Bar charts for test accuracy of all tasks.
    """

    def __init__(
        self,
        save_dir: str,
        test_acc_csv_name: str = "acc.csv",
        test_acc_plot_name: str | None = None,
    ) -> None:
        r"""
        **Args:**
        - **save_dir** (`str`): The directory where data and figures of metrics will be saved. Better inside the output folder.
        - **test_acc_csv_name** (`str`): file name to save test accuracy of all tasks and average accuracy as CSV file.
        - **test_acc_plot_name** (`str` | `None`): file name to save accuracy plot. If `None`, no file will be saved.
        """
        super().__init__(save_dir=save_dir)

        # paths
        self.test_acc_csv_path: str = os.path.join(save_dir, test_acc_csv_name)
        r"""The path to save test accuracy of all tasks and average accuracy CSV file."""
        if test_acc_plot_name:
            self.test_acc_plot_path: str = os.path.join(save_dir, test_acc_plot_name)
            r"""The path to save test accuracy plot."""

        # training accumulated metrics
        self.acc_training_epoch: MeanMetricBatch
        r"""Classification accuracy of training epoch. Accumulated and calculated from the training batches. """

        # validation accumulated metrics
        self.acc_val: dict[int, MeanMetricBatch] = {}
        r"""Validation classification accuracy of the model after training epoch. Accumulated and calculated from the validation batches. Keys are task IDs and values are the corresponding metrics."""

        # test accumulated metrics
        self.acc_test: dict[int, MeanMetricBatch] = {}
        r"""Test classification accuracy of all tasks. Accumulated and calculated from the test batches. Keys are task IDs and values are the corresponding metrics."""

    def on_fit_start(self, trainer: Trainer, pl_module: MTLAlgorithm) -> None:
        r"""Initialize training and validation metrics."""

        # initialize training metrics
        self.acc_training_epoch = MeanMetricBatch()

        # initialize validation metrics
        self.acc_val = {
            task_id: MeanMetricBatch() for task_id in trainer.datamodule.train_tasks
        }

    def on_train_batch_end(
        self,
        trainer: Trainer,
        pl_module: MTLAlgorithm,
        outputs: dict[str, Any],
        batch: Any,
        batch_idx: int,
    ) -> None:
        r"""Record training metrics from training batch, log metrics of training batch and accumulated metrics of the epoch to Lightning loggers.

        **Args:**
        - **outputs** (`dict[str, Any]`): the outputs of the training step, the returns of the `training_step()` method in the `MTLAlgorithm`.
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
        pl_module: MTLAlgorithm,
    ) -> None:
        r"""Log metrics of training epoch to plot learning curves and reset the metrics accumulation at the end of training epoch."""

        # reset the metrics of training epoch as there are more epochs to go and not only one epoch like in the validation and test
        self.acc_training_epoch.reset()

    def on_validation_batch_end(
        self,
        trainer: Trainer,
        pl_module: MTLAlgorithm,
        outputs: dict[str, Any],
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        r"""Accumulating metrics from validation batch. We don't need to log and monitor the metrics of validation batches.

        **Args:**
        - **outputs** (`dict[str, Any]`): the outputs of the validation step, which is the returns of the `validation_step()` method in the `MTLAlgorithm`.
        - **batch** (`Any`): the validation data batch.
        - **dataloader_idx** (`int`): the task ID of the validation dataloader. A default value of 0 is given otherwise the LightningModule will raise a `RuntimeError`.
        """

        # get the batch size
        batch_size = len(batch)

        val_task_id = pl_module.get_val_task_id_from_dataloader_idx(dataloader_idx)

        # get the metrics values of the batch from the outputs
        acc_batch = outputs["acc"]

        # update the accumulated metrics in order to calculate the validation metrics
        self.acc_val[val_task_id].update(acc_batch, batch_size)

    def on_test_start(
        self,
        trainer: Trainer,
        pl_module: MTLAlgorithm,
    ) -> None:
        r"""Initialize the metrics for testing each seen task in the beginning of a task's testing."""

        # initialize test metrics for current and previous tasks
        self.acc_test = {
            task_id: MeanMetricBatch() for task_id in trainer.datamodule.eval_tasks
        }

    def on_test_batch_end(
        self,
        trainer: Trainer,
        pl_module: MTLAlgorithm,
        outputs: dict[str, Any],
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        r"""Accumulating metrics from test batch. We don't need to log and monitor the metrics of test batches.

        **Args:**
        - **outputs** (`dict[str, Any]`): the outputs of the test step, which is the returns of the `test_step()` method in the `MTLAlgorithm`.
        - **batch** (`Any`): the validation data batch.
        - **dataloader_idx** (`int`): the task ID of seen tasks to be tested. A default value of 0 is given otherwise the LightningModule will raise a `RuntimeError`.
        """

        # get the batch size
        batch_size = len(batch)

        test_task_id = pl_module.get_test_task_id_from_dataloader_idx(dataloader_idx)

        # get the metrics values of the batch from the outputs
        acc_batch = outputs["acc"]

        # update the accumulated metrics in order to calculate the metrics of the epoch
        self.acc_test[test_task_id].update(acc_batch, batch_size)

    def on_test_epoch_end(
        self,
        trainer: Trainer,
        pl_module: MTLAlgorithm,
    ) -> None:
        r"""Save and plot test metrics at the end of test."""

        # save (update) the test metrics to CSV files
        self.save_test_acc_to_csv(
            csv_path=self.test_acc_csv_path,
        )

        # plot the test metrics
        if hasattr(self, "test_acc_plot_path"):
            self.plot_test_acc_from_csv(
                csv_path=self.test_acc_csv_path,
                plot_path=self.test_acc_plot_path,
            )

    def save_test_acc_to_csv(
        self,
        csv_path: str,
    ) -> None:
        r"""Save the test accuracy metrics of all tasks in multi-task learning to an CSV file.

        **Args:**
        - **csv_path** (`str`): save the test metric to path. E.g. './outputs/expr_name/1970-01-01_00-00-00/results/acc.csv'.
        """
        all_task_ids = list(self.acc_test.keys())
        fieldnames = ["average_accuracy"] + [
            f"test_on_task_{task_id}" for task_id in all_task_ids
        ]
        new_line = {}

        # construct the columns and calculate the average accuracy over tasks at the same time
        average_accuracy_over_tasks = MeanMetric().to(
            device=next(iter(self.acc_test.values())).device
        )
        for task_id in all_task_ids:
            acc = self.acc_test[task_id].compute().item()
            new_line[f"test_on_task_{task_id}"] = acc
            average_accuracy_over_tasks(acc)
        new_line["average_accuracy"] = average_accuracy_over_tasks.compute().item()

        # write
        with open(csv_path, "w", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerow(new_line)

    def plot_test_acc_from_csv(self, csv_path: str, plot_path: str) -> None:
        """Plot the test accuracy bar chart of all tasks in multi-task learning from saved CSV file and save the plot to the designated directory.

        **Args:**
        - **csv_path** (`str`): the path to the csv file where the `utils.save_test_acc_csv()` saved the test accuracy metric.
        - **plot_path** (`str`): the path to save plot. Better same as the output directory of the experiment. E.g. './outputs/expr_name/1970-01-01_00-00-00/acc.png'.
        """
        data = pd.read_csv(csv_path)

        # extract all accuracy columns including average
        all_columns = data.columns.tolist()
        task_ids = list(range(len(all_columns)))  # assign index-based positions
        labels = [
            (
                col.replace("test_on_task_", "Task ")
                if "test_on_task_" in col
                else "Average"
            )
            for col in all_columns
        ]
        accuracies = data.iloc[0][all_columns].values

        # plot the accuracy bar chart over tasks
        fig, ax = plt.subplots(figsize=(16, 9))
        ax.bar(
            task_ids,
            accuracies,
            color="skyblue",
            edgecolor="black",
        )
        ax.set_xlabel("Task", fontsize=16)
        ax.set_ylabel("Accuracy", fontsize=16)
        ax.grid(True)
        ax.set_xticks(task_ids)
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=14)
        ax.set_yticks([i * 0.05 for i in range(21)])
        ax.set_yticklabels(
            [f"{tick:.2f}" for tick in [i * 0.05 for i in range(21)]], fontsize=14
        )
        fig.tight_layout()
        fig.savefig(plot_path)
        plt.close(fig)
