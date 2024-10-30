"""
The submodule in `callbacks` for `MetricsCallback`.
"""

__all__ = ["MetricsCallback"]

import csv
import logging
import os
from typing import Any

import pandas as pd
import torch
from lightning import Callback, Trainer
from matplotlib import pyplot as plt
from torch import Tensor
from torchmetrics import MeanMetric, Metric

from clarena.cl_algorithms import CLAlgorithm

# always get logger for built-in logging in each module
pylogger = logging.getLogger(__name__)


class MetricsCallback(Callback):
    """Metrics Callback provides class for logging monitored metrics to Lightning loggers and else.

    Put `self.log()` here if you don't want to mess up the `CLAlgorithm` (`LightningModule`) with a huge amount of logging.
    """

    def __init__(
        self,
        test_results_output_dir: str,
    ):
        """Initialise the Metrics Callback.

        **Args:**
        - **test_results_output_dir** (`str`): the directory to save test results as documents. Better at the output directory.
        """

        if not os.path.exists(test_results_output_dir):
            os.makedirs(test_results_output_dir, exist_ok=True)

        self.test_results_output_dir = test_results_output_dir
        """Store the `test_results_output_dir` argument."""

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

    def on_fit_start(self, trainer: Trainer, pl_module: CLAlgorithm):
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
    ):
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
    ):
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
    ):
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
    ):
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
    ):
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
    ):
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
    ):
        """Log metrics of test to csv files learning curves and reset the metrics accumulation at the end of test epoch."""

        self._write_to_acc_csv()
        self._plot_acc_csv()
        self._write_to_loss_cls_csv()
        self._plot_loss_cls_csv()

    def _write_to_acc_csv(self):
        """Write the test accuracy metrics of `self.task_id` to a csv file in `self.test_metrics_save_dir`."""

        new_line = {"after_training_task": self.task_id}  # the first column

        # write to the columns and calculate the average accuracy over tasks at the same time
        average_accuracy_over_tasks = MeanMetric()
        for task_id in range(1, self.task_id + 1):
            # task_id = dataloader_idx
            acc = self.acc_test[task_id].compute().item()
            new_line[f"test_on_task_{task_id}"] = acc
            average_accuracy_over_tasks(acc)
        new_line["average_accuracy"] = average_accuracy_over_tasks.compute().item()

        fieldnames = ["after_training_task", "average_accuracy"] + [
            f"test_on_task_{task_id}" for task_id in range(1, self.task_id + 1)
        ]

        # write to the csv file
        acc_csv_path = os.path.join(self.test_results_output_dir, "acc.csv")
        is_first = not os.path.exists(acc_csv_path)
        if not is_first:
            with open(acc_csv_path, "r", encoding="utf-8") as file:
                lines = file.readlines()
                del lines[0]
        # write header
        with open(acc_csv_path, "w", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
        # write metrics
        with open(acc_csv_path, "a", encoding="utf-8") as file:
            if not is_first:
                file.writelines(lines)
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writerow(new_line)

    def _write_to_loss_cls_csv(self):
        """Write the test classification loss metrics of `self.task_id` in `self.test_metrics_save_dir`."""
        new_line = {"after_training_task": self.task_id}  # the first column

        # write to the columns and calculate the average classification loss over tasks at the same time
        average_classification_loss_over_tasks = MeanMetric()
        for task_id in range(1, self.task_id + 1):
            # task_id = dataloader_idx
            loss_cls = self.loss_cls_test[task_id].compute().item()
            new_line[f"test_on_task_{task_id}"] = loss_cls
            average_classification_loss_over_tasks(loss_cls)
        new_line["average_classification_loss"] = (
            average_classification_loss_over_tasks.compute().item()
        )

        fieldnames = ["after_training_task", "average_classification_loss"] + [
            f"test_on_task_{task_id}" for task_id in range(1, self.task_id + 1)
        ]

        # write to the csv file
        loss_cls_csv_path = os.path.join(self.test_results_output_dir, "loss_cls.csv")
        is_first = not os.path.exists(loss_cls_csv_path)
        if not is_first:
            with open(loss_cls_csv_path, "r", encoding="utf-8") as file:
                lines = file.readlines()
                del lines[0]
        # write header
        with open(loss_cls_csv_path, "w", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
        # write metrics
        with open(loss_cls_csv_path, "a", encoding="utf-8") as file:
            if not is_first:
                file.writelines(lines)
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writerow(new_line)

    def _plot_acc_csv(self):
        """Plot the test accuracy metrics from the csv file in `self.test_metrics_save_dir`."""
        acc_csv_path = os.path.join(self.test_results_output_dir, "acc.csv")
        acc_fig1_path = os.path.join(self.test_results_output_dir, "ave_acc.png")
        acc_fig2_path = os.path.join(self.test_results_output_dir, "acc.png")
        data = pd.read_csv(acc_csv_path)

        # plot the average accuracy curve over different training tasks
        fig, ax = plt.subplots(figsize=(16, 9))
        ax.plot(
            data["after_training_task"],
            data["average_accuracy"],
            marker="o",
            linewidth=2,
        )
        ax.set_xlabel("After training task $t$")
        ax.set_xlabel("Average Accuracy (AA)")
        ax.grid(True)
        ax.set_xticks(range(1, self.task_id + 1))
        ax.set_yticks([i * 0.05 for i in range(21)])
        fig.savefig(acc_fig1_path)

        # plot the accuracy matrix
        fig, ax = plt.subplots()
        cax = ax.imshow(
            data.drop(["after_training_task", "average_accuracy"], axis=1),
            interpolation="nearest",
            cmap="Greens",
        )
        fig.colorbar(cax)
        for i in range(self.task_id + 1):
            for j in range(1, i + 1):
                ax.text(
                    j - 1,
                    i - 1,
                    f'{data.loc[i - 1,f"test_on_task_{j}"]:.3f}',
                    ha="center",
                    va="center",
                    color="black",
                )
        ax.set_xticks(range(self.task_id))
        ax.set_yticks(range(self.task_id))

        ax.set_xticklabels(range(1, self.task_id + 1))
        ax.set_yticklabels(range(1, self.task_id + 1))

        # Labeling the axes
        ax.set_xlabel("Testing on task τ")
        ax.set_ylabel("After training task t")
        fig.savefig(acc_fig2_path)

    def _plot_loss_cls_csv(self):
        """Plot the classification loss metrics from the csv file in `self.test_metrics_save_dir`."""
        loss_cls_csv_path = os.path.join(self.test_results_output_dir, "loss_cls.csv")
        loss_cls_fig1_path = os.path.join(
            self.test_results_output_dir, "ave_loss_cls.png"
        )
        loss_cls_fig2_path = os.path.join(self.test_results_output_dir, "loss_cls.png")
        data = pd.read_csv(loss_cls_csv_path)

        # plot the average accuracy curve over different training tasks
        fig, ax = plt.subplots(figsize=(16, 9))
        ax.plot(
            data["after_training_task"],
            data["average_classification_loss"],
            marker="o",
            linewidth=2,
        )
        ax.set_xlabel("After training task $t$")
        ax.set_xlabel("Average Classification Loss")
        ax.grid(True)
        ax.set_xticks(range(1, self.task_id + 1))
        ax.set_yticks(
            [
                i * 0.5
                for i in range(int(data["average_classification_loss"].max() / 0.5) + 1)
            ]
        )
        fig.savefig(loss_cls_fig1_path)

        # plot the accuracy matrix
        fig, ax = plt.subplots()
        cax = ax.imshow(
            data.drop(["after_training_task", "average_classification_loss"], axis=1),
            interpolation="nearest",
            cmap="Greens",
        )
        fig.colorbar(cax)
        for i in range(self.task_id + 1):
            for j in range(1, i + 1):
                ax.text(
                    j - 1,
                    i - 1,
                    f'{data.loc[i - 1,f"test_on_task_{j}"]:.3f}',
                    ha="center",
                    va="center",
                    color="black",
                )
        ax.set_xticks(range(self.task_id))
        ax.set_yticks(range(self.task_id))

        ax.set_xticklabels(range(1, self.task_id + 1))
        ax.set_yticklabels(range(1, self.task_id + 1))

        # Labeling the axes
        ax.set_xlabel("Testing on task τ")
        ax.set_ylabel("After training task t")
        fig.savefig(loss_cls_fig2_path)


class MeanMetricBatch(Metric):
    """A torchmetrics metric to calculate the mean of certain metrics across data batches."""

    def __init__(self):
        """Initialise the Mean Metric Batch. Add state variables."""
        super().__init__()

        self.add_state("sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.sum: Tensor
        """State variable created by `add_state()` to store the sum of the metric values till this batch."""

        self.add_state("num", default=torch.tensor(0), dist_reduce_fx="sum")
        self.num: Tensor
        """State variable created by `add_state()` to store the number of the data till this batch."""

    def update(self, val: torch.Tensor, batch_size: int):
        """Update and accumulate the sum of metric value and num of the data till this batch from the batch.

        **Args:**
        - **val** (`torch.Tensor`): the metric value of the batch to update the sum.
        - **batch_size** (`int`): the value to update the num, which is the batch size.
        """
        self.sum += val * batch_size
        self.num += batch_size

    def compute(self):
        """Compute the mean of the metric value till this batch."""
        return self.sum.float() / self.num
