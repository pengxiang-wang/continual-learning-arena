r"""
The submodule in `metrics` for `CLLoss`.
"""

__all__ = ["CLLoss"]

import csv
import logging
import os
from typing import Any

import pandas as pd
from lightning import Trainer
from lightning.pytorch.utilities import rank_zero_only
from matplotlib import pyplot as plt
from torchmetrics import MeanMetric

from clarena.cl_algorithms import CLAlgorithm
from clarena.metrics import MetricCallback
from clarena.utils.metrics import MeanMetricBatch

# always get logger for built-in logging in each module
pylogger = logging.getLogger(__name__)


class CLLoss(MetricCallback):
    r"""Provides all actions that are related to CL loss metrics, which include:

    - Defining, initializing and recording loss metrics.
    - Logging training and validation loss metrics to Lightning loggers in real time.
    - Saving test loss metrics to files.
    - Visualizing test loss metrics as plots.


    The callback is able to produce the following outputs:

    - CSV files for classification loss (lower triangular) matrix and average classification loss. See [here](https://pengxiang-wang.com/posts/continual-learning-metrics.html#sec-test-performance-of-previous-tasks) for details.
    - Coloured plot for test classification loss (lower triangular) matrix. See [here](https://pengxiang-wang.com/posts/continual-learning-metrics.html#sec-test-performance-of-previous-tasks) for details.
    - Curve plots for test average classification loss over different training tasks. See [here](https://pengxiang-wang.com/posts/continual-learning-metrics.html#sec-average-test-performance-over-tasks) for details.

    Please refer to the [A Summary of Continual Learning Metrics](https://pengxiang-wang.com/posts/continual-learning-metrics) to learn about this metric.
    """

    def __init__(
        self,
        save_dir: str,
        test_loss_cls_csv_name: str = "loss_cls.csv",
        test_loss_cls_matrix_plot_name: str | None = None,
        test_ave_loss_cls_plot_name: str | None = None,
    ) -> None:
        r"""
        **Args:**
        - **save_dir** (`str`): The directory where data and figures of metrics will be saved. Better inside the output folder.
        - **test_loss_cls_csv_name**(`str`): file name to save classification loss matrix and average classification loss as CSV file.
        - **test_loss_cls_matrix_plot_name** (`str` | `None`): file name to save classification loss matrix plot. If `None`, no file will be saved.
        - **test_ave_loss_cls_plot_name** (`str` | `None`): file name to save average classification loss as curve plot over different training tasks. If `None`, no file will be saved.
        """
        super().__init__(save_dir=save_dir)

        self.test_loss_cls_csv_path: str = os.path.join(
            save_dir, test_loss_cls_csv_name
        )
        r"""The path to save test classification loss matrix and average classification loss CSV file."""
        if test_loss_cls_matrix_plot_name:
            self.test_loss_cls_matrix_plot_path: str = os.path.join(
                save_dir, test_loss_cls_matrix_plot_name
            )
            r"""The path to save test classification loss matrix plot."""
        if test_ave_loss_cls_plot_name:
            self.test_ave_loss_cls_plot_path: str = os.path.join(
                save_dir, test_ave_loss_cls_plot_name
            )
            r"""The path to save test average classification loss curve plot."""

        # training accumulated metrics
        self.loss_cls_training_epoch: MeanMetricBatch
        r"""Classification loss of training epoch. Accumulated and calculated from the training batches. See [here](https://pengxiang-wang.com/posts/continual-learning-metrics.html#sec-performance-of-training-epoch) for details. """
        self.loss_training_epoch: MeanMetricBatch
        r"""Total loss of training epoch. Accumulated and calculated from the training batches. See [here](https://pengxiang-wang.com/posts/continual-learning-metrics.html#sec-performance-of-training-epoch) for details. """

        # validation accumulated metrics
        self.loss_cls_val: MeanMetricBatch
        r"""Validation classification of the model loss after training epoch. Accumulated and calculated from the validation batches. See [here](https://pengxiang-wang.com/posts/continual-learning-metrics.html#sec-validation-performace) for details. """

        # test accumulated metrics
        self.loss_cls_test: dict[int, MeanMetricBatch]
        r"""Test classification loss of the current model (`self.task_id`) on current and previous tasks. Accumulated and calculated from the test batches. Keys are task IDs and values are the corresponding metrics. It is the last row of the lower triangular matrix. See [here](https://pengxiang-wang.com/posts/continual-learning-metrics.html#sec-test-performance-of-previous-tasks) for details. """

        # task ID control
        self.task_id: int
        r"""Task ID counter indicating which task is being processed. Self updated during the task loop. Valid from 1 to `cl_dataset.num_tasks`."""

    @rank_zero_only
    def on_fit_start(self, trainer: Trainer, pl_module: CLAlgorithm) -> None:
        r"""Initialize training and validation metrics."""

        # set the current task_id from the `CLAlgorithm` object
        self.task_id = pl_module.task_id

        # get the device to put the metrics on the same device
        device = pl_module.device

        # initialize training metrics
        self.loss_cls_training_epoch = MeanMetricBatch().to(device)
        self.loss_training_epoch = MeanMetricBatch().to(device)

        # initialize validation metrics
        self.loss_cls_val = MeanMetricBatch().to(device)

    @rank_zero_only
    def on_train_batch_end(
        self,
        trainer: Trainer,
        pl_module: CLAlgorithm,
        outputs: dict[str, Any],
        batch: Any,
        batch_idx: int,
    ) -> None:
        r"""Record training metrics from training batch, log metrics of training batch and accumulated metrics of the epoch to Lightning loggers.

        **Args:**
        - **outputs** (`dict[str, Any]`): the outputs of the training step, the returns of the `training_step()` method in the `CLAlgorithm`.
        - **batch** (`Any`): the training data batch.
        """
        # get the batch size
        batch_size = len(batch)

        # get training metrics values of current training batch from the outputs of the `training_step()`
        loss_cls_batch = outputs["loss_cls"]
        loss_batch = outputs["loss"]

        # update accumulated training metrics to calculate training metrics of the epoch
        self.loss_cls_training_epoch.update(loss_cls_batch, batch_size)
        self.loss_training_epoch.update(loss_batch, batch_size)

        # log training metrics of current training batch to Lightning loggers
        pl_module.log(
            f"task_{self.task_id}/train/loss_cls_batch", loss_cls_batch, prog_bar=True
        )
        pl_module.log(
            f"task_{self.task_id}/train/loss_batch", loss_batch, prog_bar=True
        )

        # log accumulated training metrics till this training batch to Lightning loggers
        pl_module.log(
            f"task_{self.task_id}/train/loss_cls",
            self.loss_cls_training_epoch.compute(),
            prog_bar=True,
        )
        pl_module.log(
            f"task_{self.task_id}/train/loss",
            self.loss_training_epoch.compute(),
            prog_bar=True,
        )

    @rank_zero_only
    def on_train_epoch_end(
        self,
        trainer: Trainer,
        pl_module: CLAlgorithm,
    ) -> None:
        r"""Log metrics of training epoch to plot learning curves and reset the metrics accumulation at the end of training epoch."""

        # log the accumulated and computed metrics of the epoch to Lightning loggers, specially for plotting learning curves
        pl_module.log(
            f"task_{self.task_id}/learning_curve/train/loss_cls",
            self.loss_cls_training_epoch.compute(),
            on_epoch=True,
            prog_bar=True,
        )

        # reset the metrics of training epoch as there are more epochs to go and not only one epoch like in the validation and test
        self.loss_cls_training_epoch.reset()
        self.loss_training_epoch.reset()

    @rank_zero_only
    def on_validation_batch_end(
        self,
        trainer: Trainer,
        pl_module: CLAlgorithm,
        outputs: dict[str, Any],
        batch: Any,
        batch_idx: int,
    ) -> None:
        r"""Accumulating metrics from validation batch. We don't need to log and monitor the metrics of validation batches.

        **Args:**
        - **outputs** (`dict[str, Any]`): the outputs of the validation step, which is the returns of the `validation_step()` method in the `CLAlgorithm`.
        - **batch** (`Any`): the validation data batch.
        """

        # get the batch size
        batch_size = len(batch)

        # get the metrics values of the batch from the outputs
        loss_cls_batch = outputs["loss_cls"]

        # update the accumulated metrics in order to calculate the validation metrics
        self.loss_cls_val.update(loss_cls_batch, batch_size)

    @rank_zero_only
    def on_validation_epoch_end(
        self,
        trainer: Trainer,
        pl_module: CLAlgorithm,
    ) -> None:
        r"""Log validation metrics to plot learning curves."""

        # log the accumulated and computed metrics of the epoch to Lightning loggers, specially for plotting learning curves
        pl_module.log(
            f"task_{self.task_id}/learning_curve/val/loss_cls",
            self.loss_cls_val.compute(),
            on_epoch=True,
            prog_bar=True,
        )

    @rank_zero_only
    def on_test_start(
        self,
        trainer: Trainer,
        pl_module: CLAlgorithm,
    ) -> None:
        r"""Initialize the metrics for testing each seen task in the beginning of a task's testing."""

        # set the current task_id again (double checking) from the `CLAlgorithm` object
        self.task_id = pl_module.task_id

        # get the device to put the metrics on the same device
        device = pl_module.device

        # initialize test metrics for current and previous tasks
        self.loss_cls_test = {
            task_id: MeanMetricBatch().to(device)
            for task_id in pl_module.processed_task_ids
        }

    @rank_zero_only
    def on_test_batch_end(
        self,
        trainer: Trainer,
        pl_module: CLAlgorithm,
        outputs: dict[str, Any],
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        r"""Accumulating metrics from test batch. We don't need to log and monitor the metrics of test batches.

        **Args:**
        - **outputs** (`dict[str, Any]`): the outputs of the test step, which is the returns of the `test_step()` method in the `CLAlgorithm`.
        - **batch** (`Any`): the test data batch.
        - **dataloader_idx** (`int`): the task ID of seen tasks to be tested. A default value of 0 is given otherwise the LightningModule will raise a `RuntimeError`.
        """

        # get the batch size
        batch_size = len(batch)

        test_task_id = pl_module.get_test_task_id_from_dataloader_idx(dataloader_idx)

        # get the metrics values of the batch from the outputs
        loss_cls_batch = outputs["loss_cls"]

        # update the accumulated metrics in order to calculate the metrics of the epoch
        self.loss_cls_test[test_task_id].update(loss_cls_batch, batch_size)

    @rank_zero_only
    def on_test_epoch_end(
        self,
        trainer: Trainer,
        pl_module: CLAlgorithm,
    ) -> None:
        r"""Save and plot test metrics at the end of test."""

        # save (update) the test metrics to CSV files
        self.update_test_loss_cls_to_csv(
            after_training_task_id=self.task_id,
            csv_path=self.test_loss_cls_csv_path,
        )

        # plot the test metrics
        if hasattr(self, "test_loss_cls_matrix_plot_path"):
            self.plot_test_loss_cls_matrix_from_csv(
                csv_path=self.test_loss_cls_csv_path,
                plot_path=self.test_loss_cls_matrix_plot_path,
            )
        if hasattr(self, "test_ave_loss_cls_plot_path"):
            self.plot_test_ave_loss_cls_curve_from_csv(
                csv_path=self.test_loss_cls_csv_path,
                plot_path=self.test_ave_loss_cls_plot_path,
            )

    def update_test_loss_cls_to_csv(
        self,
        after_training_task_id: int,
        csv_path: str,
    ) -> None:
        """Update the test classification loss metrics of seen tasks at the last line to an existing CSV file. A new file will be created if not existing.

        **Args:**
        - **after_training_task_id** (`int`): the task ID after training.
        - **csv_path** (`str`): save the test metric to path. E.g. './outputs/expr_name/1970-01-01_00-00-00/results/loss_cls.csv'.
        """
        processed_task_ids = list(self.loss_cls_test.keys())
        fieldnames = ["after_training_task", "average_classification_loss"] + [
            f"test_on_task_{task_id}" for task_id in processed_task_ids
        ]

        new_line = {
            "after_training_task": after_training_task_id
        }  # construct the first column

        # write to the columns and calculate the average classification loss over tasks at the same time
        average_classification_loss_over_tasks = MeanMetric().to(
            device=next(iter(self.loss_cls_test.values())).device
        )
        for task_id in processed_task_ids:
            loss_cls = self.loss_cls_test[task_id].compute().item()
            new_line[f"test_on_task_{task_id}"] = loss_cls
            average_classification_loss_over_tasks(loss_cls)
        new_line["average_classification_loss"] = (
            average_classification_loss_over_tasks.compute().item()
        )

        # write to the csv file
        is_first = not os.path.exists(csv_path)
        if not is_first:
            with open(csv_path, "r", encoding="utf-8") as file:
                lines = file.readlines()
                del lines[0]
        # write header
        with open(csv_path, "w", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
        # write metrics
        with open(csv_path, "a", encoding="utf-8") as file:
            if not is_first:
                file.writelines(lines)  # write the previous lines
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writerow(new_line)

    def plot_test_loss_cls_matrix_from_csv(self, csv_path: str, plot_path: str) -> None:
        """Plot the test classification loss matrix from saved CSV file and save the plot to the designated directory.

        **Args:**
        - **csv_path** (`str`): the path to the CSV file where the `utils.update_loss_cls_to_csv()` saved the test classification loss metric.
        - **plot_path** (`str`): the path to save plot. Better same as the output directory of the experiment. E.g. './outputs/expr_name/1970-01-01_00-00-00/loss_cls_matrix.png'.
        """
        data = pd.read_csv(csv_path)
        processed_task_ids = [
            int(col.replace("test_on_task_", ""))
            for col in data.columns
            if col.startswith("test_on_task_")
        ]

        # Get all columns that start with "test_on_task_"
        test_task_cols = [
            col for col in data.columns if col.startswith("test_on_task_")
        ]
        num_tasks = len(processed_task_ids)
        num_rows = len(data)

        # Build the loss matrix
        loss_matrix = data[test_task_cols].values

        fig, ax = plt.subplots(
            figsize=(2 * num_tasks, 2 * num_rows)
        )  # adaptive figure size

        cax = ax.imshow(
            loss_matrix,
            interpolation="nearest",
            cmap="Greens",
            aspect="auto",
        )

        colorbar = fig.colorbar(cax)
        yticks = colorbar.ax.get_yticks()
        colorbar.ax.set_yticks(yticks)
        colorbar.ax.set_yticklabels(
            [f"{tick:.2f}" for tick in yticks], fontsize=10 + num_tasks
        )

        # Annotate each cell
        for r in range(num_rows):
            for c in range(r + 1):
                ax.text(
                    c,
                    r,
                    f"{loss_matrix[r, c]:.3f}",
                    ha="center",
                    va="center",
                    color="black",
                    fontsize=10 + num_tasks,
                )

        ax.set_xticks(range(num_tasks))
        ax.set_yticks(range(num_rows))
        ax.set_xticklabels(processed_task_ids, fontsize=10 + num_tasks)
        ax.set_yticklabels(
            data["after_training_task"].astype(int).tolist(), fontsize=10 + num_tasks
        )

        # Labeling the axes
        ax.set_xlabel("Testing on task τ", fontsize=10 + num_tasks)
        ax.set_ylabel("After training task t", fontsize=10 + num_tasks)
        fig.tight_layout()
        fig.savefig(plot_path)
        plt.close(fig)

    def plot_test_ave_loss_cls_curve_from_csv(
        self, csv_path: str, plot_path: str
    ) -> None:
        """Plot the test average classfication loss curve over different training tasks from saved CSV file and save the plot to the designated directory.

        **Args:**
        - **csv_path** (`str`): the path to the CSV file where the `utils.update_test_acc_to_csv()` saved the test classfication loss metric.
        - **plot_path** (`str`): the path to save plot. Better same as the output directory of the experiment. E.g. './outputs/expr_name/1970-01-01_00-00-00/ave_loss_cls.png'.
        """
        data = pd.read_csv(csv_path)
        after_training_tasks = data["after_training_task"].astype(int).tolist()

        # plot the average accuracy curve over different training tasks
        fig, ax = plt.subplots(figsize=(16, 9))
        ax.plot(
            after_training_tasks,
            data["average_classification_loss"],
            marker="o",
            linewidth=2,
        )
        ax.set_xlabel("After training task $t$", fontsize=16)
        ax.set_ylabel("Average Classification Loss", fontsize=16)
        ax.grid(True)
        xticks = after_training_tasks
        yticks = [i * 0.05 for i in range(21)]
        ax.set_xticks(xticks)
        ax.set_yticks(yticks)
        ax.set_xticklabels(xticks, fontsize=16)
        ax.set_yticklabels([f"{tick:.2f}" for tick in yticks], fontsize=16)
        fig.savefig(plot_path)
        plt.close(fig)
