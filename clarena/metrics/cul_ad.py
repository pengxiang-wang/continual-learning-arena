r"""
The submodule in `metrics` for `CULAccuracyDifference`.
"""

__all__ = ["CULAccuracyDifference"]

import csv
import logging
import os
from typing import Any

import pandas as pd
from lightning import Trainer
from lightning.pytorch.utilities import rank_zero_only
from matplotlib import pyplot as plt
from torchmetrics import MeanMetric

from clarena.metrics import MetricCallback
from clarena.utils.eval import CULEvaluation
from clarena.utils.metrics import MeanMetricBatch

# always get logger for built-in logging in each module
pylogger = logging.getLogger(__name__)


class CULAccuracyDifference(MetricCallback):
    r"""Provides all actions that are related to CUL accuracy difference (AD) metric, which include:

    - Defining, initializing and recording AD metric.
    - Saving AD metric to files.
    - Visualizing AD metric as plots.

    The callback is able to produce the following outputs:

    - CSV files for AD in each task.
    - Coloured plot for AD in each task.

    Note that this callback is designed to be used with the `CULEvaluation` module, which is a special evaluation module for continual unlearning. It is not a typical test step in the algorithm, but rather a test protocol that evaluates the performance of the model on unlearned tasks.

    """

    def __init__(
        self,
        save_dir: str,
        accuracy_difference_csv_name: str = "ad.csv",
        accuracy_difference_plot_name: str | None = None,
    ) -> None:
        r"""
        **Args:**
        - **save_dir** (`str`): The directory where data and figures of metrics will be saved. Better inside the output folder.
        - **accuracy_difference_csv_name** (`str`): file name to save test accuracy difference metrics as CSV file.
        - **accuracy_difference_plot_name** (`str` | `None`): file name to save test accuracy difference metrics as plot. If `None`, no plot will be saved.

        """
        super().__init__(save_dir=save_dir)

        # paths
        self.accuracy_difference_csv_path: str = os.path.join(
            self.save_dir, accuracy_difference_csv_name
        )
        r"""The path to save the test accuracy difference metrics CSV file."""
        if accuracy_difference_plot_name:
            self.accuracy_difference_plot_path: str = os.path.join(
                self.save_dir, accuracy_difference_plot_name
            )
            r"""The path to save the test accuracy difference metrics plot file."""

        # test accumulated metrics
        self.accuracy_difference: dict[int, MeanMetricBatch] = {}
        r"""Accuracy difference (between main and full model) metrics for each seen task. Accumulated and calculated from the test batches. Keys are task IDs and values are the corresponding metrics."""

        # task ID control
        self.task_id: int
        r"""Task ID counter indicating which task is being processed. Self updated during the task loop. Valid from 1 to `cl_dataset.num_tasks`."""

    @rank_zero_only
    def on_test_start(
        self,
        trainer: Trainer,
        pl_module: CULEvaluation,
    ) -> None:
        r"""Initialize the metrics for testing each seen task in the beginning of a task's testing."""

        # get the device to put the metrics on the same device
        device = pl_module.device

        # initialize test metrics for evaluation tasks
        self.accuracy_difference = {
            task_id: MeanMetricBatch().to(device)
            for task_id in pl_module.ad_eval_task_ids
        }

    @rank_zero_only
    def on_test_batch_end(
        self,
        trainer: Trainer,
        pl_module: CULEvaluation,
        outputs: dict[str, Any],
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        r"""Accumulating metrics from test batch. We don't need to log and monitor the metrics of test batches.

        **Args:**
        - **outputs** (`dict[str, Any]`): the outputs of the test step, which is the returns of the `test_step()` method in the `CULEvaluation`.
        - **batch** (`Any`): the validation data batch.
        - **dataloader_idx** (`int`): the task ID of seen tasks to be tested. A default value of 0 is given otherwise the LightningModule will raise a `RuntimeError`.
        """

        # get the batch size
        batch_size = len(batch)

        test_task_id = pl_module.get_test_task_id_from_dataloader_idx(dataloader_idx)

        # get the metrics values of the batch from the outputs
        acc_diff = outputs["acc_diff"]  # accuracy difference

        # update the accumulated metrics in order to calculate the metrics of the epoch
        self.accuracy_difference[test_task_id].update(acc_diff, batch_size)

    @rank_zero_only
    def on_test_epoch_end(
        self,
        trainer: Trainer,
        pl_module: CULEvaluation,
    ) -> None:
        r"""Save and plot test metrics at the end of test."""

        self.update_unlearning_accuracy_difference_to_csv(
            accuracy_difference_metric=self.accuracy_difference,
            csv_path=self.accuracy_difference_csv_path,
        )

        if hasattr(self, "accuracy_difference_plot_path"):
            self.plot_unlearning_accuracy_difference_from_csv(
                csv_path=self.accuracy_difference_csv_path,
                plot_path=self.accuracy_difference_plot_path,
            )

    def update_unlearning_accuracy_difference_to_csv(
        self,
        accuracy_difference_metric: dict[int, MeanMetricBatch],
        csv_path: str,
    ) -> None:
        r"""Update the unlearning accuracy difference metrics of unlearning tasks to CSV file.

        **Args:**
        - **accuracy_difference_metric** (`dict[int, MeanMetricBatch]`): the accuracy difference metric. Accumulated and calculated from the unlearning test batches.
        - **csv_path** (`str`): save the test metric to path. E.g. './outputs/expr_name/1970-01-01_00-00-00/results/unlearning_test_after_task_X/distance.csv'.
        """

        eval_task_ids = list(accuracy_difference_metric.keys())
        fieldnames = ["average_accuracy_difference"] + [
            f"unlearning_test_on_task_{task_id}" for task_id in eval_task_ids
        ]

        new_line = {}

        # write to the columns and calculate the average accuracy difference over tasks at the same time
        average_accuracy_difference_over_unlearned_tasks = MeanMetric().to(
            next(iter(accuracy_difference_metric.values())).device
        )
        for task_id in eval_task_ids:
            loss_cls = accuracy_difference_metric[task_id].compute().item()
            new_line[f"unlearning_test_on_task_{task_id}"] = loss_cls
            average_accuracy_difference_over_unlearned_tasks(loss_cls)
        new_line["average_accuracy_difference"] = (
            average_accuracy_difference_over_unlearned_tasks.compute().item()
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

    def plot_unlearning_accuracy_difference_from_csv(
        self, csv_path: str, plot_path: str
    ) -> None:
        """Plot the unlearning accuracy difference matrix over different unlearned tasks from saved CSV file and save the plot to the designated directory.

        **Args:**
        - **csv_path** (`str`): the path to the CSV file where the `utils.save_accuracy_difference_to_csv()` saved the accuracy difference metric.
        - **plot_path** (`str`): the path to save plot. Better same as the output directory of the experiment. E.g. './outputs/expr_name/1970-01-01_00-00-00/results/unlearning_test_after_task_X/distance.png'.
        """
        data = pd.read_csv(csv_path)

        unlearned_task_ids = [
            int(col.replace("unlearning_test_on_task_", ""))
            for col in data.columns
            if col.startswith("unlearning_test_on_task_")
        ]
        num_tasks = len(unlearned_task_ids)
        num_tests = len(data)

        # plot the accuracy matrix
        fig, ax = plt.subplots(
            figsize=(2 * (num_tasks + 1), 2 * (num_tests + 1))
        )  # adaptive figure size
        cax = ax.imshow(
            data.drop(["average_accuracy_difference"], axis=1),
            interpolation="nearest",
            cmap="Greens",
            vmin=0,
            vmax=1,
        )

        colorbar = fig.colorbar(cax)
        yticks = colorbar.ax.get_yticks()
        colorbar.ax.set_yticks(yticks)
        colorbar.ax.set_yticklabels(
            [f"{tick:.2f}" for tick in yticks], fontsize=10 + num_tasks
        )  # adaptive font size

        r = 0
        for r in range(num_tests):
            for c in range(num_tasks):
                j = unlearned_task_ids[c]
                s = (
                    f"{data.loc[r, f'unlearning_test_on_task_{j}']:.3f}"
                    if f"unlearning_test_on_task_{j}" in data.columns
                    else ""
                )
                ax.text(
                    c,
                    r,
                    s,
                    ha="center",
                    va="center",
                    color="black",
                    fontsize=10 + num_tasks,  # adaptive font size
                )

        ax.set_xticks(range(num_tasks))
        ax.set_yticks(range(num_tests))
        ax.set_xticklabels(
            unlearned_task_ids, fontsize=10 + num_tasks
        )  # adaptive font size
        ax.set_yticklabels(
            range(1, num_tests + 1), fontsize=10 + num_tests
        )  # adaptive font size

        # Labeling the axes
        ax.set_xlabel(
            "Testing unlearning on task τ", fontsize=10 + num_tasks
        )  # adaptive font size
        ax.set_ylabel(
            "Unlearning test after training task t", fontsize=10 + num_tasks
        )  # adaptive font size
        fig.savefig(plot_path)
        plt.close(fig)
