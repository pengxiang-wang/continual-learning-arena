r"""
The submodule in `metrics` for `CULAccuracyGain`.
"""

__all__ = ["CULAccuracyGain"]

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


class CULAccuracyGain(MetricCallback):
    r"""Provides all actions that are related to CUL accuracy gain (AG) metric, which include:

    - Defining, initializing and recording AG metric.
    - Saving AG metric to files.
    - Visualizing AG metric as plots.

    The callback is able to produce the following outputs:

    - CSV files for AG in each task.
    - Coloured plot for AG in each task.

    Note that this callback is designed to be used with the `CULEvaluation` module, which is a special evaluation module for continual unlearning. It is not a typical test step in the algorithm, but rather a test protocol that evaluates the performance of the model on unlearned tasks.
    """

    def __init__(
        self,
        save_dir: str,
        accuracy_gain_csv_name: str = "ag.csv",
        accuracy_gain_plot_name: str | None = None,
        average_scope: str = "remaining",
    ) -> None:
        r"""
        **Args:**
        - **save_dir** (`str`): The directory where data and figures of metrics will be saved. Better inside the output folder.
        - **accuracy_gain_csv_name** (`str`): file name to save test accuracy gain metrics as CSV file.
        - **accuracy_gain_plot_name** (`str` | `None`): file name to save test accuracy gain metrics as plot. If `None`, no plot will be saved.
        - **average_scope** (`str`): scope to compute average AG over tasks, must be one of:
            - 'all': compute average AG over all eval tasks.
            - 'remaining': compute average AG over remaining tasks (exclude unlearned tasks). This is the default option which forms the AGR metric in the AmnesiacHAT paper.
            - 'unlearned': compute average AG over unlearned tasks.
        """
        super().__init__(save_dir=save_dir)

        # paths
        self.accuracy_gain_csv_path: str = os.path.join(save_dir, accuracy_gain_csv_name)
        r"""The path to save the test accuracy gain metrics CSV file."""
        if accuracy_gain_plot_name:
            self.accuracy_gain_plot_path: str = os.path.join(
                save_dir, accuracy_gain_plot_name
            )
            r"""The path to save the test accuracy gain metrics plot file."""

        # average scope control
        self.average_scope: str = average_scope
        r"""The scope to compute average AG over tasks."""

        self.average_task_ids: list[int] = []
        r"""Task IDs used to compute the average AG."""

        # test accumulated metrics
        self.accuracy_gain: dict[int, MeanMetricBatch]
        r"""Accuracy gain metrics for each seen task. Accumulated and calculated from the test batches. Keys are task IDs and values are the corresponding metrics."""

        self.sanity_check()

    def sanity_check(self) -> None:
        r"""Sanity check."""

        if self.average_scope not in ["all", "remaining", "unlearned"]:
            raise ValueError(
                f"Invalid average_scope: {self.average_scope} in `CULAccuracyGain`. Must be one of 'all', 'remaining', or 'unlearned'."
            )

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
        self.accuracy_gain = {
            task_id: MeanMetricBatch().to(device)
            for task_id in pl_module.ag_eval_task_ids
        }

        eval_task_ids = pl_module.ag_eval_task_ids
        unlearned_task_ids = set(
            getattr(pl_module.main_model, "unlearned_task_ids", [])
        )
        if self.average_scope == "all":
            self.average_task_ids = eval_task_ids
        elif self.average_scope == "unlearned":
            self.average_task_ids = [
                task_id for task_id in eval_task_ids if task_id in unlearned_task_ids
            ]
        else:
            self.average_task_ids = [
                task_id
                for task_id in eval_task_ids
                if task_id not in unlearned_task_ids
            ]

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
        - **batch** (`Any`): the test data batch.
        - **dataloader_idx** (`int`): the task ID of seen tasks to be tested. A default value of 0 is given otherwise the LightningModule will raise a `RuntimeError`.
        """

        # get the batch size
        batch_size = len(batch)

        test_task_id = pl_module.get_test_task_id_from_dataloader_idx(dataloader_idx)

        # get the metrics values of the batch from the outputs
        acc_gain_batch = outputs["acc_gain"]  # accuracy gain

        # update the accumulated metrics in order to calculate the metrics of the epoch
        self.accuracy_gain[test_task_id].update(acc_gain_batch, batch_size)

    @rank_zero_only
    def on_test_epoch_end(
        self,
        trainer: Trainer,
        pl_module: CULEvaluation,
    ) -> None:
        r"""Save and plot test metrics at the end of test."""

        self.update_unlearning_acc_gain_to_csv(
            csv_path=self.accuracy_gain_csv_path,
        )

        if hasattr(self, "accuracy_gain_plot_path"):
            self.plot_unlearning_acc_gain_from_csv(
                csv_path=self.accuracy_gain_csv_path,
                plot_path=self.accuracy_gain_plot_path,
            )

    def update_unlearning_acc_gain_to_csv(
        self,
        csv_path: str,
    ) -> None:
        r"""Update the unlearning accuracy gain metrics of unlearning tasks to CSV file.

        **Args:**
        - **csv_path** (`str`): save the test metric to path. E.g. './outputs/expr_name/1970-01-01_00-00-00/results/unlearning_test_after_task_X/acc_gain.csv'.
        """

        eval_task_ids = list(self.accuracy_gain.keys())
        average_col_name = f"average_acc_gain_on_{self.average_scope}"
        fieldnames = [average_col_name] + [
            f"unlearning_test_on_task_{task_id}" for task_id in eval_task_ids
        ]

        new_line = {}

        # write to the columns and calculate the average accuracy gain over selected tasks
        average_acc_gain_over_tasks = MeanMetric().to(
            device=next(iter(self.accuracy_gain.values())).device
        )
        for task_id in eval_task_ids:
            acc_gain = self.accuracy_gain[task_id].compute().item()
            new_line[f"unlearning_test_on_task_{task_id}"] = acc_gain
            if task_id in self.average_task_ids:
                average_acc_gain_over_tasks(acc_gain)
        average_acc_gain = average_acc_gain_over_tasks.compute().item()
        new_line[average_col_name] = average_acc_gain

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

    def plot_unlearning_acc_gain_from_csv(self, csv_path: str, plot_path: str) -> None:
        """Plot the unlearning accuracy gain matrix over different unlearned tasks from saved CSV file and save the plot to the designated directory.

        **Args:**
        - **csv_path** (`str`): the path to the CSV file where the `update_unlearning_acc_gain_to_csv()` saved the accuracy gain metric.
        - **plot_path** (`str`): the path to save plot. Better same as the output directory of the experiment. E.g. './outputs/expr_name/1970-01-01_00-00-00/results/unlearning_test_after_task_X/acc_gain.png'.
        """
        data = pd.read_csv(csv_path)
        eval_task_ids = [
            int(col.replace("unlearning_test_on_task_", ""))
            for col in data.columns
            if col.startswith("unlearning_test_on_task_")
        ]
        test_task_cols = [
            col for col in data.columns if col.startswith("unlearning_test_on_task_")
        ]
        num_tasks = len(eval_task_ids)
        num_rows = len(data)

        # Build the accuracy gain matrix
        acc_gain_matrix = data[test_task_cols].values

        # plot the accuracy gain matrix
        fig, ax = plt.subplots(
            figsize=(2 * num_tasks, 2 * num_rows)
        )  # adaptive figure size
        cax = ax.imshow(
            acc_gain_matrix,
            interpolation="nearest",
            cmap="Greens",
            vmin=0,
            vmax=1,
            aspect="auto",
        )

        colorbar = fig.colorbar(cax)
        yticks = colorbar.ax.get_yticks()
        colorbar.ax.set_yticks(yticks)
        colorbar.ax.set_yticklabels(
            [f"{tick:.2f}" for tick in yticks], fontsize=10 + num_tasks
        )  # adaptive font size

        for r in range(num_rows):
            for c in range(num_tasks):
                ax.text(
                    c,
                    r,
                    f"{acc_gain_matrix[r, c]:.3f}",
                    ha="center",
                    va="center",
                    color="black",
                    fontsize=10 + num_tasks,  # adaptive font size
                )

        ax.set_xticks(range(num_tasks))
        ax.set_yticks(range(num_rows))
        ax.set_xticklabels(
            eval_task_ids, fontsize=10 + num_tasks
        )  # adaptive font size
        ax.set_yticklabels(
            range(1, num_rows + 1), fontsize=10 + num_rows
        )  # adaptive font size

        # Labeling the axes
        ax.set_xlabel(
            "Testing unlearning on task τ", fontsize=10 + num_tasks
        )  # adaptive font size
        ax.set_ylabel(
            "Unlearning test after training task t", fontsize=10 + num_tasks
        )  # adaptive font size
        fig.tight_layout()
        fig.savefig(plot_path)
        plt.close(fig)
