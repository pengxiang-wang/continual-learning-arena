r"""
The submodule in `callbacks` for `CULDistributionDistance`.
"""

__all__ = ["CULDistributionDistance"]


import csv
import logging
import os
from typing import Any

import pandas as pd
import torch
from lightning import Trainer
from lightning.pytorch.utilities import rank_zero_only
from matplotlib import pyplot as plt
from torchmetrics import MeanMetric

from clarena.metrics import MetricCallback
from clarena.utils.eval import CULEvaluation
from clarena.utils.metrics import MeanMetricBatch

# always get logger for built-in logging in each module
pylogger = logging.getLogger(__name__)


class CULDistributionDistance(MetricCallback):
    r"""Provides all actions that are related to CUL distribution distance (DD) metric, which include:

    - Defining, initializing and recording DD metric.
    - Saving DD metric to files.
    - Visualizing DD metric as plots.

    The callback is able to produce the following outputs:

    - CSV files for DD in each task.
    - Coloured plot for DD in each task.

    Note that this callback is designed to be used with the `CULEvaluation` module, which is a special evaluation module for continual unlearning. It is not a typical test step in the algorithm, but rather a test protocol that evaluates the performance of the model on unlearned tasks.

    """

    def __init__(
        self,
        save_dir: str,
        distribution_distance_type: str,
        distribution_distance_csv_name: str = "dd.csv",
        distribution_distance_plot_name: str | None = None,
    ) -> None:
        r"""
        **Args:**
        - **save_dir** (`str`): The directory where data and figures of metrics will be saved. Better inside the output folder.
        - **distribution_distance_type** (`str`): the type of distribution distance to use; one of:
            - 'euclidean': Eulidean distance.
            - 'cosine': Cosine distance.
            - 'manhattan': Manhattan distance.
        - **distribution_distance_csv_name** (`str`): file name to save test distribution distance metrics as CSV file.
        - **distribution_distance_plot_name** (`str` | `None`): file name to save test distribution distance metrics as plot. If `None`, no plot will be saved.

        """
        super().__init__(save_dir=save_dir)

        self.distribution_distance_type: str = distribution_distance_type
        r"""The type of distribution distance to use. """

        # paths
        self.distribution_distance_csv_path: str = os.path.join(
            self.save_dir, distribution_distance_csv_name
        )
        r"""The path to save the test distribution distance metrics CSV file."""
        if distribution_distance_plot_name:
            self.distribution_distance_plot_path: str = os.path.join(
                self.save_dir, distribution_distance_plot_name
            )
            r"""The path to save the test distribution distance metrics plot file."""

        # test accumulated metrics
        self.distribution_distance: dict[int, MeanMetricBatch]
        r"""Distribution distance unlearning metrics for each seen task. Accumulated and calculated from the test batches. Keys are task IDs and values are the corresponding metrics."""

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
        self.distribution_distance = {
            task_id: MeanMetricBatch().to(device)
            for task_id in pl_module.dd_eval_task_ids
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

        # get the raw outputs from the outputs dictionary
        agg_out_main = outputs["agg_out_main"]  # aggregated outputs from the main model
        agg_out_ref = outputs[
            "agg_out_ref"
        ]  # aggregated outputs from the reference model

        if agg_out_main.dim() != 2:
            raise ValueError(
                f"Expected aggregated outputs to be (batch_size, flattened feature), i.e. 2 dimension, but got {agg_out_main.dim()}."
            )

        if agg_out_ref.dim() != 2:
            raise ValueError(
                f"Expected aggregated outputs to be (batch_size, flattened feature), i.e. 2 dimension, but got {agg_out_ref.dim()}."
            )

        # calculate the distribution distance between the main and reference model outputs
        if self.distribution_distance_type == "euclidean":
            distance = torch.torch.norm(
                agg_out_main - agg_out_ref, p=2, dim=-1
            ).mean()  # Euclidean distance

        elif self.distribution_distance_type == "cosine":
            distance = (
                1
                - torch.nn.functional.cosine_similarity(
                    agg_out_main, agg_out_ref, dim=-1
                )
            ).mean()  # cosine distance
        elif self.distribution_distance_type == "manhattan":
            distance = torch.norm(
                agg_out_main - agg_out_ref, p=1, dim=-1
            ).mean()  # Manhattan distance
        else:
            distance = None

        # update the accumulated metrics in order to calculate the metrics of the epoch
        self.distribution_distance[test_task_id].update(distance, batch_size)

    @rank_zero_only
    def on_test_epoch_end(
        self,
        trainer: Trainer,
        pl_module: CULEvaluation,
    ) -> None:
        r"""Save and plot test metrics at the end of test."""

        self.update_distribution_distance_to_csv(
            distance_metric=self.distribution_distance,
            csv_path=self.distribution_distance_csv_path,
        )

        if hasattr(self, "distribution_distance_plot_path"):
            self.plot_distribution_distance_from_csv(
                csv_path=self.distribution_distance_csv_path,
                plot_path=self.distribution_distance_plot_path,
            )

    def update_distribution_distance_to_csv(
        self,
        distance_metric: dict[int, MeanMetricBatch],
        csv_path: str,
    ) -> None:
        r"""Update the unlearning test distance metrics of unlearning tasks to CSV file.

        **Args:**
        - **distance_metric** (`dict[int, MeanMetricBatch]`): the distance metric of unlearned tasks. Accumulated and calculated from the unlearning test batches.
        - **csv_path** (`str`): save the test metric to path. E.g. './outputs/expr_name/1970-01-01_00-00-00/results/unlearning_test_after_task_X/distance.csv'.
        """

        eval_task_ids = list(distance_metric.keys())
        fieldnames = ["average_distribution_distance"] + [
            f"unlearning_test_on_task_{task_id}" for task_id in eval_task_ids
        ]

        new_line = {}

        # write to the columns and calculate the average distribution distance over tasks at the same time
        average_distribution_distance_over_unlearned_tasks = MeanMetric().to(
            next(iter(distance_metric.values())).device
        )
        for task_id in eval_task_ids:
            loss_cls = distance_metric[task_id].compute().item()
            new_line[f"unlearning_test_on_task_{task_id}"] = loss_cls
            average_distribution_distance_over_unlearned_tasks(loss_cls)
        new_line["average_distribution_distance"] = (
            average_distribution_distance_over_unlearned_tasks.compute().item()
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

    def plot_distribution_distance_from_csv(
        self, csv_path: str, plot_path: str
    ) -> None:
        """Plot the unlearning test distance matrix over different unlearned tasks from saved CSV file and save the plot to the designated directory.

        **Args:**
        - **csv_path** (`str`): the path to the CSV file where the `utils.save_distribution_distance_to_csv()` saved the unlearning test distance metric.
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
            data.drop(["average_distribution_distance"], axis=1),
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
        plt.close(fig)
