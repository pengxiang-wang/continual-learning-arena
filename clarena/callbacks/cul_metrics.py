r"""
The submodule in `callbacks` for `CULMetricsCallback`.
"""

__all__ = ["CULMetricsCallback"]

import logging
import multiprocessing
import os
from typing import Any

import torch
from lightning import Callback, Trainer
from omegaconf import DictConfig
from sympy import Q
from torchmetrics import MeanMetric

from clarena.base import CLExperiment
from clarena.callbacks import CLMetricsCallback
from clarena.cl_algorithms import CLAlgorithm
from clarena.cl_datasets.base import CLDataset
from clarena.clrun import clrun_from_cfg
from clarena.unlearning_algorithms import UnlearningAlgorithm
from clarena.utils import MeanMetricBatch, js_div, plot, save

# always get logger for built-in logging in each module
pylogger = logging.getLogger(__name__)


class CULMetricsCallback(CLMetricsCallback):
    r"""Provides all actions that are related to CUL metrics, which include:

    - Defining, initialising and recording metrics.
    - Logging training and validation metrics to Lightning loggers in real time.
    - Saving test metrics to files.
    - Visualising test metrics as plots.

    Lightning provides `self.log()` to log metrics in `LightningModule` where our `CLAlgorithm` based. You can put `self.log()` here if you don't want to mess up the `CLAlgorithm` with a huge amount of logging codes.

    The callback is able to produce the following outputs:

    """

    def __init__(
        self,
        save_dir: str,
        test_acc_csv_name: str,
        test_loss_cls_csv_name: str,
        test_acc_matrix_plot_name: str | None = None,
        test_loss_cls_matrix_plot_name: str | None = None,
        test_ave_acc_excluding_unlearned_plot_name: str | None = None,
        test_ave_loss_cls_excluding_unlearned_plot_name: str | None = None,
        unlearning_test_after_task_ids: list[int] | None = None,
    ) -> None:
        r"""Initialise the `CLMetricsCallback`.

        **Args:**
        - **save_dir** (`str` | `None`): the directory to save the test metrics files and plots. Better inside the output folder.
        - **test_acc_csv_name** (`str`): file name to save test accuracy matrix and average accuracy as CSV file.
        - **test_loss_cls_csv_name**(`str`): file name to save classification loss matrix and average classification loss as CSV file.
        - **test_acc_matrix_plot_name** (`str` | `None`): file name to save accuracy matrix plot. If `None`, no file will be saved.
        - **test_loss_cls_matrix_plot_name** (`str` | `None`): file name to save classification loss matrix plot. If `None`, no file will be saved.
        - **test_ave_acc_excluding_unlearned_plot_name** (`str` | `None`): file name to save average accuracy as curve plot over different training tasks excluding the unlearned tasks. If `None`, no file will be saved.
        - **test_ave_loss_cls_excluding_unlearned_plot_name** (`str` | `None`): file name to save average classification loss as curve plot over different training tasks excluding the unlearned tasks. If `None`, no file will be saved.
        - **unlearning_test_after_task_ids** (`list[int]` | `None`): the task IDs that after which need to evaluate unlearning metrics. If `None`, none will be evaluated.
        """
        CLMetricsCallback.__init__(
            self,
            save_dir=save_dir,
            test_acc_csv_name=test_acc_csv_name,
            test_loss_cls_csv_name=test_loss_cls_csv_name,
            test_acc_matrix_plot_name=test_acc_matrix_plot_name,
            test_loss_cls_matrix_plot_name=test_loss_cls_matrix_plot_name,
            test_ave_acc_plot_name=None,
            test_ave_loss_cls_plot_name=None,
        )
        if test_ave_acc_excluding_unlearned_plot_name:
            self.test_ave_acc_excluding_unlearned_plot_path: str = os.path.join(
                save_dir, test_ave_acc_excluding_unlearned_plot_name
            )
            r"""Store the path to save test average accuracy (excluding the unlearned tasks) curve plot."""
        if test_ave_loss_cls_excluding_unlearned_plot_name:
            self.test_ave_loss_cls_excluding_unlearned_plot_path: str = os.path.join(
                save_dir, test_ave_loss_cls_excluding_unlearned_plot_name
            )
            r"""Store the path to save test average classification loss (excluding the unlearned tasks) curve plot."""
        self.unlearning_test_after_task_ids: list[int] = (
            unlearning_test_after_task_ids
            if unlearning_test_after_task_ids is not None
            else []
        )
        r"""Store the task IDs that after which need to evaluate unlearning metrics."""

        self.cfg_unlearning_test_reference: DictConfig
        r"""Store the configuration of the reference experiment excluding unlearning tasks for unlearning test."""

    def on_test_epoch_end(
        self,
        trainer: Trainer,
        pl_module: CLAlgorithm,
    ) -> None:
        r"""Save and plot test metrics at the end of test."""

        # save (update) the test metrics to CSV files
        save.update_test_acc_to_csv(
            test_acc_metric=self.acc_test,
            csv_path=self.test_acc_csv_path,
            skipped_task_ids_for_ave=pl_module.unlearned_task_ids,  # skip the unlearned tasks to calculate average accuracy
        )
        save.update_test_loss_cls_to_csv(
            test_loss_cls_metric=self.loss_cls_test,
            csv_path=self.test_loss_cls_csv_path,
            skipped_task_ids_for_ave=pl_module.unlearned_task_ids,  # skip the unlearned tasks to calculate average classification loss
        )

        # plot the test metrics
        if self.test_acc_matrix_plot_path:
            plot.plot_test_acc_matrix_from_csv(
                csv_path=self.test_acc_csv_path,
                plot_path=self.test_acc_matrix_plot_path,
            )
        if self.test_loss_cls_matrix_plot_path:
            plot.plot_test_loss_cls_matrix_from_csv(
                csv_path=self.test_loss_cls_csv_path,
                plot_path=self.test_loss_cls_matrix_plot_path,
            )
        if self.test_ave_acc_excluding_unlearned_plot_path:
            plot.plot_test_ave_acc_curve_from_csv(
                csv_path=self.test_acc_csv_path,
                plot_path=self.test_ave_acc_excluding_unlearned_plot_path,
            )
        if self.test_ave_loss_cls_excluding_unlearned_plot_path:
            plot.plot_test_ave_loss_cls_curve_from_csv(
                csv_path=self.test_loss_cls_csv_path,
                plot_path=self.test_ave_loss_cls_excluding_unlearned_plot_path,
            )

        if self.task_id in self.unlearning_test_after_task_ids:

            not_unlearned_task_ids = [
                t
                for t in range(1, self.task_id + 1)
                if t not in pl_module.unlearned_task_ids
            ]

            print("NNNNNNNN", not_unlearned_task_ids)

            queue = multiprocessing.Queue()
            child_process = multiprocessing.Process(
                target=run_unlearning_test_reference_experiment,
                args=(
                    pl_module.cfg_unlearning_test_reference,
                    not_unlearned_task_ids,
                    queue,
                ),
                daemon=False,
            )
            child_process.start()
            child_process.join()

            expr_unlearning_test_reference = queue.get()

            model_unlearning_test_reference = expr_unlearning_test_reference.model

            self.unlearning_metrics_test(
                pl_module,
                model_unlearning_test_reference,
                trainer.datamodule,
                pl_module.unlearned_task_ids,
            )

    def unlearning_metrics_test(
        self,
        model: CLAlgorithm,
        model_unlearning_test_reference: CLAlgorithm,
        datamodule: CLDataset,
        unlearned_task_ids: list[int],
    ) -> None:
        r"""Evaluate the unlearning metrics for the current model and the reference model."""

        pylogger.info("Evaluating unlearning metrics...")

        # initialise the unlearning metrics
        distribution_distance: dict[str, MeanMetricBatch] = {}
        JSD = MeanMetric()

        for unlearned_task_id in unlearned_task_ids:
            distribution_distance[unlearned_task_id] = MeanMetricBatch()

            test_dataloader = datamodule.test_dataloader()[f"{unlearned_task_id}"]

            model.to("cpu")
            model.eval()
            model_unlearning_test_reference.eval()

            for batch in test_dataloader:

                x, _ = batch

                batch_size = len(batch)

                with torch.no_grad():
                    logits, _ = model(x, stage="test", task_id=unlearned_task_id)
                    logits_unlearning_test_reference, _ = (
                        model_unlearning_test_reference(
                            x, stage="test", task_id=unlearned_task_id
                        )
                    )
                    js = js_div(
                        logits,
                        logits_unlearning_test_reference,
                    )

                print("js", js)

                # calculate the distribution distance
                distribution_distance[unlearned_task_id].update(
                    js,
                    batch_size,
                )

            JSD(distribution_distance[unlearned_task_id])

        # log the unlearning metrics

        print("JSDDDDDD", JSD.compute())


def run_unlearning_test_reference_experiment(
    cfg_unlearning_test_reference,
    not_unlearned_task_ids,
    queue,
) -> None:
    r"""Run the reference experiment for unlearning test.
    Must be defined outside
    """

    expr_unlearning_test_reference = clrun_from_cfg(
        cfg_unlearning_test_reference, task_ids=not_unlearned_task_ids
    )
    queue.put(expr_unlearning_test_reference)
