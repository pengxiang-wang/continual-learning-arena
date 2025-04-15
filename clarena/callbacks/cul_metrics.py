r"""
The submodule in `callbacks` for `CULMetricsCallback`.
"""

__all__ = ["CULMetricsCallback"]

import logging
import multiprocessing
import os
from copy import deepcopy
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
        unlearning_test_distance_csv_name: str | None = None,
        unlearning_test_distance_plot_name: str | None = None,
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
        - **unlearning_test_distance_csv_name** (`str` | `None`): file name to save unlearning test distance as CSV file. If `None`, no file will be saved.
        - **unlearning_test_distance_plot_name** (`str` | `None`): file name to save unlearning test distance as matrix plot. If `None`, no file will be saved.
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

        # CL test now excludes the unlearned tasks
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

        # paths for unlearning test results
        if unlearning_test_distance_csv_name:
            self.unlearning_test_distance_csv_path: str = os.path.join(
                save_dir,
                unlearning_test_distance_csv_name,
            )
            r"""Store the path to save unlearning test distance as CSV file."""
        if unlearning_test_distance_plot_name:
            self.unlearning_test_distance_plot_path: str = os.path.join(
                save_dir,
                unlearning_test_distance_plot_name,
            )
            r"""Store the path to save unlearning test distance as matrix plot."""

        # for unlearning test
        self.cfg_unlearning_test_reference: DictConfig
        r"""Store the configuration of the reference experiment excluding unlearning tasks for unlearning test."""

        # unlearning metrics
        self.distribution_distance_unlearning_test: dict[str, MeanMetricBatch]
        r"""Distribution distance between the current model and the reference model on test data of different unlearned task IDs, of the current unlearning test requested in `self.unlearning_test_after_task_ids`. Keys are unlearned task IDs (string type) and values are the corresponding distribution distance."""

    def on_test_epoch_end(
        self,
        trainer: Trainer,
        pl_module: CLAlgorithm,
    ) -> None:
        r"""Save and plot test metrics at the end of test."""

        # save (update) the test metrics to CSV files
        save.update_test_acc_to_csv(
            after_training_task_id=self.task_id,
            test_acc_metric=self.acc_test,
            csv_path=self.test_acc_csv_path,
            skipped_task_ids_for_ave=pl_module.unlearned_task_ids,  # skip the unlearned tasks to calculate average accuracy
        )
        save.update_test_loss_cls_to_csv(
            after_training_task_id=self.task_id,
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
            # run test for unlearning metrics

            not_unlearned_task_ids = [
                t
                for t in range(1, self.task_id + 1)
                if t not in pl_module.unlearned_task_ids
            ]

            # create a new process to run the unlearning test reference experiment
            queue = multiprocessing.Queue()
            child_process = multiprocessing.Process(
                target=run_unlearning_test,
                args=(
                    self,
                    trainer,
                    deepcopy(pl_module).to("cpu"),
                    pl_module.cfg_unlearning_test_reference,
                    not_unlearned_task_ids,
                    queue,
                ),
                daemon=False,
            )
            child_process.start()
            child_process.join()
            # expr_unlearning_test_reference = queue.get()

            # run_unlearning_test(
            #     self,
            #     trainer,
            #     pl_module,
            #     pl_module.cfg_unlearning_test_reference,
            #     not_unlearned_task_ids,
            #     None,
            # )

    def unlearning_metrics_test(
        self,
        unlearning_test_after_task_id: int,
        model: CLAlgorithm,
        model_unlearning_test_reference: CLAlgorithm,
        datamodule: CLDataset,
        unlearned_task_ids: list[int],
    ) -> None:
        r"""Evaluate the unlearning metrics for the current model and the reference model.

        **Args:**
        - **unlearning_test_after_task_id** (`int`): the task ID after which the unlearning test is evaluated.
        - **model** (`CLAlgorithm`): the current model.
        - **model_unlearning_test_reference** (`CLAlgorithm`): the reference model.
        - **datamodule** (`CLDataset`): the datamodule. This is to get the test data.
        - **unlearned_task_ids** (`list[int]`): the task IDs that have been unlearned in the experiment.
        """
        pylogger.info("Evaluating unlearning metrics...")

        # initialise unlearning test metrics for unlearned tasks
        self.distribution_distance_unlearning_test = {
            f"{task_id}": MeanMetricBatch() for task_id in unlearned_task_ids
        }

        for unlearned_task_id in unlearned_task_ids:
            # test on the unlearned task

            test_dataloader = datamodule.test_dataloader()[
                f"{unlearned_task_id}"
            ]  # get the test data

            # set the model to evaluation mode
            model.to("cpu")
            model.eval()
            model_unlearning_test_reference.eval()

            for batch in test_dataloader:
                # unlearning test step
                x, _ = batch
                batch_size = len(batch)

                with torch.no_grad():

                    # get the aggregated backbone output (instead of logits)
                    aggregated_backbone_output = model.aggregated_backbone_output(x)
                    aggregated_backbone_output_unlearning_test_reference = (
                        model_unlearning_test_reference.aggregated_backbone_output(x)
                    )

                    # calculate the Jensen-Shannon divergence as distribution distance
                    js = js_div(
                        aggregated_backbone_output,
                        aggregated_backbone_output_unlearning_test_reference,
                    )

                print("js", js)

                # update the accumulated metrics in order to calculate the metrics of the epoch
                self.distribution_distance_unlearning_test[
                    f"{unlearned_task_id}"
                ].update(
                    js,
                    batch_size,
                )

        save.update_unlearning_test_distance_to_csv(
            unlearning_test_after_task_id=unlearning_test_after_task_id,
            distance_metric=self.distribution_distance_unlearning_test,
            csv_path=self.unlearning_test_distance_csv_path,
        )
        plot.plot_unlearning_test_distance_from_csv(
            csv_path=self.unlearning_test_distance_csv_path,
            plot_path=self.unlearning_test_distance_plot_path,
        )


def run_unlearning_test(
    cul_metrics_callback: CULMetricsCallback,
    trainer: Trainer,
    pl_module: CLAlgorithm,
    cfg_unlearning_test_reference: DictConfig,
    not_unlearned_task_ids: list[int],
    queue: multiprocessing.Queue,
) -> None:
    r"""The function to run the reference experiment for unlearning test.
    Must be defined outside the class.

    **Args:**
    - **cfg_unlearning_test_reference** (`DictConfig`): the configuration of the reference experiment for unlearning test.
    - **not_unlearned_task_ids** (`list[int]`): the task IDs that have not been unlearned in the experiment.
    - **queue** (`multiprocessing.Queue`): the queue to store the reference experiment.
    """
    expr_unlearning_test_reference = clrun_from_cfg(
        cfg_unlearning_test_reference, task_ids=not_unlearned_task_ids
    )

    # get the reference model
    model_unlearning_test_reference = expr_unlearning_test_reference.model

    # run unlearning test between current and reference model
    cul_metrics_callback.unlearning_metrics_test(
        unlearning_test_after_task_id=cul_metrics_callback.task_id,
        model=pl_module,
        model_unlearning_test_reference=model_unlearning_test_reference,
        datamodule=trainer.datamodule,
        unlearned_task_ids=pl_module.unlearned_task_ids,
    )

    # queue.put(expr_unlearning_test_reference)
