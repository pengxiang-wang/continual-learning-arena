r"""
The submodule in `callbacks` for `JLMetricsCallback`.
"""

__all__ = ["JLMetricsCallback"]

import logging
import os
from typing import Any

from lightning import Callback, Trainer

from clarena.cl_algorithms import JointLearning
from clarena.utils import MeanMetricBatch, plot, save

# always get logger for built-in logging in each module
pylogger = logging.getLogger(__name__)


class JLMetricsCallback(Callback):
    r"""Provides all actions that are related to JL metrics, which include:

    - Defining, initialising and recording metrics.
    - Logging training and validation metrics to Lightning loggers in real time.
    - Saving test metrics to files.
    - Visualising test metrics as plots.

    Lightning provides `self.log()` to log metrics in `LightningModule` where our `JointLearning` based.

    The callback is able to produce the following outputs:

    - CSV files for test accuracy and classification loss of all tasks, average accuracy and classification loss.
    - Bar charts for test accuracy and classification loss of all tasks.
    """

    def __init__(
        self,
        save_dir: str,
        test_acc_csv_name: str,
        test_loss_cls_csv_name: str,
        test_acc_plot_name: str | None = None,
        test_loss_cls_plot_name: str | None = None,
    ) -> None:
        r"""Initialise the `JLMetricsCallback`.

        **Args:**
        - **save_dir** (`str`): the directory to save the test metrics files and plots. Better inside the output folder.
        - **test_acc_csv_name** (`str`): file name to save test accuracy of all tasks and average accuracy as CSV file.
        - **test_loss_cls_csv_name**(`str`): file name to save classification loss of all tasks and average classification loss as CSV file.
        - **test_acc_plot_name** (`str` | `None`): file name to save accuracy plot. If `None`, no file will be saved.
        - **test_loss_cls_plot_name** (`str` | `None`): file name to save classification loss plot. If `None`, no file will be saved.
        """
        Callback.__init__(self)

        os.makedirs(save_dir, exist_ok=True)

        # paths
        self.save_dir: str = save_dir
        r"""Store the directory to save the test metrics files and plots."""
        self.test_acc_csv_path: str = os.path.join(save_dir, test_acc_csv_name)
        r"""Store the path to save test accuracy of all tasks and average accuracy CSV file."""
        self.test_loss_cls_csv_path: str = os.path.join(
            save_dir, test_loss_cls_csv_name
        )
        r"""Store the path to save test classification loss of all tasks and average classification loss CSV file."""
        if test_loss_cls_plot_name:
            self.test_acc_plot_path: str = os.path.join(save_dir, test_acc_plot_name)
            r"""Store the path to save test accuracy plot."""
        if test_loss_cls_plot_name:
            self.test_loss_cls_plot_path: str = os.path.join(
                save_dir, test_loss_cls_plot_name
            )
            r"""Store the path to save test classification loss plot."""

        # training accumulated metrics
        self.acc_training_epoch: MeanMetricBatch
        r"""Classification accuracy of training epoch. Accumulated and calculated from the training batches. """
        self.loss_cls_training_epoch: MeanMetricBatch
        r"""Classification loss of training epoch. Accumulated and calculated from the training batches. """

        # validation accumulated metrics
        self.acc_val: MeanMetricBatch
        r"""Validation classification accuracy of the model after training epoch. Accumulated and calculated from the validation batches. """
        self.loss_cls_val: MeanMetricBatch
        r"""Validation classification of the model loss after training epoch. Accumulated and calculated from the validation batches. """

        # test accumulated metrics
        self.acc_test: dict[str, MeanMetricBatch] = {}
        r"""Test classification accuracy of all tasks. Accumulated and calculated from the test batches. Keys are task IDs (string type) and values are the corresponding metrics. """
        self.loss_cls_test: dict[str, MeanMetricBatch]
        r"""Test classification loss of all tasks. Accumulated and calculated from the test batches. Keys are task IDs (string type) and values are the corresponding metrics. """

    def on_fit_start(self, trainer: Trainer, pl_module: JointLearning) -> None:
        r"""Initialise training and validation metrics."""

        # initialise training metrics
        self.loss_cls_training_epoch = MeanMetricBatch()
        self.acc_training_epoch = MeanMetricBatch()

        # initialise validation metrics
        self.loss_cls_val = MeanMetricBatch()
        self.acc_val = MeanMetricBatch()

    def on_train_batch_end(
        self,
        trainer: Trainer,
        pl_module: JointLearning,
        outputs: dict[str, Any],
        batch: Any,
        batch_idx: int,
    ) -> None:
        r"""Record training metrics from training batch, log metrics of training batch and accumulated metrics of the epoch to Lightning loggers.

        **Args:**
        - **outputs** (`dict[str, Any]`): the outputs of the training step, the returns of the `training_step()` method in the `JointLearning`.
        - **batch** (`Any`): the training data batch.
        """
        # get the batch size
        batch_size = len(batch)

        # get training metrics values of current training batch from the outputs of the `training_step()`
        loss_cls_batch = outputs["loss_cls"]
        acc_batch = outputs["acc"]

        # update accumulated training metrics to calculate training metrics of the epoch
        self.loss_cls_training_epoch.update(loss_cls_batch, batch_size)
        self.acc_training_epoch.update(acc_batch, batch_size)

        # log training metrics of current training batch to Lightning loggers
        pl_module.log("train/loss_cls_batch", loss_cls_batch, prog_bar=True)
        pl_module.log("train/acc_batch", acc_batch, prog_bar=True)

        # log accumulated training metrics till this training batch to Lightning loggers
        pl_module.log(
            "task/train/loss_cls",
            self.loss_cls_training_epoch.compute(),
            prog_bar=True,
        )
        pl_module.log(
            "task/train/acc",
            self.acc_training_epoch.compute(),
            prog_bar=True,
        )

    def on_train_epoch_end(
        self,
        trainer: Trainer,
        pl_module: JointLearning,
    ) -> None:
        r"""Log metrics of training epoch to plot learning curves and reset the metrics accumulation at the end of training epoch."""

        # log the accumulated and computed metrics of the epoch to Lightning loggers, specially for plotting learning curves
        pl_module.log(
            "learning_curve/train/loss_cls",
            self.loss_cls_training_epoch.compute(),
            on_epoch=True,
            prog_bar=True,
        )
        pl_module.log(
            "learning_curve/train/acc",
            self.acc_training_epoch.compute(),
            on_epoch=True,
            prog_bar=True,
        )

        # reset the metrics of training epoch as there are more epochs to go and not only one epoch like in the validation and test
        self.loss_cls_training_epoch.reset()
        self.acc_training_epoch.reset()

    def on_validation_batch_end(
        self,
        trainer: Trainer,
        pl_module: JointLearning,
        outputs: dict[str, Any],
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        r"""Accumulating metrics from validation batch. We don't need to log and monitor the metrics of validation batches.

        **Args:**
        - **outputs** (`dict[str, Any]`): the outputs of the validation step, which is the returns of the `validation_step()` method in the `JointLearning`.
        - **batch** (`Any`): the validation data batch.
        - **dataloader_idx** (`int`): the task ID of the validation dataloader. A default value of 0 is given otherwise the LightningModule will raise a `RuntimeError`.
        """

        # get the batch size
        batch_size = len(batch)

        # get the metrics values of the batch from the outputs
        loss_cls_batch = outputs["loss_cls"]
        acc_batch = outputs["acc"]

        # update the accumulated metrics in order to calculate the validation metrics
        self.loss_cls_val.update(loss_cls_batch, batch_size)
        self.acc_val.update(acc_batch, batch_size)

    def on_validation_epoch_end(
        self,
        trainer: Trainer,
        pl_module: JointLearning,
    ) -> None:
        r"""Log validation metrics to plot learning curves."""

        # log the accumulated and computed metrics of the epoch to Lightning loggers, specially for plotting learning curves
        pl_module.log(
            "learning_curve/val/loss_cls",
            self.loss_cls_val.compute(),
            on_epoch=True,
            prog_bar=True,
        )
        pl_module.log(
            "learning_curve/val/acc",
            self.acc_val.compute(),
            on_epoch=True,
            prog_bar=True,
        )

    def on_test_start(
        self,
        trainer: Trainer,
        pl_module: JointLearning,
    ) -> None:
        r"""Initialise the metrics for testing each seen task in the beginning of a task's testing."""

        # initialise test metrics for current and previous tasks
        self.loss_cls_test = {
            f"{task_id}": MeanMetricBatch()
            for task_id in range(1, pl_module.num_tasks + 1)
        }
        self.acc_test = {
            f"{task_id}": MeanMetricBatch()
            for task_id in range(1, pl_module.num_tasks + 1)
        }

    def on_test_batch_end(
        self,
        trainer: Trainer,
        pl_module: JointLearning,
        outputs: dict[str, Any],
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        r"""Accumulating metrics from test batch. We don't need to log and monitor the metrics of test batches.

        **Args:**
        - **outputs** (`dict[str, Any]`): the outputs of the test step, which is the returns of the `test_step()` method in the `JointLearning`.
        - **batch** (`Any`): the validation data batch.
        - **dataloader_idx** (`int`): the task ID of seen tasks to be tested. A default value of 0 is given otherwise the LightningModule will raise a `RuntimeError`.
        """

        # get the batch size
        batch_size = len(batch)

        test_task_id = pl_module.get_test_task_id_from_dataloader_idx(dataloader_idx)

        # get the metrics values of the batch from the outputs
        loss_cls_batch = outputs["loss_cls"]
        acc_batch = outputs["acc"]

        # update the accumulated metrics in order to calculate the metrics of the epoch
        self.acc_test[f"{test_task_id}"].update(acc_batch, batch_size)
        self.loss_cls_test[f"{test_task_id}"].update(loss_cls_batch, batch_size)

    def on_test_epoch_end(
        self,
        trainer: Trainer,
        pl_module: JointLearning,
    ) -> None:
        r"""Save and plot test metrics at the end of test."""

        # save (update) the test metrics to CSV files
        save.save_joint_test_acc_to_csv(
            test_acc_metric=self.acc_test,
            csv_path=self.test_acc_csv_path,
        )
        save.save_joint_test_loss_cls_to_csv(
            test_loss_cls_metric=self.loss_cls_test,
            csv_path=self.test_loss_cls_csv_path,
        )

        # plot the test metrics
        if self.test_acc_plot_path:
            plot.plot_joint_test_acc_from_csv(
                csv_path=self.test_acc_csv_path,
                plot_path=self.test_acc_plot_path,
            )
        # if self.test_loss_cls_plot_path:
        #     plot.plot_joint_test_loss_cls_from_csv(
        #         csv_path=self.test_loss_cls_csv_path,
        #         plot_path=self.test_loss_cls_plot_path,
        #     )
