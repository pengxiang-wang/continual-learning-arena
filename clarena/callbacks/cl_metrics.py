r"""
The submodule in `callbacks` for `CLMetricsCallback`.
"""

__all__ = ["CLMetricsCallback"]

import logging
import os
from typing import Any

from lightning import Callback, Trainer

from clarena.cl_algorithms import CLAlgorithm
from clarena.utils import MeanMetricBatch, plot, save

# always get logger for built-in logging in each module
pylogger = logging.getLogger(__name__)


class CLMetricsCallback(Callback):
    r"""Provides all actions that are related to CL metrics, which include:

    - Defining, initialising and recording metrics.
    - Logging training and validation metrics to Lightning loggers in real time.
    - Saving test metrics to files.
    - Visualising test metrics as plots.

    Please refer to the [A Summary of Continual Learning Metrics](https://pengxiang-wang.com/posts/continual-learning-metrics) to learn what continual learning metrics mean.

    Lightning provides `self.log()` to log metrics in `LightningModule` where our `CLAlgorithm` based. You can put `self.log()` here if you don't want to mess up the `CLAlgorithm` with a huge amount of logging codes.

    The callback is able to produce the following outputs:

    - CSV files for test accuracy and classification loss (lower triangular) matrix, average accuracy and classification loss. See [here](https://pengxiang-wang.com/posts/continual-learning-metrics.html#sec-test-performance-of-previous-tasks) for details.
    - Coloured plot for test accuracy and classification loss (lower triangular) matrix. See [here](https://pengxiang-wang.com/posts/continual-learning-metrics.html#sec-test-performance-of-previous-tasks) for details.
    - Curve plots for test average accuracy and classification loss over different training tasks. See [here](https://pengxiang-wang.com/posts/continual-learning-metrics.html#sec-average-test-performance-over-tasks) for details.


    """

    def __init__(
        self,
        save_dir: str,
        test_acc_csv_name: str | None = None,
        test_loss_cls_csv_name: str | None = None,
        test_acc_matrix_plot_name: str | None = None,
        test_loss_cls_matrix_plot_name: str | None = None,
        test_ave_acc_plot_name: str | None = None,
        test_ave_loss_cls_plot_name: str | None = None,
    ) -> None:
        r"""Initialise the `CLMetricsCallback`.

        **Args:**
        - **save_dir** (`str` | `None`): the directory to save the test metrics files and plots. Better inside the output folder.
        - **test_acc_csv_name** (`str` | `None`): file name to save test accuracy matrix and average accuracy as CSV file. If `None`, no file will be saved.
        - **test_loss_cls_csv_name**(`str` | `None`): file name to save classification loss matrix and average classification loss as CSV file. If `None`, no file will be saved.
        - **test_acc_matrix_plot_name** (`str` | `None`): file name to save accuracy matrix plot. If `None`, no file will be saved.
        - **test_loss_cls_matrix_plot_name** (`str` | `None`): file name to save classification loss matrix plot. If `None`, no file will be saved.
        - **test_ave_acc_plot_name** (`str` | `None`): file name to save average accuracy as curve plot over different training tasks. If `None`, no file will be saved.
        - **test_ave_loss_cls_plot_name** (`str` | `None`): file name to save average classification loss as curve plot over different training tasks. If `None`, no file will be saved.
        """
        Callback.__init__(self)

        os.makedirs(save_dir, exist_ok=True)

        # paths
        self.test_acc_csv_path: str = os.path.join(save_dir, test_acc_csv_name)
        r"""Store the path to save test accuracy matrix and average accuracy CSV file."""
        self.test_loss_cls_csv_path: str = os.path.join(
            save_dir, test_loss_cls_csv_name
        )
        r"""Store the path to save test classification loss and average accuracy CSV file."""
        self.test_acc_matrix_plot_path: str = os.path.join(
            save_dir, test_acc_matrix_plot_name
        )
        r"""Store the path to save test accuracy matrix plot."""
        self.test_loss_cls_matrix_plot_path: str = os.path.join(
            save_dir, test_loss_cls_matrix_plot_name
        )
        r"""Store the path to save test classification loss matrix plot."""
        self.test_ave_acc_plot_path: str = os.path.join(
            save_dir, test_ave_acc_plot_name
        )
        r"""Store the path to save test average accuracy curve plot."""
        self.test_ave_loss_cls_plot_path: str = os.path.join(
            save_dir, test_ave_loss_cls_plot_name
        )
        r"""Store the path to save test average classification loss curve plot."""

        # training accumulated metrics
        self.acc_training_epoch: MeanMetricBatch
        r"""Classification accuracy of training epoch. Accumulated and calculated from the training batches. See [here](https://pengxiang-wang.com/posts/continual-learning-metrics.html#sec-performance-of-training-epoch) for details. """
        self.loss_cls_training_epoch: MeanMetricBatch
        r"""Classification loss of training epoch. Accumulated and calculated from the training batches. See [here](https://pengxiang-wang.com/posts/continual-learning-metrics.html#sec-performance-of-training-epoch) for details. """
        self.loss_training_epoch: MeanMetricBatch
        r"""Total loss of training epoch. Accumulated and calculated from the training batches. See [here](https://pengxiang-wang.com/posts/continual-learning-metrics.html#sec-performance-of-training-epoch) for details. """

        # validation accumulated metrics
        self.acc_val: MeanMetricBatch
        r"""Validation classification accuracy of the model after training epoch. Accumulated and calculated from the validation batches. See [here](https://pengxiang-wang.com/posts/continual-learning-metrics.html#sec-validation-performace) for details. """
        self.loss_cls_val: MeanMetricBatch
        r"""Validation classification of the model loss after training epoch. Accumulated and calculated from the validation batches. See [here](https://pengxiang-wang.com/posts/continual-learning-metrics.html#sec-validation-performace) for details. """

        # test accumulated metrics
        self.acc_test: dict[str, MeanMetricBatch]
        r"""Test classification accuracy of the current model (`self.task_id`) on current and previous tasks. Accumulated and calculated from the test batches. Keys are task IDs (string type) and values are the corresponding metrics. It is the last row of the lower triangular matrix. See [here](https://pengxiang-wang.com/posts/continual-learning-metrics.html#sec-test-performance-of-previous-tasks) for details. """
        self.loss_cls_test: dict[str, MeanMetricBatch]
        r"""Test classification loss of the current model (`self.task_id`) on current and previous tasks. Accumulated and calculated from the test batches. Keys are task IDs (string type) and values are the corresponding metrics. It is the last row of the lower triangular matrix. See [here](https://pengxiang-wang.com/posts/continual-learning-metrics.html#sec-test-performance-of-previous-tasks) for details. """

        self.task_id: int
        r"""Task ID counter indicating which task is being processed. Self updated during the task loop."""

    def on_fit_start(self, trainer: Trainer, pl_module: CLAlgorithm) -> None:
        r"""Initialise training and validation metrics."""

        # set the current task_id from the `CLAlgorithm` object
        self.task_id = pl_module.task_id

        # initialise training metrics
        self.loss_cls_training_epoch = MeanMetricBatch()
        self.loss_training_epoch = MeanMetricBatch()
        self.acc_training_epoch = MeanMetricBatch()

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
        acc_batch = outputs["acc"]

        # update accumulated training metrics to calculate training metrics of the epoch
        self.loss_cls_training_epoch.update(loss_cls_batch, batch_size)
        self.loss_training_epoch.update(loss_cls_batch, batch_size)
        self.acc_training_epoch.update(acc_batch, batch_size)

        # log training metrics of current training batch to Lightning loggers
        pl_module.log(
            f"task_{self.task_id}/train/loss_cls_batch", loss_cls_batch, prog_bar=True
        )
        pl_module.log(
            f"task_{self.task_id}/train/loss_batch", loss_batch, prog_bar=True
        )
        pl_module.log(f"task_{self.task_id}/train/acc_batch", acc_batch, prog_bar=True)

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
        pl_module.log(
            f"task_{self.task_id}/train/acc",
            self.acc_training_epoch.compute(),
            prog_bar=True,
        )

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
        pl_module.log(
            f"task_{self.task_id}/learning_curve/train/acc",
            self.acc_training_epoch.compute(),
            on_epoch=True,
            prog_bar=True,
        )

        # reset the metrics of training epoch as there are more epochs to go and not only one epoch like in the validation and test
        self.loss_cls_training_epoch.reset()
        self.loss_training_epoch.reset()
        self.acc_training_epoch.reset()

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
        acc_batch = outputs["acc"]

        # update the accumulated metrics in order to calculate the validation metrics
        self.loss_cls_val.update(loss_cls_batch, batch_size)
        self.acc_val.update(acc_batch, batch_size)

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
    ) -> None:
        r"""Initialise the metrics for testing each seen task in the beginning of a task's testing."""

        # set the current task_id again (double checking) from the `CLAlgorithm` object
        self.task_id = pl_module.task_id

        # initialise test metrics for current and previous tasks
        self.loss_cls_test = {
            f"{task_id}": MeanMetricBatch() for task_id in range(1, self.task_id + 1)
        }
        self.acc_test = {
            f"{task_id}": MeanMetricBatch() for task_id in range(1, self.task_id + 1)
        }

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
        - **batch** (`Any`): the validation data batch.
        - **dataloader_idx** (`int`): the task ID of seen tasks to be tested. A default value of 0 is given otherwise the LightningModule will raise a `RuntimeError`.
        """

        # get the batch size
        batch_size = len(batch)

        task_id = dataloader_idx + 1  # our `task_id` system starts from 1

        # get the metrics values of the batch from the outputs
        loss_cls_batch = outputs["loss_cls"]
        acc_batch = outputs["acc"]

        # update the accumulated metrics in order to calculate the metrics of the epoch
        self.loss_cls_test[f"{task_id}"].update(loss_cls_batch, batch_size)
        self.acc_test[f"{task_id}"].update(acc_batch, batch_size)

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
        )
        save.update_test_loss_cls_to_csv(
            test_loss_cls_metric=self.loss_cls_test,
            csv_path=self.test_loss_cls_csv_path,
        )

        # plot the test metrics
        plot.plot_test_acc_matrix_from_csv(
            csv_path=self.test_acc_csv_path,
            task_id=self.task_id,
            plot_path=self.test_acc_matrix_plot_path,
        )
        plot.plot_test_loss_cls_matrix_from_csv(
            csv_path=self.test_loss_cls_csv_path,
            task_id=self.task_id,
            plot_path=self.test_loss_cls_matrix_plot_path,
        )
        plot.plot_test_ave_acc_curve_from_csv(
            csv_path=self.test_acc_csv_path,
            task_id=self.task_id,
            plot_path=self.test_ave_acc_plot_path,
        )
        plot.plot_test_ave_loss_cls_curve_from_csv(
            csv_path=self.test_loss_cls_csv_path,
            task_id=self.task_id,
            plot_path=self.test_ave_loss_cls_plot_path,
        )
