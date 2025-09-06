r"""
The submodule in `metrics` for `HATMasks`.
"""

__all__ = ["HATMasks"]

import logging
import os
from typing import Any

from lightning import Trainer
from lightning.pytorch.utilities import rank_zero_only
from matplotlib import pyplot as plt
from torch import Tensor

from clarena.cl_algorithms import HAT
from clarena.metrics import MetricCallback

# always get logger for built-in logging in each module
pylogger = logging.getLogger(__name__)


class HATMasks(MetricCallback):
    r"""Provides all actions that are related to masks of [HAT (Hard Attention to the Task)](http://proceedings.mlr.press/v80/serra18a) algorithm and its extensions, which include:

    - Visualizing mask and cumulative mask figures during training and testing as figures.

    The callback is able to produce the following outputs:

    - Figures of both training and test, masks and cumulative masks.

    """

    def __init__(
        self,
        save_dir: str,
        test_masks_dir_name: str | None = None,
        test_cumulative_masks_dir_name: str | None = None,
        training_masks_dir_name: str | None = None,
        plot_training_mask_every_n_steps: int | None = None,
    ) -> None:
        r"""
        **Args:**
        - **save_dir** (`str`): the directory to save the mask figures. Better inside the output folder.
        - **test_masks_dir_name** (`str` | `None`): the relative path to `save_dir` to save the test mask figures. If `None`, no file will be saved.
        - **test_cumulative_masks_dir_name** (`str` | `None`): the directory to save the test cumulative mask figures. If `None`, no file will be saved.
        - **training_masks_dir_name** (`str` | `None`): the directory to save the training mask figures. If `None`, no file will be saved.
        - **plot_training_mask_every_n_steps** (`int` | `None`): the frequency of plotting training mask figures in terms of number of batches during training. Only applies when `training_masks_dir_name` is not `None`.
        """
        super().__init__(save_dir=save_dir)

        # paths
        if test_masks_dir_name is not None:
            self.test_masks_dir: str = os.path.join(self.save_dir, test_masks_dir_name)
            r"""The directory to save the test mask figures."""
        if test_cumulative_masks_dir_name is not None:
            self.test_cumulative_masks_dir: str = os.path.join(
                self.save_dir, test_cumulative_masks_dir_name
            )
            r"""The directory to save the test cumulative mask figures."""
        if training_masks_dir_name is not None:
            self.training_masks_dir: str = os.path.join(
                self.save_dir, training_masks_dir_name
            )
            r"""The directory to save the training mask figures."""

        # other settings
        self.plot_training_mask_every_n_steps: int = plot_training_mask_every_n_steps
        r"""The frequency of plotting training masks in terms of number of batches."""

        # task ID control
        self.task_id: int
        r"""Task ID counter indicating which task is being processed. Self updated during the task loop. Valid from 1 to `cl_dataset.num_tasks`."""

    def on_fit_start(self, trainer: Trainer, pl_module: HAT) -> None:
        r"""Get the current task ID in the beginning of a task's fitting (training and validation). Sanity check the `pl_module` to be `HAT`.

        **Raises:**
        -**TypeError**: when the `pl_module` is not `HAT`.
        """

        # get the current task_id from the `CLAlgorithm` object
        self.task_id = pl_module.task_id

        # sanity check
        if not isinstance(pl_module, HAT):
            raise TypeError("The `CLAlgorithm` should be `HAT` to apply `HATMasks`!")

    @rank_zero_only
    def on_train_batch_end(
        self,
        trainer: Trainer,
        pl_module: HAT,
        outputs: dict[str, Any],
        batch: Any,
        batch_idx: int,
    ) -> None:
        r"""Plot training mask after training batch.

        **Args:**
        - **outputs** (`dict[str, Any]`): the outputs of the training step, which is the returns of the `training_step()` method in the `HAT`.
        - **batch** (`Any`): the training data batch.
        - **batch_idx** (`int`): the index of the current batch. This is for the file name of mask figures.
        """

        # get the mask over the model after training the batch
        mask = outputs["mask"]

        # plot the mask
        if hasattr(self, "training_masks_dir"):
            if batch_idx % self.plot_training_mask_every_n_steps == 0:
                self.plot_hat_mask(
                    mask=mask,
                    plot_dir=self.training_masks_dir,
                    task_id=self.task_id,
                    step=batch_idx,
                )

    @rank_zero_only
    def on_test_start(self, trainer: Trainer, pl_module: HAT) -> None:
        r"""Plot test mask and cumulative mask figures."""

        # test mask
        if hasattr(self, "test_masks_dir"):
            mask = pl_module.masks[self.task_id]
            self.plot_hat_mask(
                mask=mask, plot_dir=self.test_masks_dir, task_id=self.task_id
            )

        # cumulative mask
        if hasattr(self, "test_cumulative_masks_dir"):
            cumulative_mask = pl_module.cumulative_mask_for_previous_tasks
            self.plot_hat_mask(
                mask=cumulative_mask,
                plot_dir=self.test_cumulative_masks_dir,
                task_id=self.task_id,
            )

    def plot_hat_mask(
        self,
        mask: dict[str, Tensor],
        plot_dir: str,
        task_id: int,
        step: int | None = None,
    ) -> None:
        """Plot mask in [HAT (Hard Attention to the Task)](http://proceedings.mlr.press/v80/serra18a)) algorithm. This includes the mask and cumulative mask.

        **Args:**
        - **mask** (`dict[str, Tensor]`): the hard attention (whose values are 0 or 1) mask. Keys (`str`) are layer name and values (`Tensor`) are the mask tensors. The mask tensor has size (number of units, ).
        - **plot_dir** (`str`): the directory to save plot. Better same as the output directory of the experiment.
        - **task_id** (`int`): the task ID of the mask to be plotted. This is to form the plot name.
        - **step** (`int`): the training step (batch index) of the mask to be plotted. Apply to the training mask only. This is to form the plot name. Keep `None` for not showing the step in the plot name.
        """

        for layer_name, m in mask.items():
            layer_name = layer_name.replace(
                "/", "."
            )  # the layer name contains '/', which is not allowed in the file name. We replace it back with '.'.

            m = m.view(
                1, -1
            )  # reshape the 1D mask to 2D so can be plotted by image show

            fig = plt.figure()
            plt.imshow(
                m.detach().cpu(), aspect="auto", cmap="Greys"
            )  # can only convert to tensors in CPU to numpy arrays
            plt.yticks()  # hide yticks
            plt.colorbar()
            if step:
                plot_name = f"{layer_name}_task{task_id}_step{step}.png"
            else:
                plot_name = f"{layer_name}_task{task_id}.png"
            plot_path = os.path.join(plot_dir, plot_name)
            fig.savefig(plot_path)
            plt.close(fig)
            plt.close(fig)
            plt.close(fig)
