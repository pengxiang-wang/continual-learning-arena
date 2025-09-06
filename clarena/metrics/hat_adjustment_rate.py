r"""
The submodule in `metrics` for `HATAdjustmentRate`.
"""

__all__ = ["HATAdjustmentRate"]

import logging
import os
from typing import Any

from lightning import Trainer
from lightning.pytorch.utilities import rank_zero_only
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
from torch import Tensor

from clarena.cl_algorithms import HAT
from clarena.metrics import MetricCallback

# always get logger for built-in logging in each module
pylogger = logging.getLogger(__name__)


class HATAdjustmentRate(MetricCallback):
    r"""Provides all actions that are related to adjustment rate of [HAT (Hard Attention to the Task)](http://proceedings.mlr.press/v80/serra18a) algorithm and its extensions, which include:

    - Visualizing adjustment rate during training as figures.

    The callback is able to produce the following outputs:

    - Figures of training adjustment rate.

    """

    def __init__(
        self,
        save_dir: str,
        plot_adjustment_rate_every_n_steps: int | None = None,
    ) -> None:
        r"""
        **Args:**
        - **save_dir** (`str` | `None`): The directory to save the adjustment rate figures. Better inside the output folder.
        - **plot_adjustment_rate_every_n_steps** (`int` | `None`): the frequency of plotting adjustment rate figures in terms of number of batches during training.
        """
        super().__init__(save_dir=save_dir)

        # other settings
        self.plot_adjustment_rate_every_n_steps: int = (
            plot_adjustment_rate_every_n_steps
        )
        r"""The frequency of plotting adjustment rate in terms of number of batches."""

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
            raise TypeError(
                "The `CLAlgorithm` should be `HAT` to apply `HATAdjustmentRate`!"
            )

    @rank_zero_only
    def on_train_batch_end(
        self,
        trainer: Trainer,
        pl_module: HAT,
        outputs: dict[str, Any],
        batch: Any,
        batch_idx: int,
    ) -> None:
        r"""Plot adjustment rate after training batch.

        **Args:**
        - **outputs** (`dict[str, Any]`): the outputs of the training step, which is the returns of the `training_step()` method in the `HAT`.
        - **batch** (`Any`): the training data batch.
        - **batch_idx** (`int`): the index of the current batch. This is for the file name of mask figures.
        """

        # get the adjustment rate
        adjustment_rate_weight = outputs["adjustment_rate_weight"]
        adjustment_rate_bias = outputs["adjustment_rate_bias"]

        # plot the adjustment rate
        if self.save_dir is not None:
            if self.task_id > 1:
                if batch_idx % self.plot_adjustment_rate_every_n_steps == 0:
                    self.plot_hat_adjustment_rate(
                        adjustment_rate=adjustment_rate_weight,
                        weight_or_bias="weight",
                        step=batch_idx,
                    )
                    self.plot_hat_adjustment_rate(
                        adjustment_rate=adjustment_rate_bias,
                        weight_or_bias="bias",
                        step=batch_idx,
                    )

    def plot_hat_adjustment_rate(
        self,
        adjustment_rate: dict[str, Tensor],
        weight_or_bias: str,
        step: int | None = None,
    ) -> None:
        """Plot adjustment rate in [HAT (Hard Attention to the Task)](http://proceedings.mlr.press/v80/serra18a)) algorithm. This includes the adjustment rate weight and adjustment rate bias (if applicable).

        **Args:**
        - **adjustment_rate** (`dict[str, Tensor]`): the adjustment rate. Keys (`str`) are layer names and values (`Tensor`) are the adjustment rate tensors. If it's adjustment rate weight, it has size same as weights. If it's adjustment rate bias, it has size same as biases.
        - **weight_or_bias** (`str`): the type of adjustment rate. It can be either 'weight' or 'bias'. This is to form the plot name.
        - **step** (`int`): the training step (batch index) of the adjustment rate to be plotted. This is to form the plot name. Keep `None` for not showing the step in the plot name.
        """

        for layer_name, a in adjustment_rate.items():
            layer_name = layer_name.replace(
                "/", "."
            )  # the layer name contains '/', which is not allowed in the file name. We replace it back with '.'.

            if weight_or_bias == "bias":
                a = a.view(
                    1, -1
                )  # reshape the 1D mask to 2D so can be plotted by image show

            fig = plt.figure()
            plt.imshow(
                a.detach().cpu(),
                aspect="auto",
                cmap="Wistia",
                norm=LogNorm(vmin=1e-7, vmax=1e-5),
            )  # can only convert to tensors in CPU to numpy arrays
            plt.yticks()  # hide yticks
            plt.colorbar()
            if step:
                plot_name = (
                    f"{layer_name}_{weight_or_bias}_task{self.task_id}_step{step}.png"
                )
            else:
                plot_name = f"{layer_name}_{weight_or_bias}_task{self.task_id}.png"
            plot_path = os.path.join(self.save_dir, plot_name)
            fig.savefig(plot_path)
            plt.close(fig)
