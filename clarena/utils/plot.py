"""The submodule in `utils` for plotting utils."""

__all__ = [
    "plot_test_ave_acc_curve_from_csv",
    "plot_test_acc_matrix_from_csv",
    "plot_test_ave_loss_cls_curve_from_csv",
    "plot_test_loss_cls_matrix_from_csv",
    "plot_hat_mask",
]


import os

import pandas as pd
from matplotlib import pyplot as plt
from torch import Tensor


def plot_test_ave_acc_curve_from_csv(
    csv_path: str, task_id: int, plot_path: str
) -> None:
    """Plot the test average accuracy curve over different training tasks from saved csv file and save the plot to the designated directory.

    **Args:**
    - **csv_path** (`str`): the path to the csv file where the `utils.update_test_acc_to_csv()` saved the test accuracy metric.
    - **task_id** (`int`): plot the test average accuracy metric from task 1 to `task_id`.
    - **plot_path** (`str`): the path to save plot. Better same as the output directory of the experiment. E.g. './outputs/expr_name/1970-01-01_00-00-00/ave_acc.png'.
    """
    data = pd.read_csv(csv_path)

    # plot the average accuracy curve over different training tasks
    fig, ax = plt.subplots(figsize=(16, 9))
    ax.plot(
        data["after_training_task"],
        data["average_accuracy"],
        marker="o",
        linewidth=2,
    )
    ax.set_xlabel("After training task $t$", fontsize=16)
    ax.set_xlabel("Average Accuracy (AA)", fontsize=16)
    ax.grid(True)
    xticks = [int(i) for i in range(1, task_id + 1)]
    yticks = [i * 0.05 for i in range(21)]
    ax.set_xticks(xticks)
    ax.set_yticks(yticks)
    ax.set_xticklabels(xticks, fontsize=16)
    ax.set_yticklabels([f"{tick:.2f}" for tick in yticks], fontsize=16)
    fig.savefig(plot_path)
    plt.close(fig)


def plot_test_acc_matrix_from_csv(csv_path: str, task_id: int, plot_path: str) -> None:
    """Plot the test accuracy matrix from saved csv file and save the plot to the designated directory.

    **Args:**
    - **csv_path** (`str`): the path to the csv file where the `utils.update_test_acc_to_csv()` saved the test accuracy metric.
    - **task_id** (`int`): plot the test accuracy metric from task 1 to `task_id`.
    - **plot_path** (`str`): the path to save plot. Better same as the output directory of the experiment. E.g. './outputs/expr_name/1970-01-01_00-00-00/acc_matrix.png'.
    """
    data = pd.read_csv(csv_path)

    # plot the accuracy matrix
    fig, ax = plt.subplots(
        figsize=(2 * (task_id + 1), 2 * (task_id + 1))
    )  # adaptive figure size
    cax = ax.imshow(
        data.drop(["after_training_task", "average_accuracy"], axis=1),
        interpolation="nearest",
        cmap="Greens",
    )
    colorbar = fig.colorbar(cax)
    yticks = colorbar.ax.get_yticks()
    colorbar.ax.set_yticks(yticks)
    colorbar.ax.set_yticklabels(
        [f"{tick:.2f}" for tick in yticks], fontsize=10 + task_id
    )  # adaptive font size

    for i in range(task_id + 1):
        for j in range(1, i + 1):
            ax.text(
                j - 1,
                i - 1,
                f'{data.loc[i - 1,f"test_on_task_{j}"]:.3f}',
                ha="center",
                va="center",
                color="black",
                fontsize=10 + task_id,  # adaptive font size
            )

    ax.set_xticks(range(task_id))
    ax.set_yticks(range(task_id))
    ax.set_xticklabels(
        range(1, task_id + 1), fontsize=10 + task_id
    )  # adaptive font size
    ax.set_yticklabels(
        range(1, task_id + 1), fontsize=10 + task_id
    )  # adaptive font size

    # Labeling the axes
    ax.set_xlabel("Testing on task τ", fontsize=10 + task_id)  # adaptive font size
    ax.set_ylabel("After training task t", fontsize=10 + task_id)  # adaptive font size
    fig.savefig(plot_path)
    plt.close(fig)


def plot_test_ave_loss_cls_curve_from_csv(
    csv_path: str, task_id: int, plot_path: str
) -> None:
    """Plot the test average classification loss curve over different training tasks from saved csv file and save the plot to the designated directory.

    **Args:**
    - **csv_path** (`str`): the path to the csv file where the `utils.update_loss_cls_to_csv()` saved the test classification loss metric.
    - **task_id** (`int`): plot the test average accuracy metric from task 1 to `task_id`.
    - **plot_path** (`str`): the path to save plot. Better same as the output directory of the experiment. E.g. './outputs/expr_name/1970-01-01_00-00-00/ave_loss_cls.png'.
    """
    data = pd.read_csv(csv_path)

    # plot the average accuracy curve over different training tasks
    fig, ax = plt.subplots(figsize=(16, 9))
    ax.plot(
        data["after_training_task"],
        data["average_classification_loss"],
        marker="o",
        linewidth=2,
    )
    ax.set_xlabel("After training task $t$", fontsize=16)
    ax.set_xlabel("Average Classification Loss", fontsize=16)
    ax.grid(True)

    xticks = [int(i) for i in range(1, task_id + 1)]
    yticks = [
        i * 0.5 for i in range(int(data["average_classification_loss"].max() / 0.5) + 1)
    ]
    ax.set_xticks(xticks)
    ax.set_yticks(yticks)
    ax.set_xticklabels(xticks, fontsize=16)
    ax.set_yticklabels([f"{tick:.1f}" for tick in yticks], fontsize=16)
    fig.savefig(plot_path)
    plt.close(fig)


def plot_test_loss_cls_matrix_from_csv(
    csv_path: str, task_id: int, plot_path: str
) -> None:
    """Plot the test classification loss matrix from saved csv file and save the plot to the designated directory.

    **Args:**
    - **csv_path** (`str`): the path to the csv file where the `utils.update_loss_cls_to_csv()` saved the test classification loss metric.
    - **task_id** (`int`): plot the test classification loss metric from task 1 to `task_id`.
    - **plot_path** (`str`): the path to save plot. Better same as the output directory of the experiment. E.g. './outputs/expr_name/1970-01-01_00-00-00/loss_cls_matrix.png'.
    """
    data = pd.read_csv(csv_path)

    # plot the accuracy matrix
    fig, ax = plt.subplots(
        figsize=(2 * (task_id + 1), 2 * (task_id + 1))
    )  # adaptive figure size
    cax = ax.imshow(
        data.drop(["after_training_task", "average_classification_loss"], axis=1),
        interpolation="nearest",
        cmap="Greens",
    )
    colorbar = fig.colorbar(cax)
    yticks = colorbar.ax.get_yticks()
    colorbar.ax.set_yticks(yticks)
    colorbar.ax.set_yticklabels(
        [f"{tick:.2f}" for tick in yticks], fontsize=10 + task_id
    )  # adaptive font size

    for i in range(task_id + 1):
        for j in range(1, i + 1):
            ax.text(
                j - 1,
                i - 1,
                f'{data.loc[i - 1,f"test_on_task_{j}"]:.3f}',
                ha="center",
                va="center",
                color="black",
                fontsize=10 + task_id,  # adaptive font size
            )
    ax.set_xticks(range(task_id))
    ax.set_yticks(range(task_id))

    ax.set_xticklabels(
        range(1, task_id + 1), fontsize=10 + task_id
    )  # adaptive font size
    ax.set_yticklabels(
        range(1, task_id + 1), fontsize=10 + task_id
    )  # adaptive font size

    # Labeling the axes
    ax.set_xlabel("Testing on task τ", fontsize=10 + task_id)  # adaptive font size
    ax.set_ylabel("After training task t", fontsize=10 + task_id)  # adaptive font size
    fig.savefig(plot_path)
    plt.close(fig)


def plot_hat_mask(
    mask: dict[str, Tensor], plot_dir: str, task_id: int, step: int | None = None
) -> None:
    """Plot mask in [HAT (Hard Attention to the Task)](http://proceedings.mlr.press/v80/serra18a)) algorithm. This includes the mask and cumulative mask.

    **Args:**
    - **mask** (`dict[str, Tensor]`): the hard attention (whose values are 0 or 1) mask. Key (`str`) is layer name, value (`Tensor`) is the mask tensor. The mask tensor has size (number of units).
    - **plot_dir** (`str`): the directory to save plot. Better same as the output directory of the experiment.
    - **task_id** (`int`): the task ID of the mask to be plotted. This is to form the plot name.
    - **step** (`int`): the training step (batch index) of the mask to be plotted. Apply to the training mask only. This is to form the plot name. Keep `None` for not showing the step in the plot name.
    """

    for layer_name, m in mask.items():
        layer_name = layer_name.replace(
            "/", "."
        )  # the layer name contains '/', which is not allowed in the file name. We replace it back with '.'.

        m = m.view(1, -1)  # reshape the 1D mask to 2D so can be plotted by image show

        fig = plt.figure()
        plt.imshow(
            m.detach().cpu(), aspect=10, cmap="Greys"
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
