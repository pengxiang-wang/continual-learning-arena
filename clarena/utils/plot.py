"""The submodule in `utils` for plotting utils."""

__all__ = [
    "plot_test_acc_matrix_from_csv",
    "plot_test_loss_cls_matrix_from_csv",
    "plot_test_ave_acc_curve_from_csv",
    "plot_test_ave_loss_cls_curve_from_csv",
    "plot_hat_mask",
    "plot_hat_adjustment_rate",
    "plot_unlearning_test_distance_from_csv",
]


import os

import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
from torch import Tensor


def plot_test_acc_matrix_from_csv(csv_path: str, plot_path: str) -> None:
    """Plot the test accuracy matrix from saved CSV file and save the plot to the designated directory.

    **Args:**
    - **csv_path** (`str`): the path to the CSV file where the `utils.update_test_acc_to_csv()` saved the test accuracy metric.
    - **plot_path** (`str`): the path to save plot. Better same as the output directory of the experiment. E.g. './outputs/expr_name/1970-01-01_00-00-00/acc_matrix.png'.
    """
    data = pd.read_csv(csv_path)
    seen_task_ids = [
        int(col.replace("test_on_task_", ""))
        for col in data.columns
        if col.startswith("test_on_task_")
    ]
    num_tasks = len(seen_task_ids)

    # plot the accuracy matrix
    fig, ax = plt.subplots(
        figsize=(2 * (num_tasks + 1), 2 * (num_tasks + 1))
    )  # adaptive figure size
    cax = ax.imshow(
        data.drop(["after_training_task", "average_accuracy"], axis=1),
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

    for r in range(num_tasks):
        for c in range(r + 1):
            j = seen_task_ids[c]
            ax.text(
                c,
                r,
                f"{data.loc[r, f"test_on_task_{j}"]:.3f}",
                ha="center",
                va="center",
                color="black",
                fontsize=10 + num_tasks,  # adaptive font size
            )

    ax.set_xticks(range(num_tasks))
    ax.set_yticks(range(num_tasks))
    ax.set_xticklabels(seen_task_ids, fontsize=10 + num_tasks)  # adaptive font size
    ax.set_yticklabels(seen_task_ids, fontsize=10 + num_tasks)  # adaptive font size

    # Labeling the axes
    ax.set_xlabel("Testing on task τ", fontsize=10 + num_tasks)  # adaptive font size
    ax.set_ylabel(
        "After training task t", fontsize=10 + num_tasks
    )  # adaptive font size
    fig.savefig(plot_path)
    plt.close(fig)


def plot_test_loss_cls_matrix_from_csv(csv_path: str, plot_path: str) -> None:
    """Plot the test classification loss matrix from saved CSV file and save the plot to the designated directory.

    **Args:**
    - **csv_path** (`str`): the path to the CSV file where the `utils.update_loss_cls_to_csv()` saved the test classification loss metric.
    - **plot_path** (`str`): the path to save plot. Better same as the output directory of the experiment. E.g. './outputs/expr_name/1970-01-01_00-00-00/loss_cls_matrix.png'.
    """
    data = pd.read_csv(csv_path)
    seen_task_ids = [
        int(col.replace("test_on_task_", ""))
        for col in data.columns
        if col.startswith("test_on_task_")
    ]
    num_tasks = len(seen_task_ids)

    # plot the accuracy matrix
    fig, ax = plt.subplots(
        figsize=(2 * (num_tasks + 1), 2 * (num_tasks + 1))
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
        [f"{tick:.2f}" for tick in yticks], fontsize=10 + num_tasks
    )  # adaptive font size

    for r in range(num_tasks):
        for c in range(r + 1):
            j = seen_task_ids[c]
            ax.text(
                c,
                r,
                f"{data.loc[r, f"test_on_task_{j}"]:.3f}",
                ha="center",
                va="center",
                color="black",
                fontsize=10 + num_tasks,  # adaptive font size
            )
    ax.set_xticks(range(num_tasks))
    ax.set_yticks(range(num_tasks))

    ax.set_xticklabels(seen_task_ids, fontsize=10 + num_tasks)  # adaptive font size
    ax.set_yticklabels(seen_task_ids, fontsize=10 + num_tasks)  # adaptive font size

    # Labeling the axes
    ax.set_xlabel("Testing on task τ", fontsize=10 + num_tasks)  # adaptive font size
    ax.set_ylabel(
        "After training task t", fontsize=10 + num_tasks
    )  # adaptive font size
    fig.savefig(plot_path)
    plt.close(fig)


def plot_test_ave_acc_curve_from_csv(csv_path: str, plot_path: str) -> None:
    """Plot the test average accuracy curve over different training tasks from saved CSV file and save the plot to the designated directory.

    **Args:**
    - **csv_path** (`str`): the path to the CSV file where the `utils.update_test_acc_to_csv()` saved the test accuracy metric.
    - **plot_path** (`str`): the path to save plot. Better same as the output directory of the experiment. E.g. './outputs/expr_name/1970-01-01_00-00-00/ave_acc.png'.
    """
    data = pd.read_csv(csv_path)
    num_tasks = len(data)

    # plot the average accuracy curve over different training tasks
    fig, ax = plt.subplots(figsize=(16, 9))
    ax.plot(
        data["after_training_task"],
        data["average_accuracy"],
        marker="o",
        linewidth=2,
    )
    ax.set_xlabel("After training task $t$", fontsize=16)
    ax.set_ylabel("Average Accuracy (AA)", fontsize=16)
    ax.grid(True)
    xticks = [int(i) for i in range(1, num_tasks + 1)]
    yticks = [i * 0.05 for i in range(21)]
    ax.set_xticks(xticks)
    ax.set_yticks(yticks)
    ax.set_xticklabels(xticks, fontsize=16)
    ax.set_yticklabels([f"{tick:.2f}" for tick in yticks], fontsize=16)
    fig.savefig(plot_path)
    plt.close(fig)


def plot_test_ave_loss_cls_curve_from_csv(csv_path: str, plot_path: str) -> None:
    """Plot the test average classification loss curve over different training tasks from saved CSV file and save the plot to the designated directory.

    **Args:**
    - **csv_path** (`str`): the path to the CSV file where the `utils.update_loss_cls_to_csv()` saved the test classification loss metric.
    - **plot_path** (`str`): the path to save plot. Better same as the output directory of the experiment. E.g. './outputs/expr_name/1970-01-01_00-00-00/ave_loss_cls.png'.
    """
    data = pd.read_csv(csv_path)
    num_tasks = len(data)

    # plot the average accuracy curve over different training tasks
    fig, ax = plt.subplots(figsize=(16, 9))
    ax.plot(
        data["after_training_task"],
        data["average_classification_loss"],
        marker="o",
        linewidth=2,
    )
    ax.set_xlabel("After training task $t$", fontsize=16)
    ax.set_ylabel("Average Classification Loss", fontsize=16)
    ax.grid(True)

    xticks = [int(i) for i in range(1, num_tasks + 1)]
    yticks = [
        i * 0.5 for i in range(int(data["average_classification_loss"].max() / 0.5) + 1)
    ]
    ax.set_xticks(xticks)
    ax.set_yticks(yticks)
    ax.set_xticklabels(xticks, fontsize=16)
    ax.set_yticklabels([f"{tick:.1f}" for tick in yticks], fontsize=16)
    fig.savefig(plot_path)
    plt.close(fig)


def plot_joint_test_acc_from_csv(csv_path: str, plot_path: str) -> None:
    """Plot the test accuracy bar chart of all tasks in joint learning from saved CSV file and save the plot to the designated directory.

    **Args:**
    - **csv_path** (`str`): the path to the csv file where the `utils.save_joint_test_acc_csv()` saved the test accuracy metric.
    - **plot_path** (`str`): the path to save plot. Better same as the output directory of the experiment. E.g. './outputs/expr_name/1970-01-01_00-00-00/acc.png'.
    """
    data = pd.read_csv(csv_path)

    # extract all accuracy columns including average
    all_columns = data.columns.tolist()
    task_ids = list(range(len(all_columns)))  # assign index-based positions
    labels = [
        col.replace("test_on_task_", "Task ") if "test_on_task_" in col else "Average"
        for col in all_columns
    ]
    accuracies = data.iloc[0][all_columns].values

    # plot the accuracy bar chart over tasks
    fig, ax = plt.subplots(figsize=(16, 9))
    ax.bar(
        task_ids,
        accuracies,
        color="skyblue",
        edgecolor="black",
    )
    ax.set_xlabel("Task", fontsize=16)
    ax.set_ylabel("Accuracy", fontsize=16)
    ax.grid(True)
    ax.set_xticks(task_ids)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=14)
    ax.set_yticks([i * 0.05 for i in range(21)])
    ax.set_yticklabels(
        [f"{tick:.2f}" for tick in [i * 0.05 for i in range(21)]], fontsize=14
    )
    fig.tight_layout()
    fig.savefig(plot_path)
    plt.close(fig)


def plot_hat_mask(
    mask: dict[str, Tensor], plot_dir: str, task_id: int, step: int | None = None
) -> None:
    """Plot mask in [HAT (Hard Attention to the Task)](http://proceedings.mlr.press/v80/serra18a)) algorithm. This includes the mask and cumulative mask.

    **Args:**
    - **mask** (`dict[str, Tensor]`): the hard attention (whose values are 0 or 1) mask. Key (`str`) is layer name, value (`Tensor`) is the mask tensor. The mask tensor has size (number of units, ).
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


def plot_hat_adjustment_rate(
    adjustment_rate: dict[str, Tensor],
    weight_or_bias: str,
    plot_dir: str,
    task_id: int,
    step: int | None = None,
) -> None:
    """Plot adjustment rate in [HAT (Hard Attention to the Task)](http://proceedings.mlr.press/v80/serra18a)) algorithm. This includes the adjustment rate weight and adjustment rate bias (if applicable).

    **Args:**
    - **adjustment_rate** (`dict[str, Tensor]`): the adjustment rate. Key (`str`) is layer name, value (`Tensor`) is the adjustment rate tensor. If it's adjustment rate weight, it has size same as weights. If it's adjustment rate bias, it has size same as biases.
    - **weight_or_bias** (`str`): the type of adjustment rate. It can be either 'weight' or 'bias'. This is to form the plot name.
    - **plot_dir** (`str`): the directory to save plot. Better same as the output directory of the experiment.
    - **task_id** (`int`): the task ID of the adjustment rate to be plotted. This is to form the plot name.
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
            plot_name = f"{layer_name}_{weight_or_bias}_task{task_id}_step{step}.png"
        else:
            plot_name = f"{layer_name}_{weight_or_bias}_task{task_id}.png"
        plot_path = os.path.join(plot_dir, plot_name)
        fig.savefig(plot_path)
        plt.close(fig)


def plot_unlearning_test_distance_from_csv(csv_path: str, plot_path: str) -> None:
    """Plot the unlearning test distance matrix over different unlearned tasks from saved CSV file and save the plot to the designated directory.

    **Args:**
    - **csv_path** (`str`): the path to the CSV file where the `utils.save_unlearning_test_distance_to_csv()` saved the unlearning test distance metric.
    - **plot_path** (`str`): the path to save plot. Better same as the output directory of the experiment. E.g. './outputs/expr_name/1970-01-01_00-00-00/results/unlearning_test_after_task_X/distance.png'.
    """
    data = pd.read_csv(csv_path)
    unlearned_task_ids = [
        int(col.replace("unlearning_test_on_task_", ""))
        for col in data.columns
        if col.startswith("unlearning_test_on_task_")
    ]
    num_tasks = len(unlearned_task_ids)

    # plot the accuracy matrix
    fig, ax = plt.subplots(
        figsize=(2 * (num_tasks + 1), 2 * (num_tasks + 1))
    )  # adaptive figure size
    cax = ax.imshow(
        data.drop(
            ["unlearning_test_after_task", "average_distribution_distance"], axis=1
        ),
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

    for r in range(num_tasks):
        for c in range(r + 1):
            j = unlearned_task_ids[c]
            ax.text(
                c,
                r,
                f"{data.loc[r, f"unlearning_test_on_task_{j}"]:.3f}",
                ha="center",
                va="center",
                color="black",
                fontsize=10 + num_tasks,  # adaptive font size
            )

    ax.set_xticks(range(num_tasks))
    ax.set_yticks(range(num_tasks))
    ax.set_xticklabels(
        unlearned_task_ids, fontsize=10 + num_tasks
    )  # adaptive font size
    ax.set_yticklabels(
        unlearned_task_ids, fontsize=10 + num_tasks
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
