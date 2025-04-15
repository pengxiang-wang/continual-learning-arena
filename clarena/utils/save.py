"""The submodule in `utils` for saving data in the experiment as files."""

__all__ = [
    "update_test_acc_to_csv",
    "update_test_loss_cls_to_csv",
]


import csv
import os

from torchmetrics import MeanMetric

from clarena.utils import MeanMetricBatch


def update_test_acc_to_csv(
    after_training_task_id: int,
    test_acc_metric: dict[str, MeanMetricBatch],
    csv_path: str,
    skipped_task_ids_for_ave: list[int] | None = None,
) -> None:
    r"""Update the test accuracy metrics of seen tasks at the last line to an existing csv file. A new file will be created if not existing. Used in `CLMetricsCallback`.

    **Args:**
    - **after_training_task_id** (`int`): the task ID after training.
    - **test_acc_metric** (`dict[str, MeanMetricBatch]`): classification accuracy of the test data of each seen task. Accumulated and calculated from the test batches.
    - **csv_path** (`str`): save the test metric to path. E.g. './outputs/expr_name/1970-01-01_00-00-00/results/acc.csv'.
    - **skipped_task_ids_for_ave** (`list[int]` | `None`): the task IDs that are skipped to calculate average accuracy. If `None`, no task is skipped. Default: `None`.
    """
    seen_task_ids = list(test_acc_metric.keys())
    fieldnames = ["after_training_task", "average_accuracy"] + [
        f"test_on_task_{task_id}" for task_id in seen_task_ids
    ]

    new_line = {
        "after_training_task": after_training_task_id
    }  # construct the first column

    # construct the columns and calculate the average accuracy over tasks at the same time
    average_accuracy_over_tasks = MeanMetric()
    for task_id in seen_task_ids:
        acc = test_acc_metric[f"{task_id}"].compute().item()
        new_line[f"test_on_task_{task_id}"] = acc
        if skipped_task_ids_for_ave is None or task_id not in skipped_task_ids_for_ave:
            average_accuracy_over_tasks(acc)
    new_line["average_accuracy"] = average_accuracy_over_tasks.compute().item()

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
            file.writelines(lines)
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writerow(new_line)


def update_test_loss_cls_to_csv(
    after_training_task_id: int,
    test_loss_cls_metric: dict[str, MeanMetricBatch],
    csv_path: str,
    skipped_task_ids_for_ave: list[int] | None = None,
) -> None:
    """Update the test classification loss metrics of seen tasks at the last line to an existing csv file. A new file will be created if not existing. Used in `CLMetricsCallback`.

    **Args:**
    - **after_training_task_id** (`int`): the task ID after training.
    - **test_loss_cls_metric** (`dict[str, MeanMetricBatch]`): classification loss of the test data of each seen task. Accumulated and calculated from the test batches.
    - **csv_path** (`str`): save the test metric to path. E.g. './outputs/expr_name/1970-01-01_00-00-00/results/loss_cls.csv'.
    - **skipped_task_ids_for_ave** (`list[int]` | `None`): the task IDs that are skipped to calculate average accuracy. If `None`, no task is skipped. Default: `None`.
    """
    seen_task_ids = list(test_loss_cls_metric.keys())
    fieldnames = ["after_training_task", "average_classification_loss"] + [
        f"test_on_task_{task_id}" for task_id in seen_task_ids
    ]

    new_line = {
        "after_training_task": after_training_task_id
    }  # construct the first column

    # write to the columns and calculate the average classification loss over tasks at the same time
    average_classification_loss_over_tasks = MeanMetric()
    for task_id in seen_task_ids:
        # task_id = dataloader_idx
        loss_cls = test_loss_cls_metric[f"{task_id}"].compute().item()
        new_line[f"test_on_task_{task_id}"] = loss_cls
        if skipped_task_ids_for_ave is None or task_id not in skipped_task_ids_for_ave:
            average_classification_loss_over_tasks(loss_cls)
    new_line["average_classification_loss"] = (
        average_classification_loss_over_tasks.compute().item()
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
            file.writelines(lines)
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writerow(new_line)


def update_unlearning_test_distance_to_csv(
    unlearning_test_after_task_id: int,
    distance_metric: dict[str, MeanMetricBatch],
    csv_path: str,
) -> None:
    r"""Update the unlearning test distance metrics of unlearning tasks to csv file.

    **Args:**
    - **unlearning_test_after_task_id** (`int`): the task ID after for unlearning test.
    - **distance_metric** (`dict[str, MeanMetricBatch]`): the distance metric of unlearned tasks. Accumulated and calculated from the unlearning test batches.
    - **csv_path** (`str`): save the test metric to path. E.g. './outputs/expr_name/1970-01-01_00-00-00/results/unlearning_test_after_task_X/distance.csv'.
    """

    unlearned_task_ids = list(distance_metric.keys())
    fieldnames = ["unlearning_test_after_task", "average_distribution_distance"] + [
        f"unlearning_test_on_task_{task_id}" for task_id in unlearned_task_ids
    ]

    new_line = {
        "after_training_task": unlearning_test_after_task_id
    }  # construct the first column

    # write to the columns and calculate the average distribution distance over tasks at the same time
    average_distribution_distance_over_unlearned_tasks = MeanMetric()
    for task_id in unlearned_task_ids:
        loss_cls = distance_metric[f"{task_id}"].compute().item()
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
            file.writelines(lines)
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writerow(new_line)
