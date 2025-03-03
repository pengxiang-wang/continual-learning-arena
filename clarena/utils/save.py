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
    test_acc_metric: dict[int, MeanMetricBatch], csv_path: str
) -> None:
    """Update the test accuracy metrics of task 1 to `num_tasks` at the last line to an existing csv file. A new file will be created if not existing. Used in `CLMetricsCallback`.

    **Args:**
    - **test_acc_metric** (`dict[int, MeanMetricBatch]`): classification accuracy of the test data of each seen task. Accumulated and calculated from the test batches.
    - **csv_path** (`str`): save the test metric to path. E.g. './outputs/expr_name/1970-01-01_00-00-00/acc.csv'.
    """
    num_tasks = len(test_acc_metric)
    fieldnames = ["after_training_task", "average_accuracy"] + [
        f"test_on_task_{task_id}" for task_id in range(1, num_tasks + 1)
    ]

    new_line = {"after_training_task": num_tasks}  # construct the first column

    # construct the columns and calculate the average accuracy over tasks at the same time
    average_accuracy_over_tasks = MeanMetric()
    for task_id in range(1, num_tasks + 1):
        acc = test_acc_metric[f"{task_id}"].compute().item()
        new_line[f"test_on_task_{task_id}"] = acc
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
    test_loss_cls_metric: dict[int, MeanMetricBatch], csv_path: str
) -> None:
    """Update the test classification loss metrics of task 1 to `num_tasks` at the last line to an existing csv file. A new file will be created if not existing. Used in `CLMetricsCallback`.

    **Args:**
    - **test_loss_cls_metric** (`dict[int, MeanMetricBatch]`): classification loss of the test data of each seen task. Accumulated and calculated from the test batches.
    - **csv_path** (`str`): save the test metric to path. E.g. './outputs/expr_name/1970-01-01_00-00-00/loss_cls.csv'.
    """
    num_tasks = len(test_loss_cls_metric)
    fieldnames = ["after_training_task", "average_classification_loss"] + [
        f"test_on_task_{task_id}" for task_id in range(1, num_tasks + 1)
    ]

    new_line = {"after_training_task": num_tasks}  # construct the first column

    # write to the columns and calculate the average classification loss over tasks at the same time
    average_classification_loss_over_tasks = MeanMetric()
    for task_id in range(1, num_tasks + 1):
        # task_id = dataloader_idx
        loss_cls = test_loss_cls_metric[f"{task_id}"].compute().item()
        new_line[f"test_on_task_{task_id}"] = loss_cls
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
