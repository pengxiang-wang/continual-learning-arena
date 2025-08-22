r"""
The submodule in `callbacks` for callback of saving first batch images.
"""

__all__ = ["SaveFirstBatchImages"]


import logging
import os

import torch
import torchvision
from lightning import Callback, Trainer

from clarena.cl_algorithms import CLAlgorithm
from clarena.mtl_algorithms import MTLAlgorithm
from clarena.stl_algorithms import STLAlgorithm

# always get logger for built-in logging in each module
pylogger = logging.getLogger(__name__)


class SaveFirstBatchImages(Callback):
    r"""Saves images and labels of the first batch of training data into files. In continual learning / unlearning, applies to all tasks."""

    def __init__(
        self,
        save_dir: str,
        img_prefix: str = "sample",
        labels_filename: str = "labels.txt",
        task_ids_filename: str | None = "tasks.txt",
    ) -> None:
        r"""Initialize the Save First Batch Images Callback.

        **Args:**
        - **save_dir** (`str`): the directory to save images and labels as files. Better inside the output directory.
        - **img_prefix** (`str`): the prefix for image files.
        - **labels_filename** (`str`): the filename for the labels file as texts.
        - **task_ids_filename** (`str` | `None`): the filename for the task IDs file as texts. Only used in MTL algorithms. If `None`, no task IDs file is saved.
        """

        os.makedirs(save_dir, exist_ok=True)

        self.save_dir: str = save_dir
        r"""Store the directory to save images and labels as files."""
        self.img_prefix: str = img_prefix
        r"""Store the prefix for image files."""
        self.labels_filename: str = labels_filename
        r"""Store the filename for the labels file."""
        self.task_ids_filename: str | None = task_ids_filename
        r"""Store the filename for the task IDs file. Only used in MTL algorithms. If `None`, no task IDs file is saved."""

        self.called: bool = False
        r"""Flag to avoid calling the callback multiple times."""

    def on_train_start(
        self, trainer: Trainer, pl_module: CLAlgorithm | MTLAlgorithm | STLAlgorithm
    ) -> None:
        r"""Save images and labels into files in the first batch of training data at the beginning of the training of the task."""
        if self.called:
            return  # flag to avoid calling the callback multiple times

        dataloader = trainer.train_dataloader
        batch = next(iter(dataloader))  # get the first batch

        if isinstance(pl_module, CLAlgorithm):
            image_batch, label_batch = batch
            image_samples = list(torch.unbind(image_batch, dim=0))
            label_samples = list(torch.unbind(label_batch, dim=0))

            # save images and labels as documents
            save_dir_task = os.path.join(self.save_dir, f"task_{pl_module.task_id}")
            os.makedirs(save_dir_task, exist_ok=True)
            labels_file = open(
                os.path.join(save_dir_task, self.labels_filename),
                "w",
                encoding="utf-8",
            )
            for i, (image, label) in enumerate(zip(image_samples, label_samples)):
                torchvision.utils.save_image(
                    image, os.path.join(save_dir_task, f"{self.img_prefix}_{i}.png")
                )
                labels_file.write(f"{self.img_prefix}_{i}.png: {label}\n")
            labels_file.close()
        elif isinstance(pl_module, MTLAlgorithm):
            image_batch, label_batch, tasks_batch = batch
            image_samples = list(torch.unbind(image_batch, dim=0))
            label_samples = list(torch.unbind(label_batch, dim=0))
            task_samples = list(torch.unbind(tasks_batch, dim=0))

            # save images, labels and task_ids as documents
            labels_file = open(
                os.path.join(self.save_dir, self.labels_filename), "w", encoding="utf-8"
            )
            task_ids_file = open(
                os.path.join(self.save_dir, self.task_ids_filename),
                "w",
                encoding="utf-8",
            )
            for i, (image, label, task_id) in enumerate(
                zip(image_samples, label_samples, task_samples)
            ):
                torchvision.utils.save_image(
                    image, os.path.join(self.save_dir, f"{self.img_prefix}_{i}.png")
                )
                labels_file.write(f"{self.img_prefix}_{i}.png: {label}\n")
                task_ids_file.write(f"{self.img_prefix}_{i}.png: {task_id}\n")
            labels_file.close()
            task_ids_file.close()

        elif isinstance(pl_module, STLAlgorithm):
            image_batch, label_batch = batch
            image_samples = list(torch.unbind(image_batch, dim=0))
            label_samples = list(torch.unbind(label_batch, dim=0))

            # save images and labels as documents
            labels_file = open(
                os.path.join(self.save_dir, self.labels_filename), "w", encoding="utf-8"
            )
            for i, (image, label) in enumerate(zip(image_samples, label_samples)):
                torchvision.utils.save_image(
                    image, os.path.join(self.save_dir, f"{self.img_prefix}_{i}.png")
                )
                labels_file.write(f"{self.img_prefix}_{i}.png: {label}\n")
            labels_file.close()

        self.called = True  # flag to avoid calling the callback multiple times
