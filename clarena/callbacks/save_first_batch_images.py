r"""
The submodule in `callbacks` for `SaveFirstBatchImagesCallback`.
"""

__all__ = ["SaveFirstBatchImagesCallback"]


import logging
import os

import torch
import torchvision
from lightning import Callback, Trainer

from clarena.cl_algorithms import CLAlgorithm

# always get logger for built-in logging in each module
pylogger = logging.getLogger(__name__)


class SaveFirstBatchImagesCallback(Callback):
    r"""Saves images and labels into files in the first batch of training data."""

    def __init__(
        self,
        save_dir: str,
        img_prefix: str = "sample",
        labels_filename: str = "labels.txt",
    ) -> None:
        r"""Initialise the Image Show Callback.

        **Args:**
        - **save_dir** (`str`): the directory to save images and labels as documents. Better inside the output directory.
        - **img_prefix** (`str`): the prefix for image files.
        - **labels_filename** (`str`): the filename for the labels file.
        """
        Callback.__init__(self)

        self.save_dir: str = save_dir
        r"""Store the directory to save images and labels as documents."""
        self.img_prefix: str = img_prefix
        r"""Store the prefix for image files."""
        self.labels_filename: str = labels_filename
        r"""Store the filename for the labels file."""

        self.called: bool = False
        r"""Flag to avoid calling the callback multiple times."""

    def on_train_start(self, trainer: Trainer, pl_module: CLAlgorithm) -> None:
        r"""Save images and labels into files in the first batch of training data at the beginning of the training of the task."""
        if self.called:
            return  # flag to avoid calling the callback multiple times

        dataloader = trainer.train_dataloader
        image_batch, label_batch = next(iter(dataloader))  # the first batch
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

        self.called = True  # flag to avoid calling the callback multiple times
