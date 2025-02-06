r"""
The submodule in `callbacks` for `ImageShowCallback`.
"""

__all__ = ["ImageShowCallback"]


import logging
import os

import torch
import torchvision
from lightning import Callback, Trainer

from clarena.cl_algorithms import CLAlgorithm

# always get logger for built-in logging in each module
pylogger = logging.getLogger(__name__)


class ImageShowCallback(Callback):
    r"""Image Show Callback shows images and labels in the first batch of training data in different ways."""

    def __init__(
        self,
        save: bool,
        save_dir: str,
        log_to_tensorboard: bool,
        img_prefix: str = "sample",
        labels_filename: str = "labels.txt",
    ) -> None:
        r"""Initialise the Image Show Callback.

        **Args:**
        - **save** (`bool`): whether to save images and labels as documents to output.
        - **save_dir** (`str`): the directory to save images and labels as documents. Better at the output directory.
        - **log_to_tensorboard** (`bool`): whether to log images and labels to TensorBoard.
        - **img_prefix** (`str`): the prefix for image files.
        - **labels_filename** (`str`): the filename for labels file.
        """
        super().__init__()

        self.save: bool = save
        r"""Store the `save` argument."""
        self.save_dir: str = save_dir
        r"""Store the `save_dir` argument."""
        self.log_to_tensorboard: bool = log_to_tensorboard
        r"""Store the `log_to_tensorboard` argument."""
        self.img_prefix: str = img_prefix
        r"""Store the `img_prefix` argument."""
        self.labels_filename: str = labels_filename
        r"""Store the `labels_filename` argument."""

        self.called: bool = False
        r"""Flag to avoid calling the callback multiple times."""

    def on_train_start(self, trainer: Trainer, pl_module: CLAlgorithm) -> None:
        r"""Show images and labels in the first batch of training data of a task in different ways in the beginning of the training of the task."""
        if self.called:
            return  # flag to avoid calling the callback multiple times

        dataloader = trainer.train_dataloader
        image_batch, label_batch = next(iter(dataloader))  # the first batch
        image_samples = list(torch.unbind(image_batch, dim=0))
        label_samples = list(torch.unbind(label_batch, dim=0))

        if self.save:
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
