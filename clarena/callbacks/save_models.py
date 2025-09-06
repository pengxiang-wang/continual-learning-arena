r"""
The submodule in `callbacks` for callback of saving models.
"""

__all__ = ["SaveModels"]

import logging
import os

import torch
from lightning import Callback, Trainer

from clarena.cl_algorithms import CLAlgorithm
from clarena.mtl_algorithms.base import MTLAlgorithm
from clarena.stl_algorithms.base import STLAlgorithm

# always get logger for built-in logging in each module
pylogger = logging.getLogger(__name__)


class SaveModels(Callback):
    r"""Saves the model at the end of training. In continual learning / unlearning, applies to all tasks."""

    def __init__(self, save_dir: str, save_after_each_task: bool = False) -> None:
        r"""Initialize the SaveModel callback.

        **Args:**
        - **save_path** (`str`): the path to save the model.
        """
        self.save_dir = save_dir
        r"""Store the path to save the model."""

        os.makedirs(self.save_dir, exist_ok=True)

        self.save_after_each_task = save_after_each_task
        r"""Whether to save the model after each task in continual learning / unlearning."""

    def on_fit_end(
        self, trainer: Trainer, pl_module: CLAlgorithm | MTLAlgorithm | STLAlgorithm
    ) -> None:
        r"""Save the model at the end of each training task."""
        save_path = None
        if isinstance(pl_module, CLAlgorithm):
            if self.save_after_each_task:
                save_path = os.path.join(
                    self.save_dir, f"model_after_task_{pl_module.task_id}.pth"
                )
            else:
                save_path = os.path.join(self.save_dir, "cl_model.pth")
        elif isinstance(pl_module, MTLAlgorithm):
            save_path = os.path.join(self.save_dir, "mtl_model.pth")
        elif isinstance(pl_module, STLAlgorithm):
            save_path = os.path.join(self.save_dir, "stl_model.pth")

        torch.save(pl_module, save_path)

        if isinstance(pl_module, CLAlgorithm):
            torch.save(pl_module, os.path.join(self.save_dir, "cl_model.pth"))
        pylogger.info("Model saved!")
