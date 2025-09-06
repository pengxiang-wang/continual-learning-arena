r"""
The submodule in `pipelines` for continual learning main evaluation.
"""

__all__ = ["CLMainEvaluation"]

import logging

import hydra
import lightning as L
import torch
from lightning import Callback, Trainer
from omegaconf import DictConfig, ListConfig

from clarena.cl_datasets import CLDataset

# always get logger for built-in logging in each module
pylogger = logging.getLogger(__name__)


class CLMainEvaluation:
    r"""The base class for continual learning main evaluation."""

    def __init__(self, cfg: DictConfig) -> None:
        r"""
        **Args:**
        - **cfg** (`DictConfig`): the config dict for the continual learning main evaluation.
        """
        self.cfg: DictConfig = cfg
        r"""The complete config dict."""

        CLMainEvaluation.sanity_check(self)

        # required config fields
        self.main_model_path: str = cfg.main_model_path
        r"""The file path of the model to evaluate."""
        self.cl_paradigm: str = cfg.cl_paradigm
        r"""The continual learning paradigm."""
        self.eval_tasks: list[int] = (
            cfg.eval_tasks
            if isinstance(cfg.eval_tasks, ListConfig)
            else list(range(1, cfg.eval_tasks + 1))
        )
        r"""The list of task IDs to evaluate."""
        self.global_seed: int = cfg.global_seed
        r"""The global seed for the entire experiment."""
        self.output_dir: str = cfg.output_dir
        r"""The folder for storing the experiment results."""

        # components
        self.cl_dataset: CLDataset
        r"""CL dataset object."""
        self.callbacks: list[Callback]
        r"""Callback objects."""
        self.trainer: Trainer
        r"""Trainer object."""

    def sanity_check(self) -> None:
        r"""Sanity check for config."""

        # check required config fields
        required_config_fields = [
            "pipeline",
            "main_model_path",
            "eval_tasks",
            "cl_paradigm",
            "global_seed",
            "cl_dataset",
            "trainer",
            "metrics",
            "callbacks",
            "output_dir",
            # "hydra" is excluded as it doesn't appear
            "misc",
        ]
        for field in required_config_fields:
            if not self.cfg.get(field):
                raise KeyError(
                    f"Field `{field}` is required in the experiment index config."
                )

        # check cl_paradigm
        if self.cfg.cl_paradigm not in ["TIL", "CIL"]:
            raise ValueError(
                f"Field `cl_paradigm` should be either 'TIL' or 'CIL' but got {self.cfg.cl_paradigm}!"
            )

        # check eval_tasks
        if self.cfg.cl_dataset.get("num_tasks"):
            num_tasks = self.cfg.cl_dataset.get("num_tasks")
        elif self.cfg.cl_dataset.get("class_split"):
            num_tasks = len(self.cfg.cl_dataset.class_split)
        elif self.cfg.cl_dataset.get("datasets"):
            num_tasks = len(self.cfg.cl_dataset.datasets)
        else:
            raise KeyError(
                "`num_tasks` is required in cl_dataset config. Please specify `num_tasks` (for `CLPermutedDataset`) or `class_split` (for `CLSplitDataset`) or `datasets` (for `CLCombinedDataset`) in cl_dataset config."
            )

        eval_tasks = self.cfg.eval_tasks
        if isinstance(eval_tasks, ListConfig):
            if len(eval_tasks) < 1:
                raise ValueError("`eval_tasks` must contain at least one task.")
            if any(t < 1 or t > num_tasks for t in eval_tasks):
                raise ValueError(
                    f"All task IDs in `eval_tasks` must be between 1 and {num_tasks}."
                )
        elif isinstance(eval_tasks, int):
            if eval_tasks < 0 or eval_tasks > num_tasks:
                raise ValueError(
                    f"`eval_tasks` as integer must be between 0 and {num_tasks}."
                )
        else:
            raise TypeError(
                "`eval_tasks` must be either a list of integers or an integer."
            )

    def instantiate_cl_dataset(self, cl_dataset_cfg: DictConfig) -> None:
        r"""Instantiate the CL dataset object from `cl_dataset_cfg`."""
        pylogger.debug(
            "Instantiating CL dataset <%s> (clarena.cl_datasets.CLDataset)...",
            cl_dataset_cfg.get("_target_"),
        )
        self.cl_dataset = hydra.utils.instantiate(cl_dataset_cfg)
        pylogger.debug(
            "CL dataset <%s> (clarena.cl_datasets.CLDataset) instantiated!",
            cl_dataset_cfg.get("_target_"),
        )

    def instantiate_callbacks(
        self, metrics_cfg: DictConfig, callbacks_cfg: DictConfig
    ) -> None:
        r"""Instantiate the list of callbacks objects from `metrics_cfg` and `callbacks_cfg`. Note that `metrics_cfg` is a list of metric callbacks and `callbacks_cfg` is a list of callbacks other the metric callbacks. The instantiated callbacks contain both metric callbacks and other callbacks."""
        pylogger.debug("Instantiating callbacks (lightning.Callback)...")

        # instantiate metric callbacks
        metric_callbacks: list[Callback] = [
            hydra.utils.instantiate(callback) for callback in metrics_cfg
        ]

        # instantiate other callbacks
        other_callbacks: list[Callback] = [
            hydra.utils.instantiate(callback) for callback in callbacks_cfg
        ]

        # add metric callbacks to the list of callbacks
        self.callbacks = metric_callbacks + other_callbacks
        pylogger.debug("Callbacks (lightning.Callback) instantiated!")

    def instantiate_trainer(
        self,
        trainer_cfg: DictConfig,
        callbacks: list[Callback],
    ) -> None:
        r"""Instantiate the trainer object from `trainer_cfg` and `callbacks`."""

        pylogger.debug("Instantiating trainer (lightning.Trainer)...")
        self.trainer = hydra.utils.instantiate(
            trainer_cfg,
            callbacks=callbacks,
        )
        pylogger.debug("Trainer (lightning.Trainer) instantiated!")

    def set_global_seed(self, global_seed: int) -> None:
        r"""Set the `global_seed` for the entire evaluation."""
        L.seed_everything(self.global_seed, workers=True)
        pylogger.debug("Global seed is set as %d.", global_seed)

    def run(self) -> None:
        r"""The main method to run the continual learning main evaluation."""

        self.set_global_seed(self.global_seed)

        # load the model from file
        model = torch.load(self.main_model_path)

        # components
        self.instantiate_cl_dataset(cl_dataset_cfg=self.cfg.cl_dataset)
        self.cl_dataset.set_cl_paradigm(cl_paradigm=self.cl_paradigm)
        self.instantiate_callbacks(
            metrics_cfg=self.cfg.metrics, callbacks_cfg=self.cfg.callbacks
        )
        self.instantiate_trainer(
            trainer_cfg=self.cfg.trainer,
            callbacks=self.callbacks,
        )  # trainer should be instantiated after callbacks

        # setup tasks for dataset
        self.cl_dataset.setup_tasks_eval(eval_tasks=self.eval_tasks)

        # evaluation skipping training and validation
        self.trainer.test(
            model=model,
            datamodule=self.cl_dataset,
        )
