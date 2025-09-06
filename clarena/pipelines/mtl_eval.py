r"""The submodule in `pipelines` for multi-task learning evaluation."""

__all__ = ["MTLEvaluation"]

import logging

import hydra
import lightning as L
import torch
from lightning import Callback, Trainer
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig, ListConfig

from clarena.mtl_datasets import MTLDataset

# always get logger for built-in logging in each module
pylogger = logging.getLogger(__name__)


class MTLEvaluation:
    r"""The base class for multi-task learning evaluation."""

    def __init__(self, cfg: DictConfig) -> None:
        r"""
        **Args:**
        - **cfg** (`DictConfig`): the config dict for the multi-task learning evaluation.
        """
        self.cfg: DictConfig = cfg
        r"""The complete config dict."""

        MTLEvaluation.sanity_check(self)

        # required config fields
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
        self.model_path: str = cfg.model_path
        r"""The file path of the model to evaluate."""

        # components
        self.mtl_dataset: MTLDataset
        r"""MTL dataset object."""
        self.lightning_loggers: list[Logger]
        r"""Lightning logger objects."""
        self.callbacks: list[Callback]
        r"""Callback objects."""
        self.trainer: Trainer
        r"""Trainer object."""

    def sanity_check(self) -> None:
        r"""Sanity check for config."""

        # check required config fields
        required_config_fields = [
            "pipeline",
            "model_path",
            "eval_tasks",
            "global_seed",
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

        # get dataset number of tasks
        if self.cfg.mtl_dataset._target_ == "clarena.mtl_datasets.MTLDatasetFromCL":
            cl_dataset_cfg = self.cfg.mtl_dataset.get("cl_dataset")
            if cl_dataset_cfg.get("num_tasks"):
                num_tasks = cl_dataset_cfg.get("num_tasks")
            elif cl_dataset_cfg.get("class_split"):
                num_tasks = len(cl_dataset_cfg.class_split)
            elif cl_dataset_cfg.get("datasets"):
                num_tasks = len(cl_dataset_cfg.datasets)
            else:
                raise KeyError(
                    "`num_tasks` is required in cl_dataset config under mtl_dataset config. Please specify `num_tasks` (for `CLPermutedDataset`) or `class_split` (for `CLSplitDataset`) or `datasets` (for `CLCombinedDataset`) in cl_dataset config."
                )
        else:
            if self.cfg.mtl_dataset.get("num_tasks"):
                num_tasks = self.cfg.mtl_dataset.num_tasks
            else:
                raise KeyError(
                    "`num_tasks` is required in mtl_dataset config. Please specify `num_tasks` in mtl_dataset config."
                )

        # check eval_tasks
        eval_tasks = self.cfg.eval_tasks
        if isinstance(eval_tasks, list):
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

    def instantiate_mtl_dataset(
        self,
        mtl_dataset_cfg: DictConfig,
    ) -> None:
        r"""Instantiate the MTL dataset object from `mtl_dataset_cfg`."""
        pylogger.debug(
            "Instantiating MTL dataset <%s> (clarena.mtl_datasets.MTLDataset) ...",
            mtl_dataset_cfg.get("_target_"),
        )
        self.mtl_dataset = hydra.utils.instantiate(mtl_dataset_cfg)
        pylogger.debug(
            "MTL dataset <%s> (clarena.mtl_datasets.MTLDataset) instantiated! ",
            mtl_dataset_cfg.get("_target_"),
        )

    def instantiate_callbacks(
        self, metrics_cfg: DictConfig, callbacks_cfg: DictConfig
    ) -> None:
        r"""Instantiate the list of callbacks objects from `metrics_cfg` and `callbacks_cfg`. Note that `metrics_cfg` is a list of metric callbacks and `callbacks_cfg` is a list of callbacks other the metric callbacks. The instantiated callbacks contain both metric callbacks and other callbacks."""
        pylogger.debug("Instantiating callbacks (lightning.Callback)...")

        # instantiate metric callbacks
        metric_callbacks = [
            hydra.utils.instantiate(callback) for callback in metrics_cfg
        ]

        # instantiate other callbacks
        other_callbacks = [
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
        self.trainer = hydra.utils.instantiate(trainer_cfg, callbacks=callbacks)
        pylogger.debug("Trainer (lightning.Trainer) instantiated!")

    def set_global_seed(self, global_seed: int) -> None:
        r"""Set the `global_seed` for the entire evaluation."""
        L.seed_everything(self.global_seed, workers=True)
        pylogger.debug("Global seed is set as %d.", global_seed)

    def run(self) -> None:
        r"""The main method to run the multi-task learning evaluation."""

        self.set_global_seed(self.global_seed)

        # load the model from file
        model = torch.load(self.model_path)

        self.instantiate_mtl_dataset(mtl_dataset_cfg=self.cfg.mtl_dataset)
        self.instantiate_callbacks(
            metrics_cfg=self.cfg.metrics, callbacks_cfg=self.cfg.callbacks
        )
        self.instantiate_trainer(
            trainer_cfg=self.cfg.trainer,
            callbacks=self.callbacks,
        )  # trainer should be instantiated after callbacks

        # setup tasks for dataset
        self.mtl_dataset.setup_tasks_eval(eval_tasks=self.eval_tasks)

        # evaluation skipping training and validation
        self.trainer.test(
            model=model,
            datamodule=self.mtl_dataset,
        )
