r"""The submodule in `pipelines` for single-task learning evaluation."""

__all__ = ["STLEvaluation"]

import logging

import hydra
import lightning as L
import torch
from lightning import Callback, Trainer
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig

from clarena.pipelines import STLExperiment
from clarena.stl_datasets import STLDataset

# always get logger for built-in logging in each module
pylogger = logging.getLogger(__name__)


class STLEvaluation:
    r"""The base class for single-task learning evaluation."""

    def __init__(self, cfg: DictConfig) -> None:
        r"""
        **Args:**
        - **cfg** (`DictConfig`): the config dict for the single-task learning evaluation.
        """
        self.cfg: DictConfig = cfg
        r"""The complete config dict."""

        STLEvaluation.sanity_check(self)

        # required config fields
        self.global_seed: int = cfg.global_seed
        r"""The global seed for the entire experiment."""
        self.output_dir: str = cfg.output_dir
        r"""The folder for storing the experiment results."""
        self.model_path: str = cfg.model_path
        r"""The file path of the model to evaluate."""

        # components
        self.stl_dataset: STLDataset
        r"""STL dataset object."""
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
            "global_seed",
            "stl_dataset",
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

    def instantiate_stl_dataset(
        self,
        stl_dataset_cfg: DictConfig,
    ) -> None:
        r"""Instantiate the STL dataset object from `stl_dataset_cfg`."""
        pylogger.debug(
            "Instantiating STL dataset <%s> (clarena.stl_datasets.STLDataset)...",
            stl_dataset_cfg.get("_target_"),
        )
        self.stl_dataset = hydra.utils.instantiate(
            stl_dataset_cfg,
        )
        pylogger.debug(
            "STL dataset <%s> (clarena.stl_datasets.STLDataset) instantiated!",
            stl_dataset_cfg.get("_target_"),
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
        r"""Instantiate the trainer object from `trainer_cfg`, `lightning_loggers`, and `callbacks`."""

        pylogger.debug("Instantiating trainer (lightning.Trainer)...")
        self.trainer = hydra.utils.instantiate(
            trainer_cfg, callbacks=callbacks
        )
        pylogger.debug("Trainer (lightning.Trainer) instantiated!")

    def set_global_seed(self, global_seed: int) -> None:
        r"""Set the `global_seed` for the entire evaluation."""
        L.seed_everything(self.global_seed, workers=True)
        pylogger.debug("Global seed is set as %d.", global_seed)

    def run(self) -> None:
        r"""The main method to run the single-task learning experiment."""
        self.set_global_seed(self.global_seed)

        # load the model from file
        model = torch.load(self.model_path)

        self.instantiate_stl_dataset(stl_dataset_cfg=self.cfg.stl_dataset)
        self.instantiate_callbacks(
            metrics_cfg=self.cfg.metrics, callbacks_cfg=self.cfg.callbacks
        )
        self.instantiate_trainer(
            trainer_cfg=self.cfg.trainer,
            callbacks=self.callbacks,
        )  # trainer should be instantiated after callbacks

        # setup task for dataset
        self.stl_dataset.setup_task()

        # evaluation skipping training and validation
        self.trainer.test(
            model=model,
            datamodule=self.stl_dataset,
        )
