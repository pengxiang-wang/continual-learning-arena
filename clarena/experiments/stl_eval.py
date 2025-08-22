r"""
The submodule in `experiments` for evaluating trained single-task learning experiment.
"""

__all__ = ["STLEval"]

import logging

import hydra
import lightning as L
import torch
from lightning import Callback, Trainer
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig

from clarena.experiments.stl_train import STLTrain
from clarena.stl_datasets import STLDataset

# always get logger for built-in logging in each module
pylogger = logging.getLogger(__name__)


class STLEval:
    r"""The base class for evaluating trained single-task learning experiment.

    This runs evaluation on a trained single-task learning model read from saved model file, without any training loop.
    """

    def __init__(
        self,
        cfg: DictConfig,
    ) -> None:
        r"""Initializes the STL evaluation object with a evaluation configuration."""
        self.cfg: DictConfig = cfg
        r"""The complete config dict."""

        STLEval.sanity_check(self)

        self.global_seed: int = cfg.global_seed if cfg.get("global_seed") else None
        r"""Store the global seed for the entire experiment. Parsed from config and used to seed all random number generators."""
        self.output_dir: str = cfg.output_dir
        r"""Store the output directory to store the logs and checkpoints. Parsed from config and help any output operation to locate the correct directory."""
        self.model_path: str = cfg.model_path
        r"""Store the model path to load the model from. Parsed from config and used to load the model for evaluation."""

        # components
        self.stl_dataset: STLDataset
        r"""STL dataset object. Instantiate in `instantiate_stl_dataset()`."""
        self.trainer: Trainer
        r"""Trainer object. Instantiate in `instantiate_trainer()`."""
        self.lightning_loggers: list[Logger]
        r"""The list of initialized lightning loggers objects for current task `self.task_id`. Instantiate in `instantiate_lightning_loggers()`."""
        self.callbacks: list[Callback]
        r"""The list of initialized callbacks objects for the evaluation. Instantiate in `instantiate_callbacks()`."""

    def sanity_check(self) -> None:
        r"""Check the sanity of the config dict `self.cfg`."""

        # check required config fields
        required_config_fields = [
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
        r"""Instantiate the STL dataset object from stl_dataset config.

        **Args:**
        - **stl_dataset_cfg** (`DictConfig`): the stl_dataset config dict.
        """
        STLTrain.instantiate_stl_dataset(self, stl_dataset_cfg)

    def instantiate_trainer(self, trainer_cfg: DictConfig) -> None:
        r"""Instantiate the trainer object from trainer config.

        **Args:**
        - **trainer_cfg** (`DictConfig`): the trainer config dict.
        """
        STLTrain.instantiate_trainer(self, trainer_cfg)

    def instantiate_lightning_loggers(self, lightning_loggers_cfg: DictConfig) -> None:
        r"""Instantiate the list of lightning loggers objects from lightning_loggers config.

        **Args:**
        - **lightning_loggers_cfg** (`DictConfig`): the lightning_loggers config dict.
        """
        STLTrain.instantiate_lightning_loggers(self, lightning_loggers_cfg)

    def instantiate_callbacks(
        self, metrics_cfg: DictConfig, callbacks_cfg: DictConfig
    ) -> None:
        r"""Instantiate the list of callbacks objects from metrics and other callbacks config.

        **Args:**
        - **metrics_cfg** (`DictConfig`): the metrics config dict.
        - **callbacks_cfg** (`DictConfig`): the callbacks config dict.
        """
        STLTrain.instantiate_callbacks(self, metrics_cfg, callbacks_cfg)

    def set_global_seed(self) -> None:
        r"""Set the global seed for the entire experiment."""
        L.seed_everything(self.global_seed, workers=True)
        pylogger.debug("Global seed is set as %d.", self.global_seed)

    def instantiate(self) -> None:
        r"""Instantiate components for the STL experiment from `self.cfg`."""

        self.instantiate_stl_dataset(self.cfg.stl_dataset)
        self.instantiate_callbacks(self.cfg.metrics, self.cfg.callbacks)
        self.instantiate_lightning_loggers(self.cfg.lightning_loggers)
        self.instantiate_trainer(
            self.cfg.trainer
        )  # trainer should be instantiated after loggers and callbacks

    def setup(self) -> None:
        r"""Setup."""
        self.set_global_seed()

        self.stl_dataset.setup_task()
        self.stl_dataset.setup(stage="test")

    def run(self) -> None:
        r"""The main method to run the single-task learning experiment."""

        self.instantiate()
        self.setup()

        model = torch.load(self.model_path)

        # evaluation skipping training and validation
        self.trainer.test(
            model=model,
            datamodule=self.stl_dataset,
        )
