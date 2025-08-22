r"""
The submodule in `experiments` for single-task learning experiment.

"""

__all__ = ["STLTrain"]

import logging

import hydra
import lightning as L
from lightning import Callback, Trainer
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from clarena.backbones import Backbone
from clarena.heads import HeadSTL
from clarena.stl_algorithms import STLAlgorithm
from clarena.stl_datasets import STLDataset

# always get logger for built-in logging in each module
pylogger = logging.getLogger(__name__)


class STLTrain:
    r"""The base class for single-task learning experiment."""

    def __init__(self, cfg: DictConfig) -> None:
        r"""Initializes the STL experiment object with a single-task learning configuration.

        **Args:**
        - **cfg** (`DictConfig`): the complete config dict for the STL experiment.
        """
        self.cfg: DictConfig = cfg
        r"""The complete config dict."""

        STLTrain.sanity_check(self)

        self.global_seed: int = cfg.global_seed
        r"""The global seed for the entire experiment."""

        self.stl_dataset: STLDataset
        r"""STL dataset object. Instantiate in `instantiate_stl_dataset()`."""
        self.backbone: Backbone
        r"""Backbone network object. Instantiate in `instantiate_backbone()`."""
        self.head: HeadSTL
        r"""STL output heads object. Instantiate in `instantiate_head()`."""
        self.model: STLAlgorithm
        r"""STL model object. Instantiate in `instantiate_stl_algorithm()`."""

        self.optimizer: Optimizer
        r"""Optimizer object. Instantiate in `instantiate_optimizer()`."""
        self.lr_scheduler: LRScheduler
        r"""Learning rate scheduler object. Instantiate in `instantiate_lr_scheduler()`."""
        self.trainer: Trainer
        r"""Trainer object. Instantiate in `instantiate_trainer()`."""
        self.lightning_loggers: list[Logger]
        r"""The list of initialized lightning loggers objects. Instantiate in `instantiate_lightning_loggers()`."""
        self.callbacks: list[Callback]
        r"""The list of initialized callbacks objects. Instantiate in `instantiate_callbacks()`."""

    def sanity_check(self) -> None:
        r"""Check the sanity of the config dict `self.cfg`."""

        # check required config fields
        required_config_fields = [
            "global_seed",
            "stl_dataset",
            "stl_algorithm",
            "backbone",
            "optimizer",
            "lr_scheduler",
            "trainer",
            "metrics",
            "lightning_loggers",
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

        pylogger.debug(
            "Instantiating STL dataset <%s> (clarena.stl_datasets.STLDataset) ...",
            stl_dataset_cfg.get("_target_"),
        )
        self.stl_dataset = hydra.utils.instantiate(
            stl_dataset_cfg,
        )  # instantiate the STL dataset

        pylogger.debug(
            "STL dataset <%s> (clarena.stl_datasets.STLDataset) instantiated! ",
            stl_dataset_cfg.get("_target_"),
        )

    def instantiate_backbone(self, backbone_cfg: DictConfig) -> None:
        r"""Instantiate the MTL backbone network object from backbone config.

        **Args:**
        - **backbone_cfg** (`DictConfig`): the backbone config dict.
        """
        pylogger.debug(
            "Instantiating backbone network <%s> (clarena.backbones.Backbone)...",
            backbone_cfg.get("_target_"),
        )
        self.backbone: Backbone = hydra.utils.instantiate(backbone_cfg)
        pylogger.debug(
            "Backbone network <%s> (clarena.backbones.Backbone) instantiated!",
            backbone_cfg.get("_target_"),
        )

    def instantiate_head(self, input_dim: int) -> None:
        r"""Instantiate the STL output head object according to `output_dim` in the config.

        **Args:**
        - **input_dim** (`int`): the input dimension of the heads. Must be equal to the `output_dim` of the connected backbone.
        """
        pylogger.debug(
            "Instantiating STL head...",
        )
        self.head: HeadSTL = HeadSTL(input_dim=input_dim)
        pylogger.debug("STL head instantiated! ")

    def instantiate_stl_algorithm(self, stl_algorithm_cfg: DictConfig) -> None:
        r"""Instantiate the stl_algorithm object from stl_algorithm config.

        **Args:**
        - **stl_algorithm_cfg** (`DictConfig`): the stl_algorithm config dict.
        """
        pylogger.debug(
            "STL algorithm is set as <%s>. Instantiating <%s> (clarena.stl_algorithms.STLAlgorithm)...",
            stl_algorithm_cfg.get("_target_"),
            stl_algorithm_cfg.get("_target_"),
        )
        self.model = hydra.utils.instantiate(
            stl_algorithm_cfg,
            backbone=self.backbone,
            head=self.head,
        )
        pylogger.debug(
            "<%s> (clarena.stl_algorithms.STLAlgorithm) instantiated!",
            stl_algorithm_cfg.get("_target_"),
        )

    def instantiate_optimizer(
        self,
        optimizer_cfg: DictConfig,
    ) -> None:
        r"""Instantiate the optimizer object from optimizer config.

        **Args:**
        - **optimizer_cfg** (`DictConfig`): the optimizer config dict.
        """

        # partially instantiate optimizer as the 'params' argument is from Lightning Modules cannot be passed for now.
        pylogger.debug(
            "Partially instantiating optimizer <%s> (torch.optim.Optimizer)...",
            optimizer_cfg.get("_target_"),
        )
        self.optimizer: Optimizer = hydra.utils.instantiate(optimizer_cfg)
        pylogger.debug(
            "Optimizer <%s> (torch.optim.Optimizer) partially instantiated!",
            optimizer_cfg.get("_target_"),
        )

    def instantiate_lr_scheduler(
        self,
        lr_scheduler_cfg: DictConfig,
    ) -> None:
        r"""Instantiate the learning rate scheduler object from lr_scheduler config.

        **Args:**
        - **lr_scheduler_cfg** (`DictConfig`): the learning rate scheduler config dict.
        """

        # partially instantiate learning rate scheduler as the 'params' argument is from Lightning Modules cannot be passed for now.
        pylogger.debug(
            "Partially instantiating learning rate scheduler <%s> (torch.optim.lr_scheduler.LRScheduler) ...",
            lr_scheduler_cfg.get("_target_"),
        )
        self.lr_scheduler: Optimizer = hydra.utils.instantiate(lr_scheduler_cfg)
        pylogger.debug(
            "Learning rate scheduler <%s> (torch.optim.lr_scheduler.LRScheduler) partially instantiated!",
            lr_scheduler_cfg.get("_target_"),
        )

    def instantiate_trainer(self, trainer_cfg: DictConfig) -> None:
        r"""Instantiate the trainer object from trainer config.

        **Args:**
        - **trainer_cfg** (`DictConfig`): the trainer config dict.
        """

        pylogger.debug(
            "Instantiating trainer <%s> (lightning.Trainer)...",
            trainer_cfg.get("_target_"),
        )
        self.trainer: Trainer = hydra.utils.instantiate(
            trainer_cfg, callbacks=self.callbacks, logger=self.lightning_loggers
        )
        pylogger.debug(
            "Trainer <%s> (lightning.Trainer) instantiated!",
            trainer_cfg.get("_target_"),
        )

    def instantiate_lightning_loggers(self, lightning_loggers_cfg: DictConfig) -> None:
        r"""Instantiate the list of lightning loggers objects from lightning_loggers config.

        **Args:**
        - **lightning_loggers_cfg** (`DictConfig`): the lightning_loggers config dict.
        """
        pylogger.debug(
            "Instantiating Lightning loggers (lightning.Logger)...",
        )
        self.lightning_loggers: list[Logger] = [
            hydra.utils.instantiate(lightning_logger)
            for lightning_logger in lightning_loggers_cfg.values()
        ]
        pylogger.debug(
            "Lightning loggers (lightning.Logger) instantiated!",
        )

    def instantiate_callbacks(
        self, metrics_cfg: DictConfig, callbacks_cfg: DictConfig
    ) -> None:
        r"""Instantiate the list of callbacks objects from metrics and other callbacks config.

        **Args:**
        - **metrics_cfg** (`DictConfig`): the metrics config dict.
        - **callbacks_cfg** (`DictConfig`): the callbacks config dict.
        """

        pylogger.debug(
            "Instantiating callbacks (lightning.Callback)...",
        )

        # instantiate metric callbacks
        metric_callbacks: list[Callback] = [
            hydra.utils.instantiate(callback) for callback in metrics_cfg
        ]

        # instantiate other callbacks
        other_callbacks: list[Callback] = [
            hydra.utils.instantiate(callback) for callback in callbacks_cfg
        ]

        # add metric callbacks to the list of callbacks
        self.callbacks: list[Callback] = metric_callbacks + other_callbacks
        pylogger.debug(
            "Callbacks (lightning.Callback) instantiated!",
        )

    def set_global_seed(self) -> None:
        r"""Set the global seed for the entire experiment."""
        L.seed_everything(self.global_seed, workers=True)
        pylogger.debug("Global seed is set as %d.", self.global_seed)

    def instantiate(self) -> None:
        r"""Instantiate components for the JL experiment from `self.cfg`."""

        self.instantiate_stl_dataset(self.cfg.stl_dataset)
        self.instantiate_backbone(self.cfg.backbone)
        self.instantiate_head(input_dim=self.cfg.backbone.output_dim)
        self.instantiate_stl_algorithm(
            self.cfg.stl_algorithm
        )  # stl_algorithm should be instantiated after backbone and heads
        self.instantiate_optimizer(self.cfg.optimizer)
        self.instantiate_lr_scheduler(self.cfg.lr_scheduler)
        self.instantiate_callbacks(self.cfg.metrics, self.cfg.callbacks)
        self.instantiate_lightning_loggers(self.cfg.lightning_loggers)
        self.instantiate_trainer(
            self.cfg.trainer
        )  # trainer should be instantiated after loggers and callbacks

    def setup(self) -> None:
        r"""Setup."""
        self.set_global_seed()

        self.stl_dataset.setup_task()
        self.head.setup_task(
            num_classes=len(self.stl_dataset.get_class_map()),
        )
        self.model.setup_task(optimizer=self.optimizer, lr_scheduler=self.lr_scheduler)

    def run(self) -> None:
        r"""The main method to run the single-task learning experiment."""

        self.instantiate()
        self.setup()

        # fit the model on the STL dataset
        self.trainer.fit(
            model=self.model,
            datamodule=self.stl_dataset,
        )

        # evaluation after training and validation
        self.trainer.test(
            model=self.model,
            datamodule=self.stl_dataset,
        )
