r"""
The submodule in `pipelines` for single-task learning experiment.

"""

__all__ = ["STLExperiment"]

import logging
from typing import Any

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
from clarena.utils.cfg import select_hyperparameters_from_config

# always get logger for built-in logging in each module
pylogger = logging.getLogger(__name__)


class STLExperiment:
    r"""The base class for single-task learning experiment."""

    def __init__(self, cfg: DictConfig) -> None:
        r"""
        **Args:**
        - **cfg** (`DictConfig`): the complete config dict for the single-task learning experiment.
        """
        self.cfg: DictConfig = cfg
        r"""The complete config dict."""

        STLExperiment.sanity_check(self)

        # required config fields
        self.eval: bool = cfg.eval
        r"""Whether to include evaluation phase."""
        self.global_seed: int = cfg.global_seed
        r"""The global seed for the entire experiment."""

        # components
        self.stl_dataset: STLDataset
        r"""STL dataset object."""
        self.backbone: Backbone
        r"""Backbone network object."""
        self.head: HeadSTL
        r"""STL output heads object."""
        self.model: STLAlgorithm
        r"""STL model object."""
        self.optimizer: Optimizer
        r"""Optimizer object."""
        self.lr_scheduler: LRScheduler | None
        r"""Learning rate scheduler object."""
        self.lightning_loggers: list[Logger]
        r"""The list of initialized lightning loggers objects."""
        self.callbacks: list[Callback]
        r"""The list of initialized callbacks objects."""
        self.trainer: Trainer
        r"""Trainer object."""

    def sanity_check(self) -> None:
        r"""Sanity check for config."""

        # check required config fields
        required_config_fields = [
            "pipeline",
            "expr_name",
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

    def instantiate_backbone(self, backbone_cfg: DictConfig) -> None:
        r"""Instantiate the MTL backbone network object from `backbone_cfg`."""
        pylogger.debug(
            "Instantiating backbone network <%s> (clarena.backbones.Backbone)...",
            backbone_cfg.get("_target_"),
        )
        self.backbone = hydra.utils.instantiate(backbone_cfg)
        pylogger.debug(
            "Backbone network <%s> (clarena.backbones.Backbone) instantiated!",
            backbone_cfg.get("_target_"),
        )

    def instantiate_head(self, input_dim: int) -> None:
        r"""Instantiate the STL output head object.

        **Args:**
        - **input_dim** (`int`): the input dimension of the head. Must be equal to the `output_dim` of the connected backbone.
        """
        pylogger.debug(
            "Instantiating STL head...",
        )
        self.head = HeadSTL(input_dim=input_dim)
        pylogger.debug("STL head instantiated! ")

    def instantiate_stl_algorithm(
        self,
        stl_algorithm_cfg: DictConfig,
        backbone: Backbone,
        head: HeadSTL,
        non_algorithmic_hparams: dict[str, Any],
    ) -> None:
        r"""Instantiate the stl_algorithm object from `stl_algorithm_cfg`, `backbone`, `heads` and `non_algorithmic_hparams`."""
        pylogger.debug(
            "STL algorithm is set as <%s>. Instantiating <%s> (clarena.stl_algorithms.STLAlgorithm)...",
            stl_algorithm_cfg.get("_target_"),
            stl_algorithm_cfg.get("_target_"),
        )
        self.model = hydra.utils.instantiate(
            stl_algorithm_cfg,
            backbone=backbone,
            head=head,
            non_algorithmic_hparams=non_algorithmic_hparams,
        )
        pylogger.debug(
            "<%s> (clarena.stl_algorithms.STLAlgorithm) instantiated!",
            stl_algorithm_cfg.get("_target_"),
        )

    def instantiate_optimizer(
        self,
        optimizer_cfg: DictConfig,
    ) -> None:
        r"""Instantiate the optimizer object from `optimizer_cfg`."""

        # partially instantiate optimizer as the 'params' argument is from Lightning Modules cannot be passed for now.
        pylogger.debug(
            "Partially instantiating optimizer <%s> (torch.optim.Optimizer)...",
            optimizer_cfg.get("_target_"),
        )
        self.optimizer = hydra.utils.instantiate(optimizer_cfg)
        pylogger.debug(
            "Optimizer <%s> (torch.optim.Optimizer) partially instantiated!",
            optimizer_cfg.get("_target_"),
        )

    def instantiate_lr_scheduler(
        self,
        lr_scheduler_cfg: DictConfig,
    ) -> None:
        r"""Instantiate the learning rate scheduler object from `lr_scheduler_cfg`."""

        # partially instantiate learning rate scheduler as the 'params' argument is from Lightning Modules cannot be passed for now.
        pylogger.debug(
            "Partially instantiating learning rate scheduler <%s> (torch.optim.lr_scheduler.LRScheduler) ...",
            lr_scheduler_cfg.get("_target_"),
        )
        self.lr_scheduler = hydra.utils.instantiate(lr_scheduler_cfg)
        pylogger.debug(
            "Learning rate scheduler <%s> (torch.optim.lr_scheduler.LRScheduler) partially instantiated!",
            lr_scheduler_cfg.get("_target_"),
        )

    def instantiate_lightning_loggers(self, lightning_loggers_cfg: DictConfig) -> None:
        r"""Instantiate the list of lightning loggers objects from `lightning_loggers_cfg`."""
        pylogger.debug("Instantiating Lightning loggers (lightning.Logger)...")
        self.lightning_loggers = [
            hydra.utils.instantiate(lightning_logger)
            for lightning_logger in lightning_loggers_cfg.values()
        ]
        pylogger.debug("Lightning loggers (lightning.Logger) instantiated!")

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
        lightning_loggers: list[Logger],
        callbacks: list[Callback],
    ) -> None:
        r"""Instantiate the trainer object from `trainer_cfg`, `lightning_loggers`, and `callbacks`."""

        pylogger.debug("Instantiating trainer (lightning.Trainer)...")
        self.trainer = hydra.utils.instantiate(
            trainer_cfg, logger=lightning_loggers, callbacks=callbacks
        )
        pylogger.debug("Trainer (lightning.Trainer) instantiated!")

    def set_global_seed(self, global_seed: int) -> None:
        r"""Set the `global_seed` for the entire experiment."""
        L.seed_everything(self.global_seed, workers=True)
        pylogger.debug("Global seed is set as %d.", global_seed)

    def run(self) -> None:
        r"""The main method to run the single-task learning experiment."""
        self.set_global_seed(self.global_seed)

        self.instantiate_stl_dataset(stl_dataset_cfg=self.cfg.stl_dataset)
        self.instantiate_backbone(backbone_cfg=self.cfg.backbone)
        self.instantiate_head(input_dim=self.cfg.backbone.output_dim)
        self.instantiate_stl_algorithm(
            stl_algorithm_cfg=self.cfg.stl_algorithm,
            backbone=self.backbone,
            head=self.head,
            non_algorithmic_hparams=select_hyperparameters_from_config(
                cfg=self.cfg, type=self.cfg.pipeline
            ),
        )  # stl_algorithm should be instantiated after backbone and heads
        self.instantiate_optimizer(optimizer_cfg=self.cfg.optimizer)
        self.instantiate_lr_scheduler(lr_scheduler_cfg=self.cfg.lr_scheduler)
        self.instantiate_lightning_loggers(
            lightning_loggers_cfg=self.cfg.lightning_loggers
        )
        self.instantiate_callbacks(
            metrics_cfg=self.cfg.metrics, callbacks_cfg=self.cfg.callbacks
        )
        self.instantiate_trainer(
            trainer_cfg=self.cfg.trainer,
            lightning_loggers=self.lightning_loggers,
            callbacks=self.callbacks,
        )  # trainer should be instantiated after loggers and callbacks

        # setup task for dataset and model
        self.stl_dataset.setup_task()
        t = self.stl_dataset.get_class_map()
        print(t)
        self.model.setup_task(
            num_classes=len(self.stl_dataset.get_class_map()),
            optimizer=self.optimizer,
            lr_scheduler=self.lr_scheduler,
        )

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
