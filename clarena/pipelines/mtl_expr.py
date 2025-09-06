r"""
The submodule in `pipelines` for multi-task learning experiment.
"""

__all__ = ["MTLExperiment"]

import logging
from typing import Any

import hydra
import lightning as L
from lightning import Callback, Trainer
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from clarena.backbones import Backbone, CLBackbone
from clarena.heads import HeadsMTL
from clarena.mtl_algorithms import MTLAlgorithm
from clarena.mtl_datasets import MTLDataset
from clarena.utils.cfg import select_hyperparameters_from_config

# always get logger for built-in logging in each module
pylogger = logging.getLogger(__name__)


class MTLExperiment:
    r"""The base class for multi-task learning experiment."""

    def __init__(self, cfg: DictConfig) -> None:
        r"""
        **Args:**
        - **cfg** (`DictConfig`): the complete config dict for the multi-task learning experiment.
        """
        self.cfg: DictConfig = cfg
        r"""The complete config dict."""

        MTLExperiment.sanity_check(self)

        # required config fields
        self.train_tasks: list[int] = (
            cfg.train_tasks
            if isinstance(cfg.train_tasks, list)
            else list(range(1, cfg.train_tasks + 1))
        )
        r"""The list of tasks to train."""
        self.eval_tasks: list[int] = (
            cfg.eval_tasks
            if isinstance(cfg.eval_tasks, list)
            else list(range(1, cfg.eval_tasks + 1))
        )
        r"""The list of tasks to evaluate."""
        self.global_seed: int = cfg.global_seed
        r"""The global seed for the entire experiment."""
        self.output_dir: str = cfg.output_dir
        r"""The folder for storing the experiment results."""

        # components
        self.mtl_dataset: MTLDataset
        r"""MTL dataset object."""
        self.backbone: CLBackbone
        r"""Backbone network object."""
        self.heads: HeadsMTL
        r"""MTL output heads object."""
        self.model: MTLAlgorithm
        r"""MTL model object."""
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
            "train_tasks",
            "eval_tasks",
            "global_seed",
            "mtl_dataset",
            "mtl_algorithm",
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

        # check train_tasks
        train_tasks = self.cfg.train_tasks
        if isinstance(train_tasks, list):
            if len(train_tasks) < 1:
                raise ValueError("`train_tasks` must contain at least one task.")
            if any(t < 1 or t > num_tasks for t in train_tasks):
                raise ValueError(
                    f"All task IDs in `train_tasks` must be between 1 and {num_tasks}."
                )
        elif isinstance(train_tasks, int):
            if train_tasks < 0 or train_tasks > num_tasks:
                raise ValueError(
                    f"`train_tasks` as integer must be between 0 and {num_tasks}."
                )
        else:
            raise TypeError(
                "`train_tasks` must be either a list of integers or an integer."
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
            "Instantiating MTL dataset <%s> (clarena.mtl_datasets.MTLDataset)...",
            mtl_dataset_cfg.get("_target_"),
        )
        self.mtl_dataset = hydra.utils.instantiate(mtl_dataset_cfg)
        pylogger.debug(
            "MTL dataset <%s> (clarena.mtl_datasets.MTLDataset) instantiated!",
            mtl_dataset_cfg.get("_target_"),
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

    def instantiate_heads(
        self,
        input_dim: int,
    ) -> None:
        r"""Instantiate the MTL output heads object.

        **Args:**
        - **input_dim** (`int`): the input dimension of the heads. Must be equal to the `output_dim` of the connected backbone.
        """
        pylogger.debug(
            "Instantiating MTL heads...",
        )
        self.heads = HeadsMTL(input_dim=input_dim)
        pylogger.debug("MTL heads instantiated! ")

    def instantiate_mtl_algorithm(
        self,
        mtl_algorithm_cfg: DictConfig,
        backbone: Backbone,
        heads: HeadsMTL,
        non_algorithmic_hparams: dict[str, Any],
    ) -> None:
        r"""Instantiate the mtl_algorithm object from `mtl_algorithm_cfg`, `backbone`, `heads` and `non_algorithmic_hparams`."""
        pylogger.debug(
            "MTL algorithm is set as <%s>. Instantiating <%s> (clarena.mtl_algorithms.MTLAlgorithm)...",
            mtl_algorithm_cfg.get("_target_"),
            mtl_algorithm_cfg.get("_target_"),
        )
        self.model = hydra.utils.instantiate(
            mtl_algorithm_cfg,
            backbone=backbone,
            heads=heads,
            non_algorithmic_hparams=non_algorithmic_hparams,
        )
        pylogger.debug(
            "<%s> (clarena.mtl_algorithms.MTLAlgorithm) instantiated!",
            mtl_algorithm_cfg.get("_target_"),
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
        r"""The main method to run the multi-task learning experiment."""
        self.set_global_seed(self.global_seed)

        self.instantiate_mtl_dataset(mtl_dataset_cfg=self.cfg.mtl_dataset)
        self.instantiate_backbone(backbone_cfg=self.cfg.backbone)
        self.instantiate_heads(input_dim=self.cfg.backbone.output_dim)
        self.instantiate_mtl_algorithm(
            mtl_algorithm_cfg=self.cfg.mtl_algorithm,
            backbone=self.backbone,
            heads=self.heads,
            non_algorithmic_hparams=select_hyperparameters_from_config(
                cfg=self.cfg, type=self.cfg.pipeline
            ),
        )  # mtl_algorithm should be instantiated after backbone and heads
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
        )  # trainer should be instantiated after lightning loggers and callbacks

        # setup tasks for dataset and model
        self.mtl_dataset.setup_tasks_expr(
            train_tasks=self.train_tasks, eval_tasks=self.eval_tasks
        )
        self.model.setup_tasks(
            task_ids=self.train_tasks,
            num_classes={
                task_id: len(self.mtl_dataset.get_mtl_class_map(task_id))
                for task_id in self.train_tasks
            },
            optimizer=self.optimizer,
            lr_scheduler=self.lr_scheduler,
        )

        # train and validate the model
        self.trainer.fit(
            model=self.model,
            datamodule=self.mtl_dataset,
        )

        # evaluation after training and validation
        self.trainer.test(
            model=self.model,
            datamodule=self.mtl_dataset,
        )
