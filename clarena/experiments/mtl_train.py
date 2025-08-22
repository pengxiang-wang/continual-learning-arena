r"""
The submodule in `experiments` for multi-task learning experiment.

This module contains the `MTLTrain` class, which is the main entry point for running multi-task learning experiments in the Continual Learning Arena. It handles the instantiation of various components such as datasets, backbones, heads, algorithms, optimizers, learning rate schedulers, trainers, loggers, and callbacks.
"""

__all__ = ["MTLTrain"]

import logging

import hydra
import lightning as L
from lightning import Callback, Trainer
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from clarena.backbones import Backbone, CLBackbone
from clarena.cl_datasets import CLDataset
from clarena.heads import HeadsMTL
from clarena.mtl_algorithms import MTLAlgorithm
from clarena.mtl_datasets import MTLDataset, MTLDatasetFromCL

# always get logger for built-in logging in each module
pylogger = logging.getLogger(__name__)


class MTLTrain:
    r"""The base class for multi-task learning experiment.

    This module contains the `MTLTrain` class, which is the main entry point for running multi-task learning experiments in the Continual Learning Arena. It handles the instantiation of various components such as datasets, backbones, heads, algorithms, optimizers, learning rate schedulers, trainers, loggers, and callbacks.
    """

    def __init__(self, cfg: DictConfig) -> None:
        r"""Initializes the MTL experiment object with a multi-task learning configuration.

        **Args:**
        - **cfg** (`DictConfig`): the complete config dict for the MTL experiment.
        """
        self.cfg: DictConfig = cfg
        r"""The complete config dict."""

        MTLTrain.sanity_check(self)

        # required config fields
        self.train_tasks: list[int] = (
            cfg.train_tasks
            if isinstance(cfg.train_tasks, list)
            else list(range(1, cfg.train_tasks + 1))
        )
        r"""The list of tasks to be jointly trained."""
        self.eval_tasks: list[int] = (
            cfg.eval_tasks
            if isinstance(cfg.eval_tasks, list)
            else list(range(1, cfg.eval_tasks + 1))
        )
        r"""The list of tasks to be evaluated."""
        self.global_seed: int = cfg.global_seed
        r"""The global seed for the entire experiment."""
        self.output_dir: str = cfg.output_dir
        r"""The folder name for storing the experiment results."""

        # components
        self.mtl_dataset: MTLDataset
        r"""MTL dataset object. Instantiate in `instantiate_mtl_dataset()`. One of `mtl_dataset` and `cl_dataset` must exist."""
        self.cl_dataset: CLDataset
        r"""CL dataset object to construct MTL dataset. One of `mtl_dataset` and `cl_dataset` must exist."""
        self.backbone: CLBackbone
        r"""Backbone network object. Instantiate in `instantiate_backbone()`."""
        self.heads: HeadsMTL
        r"""MTL output heads object. Instantiate in `instantiate_heads()`."""
        self.model: MTLAlgorithm
        r"""MTL model object. Instantiate in `instantiate_mtl_algorithm()`."""
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
            "train_tasks",
            "eval_tasks",
            "global_seed",
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

        if self.cfg.get("mtl_dataset") and self.cfg.get("cl_dataset"):
            raise KeyError(
                "Field mtl_dataset and cl_dataset are both specified! Please clarify whether you want to use the specified MTL dataset or construct the MTL dataset from the CL dataset!"
            )
        if not self.cfg.get("mtl_dataset") and not self.cfg.get("cl_dataset"):
            raise KeyError(
                "Field `mtl_dataset` or `cl_dataset` is required in the experiment index config. "
            )

        if self.cfg.get("cl_dataset"):
            if self.cfg.cl_dataset.get("num_tasks"):
                num_tasks = self.cfg.cl_dataset.get("num_tasks")
            elif self.cfg.cl_dataset.get("class_split"):
                num_tasks = len(self.cfg.cl_dataset.class_split)
            elif self.cfg.cl_dataset.get("datasets"):
                num_tasks = len(self.cfg.cl_dataset.datasets)
            else:
                raise KeyError(
                    "num_tasks is required in cl_dataset config. Please specify `num_tasks` (for `CLPermutedDataset`) or `class_split` (for `CLSplitDataset`) or `datasets` (for `CLCombinedDataset`) in cl_dataset config."
                )
        elif self.cfg.get("mtl_dataset"):
            if self.cfg.mtl_dataset.get("num_tasks"):
                num_tasks = self.cfg.mtl_dataset.num_tasks
            else:
                raise KeyError(
                    "num_tasks is required in mtl_dataset config. Please specify `num_tasks` in mtl_dataset config."
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
                    f"All task IDs in `eval_after_tasks` must be between 1 and {num_tasks}."
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
        self, mtl_dataset_cfg: DictConfig | None, cl_dataset_cfg: DictConfig | None
    ) -> None:
        r"""Instantiate the MTL dataset object from mtl_dataset config, or construct the MTL dataset from cl_dataset_config.

        **Args:**
        - **mtl_dataset_cfg** (`DictConfig`): the mtl_dataset config dict. When cl_dataset_cfg is not None, it should be None.
        - **cl_dataset_cfg** (`DictConfig`): the cl_dataset config dict used to construct the MTL dataset from the CL dataset. When mtl_dataset_cfg is not None, it should be None.
        """
        if self.cfg.get("mtl_dataset"):
            pylogger.debug(
                "Instantiating MTL dataset <%s> (clarena.mtl_datasets.MTLDataset) ...",
                mtl_dataset_cfg.get("_target_"),
            )
            self.mtl_dataset = hydra.utils.instantiate(
                mtl_dataset_cfg,
            )  # instantiate the MTL dataset

            pylogger.debug(
                "MTL dataset <%s> (clarena.mtl_datasets.MTLDataset) instantiated! ",
                mtl_dataset_cfg.get("_target_"),
            )
        elif self.cfg.get("cl_dataset"):
            pylogger.debug(
                "Constructing MTL dataset (clarena.mtl_datasets.MTLDataset) from <%s> (clarena.cl_datasets.CLDataset) ...",
                cl_dataset_cfg.get("_target_"),
            )

            self.cl_dataset = hydra.utils.instantiate(
                cl_dataset_cfg
            )  # instantiate the CL dataset
            self.mtl_dataset = MTLDatasetFromCL(
                self.cl_dataset,
            )  # construct the MTL dataset from the CL dataset
            pylogger.debug(
                "MTL dataset (clarena.mtl_datasets.MTLDataset) constructed from <%s> (clarena.cl_datasets.CLDataset)! ",
                cl_dataset_cfg.get("_target_"),
            )

    def instantiate_backbone(self, backbone_cfg: DictConfig) -> None:
        r"""Instantiate the MTL backbone network object from backbone config.

        **Args:**
        - **backbone_cfg** (`DictConfig`): the backbone config dict.
        """
        pylogger.debug(
            "Instantiating backbone network <%s> (clarena.backbones.MTLBackbone)...",
            backbone_cfg.get("_target_"),
        )
        self.backbone: Backbone = hydra.utils.instantiate(backbone_cfg)
        pylogger.debug(
            "Backbone network <%s> (clarena.backbones.MTLBackbone) instantiated!",
            backbone_cfg.get("_target_"),
        )

    def instantiate_heads(
        self,
        input_dim: int,
    ) -> None:
        r"""Instantiate the MTL output heads object according to field  backbone `output_dim` in the config.

        **Args:**
        - **input_dim** (`int`): the input dimension of the heads. Must be equal to the `output_dim` of the connected backbone.
        """
        pylogger.debug(
            "Instantiating MTL heads...",
        )
        self.heads: HeadsMTL = HeadsMTL(input_dim=input_dim)
        pylogger.debug("MTL heads instantiated! ")

    def instantiate_mtl_algorithm(self, mtl_algorithm_cfg: DictConfig) -> None:
        r"""Instantiate the mtl_algorithm object from mtl_algorithm config.

        **Args:**
        - **mtl_algorithm_cfg** (`DictConfig`): the mtl_algorithm config dict.
        """
        pylogger.debug(
            "MTL algorithm is set as <%s>. Instantiating <%s> (clarena.mtl_algorithms.MTLAlgorithm)...",
            mtl_algorithm_cfg.get("_target_"),
            mtl_algorithm_cfg.get("_target_"),
        )
        self.model: MTLAlgorithm = hydra.utils.instantiate(
            mtl_algorithm_cfg,
            backbone=self.backbone,
            heads=self.heads,
        )
        pylogger.debug(
            "<%s> (clarena.mtl_algorithms.MTLAlgorithm) instantiated!",
            mtl_algorithm_cfg.get("_target_"),
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
        self.lr_scheduler: LRScheduler = hydra.utils.instantiate(lr_scheduler_cfg)
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
        r"""Instantiate components for the MTL experiment from `self.cfg`."""

        if self.cfg.get("cl_dataset"):
            self.instantiate_mtl_dataset(
                cl_dataset_cfg=self.cfg.cl_dataset, mtl_dataset_cfg=None
            )
        elif self.cfg.get("mtl_dataset"):
            self.instantiate_mtl_dataset(
                mtl_dataset_cfg=self.cfg.mtl_dataset, cl_dataset_cfg=None
            )
        self.instantiate_backbone(self.cfg.backbone)
        self.instantiate_heads(
            input_dim=self.cfg.backbone.output_dim,
        )
        self.instantiate_mtl_algorithm(
            self.cfg.mtl_algorithm
        )  # mtl_algorithm should be instantiated after backbone and heads
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

        # set up the CL dataset if exists
        if self.cfg.get("cl_dataset"):
            self.cl_dataset.set_cl_paradigm(
                cl_paradigm="TIL"
            )  # MTL requires independent heads
            for task_id in self.train_tasks:
                self.cl_dataset.setup_task_id(task_id)

        self.mtl_dataset.setup_tasks(
            train_tasks=self.train_tasks,
            eval_tasks=self.eval_tasks,
        )
        self.heads.setup_tasks(
            task_ids=self.train_tasks,
            num_classes={
                task_id: len(self.mtl_dataset.get_class_map(task_id))
                for task_id in self.train_tasks
            },
        )
        self.model.setup_tasks(self.optimizer, self.lr_scheduler)

    def run(self) -> None:
        r"""The main method to run the multi-task learning experiment."""

        self.instantiate()
        self.setup()

        # fit the model on the MTL dataset
        self.trainer.fit(
            model=self.model,
            datamodule=self.mtl_dataset,
        )

        # evaluation after training and validation
        self.trainer.test(
            model=self.model,
            datamodule=self.mtl_dataset,
        )
