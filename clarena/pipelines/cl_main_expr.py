r"""
The submodule in `pipelines` for continual learning main experiment.
"""

__all__ = ["CLMainExperiment"]

import logging
from typing import Any

import hydra
import lightning as L
from lightning import Callback, Trainer
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig, ListConfig
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from clarena.backbones import CLBackbone
from clarena.cl_algorithms import CLAlgorithm
from clarena.cl_datasets import CLDataset
from clarena.heads import HeadsCIL, HeadsTIL
from clarena.utils.cfg import select_hyperparameters_from_config

# always get logger for built-in logging in each module
pylogger = logging.getLogger(__name__)


class CLMainExperiment:
    r"""The base class for continual learning main experiment."""

    def __init__(self, cfg: DictConfig) -> None:
        r"""
        **Args:**
        - **cfg** (`DictConfig`): the complete config dict for the continual learning main experiment.
        """
        self.cfg: DictConfig = cfg
        r"""The complete config dict."""

        CLMainExperiment.sanity_check(self)

        # required config fields
        self.cl_paradigm: str = cfg.cl_paradigm
        r"""The continual learning paradigm."""
        self.train_tasks: list[int] = (
            cfg.train_tasks
            if isinstance(cfg.train_tasks, ListConfig)
            else list(range(1, cfg.train_tasks + 1))
        )
        r"""The list of task IDs to train."""
        self.eval_after_tasks: list[int] = (
            cfg.eval_after_tasks
            if isinstance(cfg.eval_after_tasks, ListConfig)
            else list(range(1, cfg.eval_after_tasks + 1))
        )
        r"""If task ID $t$ is in this list, run the evaluation process for all seen tasks after training task $t$."""
        self.global_seed: int = cfg.global_seed
        r"""The global seed for the entire experiment."""
        self.output_dir: str = cfg.output_dir
        r"""The folder for storing the experiment results."""

        # components

        # global components
        self.cl_dataset: CLDataset
        r"""CL dataset object."""
        self.backbone: CLBackbone
        r"""Backbone network object."""
        self.heads: HeadsTIL | HeadsCIL
        r"""CL output heads object."""
        self.model: CLAlgorithm
        r"""CL model object."""
        self.lightning_loggers: list[Logger]
        r"""Lightning logger objects."""
        self.callbacks: list[Callback]
        r"""Callback objects."""

        # task-specific components
        self.optimizer_t: Optimizer
        r"""Optimizer object for the current task `self.task_id`."""
        self.lr_scheduler_t: LRScheduler | None
        r"""Learning rate scheduler object for the current task `self.task_id`."""
        self.trainer_t: Trainer
        r"""Trainer object for the current task `self.task_id`."""

        # task ID control
        self.task_id: int
        r"""Task ID counter indicating which task is being processed. Self updated during the task loop. Valid from 1 to the number of tasks in the CL dataset."""
        self.processed_task_ids: list[int] = []
        r"""Task IDs that have been processed."""

    def sanity_check(self) -> None:
        r"""Sanity check for config."""

        # check required config fields
        required_config_fields = [
            "pipeline",
            "expr_name",
            "cl_paradigm",
            "train_tasks",
            "eval_after_tasks",
            "global_seed",
            "cl_dataset",
            "cl_algorithm",
            "backbone",
            "optimizer",
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

        # check cl_paradigm
        if self.cfg.cl_paradigm not in ["TIL", "CIL"]:
            raise ValueError(
                f"Field `cl_paradigm` should be either 'TIL' or 'CIL' but got {self.cfg.cl_paradigm}!"
            )

        # get dataset number of tasks
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

        # check train_tasks
        train_tasks = self.cfg.train_tasks
        if isinstance(train_tasks, ListConfig):
            if len(train_tasks) < 1:
                raise ValueError("`train_tasks` config must contain at least one task.")
            if any(t < 1 or t > num_tasks for t in train_tasks):
                raise ValueError(
                    f"All task IDs in `train_tasks` config must be between 1 and {num_tasks}."
                )
        elif isinstance(train_tasks, int):
            if train_tasks < 0 or train_tasks > num_tasks:
                raise ValueError(
                    f"`train_tasks` config as integer must be between 0 and {num_tasks}."
                )
        else:
            raise TypeError(
                "`train_tasks` config must be either a list of integers or an integer."
            )

        # check eval_after_tasks
        eval_after_tasks = self.cfg.eval_after_tasks
        if isinstance(eval_after_tasks, ListConfig):
            if len(eval_after_tasks) < 1:
                raise ValueError(
                    "`eval_after_tasks` config must contain at least one task."
                )
            if any(t < 1 or t > num_tasks for t in eval_after_tasks):
                raise ValueError(
                    f"All task IDs in `eval_after_tasks` config must be between 1 and {num_tasks}."
                )
        elif isinstance(eval_after_tasks, int):
            if eval_after_tasks < 0 or eval_after_tasks > num_tasks:
                raise ValueError(
                    f"`eval_after_tasks` config as integer must be between 0 and {num_tasks}."
                )
        else:
            raise TypeError(
                "`eval_after_tasks` config must be either a list of integers or an integer."
            )

        # check that eval_after_tasks is a subset of train_tasks
        if isinstance(train_tasks, list) and isinstance(eval_after_tasks, list):
            if not set(eval_after_tasks).issubset(set(train_tasks)):
                raise ValueError(
                    "`eval_after_tasks` config must be a subset of `train_tasks` config."
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

    def instantiate_backbone(self, backbone_cfg: DictConfig) -> None:
        r"""Instantiate the CL backbone network object from `backbone_cfg`."""
        pylogger.debug(
            "Instantiating backbone network <%s> (clarena.backbones.CLBackbone)...",
            backbone_cfg.get("_target_"),
        )
        self.backbone = hydra.utils.instantiate(backbone_cfg)
        pylogger.debug(
            "Backbone network <%s> (clarena.backbones.CLBackbone) instantiated!",
            backbone_cfg.get("_target_"),
        )

    def instantiate_heads(self, cl_paradigm: str, input_dim: int) -> None:
        r"""Instantiate the CL output heads object.

        **Args:**
        - **cl_paradigm** (`str`): the CL paradigm, either 'TIL' or 'CIL'. 'TIL' uses `HeadsTIL`, while 'CIL' uses `HeadsCIL`.
        - **input_dim** (`int`): the input dimension of the heads. Must be equal to the `output_dim` of the connected backbone.
        """
        pylogger.debug(
            "CL paradigm is set as %s. Instantiating %s heads...",
            cl_paradigm,
            cl_paradigm,
        )
        self.heads = (
            HeadsTIL(input_dim=input_dim)
            if cl_paradigm == "TIL"
            else HeadsCIL(input_dim=input_dim)
        )
        pylogger.debug("%s heads instantiated!", cl_paradigm)

    def instantiate_cl_algorithm(
        self,
        cl_algorithm_cfg: DictConfig,
        backbone: CLBackbone,
        heads: HeadsTIL | HeadsCIL,
        non_algorithmic_hparams: dict[str, Any],
    ) -> None:
        r"""Instantiate the cl_algorithm object from `cl_algorithm_cfg`, `backbone`, `heads` and `non_algorithmic_hparams`."""
        pylogger.debug(
            "CL algorithm is set as <%s>. Instantiating <%s> (clarena.cl_algorithms.CLAlgorithm)...",
            cl_algorithm_cfg.get("_target_"),
            cl_algorithm_cfg.get("_target_"),
        )
        self.model = hydra.utils.instantiate(
            cl_algorithm_cfg,
            backbone=backbone,
            heads=heads,
            non_algorithmic_hparams=non_algorithmic_hparams,
        )
        pylogger.debug(
            "<%s> (clarena.cl_algorithms.CLAlgorithm) instantiated!",
            cl_algorithm_cfg.get("_target_"),
        )

    def instantiate_optimizer(
        self,
        optimizer_cfg: DictConfig,
        task_id: int,
    ) -> None:
        r"""Instantiate the optimizer object for task `task_id` from `optimizer_cfg`."""

        # distinguish whether the optimizer config is uniform or task-specific
        if not optimizer_cfg.get("_target_"):
            pylogger.debug("Distinct optimizer config is applied to each task.")
            optimizer_cfg = optimizer_cfg[task_id]
        else:
            pylogger.debug("Uniform optimizer config is applied to all tasks.")

        # partially instantiate optimizer as the 'params' argument from Lightning Modules cannot be passed for now
        pylogger.debug(
            "Partially instantiating optimizer <%s> (torch.optim.Optimizer) for task %d...",
            optimizer_cfg.get("_target_"),
            task_id,
        )
        self.optimizer_t = hydra.utils.instantiate(optimizer_cfg)
        pylogger.debug(
            "Optimizer <%s> (torch.optim.Optimizer) partially for task %d instantiated!",
            optimizer_cfg.get("_target_"),
            task_id,
        )

    def instantiate_lr_scheduler(
        self,
        lr_scheduler_cfg: DictConfig,
        task_id: int,
    ) -> None:
        r"""Instantiate the learning rate scheduler object for task `task_id` from `lr_scheduler_cfg`."""

        # distinguish whether the learning rate scheduler config is uniform or task-specific
        if not lr_scheduler_cfg.get("_target_"):
            pylogger.debug(
                "Distinct learning rate scheduler config is applied to each task."
            )
            lr_scheduler_cfg = lr_scheduler_cfg[task_id]
        else:
            pylogger.debug(
                "Uniform learning rate scheduler config is applied to all tasks."
            )

        # partially instantiate learning rate scheduler as the 'optimizer' argument from Lightning Modules cannot be passed for now
        pylogger.debug(
            "Partially instantiating learning rate scheduler <%s> (torch.optim.lr_scheduler.LRScheduler) for task %d...",
            lr_scheduler_cfg.get("_target_"),
            task_id,
        )
        self.lr_scheduler_t = hydra.utils.instantiate(lr_scheduler_cfg)
        pylogger.debug(
            "Learning rate scheduler <%s> (torch.optim.lr_scheduler.LRScheduler) partially for task %d instantiated!",
            lr_scheduler_cfg.get("_target_"),
            task_id,
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
        self, metrics_cfg: ListConfig, callbacks_cfg: ListConfig
    ) -> None:
        r"""Instantiate the list of callbacks objects from `metrics_cfg` and `callbacks_cfg`. Note that `metrics_cfg` is a list of metric callbacks and `callbacks_cfg` is a list of callbacks other the metric callbacks. The instantiated callbacks contain both metric callbacks and other callbacks."""
        pylogger.debug(
            "Instantiating callbacks (lightning.Callback)...",
        )

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
        task_id: int,
    ) -> None:
        r"""Instantiate the trainer object for task `task_id` from `trainer_cfg`, `lightning_loggers`, and `callbacks`."""

        if not trainer_cfg.get("_target_"):
            pylogger.debug("Distinct trainer config is applied to each task.")
            trainer_cfg = trainer_cfg[task_id]
        else:
            pylogger.debug("Uniform trainer config is applied to all tasks.")

        pylogger.debug(
            "Instantiating trainer (lightning.Trainer) for task %d...",
            task_id,
        )
        self.trainer_t = hydra.utils.instantiate(
            trainer_cfg,
            logger=lightning_loggers,
            callbacks=callbacks,
        )
        pylogger.debug(
            "Trainer (lightning.Trainer) for task %d instantiated!",
            task_id,
        )

    def set_global_seed(self, global_seed: int) -> None:
        r"""Set the `global_seed` for the entire experiment."""
        L.seed_everything(self.global_seed, workers=True)
        pylogger.debug("Global seed is set as %d.", global_seed)

    def run(self) -> None:
        r"""The main method to run the continual learning main experiment."""

        self.set_global_seed(self.global_seed)

        # global components
        self.instantiate_cl_dataset(cl_dataset_cfg=self.cfg.cl_dataset)
        self.cl_dataset.set_cl_paradigm(cl_paradigm=self.cl_paradigm)
        self.instantiate_backbone(backbone_cfg=self.cfg.backbone)
        self.instantiate_heads(
            cl_paradigm=self.cl_paradigm, input_dim=self.cfg.backbone.output_dim
        )
        self.instantiate_cl_algorithm(
            cl_algorithm_cfg=self.cfg.cl_algorithm,
            backbone=self.backbone,
            heads=self.heads,
            non_algorithmic_hparams=select_hyperparameters_from_config(
                cfg=self.cfg, type=self.cfg.pipeline
            ),
        )  # cl_algorithm should be instantiated after backbone and heads
        self.instantiate_lightning_loggers(
            lightning_loggers_cfg=self.cfg.lightning_loggers
        )
        self.instantiate_callbacks(
            metrics_cfg=self.cfg.metrics,
            callbacks_cfg=self.cfg.callbacks,
        )

        # task loop
        for task_id in self.train_tasks:

            self.task_id = task_id

            # task-specific components
            self.instantiate_optimizer(
                optimizer_cfg=self.cfg.optimizer,
                task_id=task_id,
            )
            if self.cfg.get("lr_scheduler"):
                self.instantiate_lr_scheduler(
                    lr_scheduler_cfg=self.cfg.lr_scheduler,
                    task_id=task_id,
                )
            self.instantiate_trainer(
                trainer_cfg=self.cfg.trainer,
                lightning_loggers=self.lightning_loggers,
                callbacks=self.callbacks,
                task_id=task_id,
            )  # trainer should be instantiated after lightning loggers and callbacks

            # setup task ID for dataset and model
            self.cl_dataset.setup_task_id(task_id=task_id)
            self.model.setup_task_id(
                task_id=task_id,
                num_classes=len(self.cl_dataset.get_cl_class_map(self.task_id)),
                optimizer=self.optimizer_t,
                lr_scheduler=self.lr_scheduler_t,
            )

            # train and validate the model
            self.trainer_t.fit(
                model=self.model,
                datamodule=self.cl_dataset,
            )

            # evaluation after training and validation
            if task_id in self.eval_after_tasks:
                self.trainer_t.test(
                    model=self.model,
                    datamodule=self.cl_dataset,
                )

            self.processed_task_ids.append(task_id)
