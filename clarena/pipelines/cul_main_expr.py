r"""
The submodule in `pipelines` for continual unlearning main experiment.
"""

__all__ = ["CULMainExperiment"]

import logging

import hydra
from omegaconf import DictConfig, ListConfig

from clarena.cul_algorithms import CULAlgorithm
from clarena.pipelines import CLMainExperiment
from clarena.utils.cfg import select_hyperparameters_from_config

# always get logger for built-in logging in each module
pylogger = logging.getLogger(__name__)


class CULMainExperiment(CLMainExperiment):
    r"""The base class for continual unlearning main experiment."""

    def __init__(self, cfg: DictConfig) -> None:
        r"""
        **Args:**
        - **cfg** (`DictConfig`): the complete config dict for the CUL experiment.
        """
        super().__init__(
            cfg
        )  # CUL main experiment inherits all configs from CL main experiment

        CULMainExperiment.sanity_check(self)

        self.cul_algorithm: CULAlgorithm
        r"""Continual unlearning algorithm object."""

        self.unlearning_requests: dict[int, list[int]] = cfg.unlearning_requests
        r"""The unlearning requests for each task in the experiment. Keys are IDs of the tasks that request unlearning after their learning, and values are the list of the previous tasks to be unlearned. Parsed from config and used in the tasks loop."""
        self.unlearned_task_ids: set[int] = set()
        r"""The list of task IDs that have been unlearned in the experiment. Updated in the tasks loop when unlearning requests are made."""

        self.unlearnable_ages: dict[int, int | None] | int | None = (
            cfg.unlearnable_age
            if isinstance(cfg.unlearnable_age, DictConfig)
            else {
                task_id: cfg.unlearnable_age
                for task_id in range(1, cfg.train_tasks + 1)
            }
        )
        r"""The dict of task unlearnable ages. Keys are task IDs and values are the unlearnable age of the corresponding task. A task cannot be unlearned when its age (i.e., the number of tasks learned after it) exceeds this value. If `None`, the task is unlearnable at any time."""

    def sanity_check(self) -> None:
        r"""Check the sanity of the config dict `self.cfg`."""

        # check required config fields
        required_config_fields = [
            "pipeline",
            "expr_name",
            "cl_paradigm",
            "train_tasks",
            "eval_after_tasks",
            "unlearning_requests",
            "unlearnable_age",
            "global_seed",
            "cl_dataset",
            "cl_algorithm",
            "cul_algorithm",
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

        # check unlearning requests
        for task_id, unlearning_task_ids in self.cfg.unlearning_requests.items():
            if task_id not in self.train_tasks:
                raise ValueError(
                    f"Task ID {task_id} in unlearning_requests is not within the train_tasks in the experiment!"
                )
            for unlearning_task_id in unlearning_task_ids:
                if unlearning_task_id not in self.train_tasks:
                    raise ValueError(
                        f"Unlearning task ID {unlearning_task_id} in unlearning_requests is not within the train_tasks in the experiment!"
                    )

    def instantiate_cul_algorithm(self, cul_algorithm_cfg: DictConfig) -> None:
        r"""Instantiate the CUL algorithm object from `cul_algorithm_cfg`."""
        pylogger.debug(
            "Instantiating CUL algorithm <%s> (clarena.cul_algorithms.CULAlgorithm)...",
            cul_algorithm_cfg.get("_target_"),
        )
        self.cul_algorithm: CULAlgorithm = hydra.utils.instantiate(
            cul_algorithm_cfg,
            model=self.model,
        )
        pylogger.debug(
            "<%s> (clarena.cul_algorithms.CULAlgorithm) instantiated!",
            cul_algorithm_cfg.get("_target_"),
        )

    def unlearnable_task_ids(self, task_id: int) -> list[int]:
        r"""Get the list of unlearnable task IDs at task `task_id`.

        **Args:**
        - **task_id** (`int`): the target task ID to check unlearnable task IDs.

        **Returns:**
        - **unlearnable_task_ids** (`list[int]`): the list of unlearnable task IDs at task `task_id`.
        """
        unlearnable_task_ids = []
        for tid in range(1, task_id + 1):
            unlearnable_age = self.unlearnable_ages[tid]
            if (
                unlearnable_age is None or (task_id - tid) < unlearnable_age
            ) and tid not in self.unlearned_task_ids:
                unlearnable_task_ids.append(tid)

        return unlearnable_task_ids

    def task_ids_just_no_longer_unlearnable(self, task_id: int) -> list[int]:
        r"""Get the list of task IDs just turning not unlearnable at task `task_id`.

        **Args:**
        - **task_id** (`int`): the target task ID to check.

        **Returns:**
        - **task_ids_just_no_longer_unlearnable** (`list[int]`): the list of task IDs just turning not unlearnable at task `task_id`.
        """
        task_ids_just_no_longer_unlearnable = []
        for tid in range(1, task_id + 1):
            unlearnable_age = self.unlearnable_ages[tid]
            if task_id - unlearnable_age == tid and tid not in self.unlearned_task_ids:
                task_ids_just_no_longer_unlearnable.append(tid)

        return task_ids_just_no_longer_unlearnable

    def run(self) -> None:
        r"""The main method to run the continual unlearning main experiment."""

        self.set_global_seed(self.global_seed)

        # global components
        self.instantiate_cl_dataset(cl_dataset_cfg=self.cfg.cl_dataset)
        self.cl_dataset.set_cl_paradigm(cl_paradigm=self.cl_paradigm)
        self.instantiate_backbone(
            backbone_cfg=self.cfg.backbone, disable_unlearning=False
        )
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
            disable_unlearning=False,
        )  # cl_algorithm should be instantiated after backbone and heads
        self.instantiate_cul_algorithm(
            self.cfg.cul_algorithm
        )  # cul_algorithm should be instantiated after model
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
            self.cul_algorithm.setup_task_id(
                task_id=self.task_id,
                unlearning_requests=self.unlearning_requests,
                unlearnable_task_ids=self.unlearnable_task_ids(self.task_id),
                task_ids_just_no_longer_unlearnable=self.task_ids_just_no_longer_unlearnable(
                    self.task_id
                ),
            )
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

            # unlearn
            if self.task_id in self.unlearning_requests.keys():
                unlearning_task_ids = self.unlearning_requests[self.task_id]
                pylogger.info(
                    "Starting unlearning process for tasks: %s...", unlearning_task_ids
                )
                self.cul_algorithm.unlearn()
                pylogger.info("Unlearning process finished.")

            # for unlearning_task_id in self.cul_algorithm.unlearning_task_ids:
            #     self.processed_task_ids.remove(unlearning_task_id)

            self.cul_algorithm.setup_test_task_id()

            # evaluation after training and validation
            if task_id in self.eval_after_tasks:
                self.trainer_t.test(
                    model=self.model,
                    datamodule=self.cl_dataset,
                )

            self.processed_task_ids.append(task_id)
