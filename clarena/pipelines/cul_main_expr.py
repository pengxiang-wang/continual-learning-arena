r"""
The submodule in `pipelines` for continual unlearning main experiment.
"""

__all__ = ["CULMainExperiment"]

import logging

import hydra
from omegaconf import DictConfig

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

        self.permanent_mark: dict[int, bool] = (
            cfg.permanent_mark
            if cfg.get("permanent_mark")
            else {t: True for t in self.train_tasks}
        )
        r"""Whether a task is permanent for each task in the experiment. If a task is permanent, it will not be unlearned i.e. not shown in future unlearning requests. This applies to some unlearning algorithms that need to know whether a task is permanent. """

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

    def run(self) -> None:
        r"""The main method to run the continual unlearning main experiment."""

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
            self.model.setup_task_id(
                task_id=task_id,
                num_classes=len(self.cl_dataset.get_cl_class_map(self.task_id)),
                optimizer=self.optimizer_t,
                lr_scheduler=self.lr_scheduler_t,
            )
            self.cul_algorithm.setup_task_id(
                task_id=self.task_id,
                unlearning_requests=self.unlearning_requests,
                if_permanent=self.permanent_mark[self.task_id],
            )

            # train and validate the model
            self.trainer_t.fit(
                model=self.model,
                datamodule=self.cl_dataset,
            )

            # unlearn
            self.cul_algorithm.unlearn()

            self.cul_algorithm.setup_test_task_id()

            # evaluation after training and validation
            if task_id in self.eval_after_tasks:
                self.trainer_t.test(
                    model=self.model,
                    datamodule=self.cl_dataset,
                )

            self.processed_task_ids.append(task_id)
