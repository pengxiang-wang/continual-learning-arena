r"""
The submodule in `experiments` for continual unlearning main experiment.
"""

__all__ = ["CULMainTrain"]

import logging

import hydra
from omegaconf import DictConfig

from clarena.experiments import CLMainTrain
from clarena.unlearning_algorithms import CULAlgorithm

# always get logger for built-in logging in each module
pylogger = logging.getLogger(__name__)


class CULMainTrain(CLMainTrain):
    r"""The base class for continual unlearning main experiment."""

    def __init__(self, cfg: DictConfig) -> None:
        r"""Initializes the CUL experiment object with a experiment configuration.

        **Args:**
        - **cfg** (`DictConfig`): the complete config dict for the CUL experiment.
        """
        super().__init__(cfg)

        CULMainTrain.sanity_check(self)

        self.unlearning_algorithm: CULAlgorithm
        r"""Continual unlearning algorithm object. Instantiate in `instantiate_unlearning_algorithm()`."""

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
            "cl_paradigm", 
            "train_tasks", 
            "eval_after_tasks",
            "unlearning_requests",
            "global_seed",
            "cl_dataset",
            "cl_algorithm",
            "unlearning_algorithm",
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

        if not self.cfg.get("unlearning_algorithm"):
            raise KeyError(
                "Field unlearn.num_tasks should be specified in experiment config!"
            )

        if not self.cfg.get("unlearning_requests"):
            raise KeyError(
                "Field unlearning_requests should be specified in experiment config because this is a continual unlearning experiment!"
            )

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

    def instantiate_unlearning_algorithm(
        self, unlearning_algorithm_cfg: DictConfig
    ) -> None:
        r"""Instantiate the unlearning_algorithm object from unlearning_algorithm config.

        **Args:**
        - **unlearning_algorithm_cfg** (`DictConfig`): the unlearning_algorithm config dict.
        """
        pylogger.debug(
            "Unlearning algorithm is set as <%s>. Instantiating <%s> (clarena.unlearning_algorithms.UnlearningAlgorithm)...",
            unlearning_algorithm_cfg.get("_target_"),
            unlearning_algorithm_cfg.get("_target_"),
        )
        self.unlearning_algorithm: CULAlgorithm = hydra.utils.instantiate(
            unlearning_algorithm_cfg,
            model=self.model,
        )
        pylogger.debug(
            "<%s> (clarena.unlearning_algorithms.UnlearningAlgorithm) instantiated!",
            unlearning_algorithm_cfg.get("_target_"),
        )

    def instantiate_global(self) -> None:
        r"""Instantiate global components for the entire CUL experiment from `self.cfg`."""
        super().instantiate_global()
        self.instantiate_unlearning_algorithm(
            self.cfg.unlearning_algorithm
        )  # unlearning_algorithm should be instantiated after model

    def setup_task_specific(self):
        r"""Setup task-specific components to get ready for the current task `self.task_id`."""
        super().setup_task_specific()

        self.unlearning_algorithm.setup_task_id(
            task_id=self.task_id,
            unlearning_requests=self.unlearning_requests,
            if_permanent=self.permanent_mark[self.task_id],
        )

        pylogger.debug(
            "Unlearning algorithm is set up ready for task %d!",
            self.task_id,
        )

    def run_task(self, task_id: int) -> None:
        r"""Run the continual learning process for a single task `self.task_id`.

        **Args:**
        - **task_id** (`int`): the target task ID.
        """

        # train and validate the model
        if task_id in self.train_tasks:
            self.trainer_t.fit(
                model=self.model,
                datamodule=self.cl_dataset,
            )

        # unlearn
        self.unlearning_algorithm.unlearn()
        self.unlearning_algorithm.setup_test_task_id()

        # evaluation after training and validation
        if task_id in self.eval_after_tasks:
            self.trainer_t.test(
                model=self.model,
                datamodule=self.cl_dataset,
            )

        pylogger.debug(
            "Task %d is completed with unlearning requests %s!",
            self.task_id,
            self.unlearning_requests.get(self.task_id),
        )
