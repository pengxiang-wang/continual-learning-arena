r"""
The submodule in `experiments` for evaluating trained continual learning main experiment.
"""

__all__ = ["CLMainEval"]

import logging

import lightning as L
import torch
from lightning import Callback, Trainer
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig, ListConfig

from clarena.cl_datasets import CLDataset
from clarena.experiments import CLMainTrain

# always get logger for built-in logging in each module
pylogger = logging.getLogger(__name__)


class CLMainEval:
    r"""The base class for evaluating trained continual learning main experiment. It is the evaluation process on a trained continual learning model read from saved model file without training and validation.

    This class controls the entire continual learning evaluation lifecycle including dataset setup, model loading, and evaluation across sequential tasks.
    """

    def __init__(
        self,
        cfg: DictConfig,
    ) -> None:
        r"""
        **Args:**
        - **cfg** (`DictConfig`): the config dict for the continual learning main evaluation experiment.
        """
        self.cfg: DictConfig = cfg
        r"""The complete config dict."""

        CLMainEval.sanity_check(self)

        # required config fields
        self.cl_paradigm: str = cfg.cl_paradigm
        r"""The continual learning paradigm."""
        self.eval_tasks: list[int] = (
            cfg.eval_tasks
            if isinstance(cfg.eval_tasks, ListConfig)
            else list(range(1, cfg.eval_tasks + 1))
        )
        r"""The list of task IDs to evaluate."""
        self.global_seed: int = cfg.global_seed
        r"""The global seed for the entire experiment."""
        self.output_dir: str = cfg.output_dir
        r"""The folder name for storing the experiment results."""
        self.main_model_path: str = cfg.main_model_path
        r"""The file path of the model to evaluate."""

        # components
        self.cl_dataset: CLDataset
        r"""CL dataset object."""
        self.lightning_loggers: list[Logger]
        r"""Lightning logger objects."""
        self.callbacks: list[Callback]
        r"""Callback objects."""
        self.trainer: Trainer
        r"""Trainer object."""

        # task ID control
        self.task_id: int
        r"""Task ID counter indicating which task is being processed. Self updated during the task loop. Valid from 1 to `cl_dataset.num_tasks`."""
        self.processed_task_ids: list[int] = []
        r"""Task IDs that have been processed in the experiment."""

    def sanity_check(self) -> None:
        r"""Sanity check."""

        # check required config fields
        required_config_fields = [
            "main_model_path",
            "eval_tasks",
            "cl_paradigm",
            "global_seed",
            "cl_dataset",
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
                f"Field cl_paradigm should be either 'TIL' or 'CIL' but got {self.cfg.cl_paradigm}!"
            )

        # check eval_tasks
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

        eval_tasks = self.cfg.eval_tasks
        if isinstance(eval_tasks, ListConfig):
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

    def instantiate_cl_dataset(self, cl_dataset_cfg: DictConfig) -> None:
        r"""Instantiate the CL dataset object from cl_dataset config.

        **Args:**
        - **cl_dataset_cfg** (`DictConfig`): the cl_dataset config dict.
        """
        CLMainTrain.instantiate_cl_dataset(self, cl_dataset_cfg)

    def instantiate_trainer(self, trainer_cfg: DictConfig, task_id: int) -> None:
        r"""Instantiate the trainer object for task `task_id` from trainer config.

        **Args:**
        - **trainer_cfg** (`DictConfig`): the trainer config dict. It can be a dict containing trainer config for each task; otherwise, it's an uniform trainer config for all tasks (but different objects).
        - **task_id** (`int`): the target task ID.
        """
        CLMainTrain.instantiate_trainer(self, trainer_cfg, task_id)

    def instantiate_lightning_loggers(
        self, lightning_loggers_cfg: DictConfig, task_id: int
    ) -> None:
        r"""Instantiate the list of lightning loggers objects for task `task_id` from lightning_loggers config.

        **Args:**
        - **lightning_loggers_cfg** (`DictConfig`): the lightning_loggers config dict. All tasks share the same lightning_loggers config but different objects.
        - **task_id** (`int`): the target task ID.
        """
        CLMainTrain.instantiate_lightning_loggers(self, lightning_loggers_cfg, task_id)

    def instantiate_callbacks(
        self, metrics_cfg: DictConfig, callbacks_cfg: DictConfig, task_id: int
    ) -> None:
        r"""Instantiate the list of callbacks objects from metrics and other callbacks config.

        **Args:**
        - **metrics_cfg** (`DictConfig`): the metrics config dict. All tasks share the same callbacks config but different objects.
        - **callbacks_cfg** (`DictConfig`): the callbacks config dict. All tasks share the same callbacks config but different objects.
        - **task_id** (`int`): the target task ID.
        """
        CLMainTrain.instantiate_callbacks(self, metrics_cfg, callbacks_cfg, task_id)

    def set_global_seed(self) -> None:
        r"""Set the global seed for the entire experiment."""
        L.seed_everything(self.global_seed, workers=True)
        pylogger.debug("Global seed is set as %d.", self.global_seed)

    def setup_task_id(self, task_id: int) -> None:
        r"""Set up current task_id in the beginning of the continual learning process of a new task.

        **Args:**
        - **task_id** (`int`): current task_id.
        """
        self.task_id = task_id
        self.processed_task_ids.append(task_id)

    def instantiate_global(self) -> None:
        r"""Instantiate dataset."""

        self.instantiate_cl_dataset(self.cfg.cl_dataset)

    def setup_global(self) -> None:
        r"""Let CL dataset know the CL paradigm to define its CL class map."""

        self.set_global_seed()
        self.cl_dataset.set_cl_paradigm(cl_paradigm=self.cl_paradigm)

    def instantiate_task_specific(self, task_id: int) -> None:
        r"""Instantiate task-specific components for the task `task_id` from `self.cfg`.

        **Args:**
        - **task_id** (`int`): the target task ID.
        """

        self.instantiate_callbacks(self.cfg.metrics, self.cfg.callbacks, task_id)
        self.instantiate_lightning_loggers(self.cfg.lightning_loggers, task_id)
        self.instantiate_trainer(
            self.cfg.trainer, task_id
        )  # trainer should be instantiated after loggers and callbacks

    def setup_task_specific(self, task_id: int) -> None:
        r"""Setup task-specific components to get ready for the task `task_id`.

        **Args:**
        - **task_id** (`int`): the target task ID.
        """

        self.cl_dataset.setup_task_id(task_id)
        self.cl_dataset.setup(stage="test")

    def run(self) -> None:
        r"""
        The main method to run the continual learning evaluation experiment.
        """
        self.instantiate_global()
        self.setup_global()

        model = torch.load(self.main_model_path)

        # task loop
        for task_id in self.eval_tasks:
            self.setup_task_id(task_id)
            self.setup_task_specific(task_id)

        # evaluation skipping training and validation
        self.trainer.test(
            model=model,
            datamodule=self.cl_dataset,
        )
