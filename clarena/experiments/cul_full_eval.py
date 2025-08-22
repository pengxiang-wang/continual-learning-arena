r"""
The submodule in `experiments` for full evaluating trained continual unlearning experiment.
"""

__all__ = ["CULFullEval"]

import logging

import hydra
import lightning as L
import torch
from lightning import Callback, LightningDataModule, Trainer
from omegaconf import DictConfig, ListConfig

from clarena.cl_datasets import CLDataset
from clarena.utils.eval import CULEvaluation

# always get logger for built-in logging in each module
pylogger = logging.getLogger(__name__)


class CULFullEval:
    r"""The base class for full evaluating trained continual unlearning experiment.

    This runs evaluation (`test` of the `Trainer`) on a trained continual unlearning and reference model read from saved model file, without any training loop.
    """

    def __init__(self, cfg: DictConfig) -> None:
        r"""Initializes the CUL evaluation object with a evaluation configuration."""

        self.cfg: DictConfig = cfg
        r"""The complete config dict."""

        CULFullEval.sanity_check(self)

        self.main_model_path: str = cfg.main_model_path
        r"""The path to the model file to load the main model from."""
        self.refretrain_model_path: str = cfg.refretrain_model_path
        r"""The path to the model file to load the reference retraining model from."""
        self.reforiginal_model_path: str = cfg.reforiginal_model_path
        r"""The path to the model file to load the reference original model from."""

        self.output_dir: str = cfg.output_dir
        r"""The main output directory to store the logs and checkpoints. Parsed from config and help any output operation to locate the correct directory."""

        self.cl_paradigm: str = cfg.cl_paradigm
        r"""The continual learning paradigm, either 'TIL' (Task-Incremental Learning) or 'CIL' (Class-Incremental Learning)."""
        self.dd_eval_tasks: list[int] = (
            cfg.dd_eval_tasks
            if isinstance(cfg.dd_eval_tasks, ListConfig)
            else list(range(1, cfg.dd_eval_tasks + 1))
        )
        r"""The list of tasks to be evaluated for JSD. Parsed from config and used in the evaluation loop. """
        self.ad_eval_tasks: list[int] = (
            cfg.ad_eval_tasks
            if isinstance(cfg.ad_eval_tasks, ListConfig)
            else list(range(1, cfg.ad_eval_tasks + 1))
        )
        r"""The list of tasks to be evaluated for AD. Parsed from config and used in the evaluation loop. """
        self.global_seed: int = cfg.global_seed if cfg.get("global_seed") else None
        r"""Store the global seed for the entire experiment. Parsed from config and used to seed all random number generators."""

        # components
        self.cl_dataset: CLDataset
        r"""CL dataset object. Instantiate in `instantiate_cl_dataset()`."""
        self.evaluation_module: CULEvaluation
        r"""Evaluation module for continual unlearning. Instantiate in `instantiate_evaluation_module()`."""
        self.trainer: Trainer
        r"""Trainer object. Instantiate in `instantiate_trainer()`."""
        self.callbacks: list[Callback]
        r"""The list of initialized callbacks objects for the evaluation. Instantiate in `instantiate_callbacks()`."""

        # task control
        self.task_id: int
        r"""Task ID counter indicating which task is being processed. Self updated during the task loop. Starting from 1. """
        self.processed_task_ids: list[int] = []
        r"""Task IDs that have been processed in the experiment."""

    def sanity_check(self) -> None:
        r"""Check the sanity of the config dict `self.cfg`."""

        # check required config fields
        required_config_fields = [
            "main_model_path",
            "cl_paradigm",
            "global_seed",
            "cl_dataset",
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

        if self.cfg.cl_paradigm not in ["TIL", "CIL"]:
            raise ValueError(
                f"Field cl_paradigm should be either 'TIL' or 'CIL' but got {self.cfg.cl_paradigm}!"
            )

        if not self.cfg.get("refretrain_model_path"):
            pylogger.warning(
                "`refretrain_model_path` not provided. Distribution Distance (DD) cannot be calculated."
            )

        if not self.cfg.get("reforiginal_model_path"):
            pylogger.warning(
                "`reforiginal_model_path` not provided. Accuracy Difference (AD) cannot be calculated."
            )

    def instantiate_cl_dataset(self, cl_dataset_cfg: DictConfig) -> None:
        r"""Instantiate the CL dataset object from cl_dataset config.

        **Args:**
        - **cl_dataset_cfg** (`DictConfig`): the cl_dataset config dict.
        """
        pylogger.debug(
            "Instantiating CL dataset <%s> (clarena.cl_datasets.CLDataset)...",
            cl_dataset_cfg.get("_target_"),
        )
        self.cl_dataset: CLDataset = hydra.utils.instantiate(cl_dataset_cfg)
        pylogger.debug(
            "CL dataset <%s> (clarena.cl_datasets.CLDataset) instantiated!",
            cl_dataset_cfg.get("_target_"),
        )

    def instantiate_evaluation_module(
        self,
    ) -> None:
        r"""Instantiate the evaluation module object."""
        pylogger.debug(
            "Instantiating evaluation module (clarena.utils.eval.CULEvaluation)...",
        )

        main_model = torch.load(self.main_model_path)
        ref_model = torch.load(self.refretrain_model_path)
        full_model = torch.load(self.reforiginal_model_path)

        self.evaluation_module = CULEvaluation(
            main_model,
            ref_model,
            full_model,
            dd_eval_task_ids=self.dd_eval_tasks,
            ad_eval_task_ids=self.ad_eval_tasks,
        )
        pylogger.debug(
            "Evaluation module (clarena.utils.eval.CULEvaluation) instantiated!"
        )

    def instantiate_trainer(self, trainer_cfg: DictConfig, task_id: int) -> None:
        r"""Instantiate the trainer object from trainer config.

        **Args:**
        - **trainer_cfg** (`DictConfig`): the trainer config dict. All tasks share the same trainer config but different objects.
        - **task_id** (`int`): the target task ID.
        """

        pylogger.debug(
            "Instantiating trainer <%s> (lightning.Trainer) for task %d...",
            trainer_cfg.get("_target_"),
            task_id,
        )
        self.trainer: Trainer = hydra.utils.instantiate(
            trainer_cfg,
            callbacks=self.callbacks,
        )
        pylogger.debug(
            "Trainer <%s> (lightning.Trainer) for task %d instantiated!",
            trainer_cfg.get("_target_"),
            task_id,
        )

    def instantiate_callbacks(
        self, metrics_cfg: DictConfig, callbacks_cfg: DictConfig, task_id: int
    ) -> None:
        r"""Instantiate the list of callbacks objects from metrics and other callbacks config.

        **Args:**
        - **metrics_cfg** (`DictConfig`): the metrics config dict. All tasks share the same callbacks config but different objects.
        - **callbacks_cfg** (`DictConfig`): the callbacks config dict. All tasks share the same callbacks config but different objects.
        - **task_id** (`int`): the target task ID.
        """
        pylogger.debug(
            "Instantiating callbacks (lightning.Callback) for task %d...", task_id
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
            "Callbacks (lightning.Callback) for task %d instantiated!", task_id
        )

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
        self.instantiate_evaluation_module()

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
        self.instantiate_trainer(
            self.cfg.trainer, task_id
        )  # trainer should be instantiated after callbacks

    def setup_task_specific(self, task_id: int) -> None:
        r"""Setup task-specific components to get ready for the task `task_id`.

        **Args:**
        - **task_id** (`int`): the target task ID.
        """

        self.cl_dataset.setup_task_id(task_id)
        self.cl_dataset.setup(stage="test")

        self.evaluation_module.setup_task_id(task_id)

    def run(self) -> None:
        r"""
        Run the evaluation: calls trainer.test() on the loaded model
        and the instantiated CLDataModule.
        """
        self.instantiate_global()
        self.setup_global()

        # task loop
        for task_id in self.ad_eval_tasks:  # ad_eval_tasks includes all tasks

            self.setup_task_id(task_id)
            self.instantiate_task_specific(task_id)
            self.setup_task_specific(task_id)

        # evaluation
        self.trainer.test(
            model=self.evaluation_module,
            datamodule=self.cl_dataset,
        )
