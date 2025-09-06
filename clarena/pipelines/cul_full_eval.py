r"""
The submodule in `pipelines` for continual unlearning full evaluation.
"""

__all__ = ["CULFullEvaluation"]

import logging

import hydra
import lightning as L
import torch
from lightning import Callback, Trainer
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig, ListConfig

from clarena.cl_datasets import CLDataset
from clarena.utils.eval import CULEvaluation

# always get logger for built-in logging in each module
pylogger = logging.getLogger(__name__)


class CULFullEvaluation:
    r"""The base class for continual unlearning full evaluation."""

    def __init__(self, cfg: DictConfig) -> None:
        r"""**Args:**
        - **cfg** (`DictConfig`): the complete config dict for the continual unlearning main evaluation.
        """

        self.cfg: DictConfig = cfg
        r"""The complete config dict."""

        CULFullEvaluation.sanity_check(self)

        # required config fields
        self.main_model_path: str = cfg.main_model_path
        r"""The path to the model file to load the main model from."""
        self.refretrain_model_path: str = cfg.refretrain_model_path
        r"""The path to the model file to load the reference retrain model from."""
        self.reforiginal_model_path: str = cfg.reforiginal_model_path
        r"""The path to the model file to load the reference original model from."""
        self.cl_paradigm: str = cfg.cl_paradigm
        r"""The continual learning paradigm."""
        self.dd_eval_tasks: list[int] = (
            cfg.dd_eval_tasks
            if isinstance(cfg.dd_eval_tasks, ListConfig)
            else list(range(1, cfg.dd_eval_tasks + 1))
        )
        r"""The list of tasks to be evaluated for DD."""
        self.ad_eval_tasks: list[int] = (
            cfg.ad_eval_tasks
            if isinstance(cfg.ad_eval_tasks, ListConfig)
            else list(range(1, cfg.ad_eval_tasks + 1))
        )
        r"""The list of tasks to be evaluated for AD."""
        self.global_seed: int = cfg.global_seed
        r"""The global seed for the entire experiment."""
        self.output_dir: str = cfg.output_dir
        r"""The folder for storing the experiment results."""

        # components
        self.cl_dataset: CLDataset
        r"""CL dataset object."""
        self.evaluation_module: CULEvaluation
        r"""Evaluation module for continual unlearning full evaluation."""
        self.lightning_loggers: list[Logger]
        r"""Lightning logger objects."""
        self.callbacks: list[Callback]
        r"""Callback objects."""
        self.trainer: Trainer
        r"""Trainer object."""

    def sanity_check(self) -> None:
        r"""Sanity check for config."""

        # check required config fields
        required_config_fields = [
            "pipeline",
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

        # check cl_paradigm
        if self.cfg.cl_paradigm not in ["TIL", "CIL"]:
            raise ValueError(
                f"Field cl_paradigm should be either 'TIL' or 'CIL' but got {self.cfg.cl_paradigm}!"
            )

        # warn if any reference experiment result is not provided
        if not self.cfg.get("refretrain_model_path"):
            pylogger.warning(
                "`refretrain_model_path` not provided. Distribution Distance (DD) cannot be calculated."
            )

        if not self.cfg.get("reforiginal_model_path"):
            pylogger.warning(
                "`reforiginal_model_path` not provided. Accuracy Difference (AD) cannot be calculated."
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

    def instantiate_evaluation_module(self) -> None:
        r"""Instantiate the evaluation module object."""
        pylogger.debug(
            "Instantiating evaluation module (clarena.utils.eval.CULEvaluation)...",
        )

        main_model = torch.load(self.main_model_path)
        refretrain_model = torch.load(self.refretrain_model_path)
        reforiginal_model = torch.load(self.reforiginal_model_path)

        self.evaluation_module = CULEvaluation(
            main_model=main_model,
            refretrain_model=refretrain_model,
            reforiginal_model=reforiginal_model,
            dd_eval_task_ids=self.dd_eval_tasks,
            ad_eval_task_ids=self.ad_eval_tasks,
        )
        pylogger.debug(
            "Evaluation module (clarena.utils.eval.CULEvaluation) instantiated!"
        )

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
        callbacks: list[Callback],
    ) -> None:
        r"""Instantiate the trainer object from `trainer_cfg` and `callbacks`."""

        pylogger.debug("Instantiating trainer (lightning.Trainer)...")
        self.trainer = hydra.utils.instantiate(
            trainer_cfg,
            callbacks=callbacks,
        )
        pylogger.debug("Trainer (lightning.Trainer) instantiated!")

    def set_global_seed(self, global_seed: int) -> None:
        r"""Set the `global_seed` for the entire evaluation."""
        L.seed_everything(self.global_seed, workers=True)
        pylogger.debug("Global seed is set as %d.", global_seed)

    def run(self) -> None:
        r"""The main method to run the continual unlearning full evaluation."""

        self.set_global_seed(self.global_seed)

        # components
        self.instantiate_cl_dataset(cl_dataset_cfg=self.cfg.cl_dataset)
        self.cl_dataset.set_cl_paradigm(cl_paradigm=self.cl_paradigm)
        self.instantiate_evaluation_module()
        self.instantiate_callbacks(
            metrics_cfg=self.cfg.metrics,
            callbacks_cfg=self.cfg.callbacks,
        )
        self.instantiate_trainer(
            trainer_cfg=self.cfg.trainer,
            callbacks=self.callbacks,
        )  # trainer should be instantiated after callbacks

        # setup tasks for dataset and evaluation module
        self.cl_dataset.setup_tasks_eval(
            eval_tasks=sorted(set(self.dd_eval_tasks + self.ad_eval_tasks))
        )

        # evaluation
        self.trainer.test(
            model=self.evaluation_module,
            datamodule=self.cl_dataset,
        )
