r"""
The submodule in `experiments` for evaluating trained multi-task learning experiment.
"""

__all__ = ["MTLEval"]

import logging

import lightning as L
import torch
from lightning import Callback, Trainer
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig

from clarena.cl_datasets import CLDataset
from clarena.experiments import MTLTrain
from clarena.mtl_datasets import MTLDataset

# always get logger for built-in logging in each module
pylogger = logging.getLogger(__name__)


class MTLEval:
    r"""The base class for evaluating trained multi-task learning experiment.

    This runs evaluation on a trained multi-task learning model read from saved model file, without any training loop.
    """

    def __init__(
        self,
        cfg: DictConfig,
    ) -> None:
        r"""Initializes the MTL evaluation object with a evaluation configuration."""
        self.cfg: DictConfig = cfg
        r"""The complete config dict."""

        MTLEval.sanity_check(self)

        self.eval_tasks: list[int] = (
            cfg.eval_tasks
            if isinstance(cfg.eval_tasks, list)
            else list(range(1, cfg.eval_tasks + 1))
        )
        r"""Store the list of tasks to be evaluated. Parsed from config and used in the evaluation loop. """
        self.global_seed: int = cfg.global_seed if cfg.get("global_seed") else None
        r"""Store the global seed for the entire experiment. Parsed from config and used to seed all random number generators."""
        self.output_dir: str = cfg.output_dir
        r"""Store the output directory to store the logs and checkpoints. Parsed from config and help any output operation to locate the correct directory."""
        self.model_path: str = cfg.model_path
        r"""Store the model path to load the model from. Parsed from config and used to load the model for evaluation."""

        # components
        self.mtl_dataset: MTLDataset
        r"""MTL dataset object. Instantiate in `instantiate_mtl_dataset()`. One of `mtl_dataset` and `cl_dataset` must exist."""
        self.cl_dataset: CLDataset
        r"""CL dataset object to construct MTL dataset. One of `mtl_dataset` and `cl_dataset` must exist."""
        self.trainer: Trainer
        r"""Trainer object. Instantiate in `instantiate_trainer()`."""
        self.lightning_loggers: list[Logger]
        r"""The list of initialized lightning loggers objects for current task `self.task_id`. Instantiate in `instantiate_lightning_loggers()`."""
        self.callbacks: list[Callback]
        r"""The list of initialized callbacks objects for the evaluation. Instantiate in `instantiate_callbacks()`."""

    def sanity_check(self) -> None:
        r"""Check the sanity of the config dict `self.cfg`."""

        # check required config fields
        required_config_fields = [
            "model_path",
            "eval_tasks",
            "global_seed",
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
        MTLTrain.instantiate_mtl_dataset(self, mtl_dataset_cfg, cl_dataset_cfg)

    def instantiate_trainer(self, trainer_cfg: DictConfig) -> None:
        r"""Instantiate the trainer object from trainer config.

        **Args:**
        - **trainer_cfg** (`DictConfig`): the trainer config dict.
        """
        MTLTrain.instantiate_trainer(self, trainer_cfg)

    def instantiate_lightning_loggers(self, lightning_loggers_cfg: DictConfig) -> None:
        r"""Instantiate the list of lightning loggers objects from lightning_loggers config.

        **Args:**
        - **lightning_loggers_cfg** (`DictConfig`): the lightning_loggers config dict.
        """
        MTLTrain.instantiate_lightning_loggers(self, lightning_loggers_cfg)

    def instantiate_callbacks(
        self, metrics_cfg: DictConfig, callbacks_cfg: DictConfig
    ) -> None:
        r"""Instantiate the list of callbacks objects from metrics and other callbacks config.

        **Args:**
        - **metrics_cfg** (`DictConfig`): the metrics config dict.
        - **callbacks_cfg** (`DictConfig`): the callbacks config dict.
        """
        MTLTrain.instantiate_callbacks(self, metrics_cfg, callbacks_cfg)

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
            for task_id in self.eval_tasks:
                self.cl_dataset.setup_task_id(task_id)

        self.mtl_dataset.setup_tasks(
            train_tasks=None,
            eval_tasks=self.eval_tasks,
        )
        self.mtl_dataset.setup(stage="test")

    def run(self) -> None:
        r"""The main method to run the multi-task evaluation learning experiment."""

        self.instantiate()
        self.setup()

        model = torch.load(self.model_path)

        # evaluation skipping training and validation
        self.trainer.test(
            model=model,
            datamodule=self.mtl_dataset,
        )
