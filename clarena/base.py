r"""
The module for general CL bases.
"""

import logging

import hydra
import lightning as L
from lightning import Callback, LightningDataModule, LightningModule, Trainer
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig, ListConfig
from torch import nn
from torch.optim import Optimizer

from clarena.backbones import CLBackbone
from clarena.cl_algorithms import CLAlgorithm
from clarena.cl_datasets import CLDataset
from clarena.cl_heads import HeadsCIL, HeadsTIL

# always get logger for built-in logging in each module
pylogger = logging.getLogger(__name__)


class CLExperiment:
    r"""The base class for continual learning experiments."""

    def __init__(self, cfg: DictConfig) -> None:
        r"""Initializes the CL experiment object with a complete configuration.

        **Args:**
        - **cfg** (`DictConfig`): the complete config dict for the CL experiment.
        """
        self.cfg: DictConfig = cfg
        r"""Store the complete config dict for any future reference."""

        self.cl_paradigm: str = cfg.cl_paradigm
        r"""Store the continual learning paradigm, either 'TIL' (Task-Incremental Learning) or 'CIL' (Class-Incremental Learning). Parsed from config and used to instantiate the correct heads object and set up CL dataset."""
        self.num_tasks: int = cfg.num_tasks
        r"""Store the number of tasks to be conducted in this experiment. Parsed from config and used in the tasks loop."""
        self.global_seed: int = cfg.global_seed if cfg.get("global_seed") else None
        r"""Store the global seed for the entire experiment. Parsed from config and used to seed all random number generators."""
        self.test: bool = cfg.test
        r"""Store whether to test the model after training and validation. Parsed from config and used in the tasks loop."""
        self.output_dir_name: str = cfg.output_dir_name
        r"""Store the name of the output directory to store the logs and checkpoints. Parsed from config and help any output operation to locate the correct directory."""

        self.task_id: int
        r"""Task ID counter indicating which task is being processed. Self updated during the task loop."""

        self.cl_dataset: CLDataset
        r"""CL dataset object. Instantiate in `instantiate_cl_dataset()`."""
        self.backbone: CLBackbone
        r"""Backbone network object. Instantiate in `instantiate_backbone()`."""
        self.heads: HeadsTIL | HeadsCIL
        r"""CL output heads object. Instantiate in `instantiate_heads()`."""
        self.model: CLAlgorithm
        r"""CL model object. Instantiate in `instantiate_cl_algorithm()`."""

        self.optimizer: Optimizer
        r"""Optimizer object for current task `self.task_id`. Instantiate in `instantiate_optimizer()`."""
        self.trainer: Trainer
        r"""Trainer object for current task `self.task_id`. Instantiate in `instantiate_trainer()`."""
        self.lightning_loggers: list[Logger]
        r"""The list of initialised lightning loggers objects for current task `self.task_id`. Instantiate in `instantiate_lightning_loggers()`."""
        self.callbacks: list[Callback]
        r"""The list of initialised callbacks objects for current task `self.task_id`. Instantiate in `instantiate_callbacks()`."""

        CLExperiment.sanity_check(self)

    def sanity_check(self) -> None:
        r"""Check the sanity of the config dict `self.cfg`.

        **Raises:**
        - **KeyError**: when required fields in experiment config are missing, including `cl_paradigm`, `num_tasks`, `test`, `output_dir_name`.
        - **ValueError**: when the value of `cl_paradigm` is not 'TIL' or 'CIL', or when the number of tasks is larger than the number of tasks in the CL dataset.
        """
        if not self.cfg.get("cl_paradigm"):
            raise KeyError(
                "Field cl_paradigm should be specified in experiment config!"
            )

        if self.cfg.cl_paradigm not in ["TIL", "CIL"]:
            raise ValueError(
                f"Field cl_paradigm should be either 'TIL' or 'CIL' but got {self.cfg.cl_paradigm}!"
            )

        if not self.cfg.get("num_tasks"):
            raise KeyError("Field num_tasks should be specified in experiment config!")

        if not self.cfg.cl_dataset.get("num_tasks"):
            raise KeyError("Field num_tasks should be specified in cl_dataset config!")

        if not self.cfg.num_tasks <= self.cfg.cl_dataset.num_tasks:
            raise ValueError(
                f"The experiment is set to run {self.cfg.num_tasks} tasks whereas only {self.cfg.cl_dataset.num_tasks} exists in current cl_dataset setting!"
            )

        if not self.cfg.get("test"):
            raise KeyError("Field test should be specified in experiment config!")

        if not self.cfg.get("output_dir_name"):
            raise KeyError(
                "Field output_dir_name should be specified in experiment config!"
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
        self.cl_dataset: LightningDataModule = hydra.utils.instantiate(cl_dataset_cfg)
        pylogger.debug(
            "CL dataset <%s> (clarena.cl_datasets.CLDataset) instantiated!",
            cl_dataset_cfg.get("_target_"),
        )

    def instantiate_backbone(self, backbone_cfg: DictConfig) -> None:
        r"""Instantiate the CL backbone network object from backbone config.

        **Args:**
        - **backbone_cfg** (`DictConfig`): the backbone config dict.
        """
        pylogger.debug(
            "Instantiating backbone network <%s> (clarena.backbones.CLBackbone)...",
            backbone_cfg.get("_target_"),
        )
        self.backbone: nn.Module = hydra.utils.instantiate(backbone_cfg)
        pylogger.debug(
            "Backbone network <%s> (clarena.backbones.CLBackbone) instantiated!",
            backbone_cfg.get("_target_"),
        )

    def instantiate_heads(self, cl_paradigm: str, input_dim: int) -> None:
        r"""Instantiate the CL output heads object according to field `cl_paradigm` and backbone `output_dim` in the config.

        **Args:**
        - **cl_paradigm** (`str`): the CL paradigm, either 'TIL' or 'CIL'.
        - **input_dim** (`int`): the input dimension of the heads. Must be equal to the `output_dim` of the connected backbone.
        """
        pylogger.debug(
            "CL paradigm is set as %s. Instantiating %s heads (torch.nn.Module)...",
            cl_paradigm,
            cl_paradigm,
        )
        self.heads: HeadsTIL | HeadsCIL = (
            HeadsTIL(input_dim=input_dim)
            if cl_paradigm == "TIL"
            else HeadsCIL(input_dim=input_dim)
        )
        pylogger.debug("%s heads (torch.nn.Module) instantiated! ", cl_paradigm)

    def instantiate_cl_algorithm(self, cl_algorithm_cfg: DictConfig) -> None:
        r"""Instantiate the cl_algorithm object from cl_algorithm config.

        **Args:**
        - **cl_algorithm_cfg** (`DictConfig`): the cl_algorithm config dict.
        """
        pylogger.debug(
            "CL algorithm is set as <%s>. Instantiating <%s> (clarena.cl_algorithms.CLAlgorithm)...",
            cl_algorithm_cfg.get("_target_"),
            cl_algorithm_cfg.get("_target_"),
        )
        self.model: LightningModule = hydra.utils.instantiate(
            cl_algorithm_cfg,
            backbone=self.backbone,
            heads=self.heads,
        )
        pylogger.debug(
            "<%s> (clarena.cl_algorithms.CLAlgorithm) instantiated!",
            cl_algorithm_cfg.get("_target_"),
        )

    def instantiate_optimizer(
        self, optimizer_cfg: DictConfig | ListConfig, task_id: int
    ) -> None:
        r"""Instantiate the optimizer object for task `task_id` from optimizer config.

        **Args:**
        - **optimizer_cfg** (`DictConfig` or `ListConfig`): the optimizer config dict. If it's a `ListConfig`, it should contain optimizer config for each task; otherwise, it's an uniform optimizer config for all tasks.
        - **task_id** (`int`): the target task ID.
        """
        if isinstance(optimizer_cfg, ListConfig):
            pylogger.debug("Distinct optimizer config is applied to each task.")
            optimizer_cfg = optimizer_cfg[task_id - 1]
        elif isinstance(optimizer_cfg, DictConfig):
            pylogger.debug("Uniform optimizer config is applied to all tasks.")

            # partially instantiate optimizer as the 'params' argument is from Lightning Modules cannot be passed for now.
            pylogger.debug(
                "Partially instantiating optimizer <%s> (torch.optim.Optimizer) for task %d...",
                optimizer_cfg.get("_target_"),
                task_id,
            )
            self.optimizer: Optimizer = hydra.utils.instantiate(optimizer_cfg)
            pylogger.debug(
                "Optimizer <%s> (torch.optim.Optimizer) partially for task %d instantiated!",
                optimizer_cfg.get("_target_"),
                task_id,
            )

    def instantiate_trainer(self, trainer_cfg: DictConfig, task_id: int) -> None:
        r"""Instantiate the trainer object for task `task_id` from trainer config.

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
            trainer_cfg, callbacks=self.callbacks, logger=self.lightning_loggers
        )
        pylogger.debug(
            "Trainer <%s> (lightning.Trainer) for task %d instantiated!",
            trainer_cfg.get("_target_"),
            task_id,
        )

    def instantiate_lightning_loggers(
        self, lightning_loggers_cfg: DictConfig, task_id: int
    ) -> None:
        r"""Instantiate the list of lightning loggers objects for task `task_id` from lightning_loggers config.

        **Args:**
        - **lightning_loggers_cfg** (`DictConfig`): the lightning_loggers config dict. All tasks share the same lightning_loggers config but different objects.
        - **task_id** (`int`): the target task ID.
        """
        pylogger.debug(
            "Instantiating Lightning loggers (lightning.Logger) for task %d...", task_id
        )
        self.lightning_loggers: list[Logger] = [
            hydra.utils.instantiate(
                lightning_logger, version=f"task_{task_id}"
            )  # change the directory name to "task_" prefix in lightning logs
            for lightning_logger in lightning_loggers_cfg.values()
        ]
        pylogger.debug(
            "Lightning loggers (lightning.Logger) for task %d instantiated!", task_id
        )

    def instantiate_callbacks(self, callbacks_cfg: DictConfig, task_id: int) -> None:
        r"""Instantiate the list of callbacks objects for task `task_id` from callbacks config.

        **Args:**
        - **callbacks_cfg** (`DictConfig`): the callbacks config dict. All tasks share the same callbacks config but different objects.
        - **task_id** (`int`): the target task ID.
        """
        pylogger.debug(
            "Instantiating callbacks (lightning.Callback) for task %d...", task_id
        )
        self.callbacks: list[Callback] = [
            hydra.utils.instantiate(callback) for callback in callbacks_cfg.values()
        ]
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

    def instantiate_global(self) -> None:
        r"""Instantiate global components for the entire CL experiment from `self.cfg`."""

        self.instantiate_cl_dataset(self.cfg.cl_dataset)
        self.instantiate_backbone(self.cfg.backbone)
        self.instantiate_heads(self.cfg.cl_paradigm, self.cfg.backbone.output_dim)
        self.instantiate_cl_algorithm(
            self.cfg.cl_algorithm
        )  # cl_algorithm should be instantiated after backbone and heads

    def setup_global(self) -> None:
        r"""Let CL dataset know the CL paradigm to define its CL class map."""
        self.set_global_seed()
        self.cl_dataset.set_cl_paradigm(cl_paradigm=self.cl_paradigm)

    def instantiate_task_specific(self) -> None:
        r"""Instantiate task-specific components for the current task `self.task_id` from `self.cfg`."""

        self.instantiate_optimizer(self.cfg.optimizer, self.task_id)
        self.instantiate_callbacks(self.cfg.callbacks, self.task_id)
        self.instantiate_lightning_loggers(self.cfg.lightning_loggers, self.task_id)
        self.instantiate_trainer(
            self.cfg.trainer, self.task_id
        )  # trainer should be instantiated after loggers and callbacks

    def setup_task_specific(self) -> None:
        r"""Setup task-specific components to get ready for the current task `self.task_id`."""

        self.cl_dataset.setup_task_id(self.task_id)
        self.backbone.setup_task_id(self.task_id)
        self.model.setup_task_id(
            self.task_id,
            len(self.cl_dataset.cl_class_map(self.task_id)),
            self.optimizer,
        )

        pylogger.debug(
            "Datamodule, model and loggers are all set up ready for task %d!",
            self.task_id,
        )

    def run_task(self) -> None:
        r"""Fit the model on the current task `self.task_id`. Also test the model if `self.test` is set to True."""

        self.trainer.fit(
            model=self.model,
            datamodule=self.cl_dataset,
        )

        if self.test:
            # test after training and validation
            self.trainer.test(
                model=self.model,
                datamodule=self.cl_dataset,
            )

    def run(self) -> None:
        r"""The main method to run the continual learning experiment."""

        self.instantiate_global()
        self.setup_global()

        # task loop
        for task_id in range(1, self.num_tasks + 1):  # task ID counts from 1

            self.setup_task_id(task_id)
            self.instantiate_task_specific()
            self.setup_task_specific()

            self.run_task()
