r"""
The module for general CL bases.
"""

import logging
from copy import deepcopy

import hydra
import lightning as L
from lightning import Callback, LightningDataModule, LightningModule, Trainer
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig, ListConfig, OmegaConf
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from clarena.backbones import CLBackbone
from clarena.cl_algorithms import CLAlgorithm
from clarena.cl_datasets import CLDataset
from clarena.cl_heads import HeadsCIL, HeadsTIL
from clarena.unlearning_algorithms import UnlearningAlgorithm

# always get logger for built-in logging in each module
pylogger = logging.getLogger(__name__)


class CLExperiment:
    r"""The base class for continual learning (CL) experiments."""

    def __init__(self, cfg: DictConfig) -> None:
        r"""Initializes the CL experiment object with a complete configuration.

        **Args:**
        - **cfg** (`DictConfig`): the complete config dict for the CL experiment.
        """

        self.original_cfg: DictConfig = deepcopy(cfg)
        r"""Store the original config dict for any future reference."""

        if cfg.callbacks.get("cul_metrics") and not isinstance(self, CULExperiment):
            OmegaConf.set_struct(cfg, False)
            cfg.callbacks.pop("cul_metrics")
            OmegaConf.set_struct(cfg, True)

        self.cfg: DictConfig = cfg
        r"""Store the complete config dict for any future reference."""

        CLExperiment.sanity_check(self)

        self.cl_paradigm: str = cfg.cl_paradigm
        r"""Store the continual learning paradigm, either 'TIL' (Task-Incremental Learning) or 'CIL' (Class-Incremental Learning). Parsed from config and used to instantiate the correct heads object and set up CL dataset."""
        self.num_tasks: int = cfg.num_tasks
        r"""Store the number of tasks to be conducted in this experiment. Parsed from config and used in the tasks loop."""
        self.global_seed: int = cfg.global_seed if cfg.get("global_seed") else None
        r"""Store the global seed for the entire experiment. Parsed from config and used to seed all random number generators."""
        self.test: bool = cfg.test
        r"""Store whether to test the model after training and validation. Parsed from config and used in the tasks loop."""
        self.output_dir: str = cfg.output_dir
        r"""Store the output directory to store the logs and checkpoints. Parsed from config and help any output operation to locate the correct directory."""

        self.task_id: int
        r"""Task ID counter indicating which task is being processed. Self updated during the task loop. Starting from 1. """
        self.seen_task_ids: list[int] = []
        r"""The list of task IDs that have been seen in the experiment."""

        self.cl_dataset: CLDataset
        r"""CL dataset object. Instantiate in `instantiate_cl_dataset()`."""
        self.backbone: CLBackbone
        r"""Backbone network object. Instantiate in `instantiate_backbone()`."""
        self.heads: HeadsTIL | HeadsCIL
        r"""CL output heads object. Instantiate in `instantiate_heads()`."""
        self.model: CLAlgorithm
        r"""CL model object. Instantiate in `instantiate_cl_algorithm()`."""

        self.optimizer: Optimizer
        r"""Optimizer object for current task `self.task_id`. Instantiate in `instantiate_optimizer_and_lr_scheduler()`."""
        self.lr_scheduler: LRScheduler
        r"""Learning rate scheduler object for current task `self.task_id`. Instantiate in `instantiate_optimizer_and_lr_scheduler()`."""
        self.trainer: Trainer
        r"""Trainer object for current task `self.task_id`. Instantiate in `instantiate_trainer()`."""
        self.lightning_loggers: list[Logger]
        r"""The list of initialised lightning loggers objects for current task `self.task_id`. Instantiate in `instantiate_lightning_loggers()`."""
        self.callbacks: list[Callback]
        r"""The list of initialised callbacks objects for current task `self.task_id`. Instantiate in `instantiate_callbacks()`."""

    def sanity_check(self) -> None:
        r"""Check the sanity of the config dict `self.cfg`.

        **Raises:**
        - **KeyError**: when required fields in experiment config are missing, including `cl_paradigm`, `num_tasks`, `test`, `output_dir`.
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

        if not self.cfg.get("test"):
            raise KeyError("Field test should be specified in experiment config!")

        if not self.cfg.get("output_dir"):
            raise KeyError("Field output_dir should be specified in experiment config!")

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

    def instantiate_optimizer_and_lr_scheduler(
        self,
        optimizer_cfg: DictConfig | ListConfig,
        lr_scheduler_cfg: DictConfig | ListConfig | None,
        task_id: int,
    ) -> None:
        r"""Instantiate the optimizer object for task `task_id` from optimizer config. If `lr_scheduler_cfg` is not `None`, instantiate the learning rate scheduler object as well.

        **Args:**
        - **optimizer_cfg** (`DictConfig` or `ListConfig`): the optimizer config dict. If it's a `ListConfig`, it should contain optimizer config for each task; otherwise, it's an uniform optimizer config for all tasks.
        - **lr_scheduler_cfg** (`DictConfig` or `ListConfig` or `None`): the learning rate scheduler config dict. If it's a `ListConfig`, it should contain learning rate scheduler config for each task; otherwise, it's an uniform learning rate scheduler config for all tasks. If `None`, no learning rate scheduler is used.
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

        if lr_scheduler_cfg:
            if isinstance(lr_scheduler_cfg, ListConfig):
                pylogger.debug(
                    "Distinct learning rate scheduler config is applied to each task."
                )
                lr_scheduler_cfg = lr_scheduler_cfg[task_id - 1]
            elif isinstance(lr_scheduler_cfg, DictConfig):
                pylogger.debug(
                    "Uniform learning rate scheduler config is applied to all tasks."
                )

            # partially instantiate learning rate scheduler as the 'optimizer' argument is from Lightning Modules cannot be passed for now.
            pylogger.debug(
                "Partially instantiating learning rate scheduler <%s> (torch.optim.lr_scheduler.LRScheduler) for task %d...",
                lr_scheduler_cfg.get("_target_"),
                task_id,
            )
            self.lr_scheduler: LRScheduler = hydra.utils.instantiate(lr_scheduler_cfg)
            pylogger.debug(
                "Learning rate scheduler <%s> (torch.optim.lr_scheduler.LRScheduler) partially for task %d instantiated!",
                lr_scheduler_cfg.get("_target_"),
                task_id,
            )

    def instantiate_trainer(
        self, trainer_cfg: DictConfig | ListConfig, task_id: int
    ) -> None:
        r"""Instantiate the trainer object for task `task_id` from trainer config.

        **Args:**
        - **trainer_cfg** (`DictConfig` or `ListConfig`): the trainer config dict. If it's a `ListConfig`, it should contain optimizer config for each task; otherwise, it's an uniform optimizer config for all tasks (but different objects).
        - **task_id** (`int`): the target task ID.
        """

        if isinstance(trainer_cfg, ListConfig):
            pylogger.debug("Distinct trainer config is applied to each task.")
            trainer_cfg = trainer_cfg[task_id - 1]
        elif isinstance(trainer_cfg, DictConfig):
            pylogger.debug("Uniform trainer config is applied to all tasks.")

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
        self.seen_task_ids.append(task_id)

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

        if self.cfg.get("lr_scheduler"):
            self.instantiate_optimizer_and_lr_scheduler(
                self.cfg.optimizer, self.cfg.lr_scheduler, self.task_id
            )
        else:
            self.instantiate_optimizer_and_lr_scheduler(
                self.cfg.optimizer, None, self.task_id
            )
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
            lr_scheduler=self.lr_scheduler if self.cfg.get("lr_scheduler") else None,
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

    def run(self, task_ids: list[int] | None = None) -> None:
        r"""The main method to run the continual learning experiment.

        **Args:**
        - **run_task_ids** (`list[int]` | `None`): the list of task IDs to be conducted in the experiment. If `None`, all tasks will be conducted.
        """

        self.instantiate_global()
        self.setup_global()

        run_task_ids = (
            task_ids if task_ids is not None else list(range(1, self.num_tasks + 1))
        )  # task ID counts from 1
        # task loop
        for task_id in run_task_ids:
            if task_id > self.num_tasks:
                pylogger.critical("Task ID %d is out of range! Skip it...", task_id)

            self.setup_task_id(task_id)
            self.instantiate_task_specific()
            self.setup_task_specific()

            self.run_task()


class CULExperiment(CLExperiment):
    r"""The base class for continual unlearning (CUL) experiments."""

    def __init__(self, cfg: DictConfig) -> None:
        r"""Initializes the CUL experiment object with a complete configuration.

        **Args:**
        - **cfg** (`DictConfig`): the complete config dict for the CUL experiment.
        """

        CLExperiment.__init__(self, cfg)

        if cfg.callbacks.get("cl_metrics") and isinstance(self, CULExperiment):
            OmegaConf.set_struct(cfg, False)
            cfg.callbacks.pop("cl_metrics")
            OmegaConf.set_struct(cfg, True)

        CULExperiment.sanity_check(self)

        self.unlearning_requests: dict[int, list[int]] = cfg.unlearning_requests
        r"""Store the unlearning requests for each task in the experiment. Keys are IDs of the tasks that request unlearning after their learning, and values are the list of the previous tasks to be unlearned. Parsed from config and used in the tasks loop."""

        self.unlearned_task_ids: set[int] = set()
        r"""Store the list of task IDs that have been unlearned in the experiment. Updated in the tasks loop when unlearning requests are made."""

        self.unlearning_algorithm: UnlearningAlgorithm
        r"""Continual unlearning algorithm object. Instantiate in `instantiate_unlearning_algorithm()`."""

        self.permanent_mark: dict[int, bool] = (
            cfg.permanent_mark
            if cfg.get("permanent_mark")
            else {t: True for t in range(1, self.num_tasks + 1)}
        )
        r"""Store whether a task is permanent for each task in the experiment. If a task is permanent, it will not be unlearned i.e. not shown in future unlearning requests. This applies to some unlearning algorithms that need to know whether a task is permanent. """

    def sanity_check(self) -> None:
        r"""Check the sanity of the config dict `self.cfg`.

        **Raises:**
        - **KeyError**: when required fields in experiment config are missing, including `unlearning_requests`, `unlearning_algorithm`.
        - **ValueError**: when the unlearning requests are not within the range of reasonable values.
        """

        if not self.cfg.get("unlearning_requests"):
            raise KeyError(
                "Field unlearning_requests should be specified in experiment config because this is a continual unlearning experiment!"
            )

        for task_id, unlearning_task_ids in self.cfg.unlearning_requests.items():
            if task_id not in range(1, self.num_tasks + 1):
                raise ValueError(
                    f"Task ID {task_id} in unlearning_requests is not within the range of the number of tasks in the experiment!"
                )
        if any(
            unlearning_task_id not in range(1, task_id + 1)
            for unlearning_task_id in unlearning_task_ids
        ):
            raise ValueError(
                f"Unlearning task IDs {unlearning_task_ids} for task {task_id} in unlearning_requests are not within the range till the current task!"
            )

        if not self.cfg.get("unlearning_algorithm"):
            raise KeyError(
                "Field unlearn.num_tasks should be specified in experiment config!"
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
        self.unlearning_algorithm: UnlearningAlgorithm = hydra.utils.instantiate(
            unlearning_algorithm_cfg,
            model=self.model,
        )
        pylogger.debug(
            "<%s> (clarena.unlearning_algorithms.UnlearningAlgorithm) instantiated!",
            unlearning_algorithm_cfg.get("_target_"),
        )

    def instantiate_global(self) -> None:
        r"""Instantiate global components for the entire CUL experiment from `self.cfg`."""
        CLExperiment.instantiate_global(self)
        self.instantiate_unlearning_algorithm(
            self.cfg.unlearning_algorithm
        )  # unlearning_algorithm should be instantiated after model

    def setup_task_specific(self):
        r"""Setup task-specific components to get ready for the current task `self.task_id`."""
        CLExperiment.setup_task_specific(self)

        cfg_unlearning_test_reference = deepcopy(self.original_cfg)
        cfg_unlearning_test_reference.output_dir += (
            "/unlearning_test_reference_experiment"
        )

        self.unlearning_algorithm.setup_task_id(
            task_id=self.task_id,
            unlearning_requests=self.unlearning_requests,
            cfg_unlearning_test_reference=cfg_unlearning_test_reference,
            if_permanent=self.permanent_mark[self.task_id],
        )

        pylogger.debug(
            "Unlearning algorithm is set up ready for task %d!",
            self.task_id,
        )

    def run_task(self) -> None:
        r"""Fit the model on the current task `self.task_id`. Also test the model if `self.test` is set to True. Unlearn the previous tasks if unlearning requests are made."""

        self.trainer.fit(
            model=self.model,
            datamodule=self.cl_dataset,
        )

        # unlearn
        self.unlearning_algorithm.unlearn()

        self.unlearning_algorithm.setup_test_task_id()

        if self.test:
            # test after training and validation
            self.trainer.test(
                model=self.model,
                datamodule=self.cl_dataset,
            )

        pylogger.debug(
            "Task %d is completed with unlearning requests %s!",
            self.task_id,
            self.unlearning_requests.get(self.task_id),
        )
