r"""
The submodule in `cl_algorithms` for continual learning algorithm bases.
"""

__all__ = ["CLAlgorithm", "UnlearnableCLAlgorithm", "AmnesiacCLAlgorithm"]

import logging
from copy import deepcopy
from typing import Any

from lightning import LightningModule
from torch import Tensor, nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from clarena.backbones import CLBackbone
from clarena.heads import HeadDIL, HeadsCIL, HeadsTIL

# always get logger for built-in logging in each module
pylogger = logging.getLogger(__name__)


class CLAlgorithm(LightningModule):
    r"""The base class of continual learning algorithms."""

    def __init__(
        self,
        backbone: CLBackbone,
        heads: HeadsTIL | HeadsCIL | HeadDIL,
        non_algorithmic_hparams: dict[str, Any] = {},
        **kwargs,
    ) -> None:
        r"""
        **Args:**
        - **backbone** (`CLBackbone`): backbone network.
        - **heads** (`HeadsTIL` | `HeadsCIL` | `HeadDIL`): output heads.
        - **non_algorithmic_hparams** (`dict[str, Any]`): non-algorithmic hyperparameters that are not related to the algorithm itself are passed to this `LightningModule` object from the config, such as optimizer and learning rate scheduler configurations. They are saved for Lightning APIs from `save_hyperparameters()` method. This is useful for the experiment configuration and reproducibility.
        - **kwargs**: Reserved for multiple inheritance.
        """
        super().__init__()
        self.save_hyperparameters(non_algorithmic_hparams)

        # components
        self.backbone: CLBackbone = backbone
        r"""The backbone network."""
        self.heads: HeadsTIL | HeadsCIL | HeadsCIL = heads
        r"""The output heads."""
        self.optimizer_t: Optimizer
        r"""Optimizer (partially initialized) for the current task `self.task_id`. Will be equipped with parameters in `configure_optimizers()`."""
        self.lr_scheduler_t: LRScheduler | None
        r"""Learning rate scheduler for the optimizer of the current task `self.task_id`. If `None`, no scheduler is used."""
        self.criterion = nn.CrossEntropyLoss()
        r"""Loss function between the output logits and the target labels. Default is cross-entropy loss."""

        self.if_forward_func_return_logits_only: bool = False
        r"""Whether the `forward()` method returns logits only. If `False`, it returns a dictionary containing logits and other information. Default is `False`."""

        # task ID control
        self.task_id: int
        r"""Task ID counter indicating which task is being processed. Self updated during the task loop. Valid from 1 to `cl_dataset.num_tasks`."""
        self.processed_task_ids: list[int] = []
        r"""Task IDs that have been processed."""

        CLAlgorithm.sanity_check(self)

    def sanity_check(self) -> None:
        r"""Sanity check."""

        # check backbone and heads compatibility
        if self.backbone.output_dim != self.heads.input_dim:
            raise ValueError(
                "The output_dim of the backbone must equal the input_dim of the CL heads."
            )

    def setup_task_id(
        self,
        task_id: int,
        num_classes: int,
        optimizer: Optimizer,
        lr_scheduler: LRScheduler | None,
    ) -> None:
        r"""Set up which task the CL experiment is on. This must be done before `forward()` method is called.

        **Args:**
        - **task_id** (`int`): the target task ID.
        - **num_classes** (`int`): the number of classes in the task.
        - **optimizer** (`Optimizer`): the optimizer object (partially initialized) for the task.
        - **lr_scheduler** (`LRScheduler` | `None`): the learning rate scheduler for the optimizer. If `None`, no scheduler is used.
        """
        self.task_id = task_id
        self.processed_task_ids.append(task_id)
        self.backbone.setup_task_id(task_id=task_id)
        if isinstance(self.heads, HeadsTIL) or isinstance(self.heads, HeadsCIL):
            self.heads.setup_task_id(task_id, num_classes)
        elif isinstance(self.heads, HeadDIL) and not self.heads.if_head_setup():
            self.heads.setup_task(num_classes)
        self.optimizer_t = optimizer
        self.lr_scheduler_t = lr_scheduler

    def get_test_task_id_from_dataloader_idx(self, dataloader_idx: int) -> int:
        r"""Get the test task ID from the dataloader index.

        **Args:**
        - **dataloader_idx** (`int`): the dataloader index.

        **Returns:**
        - **test_task_id** (`int`): the test task ID.
        """
        dataset_test = self.trainer.datamodule.dataset_test
        test_task_id = list(dataset_test.keys())[dataloader_idx]
        return test_task_id

    def set_forward_func_return_logits_only(
        self, forward_func_return_logits_only: bool
    ) -> None:
        r"""Set whether the `forward()` method returns logits only. This is useful for some CL algorithms that require the forward function to return logits only, such as FG-AdaHAT.

        **Args:**
        - **forward_func_return_logits_only** (`bool`): whether the `forward()` method returns logits only. If `False`, it returns a dictionary containing logits and other information.
        """
        self.if_forward_func_return_logits_only = forward_func_return_logits_only

    def preceding_layer(self, layer_name: str) -> nn.Module | None:
        r"""Get the preceding layer of the given layer (including backbone and output heads). If the given layer is the first layer, return `None`.

        **Args:**
        - **layer_name** (`str`): the name of the layer.

        **Returns:**
        - **preceding_layer** (`nn.Module` | `None`): the preceding layer.
        """

        if layer_name == "heads":
            backbone_last_layer_name = self.backbone.weighted_layer_names[-1]
            backbone_last_layer = self.backbone.get_layer_by_name(
                backbone_last_layer_name
            )
            return backbone_last_layer
        else:
            preceding_layer_name = self.backbone.preceding_layer_name(layer_name)
            preceding_layer = self.backbone.get_layer_by_name(preceding_layer_name)

        return preceding_layer

    def next_layer(self, layer_name: str) -> nn.Module | None:
        r"""Get the next layer of the given layer (including backbone and output heads). If the given layer is the last layer, return `None`.

        **Args:**
        - **layer_name** (`str`): the name of the layer.

        **Returns:**
        - **preceding_layer** (`nn.Module` | `None`): the next layer.
        """

        if layer_name == "heads":
            return None
        else:
            next_layer_name = self.backbone.next_layer_name(layer_name)
            if next_layer_name is not None:
                next_layer = self.backbone.get_layer_by_name(next_layer_name)
            else:
                next_layer = self.heads.get_head(self.task_id)

        return next_layer

    def forward(self, input: Tensor, stage: str, task_id: int | None = None) -> Tensor:
        r"""The forward pass for data from task `task_id`. Note that it is nothing to do with `forward()` method in `nn.Module`. This definition provides a template that many CL algorithm including the vanilla Finetuning algorithm use. It works both for TIL and CIL.

        **Args:**
        - **input** (`Tensor`): the input tensor from data.
        - **stage** (`str`): the stage of the forward pass; one of:
            1. 'train': training stage.
            2. 'validation': validation stage.
            3. 'test': testing stage.
        - **task_id** (`int`): the task ID where the data are from. If stage is 'train' or `validation`, it is usually from the current task `self.task_id`. If stage is 'test', it could be from any seen task. In TIL, the task IDs of test data are provided thus this argument can be used. In CIL, they are not provided, so it is just a placeholder for API consistence but never used, and best practices are not to provide this argument and leave it as the default value.

        **Returns:**
        - **logits** (`Tensor`): the output logits tensor.
        - **activations** (`dict[str, Tensor]`): the hidden features (after activation) in each weighted layer. Key (`str`) is the weighted layer name, value (`Tensor`) is the hidden feature tensor. This is used for the continual learning algorithms that need to use the hidden features for various purposes.
        """
        feature, activations = self.backbone(input, stage=stage, task_id=task_id)
        logits = self.heads(feature, task_id)
        return (
            logits if self.if_forward_func_return_logits_only else (logits, activations)
        )

    def configure_optimizers(self) -> Optimizer:
        r"""Configure optimizer hooks by Lightning. See [Lightning docs](https://lightning.ai/docs/pytorch/stable/common/lightning_module.html#configure-optimizers) for more details."""
        # finish partially initialized optimizer by specifying model parameters. The `parameters()` method of this `CLAlgorithm` (inherited from `LightningModule`) returns both backbone and heads parameters
        fully_initialized_optimizer = self.optimizer_t(params=self.parameters())

        if self.lr_scheduler_t:
            fully_initialized_lr_scheduler = self.lr_scheduler_t(
                optimizer=fully_initialized_optimizer
            )

            return {
                "optimizer": fully_initialized_optimizer,
                "lr_scheduler": {
                    "scheduler": fully_initialized_lr_scheduler,
                    "monitor": f"task_{self.task_id}/learning_curve/val/loss_cls",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }

        return {"optimizer": fully_initialized_optimizer}


class UnlearnableCLAlgorithm(CLAlgorithm):
    r"""The base class of unlearnable continual learning algorithms."""

    def __init__(
        self,
        backbone: CLBackbone,
        heads: HeadsTIL | HeadsCIL | HeadDIL,
        non_algorithmic_hparams: dict[str, Any] = {},
        disable_unlearning: bool = False,
        **kwargs,
    ) -> None:
        r"""
        **Args:**
        - **backbone** (`CLBackbone`): backbone network.
        - **heads** (`HeadsTIL` | `HeadsCIL` | `HeadDIL`): output heads.
        - **non_algorithmic_hparams** (`dict[str, Any]`): non-algorithmic hyperparameters that are not related to the algorithm itself are passed to this `LightningModule` object from the config, such as optimizer and learning rate scheduler configurations. They are saved for Lightning APIs from `save_hyperparameters()` method. This is useful for the experiment configuration and reproducibility.
        - **disable_unlearning** (`bool`): whether to disable unlearning. This is used in reference experiments following continual learning pipeline. Default is `False`.
        - **kwargs**: Reserved for multiple inheritance.
        """
        super().__init__(
            backbone=backbone,
            heads=heads,
            non_algorithmic_hparams=non_algorithmic_hparams,
            **kwargs,
        )

        self.disable_unlearning: bool = disable_unlearning
        r"""Whether to disable unlearning. This is used in reference experiments following continual learning pipeline."""

        if not self.disable_unlearning:
            self.unlearning_task_ids: list[int]
            r"""The list of task IDs that are requested to be unlearned after training `self.task_id`."""

            self.unlearned_task_ids: set[int] = set()
            r"""The list of task IDs that have been unlearned in the experiment."""

            self.unlearnable_task_ids: list[int]
            r"""The list of task IDs that are unlearnable at the current `self.task_id`."""

            self.task_ids_just_no_longer_unlearnable: list[int]
            r"""The list of task IDs that are just no longer unlearnable at the current `self.task_id`."""

            UnlearnableCLAlgorithm.sanity_check(self)

    def sanity_check(self) -> None:
        r"""Sanity check."""

    def aggregated_backbone_output(self, input: Tensor) -> Tensor:
        r"""Get the aggregated backbone output for the input data. All parts of backbones should be aggregated together.

        This output feature is used for measuring unlearning metrics, such as Distribution Distance (DD). An aggregated output involving every part of the backbone is needed to ensure the fairness of the metric.

        **Args:**
        - **input** (`Tensor`): the input tensor from data.

        **Returns:**
        - **output** (`Tensor`): the aggregated backbone output tensor.
        """
        feature = 0

        for i in self.processed_task_ids:
            feature_i = self.backbone(input, stage="unlearning_test")[0]
            feature += feature_i
        feature = feature / len(self.processed_task_ids)

        return feature


class AmnesiacCLAlgorithm(UnlearnableCLAlgorithm):
    r"""The base class of Amnesiac continual learning algorithms.

    The Amnesiac continual learning algorithm refers to the corresponding continual learning model thatd the Amnesiac continual unlearning algorithm requires. The Amnesiac continual unlearning algorithm refers to update deletion operation that directly delete the parameter updates during a task's training. This is inspired by [AmnesiacML](https://arxiv.org/abs/2010.10981) in machine unlearning. In detail, the task-wise parameter updates are stored:

    $$\theta_{l,ij}^{(t)} = \theta_{l,ij}^{(0)} + \sum_{\tau=1}^{t} \Delta \theta_{l,ij}^{(\tau)}$$

    To unlearn $u(t)$, delete these updates:

    $$\theta_{l,ij}^{(t-u(t))} = \theta_{l,ij}^{(t)} - \sum_{\tau\in u(t)}\Delta \theta_{l,ij}^{(\tau)}$$

    It is mainly used in AmnesaicHAT, but can also be used in constructing other vanilla baseline continual unlearning algorithms based on different continual learning algorithms.
    """

    def __init__(
        self,
        backbone: CLBackbone,
        heads: HeadsTIL | HeadsCIL | HeadDIL,
        non_algorithmic_hparams: dict[str, Any] = {},
        disable_unlearning: bool = False,
        **kwargs,
    ) -> None:
        r"""
        **Args:**
        - **backbone** (`CLBackbone`): backbone network.
        - **heads** (`HeadsTIL` | `HeadsCIL` | `HeadDIL`): output heads.
        - **non_algorithmic_hparams** (`dict[str, Any]`): non-algorithmic hyperparameters that are not related to the algorithm itself are passed to this `LightningModule` object from the config, such as optimizer and learning rate scheduler configurations. They are saved for Lightning APIs from `save_hyperparameters()` method. This is useful for the experiment configuration and reproducibility.
        - **disable_unlearning** (`bool`): whether to disable unlearning. This is used in reference experiments following continual learning pipeline. Default is `False`.
        - **kwargs**: Reserved for multiple inheritance.
        """
        super().__init__(
            backbone=backbone,
            heads=heads,
            non_algorithmic_hparams=non_algorithmic_hparams,
            disable_unlearning=disable_unlearning,
            **kwargs,
        )

        self.original_backbone_state_dict: dict[str, Tensor] = deepcopy(
            backbone.state_dict()
        )
        r"""Store the original backbone network state dict. It is a dict where keys are parameter names and values are the corresponding parameter update tensor for the layer. """

        self.original_heads_state_dict: dict[str, Tensor] = deepcopy(heads.state_dict())
        r"""Store the original heads state dict. It is a dict where keys are parameter names and values are the corresponding parameter update tensor for the head. """

        self.parameters_task_update: dict[int, dict[str, Tensor]] = {}
        r"""Store the parameters update in each task. Keys are task IDs and values are the corresponding parameters update tensor. Each tensor is a dict where keys are parameter names and values are the corresponding parameter update tensor for the layer. """

        self.parameters_task_update_heads: dict[int, dict[str, Tensor]] = {}
        r"""Store the heads parameters update in each task. Keys are task IDs and values are the corresponding parameters update tensor. Each tensor is a dict where keys are parameter names and values are the corresponding parameter update tensor for the head. """

        self.state_dict_task_start: dict[str, Tensor]
        r"""Store the backbone state dict at the start of training each task. """

        self.heads_state_dict_task_start: dict[str, Tensor]
        r"""Store the heads state dict at the start of training each task. """

    def _record_new_head_parameters(self) -> None:
        r"""Record the initial parameters for any newly created heads."""
        current_heads_state_dict = self.heads.state_dict()
        for param_name, param_tensor in current_heads_state_dict.items():
            if param_name not in self.original_heads_state_dict:
                self.original_heads_state_dict[param_name] = deepcopy(param_tensor)

    def setup_task_id(
        self,
        task_id: int,
        num_classes: int,
        optimizer: Optimizer,
        lr_scheduler: LRScheduler | None,
    ) -> None:
        r"""Set up which task the CL experiment is on. This must be done before `forward()` method is called."""
        super().setup_task_id(
            task_id=task_id,
            num_classes=num_classes,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
        )

        if not self.disable_unlearning:
            # record initial parameters of any newly created heads for later reconstruction
            self._record_new_head_parameters()

    def on_train_start(self):
        r"""Store the current state dict at the start of training."""
        super().on_train_start()

        if not self.disable_unlearning:
            self.state_dict_task_start = deepcopy(self.backbone.state_dict())
            self.heads_state_dict_task_start = deepcopy(self.heads.state_dict())

    def on_train_end(self):
        r"""Store the parameters update of a task at the end of its training."""
        super().on_train_end()

        if not self.disable_unlearning:
            current_state_dict = self.backbone.state_dict()
            parameters_task_t_update = {}

            # compute the parameters update for the current task
            for layer_name, current_param_tensor in current_state_dict.items():
                if layer_name.startswith("backup_backbones."):
                    continue
                parameters_task_t_update[layer_name] = (
                    current_param_tensor - self.state_dict_task_start[layer_name]
                )

            # store the parameters update for the current task
            self.parameters_task_update[self.task_id] = parameters_task_t_update

            # compute the heads parameters update for the current task
            current_heads_state_dict = self.heads.state_dict()
            parameters_task_t_update_heads = {}
            for param_name, current_param_tensor in current_heads_state_dict.items():
                if param_name not in self.heads_state_dict_task_start:
                    pylogger.warning(
                        "Head parameter %s was not found in task start state dict.",
                        param_name,
                    )
                    continue
                parameters_task_t_update_heads[param_name] = (
                    current_param_tensor - self.heads_state_dict_task_start[param_name]
                )

            # store the heads parameters update for the current task
            self.parameters_task_update_heads[self.task_id] = (
                parameters_task_t_update_heads
            )

    def construct_parameters_from_updates(self):
        r"""Delete the updates for unlearning tasks from the current parameters."""
        if not hasattr(self, "unlearning_task_ids") or not self.unlearning_task_ids:
            return

        updated_state_dict = deepcopy(self.backbone.state_dict())
        for task_id in self.unlearning_task_ids:
            param_update = self.parameters_task_update.get(task_id)
            if param_update is None:
                pylogger.warning(
                    "Attempted to delete update for task %d, but it was not found.",
                    task_id,
                )
                continue
            for layer_name, param_tensor in param_update.items():
                if layer_name in updated_state_dict:
                    updated_state_dict[layer_name] -= param_tensor
                else:
                    pylogger.warning(
                        "Backbone parameter %s was not found for task %d.",
                        layer_name,
                        task_id,
                    )

        self.backbone.load_state_dict(updated_state_dict, strict=False)

        updated_heads_state_dict = deepcopy(self.heads.state_dict())
        for task_id in self.unlearning_task_ids:
            param_update = self.parameters_task_update_heads.get(task_id)
            if param_update is None:
                continue
            for param_name, param_tensor in param_update.items():
                if param_name in updated_heads_state_dict:
                    updated_heads_state_dict[param_name] -= param_tensor
                else:
                    pylogger.warning(
                        "Head parameter %s was not found for task %d.",
                        param_name,
                        task_id,
                    )

        if updated_heads_state_dict:
            self.heads.load_state_dict(updated_heads_state_dict, strict=False)
