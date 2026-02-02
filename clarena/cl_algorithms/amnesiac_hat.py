r"""
The submodule in `cl_algorithms` for [AmnesiacHAT (Amnesiac Hard Attention to the Task)]() algorithm.
"""

__all__ = ["AmnesiacHAT"]

import logging
from copy import deepcopy
from typing import Any, Callable

import torch
from torch import Tensor
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torchvision.transforms import transforms

from clarena.backbones import AmnesiacHATBackbone
from clarena.cl_algorithms import AdaHAT, AmnesiacCLAlgorithm, DERpp
from clarena.heads import HeadsTIL, HeadDIL

# always get logger for built-in logging in each module
pylogger = logging.getLogger(__name__)


class AmnesiacHAT(AdaHAT, DERpp, AmnesiacCLAlgorithm):
    r"""AmnesiacHAT (Amnesiac Hard Attention to the Task) algorithm.

    A variant of [HAT (Hard Attention to the Task)](http://proceedings.mlr.press/v80/serra18a) enabling HAT with unlearning ability, based on the [AdaHAT (Adaptive Hard Attention to the Task)](https://link.springer.com/chapter/10.1007/978-3-031-70352-2_9) algorithm.

    This algorithm is paired with the `AmnesiacHATUnlearn` unlearning algorithm.

    We implement AmnesiacHAT as a subclass of `AdaHAT`, `DERpp` algorithm.
    """

    def __init__(
        self,
        backbone: AmnesiacHATBackbone,
        heads: HeadsTIL | HeadDIL,
        buffer_size: int,
        distillation_reg_factor: float,
        replay_ce_factor: float,
        adjustment_mode: str,
        adjustment_intensity: float,
        s_max: float,
        clamp_threshold: float,
        mask_sparsity_reg_factor: float,
        mask_sparsity_reg_mode: str = "original",
        task_embedding_init_mode: str = "N01",
        epsilon: float = 0.1,
        augmentation_transforms: Callable | transforms.Compose | None = None,
        non_algorithmic_hparams: dict[str, Any] = {},
        disable_unlearning: bool = False,
        **kwargs,
    ) -> None:
        r"""Initialize the AmnesiacHAT algorithm with the network.

        **Args:**
        - **backbone** (`AmnesiacHATBackbone`): must be a backbone network with AmnesiacHAT mechanism.
        - **heads** (`HeadsTIL` | `HeadDIL`): output heads. AmnesiacHAT algorithm only supports TIL (Task-Incremental Learning) and DIL (Domain-Incremental Learning).
        - **buffer_size** (`int`): the size of the memory buffer. For now we only support fixed size buffer.
        - **distillation_reg_factor** (`float`): hyperparameter, the distillation regularization factor. It controls the strength of preventing forgetting.
        - **replay_ce_factor** (`float`): hyperparameter, the classification loss factor for replayed samples, ($\beta$ in the [DER paper](https://arxiv.org/abs/2004.07211)). It also controls the strength of preventing forgetting.
        - **adjustment_mode** (`str`): the strategy of adjustment i.e. the mode of gradient clipping; one of:
            1. 'adahat': set the gradients of parameters linking to masked units to a soft adjustment rate in the original AdaHAT approach. This is the way that AdaHAT does, which allowes the part of network for previous tasks to be updated slightly. See equation (8) and (9) chapter 3.1 in the [AdaHAT paper](https://link.springer.com/chapter/10.1007/978-3-031-70352-2_9).
            2. 'adahat_no_sum': set the gradients of parameters linking to masked units to a soft adjustment rate in the original AdaHAT approach, but without considering the information of parameter importance i.e. summative mask. This is the way that one of the AdaHAT ablation study does. See chapter 4.3 in the [AdaHAT paper](https://link.springer.com/chapter/10.1007/978-3-031-70352-2_9).
            3. 'adahat_no_reg': set the gradients of parameters linking to masked units to a soft adjustment rate in the original AdaHAT approach, but without considering the information of network sparsity i.e. mask sparsity regularization value. This is the way that one of the AdaHAT ablation study does. See chapter 4.3 in the [AdaHAT paper](https://link.springer.com/chapter/10.1007/978-3-031-70352-2_9).
        - **adjustment_intensity** (`float`): hyperparameter, control the overall intensity of gradient adjustment. It's the $\alpha$ in equation (9) in the [AdaHAT paper](https://link.springer.com/chapter/10.1007/978-3-031-70352-2_9).
        - **s_max** (`float`): hyperparameter, the maximum scaling factor in the gate function. See chapter 2.4 "Hard Attention Training" in the [HAT paper](http://proceedings.mlr.press/v80/serra18a).
        - **clamp_threshold** (`float`): the threshold for task embedding gradient compensation. See chapter 2.5 "Embedding Gradient compensation" in the [HAT paper](http://proceedings.mlr.press/v80/serra18a).
        - **mask_sparsity_reg_factor** (`float`): hyperparameter, the regularization factor for mask sparsity.
        - **mask_sparsity_reg_mode** (`str`): the mode of mask sparsity regularization; one of:
            1. 'original' (default): the original mask sparsity regularization in HAT paper.
            2. 'cross': the cross version mask sparsity regularization.
        - **task_embedding_init_mode** (`str`): the initialization method for task embeddings; one of:
            1. 'N01' (default): standard normal distribution $N(0, 1)$.
            2. 'U-11': uniform distribution $U(-1, 1)$.
            3. 'U01': uniform distribution $U(0, 1)$.
            4. 'U-10': uniform distribution $U(-1, 0)$.
            5. 'last': inherit task embedding from last task.
        - **epsilon** (`float`): the value added to network sparsity to avoid division by zero appeared in equation (9) in the [AdaHAT paper](https://link.springer.com/chapter/10.1007/978-3-031-70352-2_9).
        - **augmentation_transforms** (`transform` or `transforms.Compose` or `None`): the transforms to apply for augmentation after replay sampling. Not to confuse with the data transforms applied to the input of training data. Can be a single transform, composed transforms, or no transform.
        - **non_algorithmic_hparams** (`dict[str, Any]`): non-algorithmic hyperparameters that are not related to the algorithm itself are passed to this `LightningModule` object from the config, such as optimizer and learning rate scheduler configurations. They are saved for Lightning APIs from `save_hyperparameters()` method. This is useful for the experiment configuration and reproducibility.
        - **disable_unlearning** (`bool`): whether to disable unlearning. This is used in reference experiments following continual learning pipeline. Default is `False`.
        - **kwargs**: Reserved for multiple inheritance.
        """
        super().__init__(
            backbone=backbone,
            heads=heads,
            buffer_size=buffer_size,
            distillation_reg_factor=distillation_reg_factor,
            replay_ce_factor=replay_ce_factor,
            adjustment_mode=adjustment_mode,
            adjustment_intensity=adjustment_intensity,
            s_max=s_max,
            clamp_threshold=clamp_threshold,
            mask_sparsity_reg_factor=mask_sparsity_reg_factor,
            mask_sparsity_reg_mode=mask_sparsity_reg_mode,
            task_embedding_init_mode=task_embedding_init_mode,
            epsilon=epsilon,
            augmentation_transforms=augmentation_transforms,
            non_algorithmic_hparams=non_algorithmic_hparams,
            disable_unlearning=disable_unlearning,
            **kwargs,
        )

        # no additional hyperparameters to save

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
        super().setup_task_id(
            task_id=task_id,
            num_classes=num_classes,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
        )

        if not self.disable_unlearning:
            backup_task_ids = [
                tid for tid in self.unlearnable_task_ids if tid != self.task_id
            ]  # exclude current task, as we don't need backup backbone for current task

            # instantiate backup backbones for unlearning tasks
            self.backbone.instantiate_backup_backbones(backup_task_ids=backup_task_ids)

            # initialize backup backbones for unlearning tasks with the parameters just before the training of each backup task
            self.initialize_backup_backbones()

    def initialize_backup_backbones(
        self,
    ) -> None:
        r"""Initialize backup backbones for unlearning tasks with the parameters just before the training of each backup task."""

        for backup_task_id in self.backbone.backup_backbones:
            backup_task_id = int(backup_task_id)

            # for each backup backbone, compute the init state dict
            state_dict_before_the_backup_task = deepcopy(
                self.original_backbone_state_dict
            )

            # construct parameters saved before the training of each backup task
            tasks_before_backup = []
            for task_id in range(1, backup_task_id):
                # tasks before the backup task
                if (
                    task_id in self.parameters_task_update
                ):  # only consider tasks that are not unlearned
                    tasks_before_backup.append(task_id)

                    # add the parameter updates from this task
                    param_update = self.parameters_task_update[task_id]
                    for layer_name, param_tensor in param_update.items():
                        if layer_name in state_dict_before_the_backup_task:
                            target_tensor = state_dict_before_the_backup_task[
                                layer_name
                            ]
                            if (
                                param_tensor.device != target_tensor.device
                                or param_tensor.dtype != target_tensor.dtype
                            ):
                                param_tensor = param_tensor.to(
                                    device=target_tensor.device,
                                    dtype=target_tensor.dtype,
                                )
                            state_dict_before_the_backup_task[
                                layer_name
                            ] += param_tensor

            backup_backbone = self.backbone.backup_backbones[f"{backup_task_id}"]
            backup_backbone.load_state_dict(
                state_dict_before_the_backup_task, strict=False
            )

            pylogger.info(
                "For backuping %s for %s, the backup backbone is initialized as original backbone ($\theta_0$) plus accumulated parameter updates from tasks %s (before the backup task %s).",
                backup_task_id,
                self.task_id,
                tasks_before_backup,
                backup_task_id,
            )

    def clip_grad_by_adjustment_in_replay_repairing(
        self,
        summative_mask_for_previous_tasks_in_replay_repairing: dict[str, Tensor],
    ) -> None:
        r"""In the unlearning replay repairing phase, if the fixing strategy is set as 'sequential', a mini continual repairing is performed. We also use AdaHAT where gradients are clipped by the adjustment rate. Note that we use the adjustment mode 'adahat_no_reg', since mask sparsity regularization makes no sense in this phase.

        Note that because the task embedding fully covers every layer in the backbone network, no parameters are left out of this system. This applies not only to parameters between layers with task embeddings, but also to those before the first layer. We design it separately in the code.

        **Args:**
        - **summative_mask_for_previous_tasks_in_replay_repairing** (`dict[str, Tensor]`): the summative mask for previous tasks used in unlearning replay repairing phase. Keys are layer names and values are the corresponding summative mask tensor for the layer.
        """

        # initialize network capacity metric
        adjustment_rate_weight = {}
        adjustment_rate_bias = {}

        # calculate the adjustment rate for gradients of the parameters, both weights and biases (if they exist). See Eq. (9) in the [AdaHAT paper](https://link.springer.com/chapter/10.1007/978-3-031-70352-2_9)
        for layer_name in self.backbone.weighted_layer_names:

            layer = self.backbone.get_layer_by_name(
                layer_name
            )  # get the layer by its name

            # placeholder for the adjustment rate to avoid the error of using it before assignment
            adjustment_rate_weight_layer = 1
            adjustment_rate_bias_layer = 1

            weight_importance, bias_importance = (
                self.backbone.get_layer_measure_parameter_wise(
                    neuron_wise_measure=summative_mask_for_previous_tasks_in_replay_repairing,
                    layer_name=layer_name,
                    aggregation_mode="min",
                )
            )  # AdaHAT depends on parameter importance rather than parameter masks (as in HAT)

            r_layer = self.adjustment_intensity / (
                self.epsilon + 0.0
            )  # adahat_no_reg does not consider network sparsity
            adjustment_rate_weight_layer = torch.div(
                r_layer, (weight_importance + r_layer)
            )
            adjustment_rate_bias_layer = torch.div(r_layer, (bias_importance + r_layer))

            # apply the adjustment rate to the gradients
            layer.weight.grad.data *= adjustment_rate_weight_layer
            if layer.bias is not None:
                layer.bias.grad.data *= adjustment_rate_bias_layer

            # store the adjustment rate for logging
            adjustment_rate_weight[layer_name] = adjustment_rate_weight_layer
            if layer.bias is not None:
                adjustment_rate_bias[layer_name] = adjustment_rate_bias_layer

    def affected_tasks_after_unlearning(self) -> list[int]:
        r"""Get the list of task IDs that are affected after unlearning the requested tasks `self.unlearning_task_ids`.

        **Returns:**
        - **affected_task_ids** (`list[int]`): the list of task IDs that are affected after unlearning the requested tasks `self.unlearning_task_ids`. In AmnesiacHAT, all tasks after the unlearning tasks are affected.
        """
        affected_task_ids: list[int] = []

        min_unlearning_task_id = min(self.unlearning_task_ids)

        for tid in range(min_unlearning_task_id, self.task_id + 1):
            if (
                tid not in self.unlearning_task_ids
                and tid not in self.unlearned_task_ids
            ):
                affected_task_ids.append(tid)

        return affected_task_ids

    def on_train_start(self):
        r"""Initialize backup backbones at the start of training."""
        AdaHAT.on_train_start(self)  # Explicitly call AdaHAT's on_train_start
        DERpp.on_train_start(self)  # Explicitly call DER++'s on_train_start
        AmnesiacCLAlgorithm.on_train_start(self)

    def forward(
        self,
        input: torch.Tensor,
        stage: str,
        task_id: int | None = None,
        batch_idx: int | None = None,
        num_batches: int | None = None,
    ):
        r"""The forward pass for data from task `task_id`. Note that it is nothing to do with `forward()` method in `nn.Module`.

        **Args:**
        - **input** (`Tensor`): The input tensor from data.
        - **stage** (`str`): the stage of the forward pass; one of:
            1. 'train_with_backup': training stage with training backup backbones for future unlearning backup compensation.
            2. 'train': training stage.
            3. 'validation': validation stage.
            4. 'test': testing stage.
            5. 'unlearning_test': unlearning testing stage.
        - **task_id** (`int`| `None`): the task ID where the data are from. If the stage is 'train' or 'validation', it should be the current task `self.task_id`. If stage is 'test', it could be from any seen task. In TIL, the task IDs of test data are provided thus this argument can be used. HAT algorithm works only for TIL.
        - **batch_idx** (`int` | `None`): the current batch index. Applies only to training stage. For other stages, it is default `None`.
        - **num_batches** (`int` | `None`): the total number of batches. Applies only to training stage. For other stages, it is default `None`.

        **Returns:**
        - **logits** (`Tensor`): the output logits tensor.
        - **backup_logits** (`dict[int, Tensor]`): the output logits tensor from backup backbones for unlearning tasks. Key (`int`) is backup task ID, value (`Tensor`) is the output logits tensor. This is returned only when `stage` is 'train' and unlearning is not disabled.
        - **mask** (`dict[str, Tensor]`): the mask for the current task. Key (`str`) is layer name, value (`Tensor`) is the mask tensor. The mask tensor has size (number of units, ).
        - **activations** (`dict[str, Tensor]`): the hidden features (after activation) in each weighted layer. Key (`str`) is the weighted layer name, value (`Tensor`) is the hidden feature tensor. This is used for the continual learning algorithms that need to use the hidden features for various purposes. Although HAT algorithm does not need this, it is still provided for API consistence for other HAT-based algorithms inherited this `forward()` method of `HAT` class.
        """

        if stage == "train_with_backup":
            feature, backup_feature, mask, activations = self.backbone(
                input,
                stage="train_with_backup",
                s_max=self.s_max,
                batch_idx=batch_idx,
                num_batches=num_batches,
                test_task_id=None,
            )
        elif stage == "train":
            feature, mask, activations = self.backbone(
                input,
                stage="train",
                s_max=self.s_max,
                batch_idx=batch_idx,
                num_batches=num_batches,
                test_task_id=None,
            )
        elif stage == "validation":
            feature, mask, activations = self.backbone(
                input,
                stage="validation",
                s_max=self.s_max,
                batch_idx=None,
                num_batches=None,
                test_task_id=None,
            )
        elif stage == "test":
            feature, mask, activations = self.backbone(
                input,
                stage="test",
                s_max=None,
                batch_idx=None,
                num_batches=None,
                test_task_id=task_id,
            )
        else:
            raise ValueError(f"Invalid stage '{stage}' for forward pass.")

        logits = self.heads(feature, task_id)
        if stage == "train_with_backup":
            backup_logits = {
                backup_task_id: self.heads(backup_feature[backup_task_id], task_id)
                for backup_task_id in backup_feature
            }
            return (
                (logits, backup_logits)
                if self.if_forward_func_return_logits_only
                else (logits, backup_logits, mask, activations)
            )
        else:
            return (
                logits
                if self.if_forward_func_return_logits_only
                else (logits, mask, activations)
            )

    def training_step(self, batch: Any, batch_idx: int) -> dict[str, Tensor]:
        r"""Training step for current task `self.task_id`.

        **Args:**
        - **batch** (`Any`): a batch of training data.
        - **batch_idx** (`int`): the index of the batch. Used for calculating annealed scalar in HAT. See Sec. 2.4 "Hard Attention Training" in the [HAT paper](http://proceedings.mlr.press/v80/serra18a).

        **Returns:**
        - **outputs** (`dict[str, Tensor]`): a dictionary containing loss and other metrics from this training step. Keys (`str`) are metric names, and values (`Tensor`) are the metrics. Must include the key 'loss' (total loss) in the case of automatic optimization, according to PyTorch Lightning. For HAT, it includes 'mask' and 'capacity' for logging.
        """
        x, y = batch
        batch_size = len(y)

        # zero the gradients before forward pass in manual optimization mode
        opt = self.optimizers()
        opt.zero_grad()

        # classification loss
        num_batches = self.trainer.num_training_batches

        if not self.disable_unlearning:
            logits, backup_logits, mask, activations = self.forward(
                x,
                stage="train_with_backup",
                batch_idx=batch_idx,
                num_batches=num_batches,
                task_id=self.task_id,
            )
        else:
            logits, mask, activations = self.forward(
                x,
                stage="train",
                batch_idx=batch_idx,
                num_batches=num_batches,
                task_id=self.task_id,
            )

        loss_cls = self.criterion(logits, y)

        if not self.disable_unlearning:

            backup_loss_cls = {
                backup_task_id: self.criterion(backup_logit, y)
                for backup_task_id, backup_logit in backup_logits.items()
            }
            backup_task_num = len(backup_loss_cls)

            backup_loss_cls_total = (
                sum(backup_loss_cls.values()) / backup_task_num
                if backup_task_num > 0
                else 0.0
            )

        else:
            backup_loss_cls_total = 0.0

        # regularization loss. See Sec. 2.6 "Promoting Low Capacity Usage" in the [HAT paper](http://proceedings.mlr.press/v80/serra18a)
        hat_mask_sparsity_reg, network_sparsity = self.mark_sparsity_reg(
            mask, self.cumulative_mask_for_previous_tasks
        )

        derpp_reg = self.compute_distillation_and_replay_ce_reg(
            backbone=self.backbone, batch_size=batch_size
        )

        # total loss. See Eq. (4) in Sec. 2.6 "Promoting Low Capacity Usage" in the [HAT paper](http://proceedings.mlr.press/v80/serra18a)
        loss = (
            loss_cls + backup_loss_cls_total + hat_mask_sparsity_reg + derpp_reg
            if not self.disable_unlearning
            else loss_cls + hat_mask_sparsity_reg + derpp_reg
        )

        # backward step (manually)
        self.manual_backward(loss)  # calculate the gradients
        # HAT hard-clips gradients using the cumulative masks. See Eq. (2) in Sec. 2.3 "Network Training" in the HAT paper.
        # Network capacity is computed along with this process (defined as the average adjustment rate over all parameters; see Sec. 4.1 in the [AdaHAT paper](https://link.springer.com/chapter/10.1007/978-3-031-70352-2_9)).

        adjustment_rate_weight, adjustment_rate_bias, capacity = (
            self.clip_grad_by_adjustment(
                network_sparsity=network_sparsity,  # passed for compatibility with AdaHAT, which inherits this method
            )
        )

        # compensate the gradients of task embedding. See Sec. 2.5 "Embedding Gradient Compensation" in the [HAT paper](http://proceedings.mlr.press/v80/serra18a)
        self.compensate_task_embedding_gradients(
            batch_idx=batch_idx,
            num_batches=num_batches,
        )
        # update parameters with the modified gradients
        opt.step()

        # predicted labels
        preds = logits.argmax(dim=1)

        # accuracy of the batch
        acc = (preds == y).float().mean()

        # add current batch to memory buffer
        self.memory_buffer.add_data(
            x, y, logits.detach(), torch.full_like(y, self.task_id)
        )

        return {
            "preds": preds,
            "loss": loss,  # return loss is essential for training step, or backpropagation will fail
            "loss_cls": loss_cls,
            "backup_loss_cls_total": backup_loss_cls_total,
            "hat_mask_sparsity_reg": hat_mask_sparsity_reg,
            "derpp_reg": derpp_reg,
            "acc": acc,
            "activations": activations,
            "logits": logits,
            "mask": mask,  # return other metrics for lightning loggers callback to handle at `on_train_batch_end()`
            "input": x,  # return the input batch for Captum to use
            "target": y,  # return the target batch for Captum to use
            "adjustment_rate_weight": adjustment_rate_weight,  # return the adjustment rate for weights and biases for logging
            "adjustment_rate_bias": adjustment_rate_bias,
            "capacity": capacity,  # return the network capacity for logging
        }

    def on_train_end(self):
        r"""Store the parameters update of a task at the end of its training."""
        AdaHAT.on_train_end(self)  # Explicitly call AdaHAT's on_train_end
        DERpp.on_train_end(self)  # Explicitly call DER++'s on_train_end
        AmnesiacCLAlgorithm.on_train_end(self)

        if not self.disable_unlearning:
            # override cumulative mask to exclude unlearned tasks
            cumulated_masks = (
                [
                    mask
                    for tid, mask in self.backbone.masks.items()
                    if tid not in self.unlearned_task_ids
                ]
                if not self.disable_unlearning
                else [mask for tid, mask in self.backbone.masks.items()]
            )

            self.cumulative_mask_for_previous_tasks = (
                self.backbone.combine_masks(cumulated_masks, mode="union")
                if cumulated_masks
                else {
                    layer_name: torch.zeros(
                        self.backbone.get_layer_by_name(layer_name).weight.shape[0]
                    )
                    for layer_name in self.backbone.weighted_layer_names
                }
            )

            # save backup backbone state dict for future unlearning backup compensation
            for backup_task_id in self.backbone.backup_backbones:
                backup_task_id = int(backup_task_id)
                self.backbone.backup_state_dicts[(backup_task_id, self.task_id)] = (
                    deepcopy(
                        self.backbone.backup_backbones[f"{backup_task_id}"].state_dict()
                    )
                )
                pylogger.info(
                    "Backup backbone state dict (backuping task %d) for task %d has been saved for future unlearning tasks.",
                    backup_task_id,
                    self.task_id,
                )
