r"""
The submodule in `cl_algorithms` for [AmnesiacHAT (Amnesiac Hard Attention to the Task)]() algorithm.
"""

__all__ = ["AmnesiacHAT"]

import logging
import math
from copy import deepcopy
from typing import Any, Callable

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torchvision.transforms import transforms

from clarena.backbones import HATMaskBackbone
from clarena.backbones.base import CLBackbone
from clarena.cl_algorithms import DER, AdaHAT, UnlearnableCLAlgorithm
from clarena.heads import HeadsTIL

# always get logger for built-in logging in each module
pylogger = logging.getLogger(__name__)


class AmnesiacHAT(AdaHAT, DER, UnlearnableCLAlgorithm):
    r"""AmnesiacHAT (Amnesiac Hard Attention to the Task) algorithm.

    A variant of [HAT (Hard Attention to the Task)](http://proceedings.mlr.press/v80/serra18a) enabling HAT with unlearning ability, based on the [AdaHAT (Adaptive Hard Attention to the Task)](https://link.springer.com/chapter/10.1007/978-3-031-70352-2_9) algorithm.

    This algorithm is paired with the `AmnesiacHATUnlearn` unlearning algorithm.

    We implement AmnesiacHAT as a subclass of `AdaHAT` algorithm.
    """

    def __init__(
        self,
        backbone: HATMaskBackbone,
        heads: HeadsTIL,
        buffer_size: int,
        distillation_reg_factor: float,
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
        **kwargs,
    ) -> None:
        r"""Initialize the AmnesiacHAT algorithm with the network.

        **Args:**
        - **backbone** (`HATMaskBackbone`): must be a backbone network with HAT mask mechanism.
        - **heads** (`HeadsTIL`): output heads. AmnesiacHAT algorithm only supports TIL (Task-Incremental Learning).
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
        - **buffer_size** (`int`): the size of the memory buffer. For now we only support fixed size buffer.
        - **distillation_reg_factor** (`float`): hyperparameter, the distillation regularization factor. It controls the strength of preventing forgetting.
        - **augmentation_transforms** (`transform` or `transforms.Compose` or `None`): the transforms to apply for augmentation after replay sampling. Not to confuse with the data transforms applied to the input of training data. Can be a single transform, composed transforms, or no transform.
        - **non_algorithmic_hparams** (`dict[str, Any]`): non-algorithmic hyperparameters that are not related to the algorithm itself are passed to this `LightningModule` object from the config, such as optimizer and learning rate scheduler configurations. They are saved for Lightning APIs from `save_hyperparameters()` method. This is useful for the experiment configuration and reproducibility.
        - **kwargs**: Reserved for multiple inheritance.
        """
        super().__init__(
            backbone=backbone,
            heads=heads,
            buffer_size=buffer_size,
            distillation_reg_factor=distillation_reg_factor,
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
            **kwargs,
        )

        # save additional hyperparameters
        self.save_hyperparameters(
            "adjustment_intensity",
            "epsilon",
        )

        self.backup_backbone: CLBackbone = deepcopy(backbone)
        r"""A backup of the backbone network parallelly trained with the main backbone. Used for fixing holes caused by unlearning."""

        self.original_backbone_state_dict: dict[str, Tensor] = deepcopy(
            backbone.state_dict()
        )
        r"""Store the original backbone network state dict. It is a dict where keys are parameter names and values are the corresponding parameter update tensor for the layer. """

        self.parameters_task_update: dict[int, dict[str, Tensor]] = {}
        r"""Store the parameters update in each task. Keys are task IDs and values are the corresponding parameters update tensor. Each tensor is a dict where keys are parameter names and values are the corresponding parameter update tensor for the layer. """

        self.state_dict_task_start: dict[str, Tensor]
        r"""Store the backbone state dict at the start of training each task. """

    def clip_grad_by_unlearning_mask(
        self,
        unlearning_mask: dict[int, Tensor],
    ) -> tuple[dict[str, Tensor], dict[str, Tensor], Tensor]:
        r"""Clip the gradients that are not masked by unlearning tasks.

        **Args:**
        - **unlearning_mask** (`dict[int, Tensor]`): the union unlearning mask for the unlearning tasks. Keys are layer names and values are the corresponding unlearning mask tensor for the layer.

        Note that because the task embedding fully covers every layer in the backbone network, no parameters are left out of this system.
        This applies not only to parameters between layers with task embeddings, but also to those before the first layer. We design it separately in the code.
        """

        for layer_name in self.backbone.weighted_layer_names:

            layer = self.backbone.get_layer_by_name(
                layer_name
            )  # get the layer by its name

            weight_mask, bias_mask = self.backbone.get_layer_measure_parameter_wise(
                neuron_wise_measure=unlearning_mask,
                layer_name=layer_name,
                aggregation_mode="min",
            )

            # apply the adjustment rate to the gradients
            layer.weight.grad.data *= weight_mask
            if layer.bias is not None:
                layer.bias.grad.data *= bias_mask

    def affected_tasks_upon_unlearning(self) -> list[int]:
        r"""Get the list of task IDs that are affected upon unlearning the requested tasks in `self.unlearning_task_ids`.

        **Returns:**
        - **affected_task_ids** (`list[int]`): the list of task IDs that are affected upon unlearning the requested tasks in `self.unlearning_task_ids`. In AmnesiacHAT, all tasks after the unlearning tasks are affected.
        """
        affected_task_ids = [
            t_id
            for t_id in range(1, self.task_id + 1)
            if t_id > min(self.unlearning_task_ids)
        ]
        return affected_task_ids

    def on_train_start(self):
        r"""Store the current state dict at the start of training."""
        super().on_train_start()
        self.state_dict_task_start = deepcopy(self.backbone.state_dict())

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
        logits, mask, activations = self.forward(
            x,
            stage="train",
            batch_idx=batch_idx,
            num_batches=num_batches,
            task_id=self.task_id,
        )
        loss_cls = self.criterion(logits, y)

        # regularization loss. See Sec. 2.6 "Promoting Low Capacity Usage" in the [HAT paper](http://proceedings.mlr.press/v80/serra18a)
        loss_reg, network_sparsity = self.mark_sparsity_reg(
            mask, self.cumulative_mask_for_previous_tasks
        )

        previous_task_ids = [
            tid for tid in self.processed_task_ids if tid != self.task_id
        ]

        # Replay distillation regularization loss.
        if not self.memory_buffer.is_empty() and previous_task_ids != []:

            # sample from memory buffer with the same batch size as current batch
            x_replay, _, logits_replay, task_labels_replay = (
                self.memory_buffer.get_data(
                    size=batch_size,
                    included_tasks=previous_task_ids,
                )
            )

            # apply augmentation transforms if any
            if self.augmentation_transforms:
                x_replay = self.augmentation_transforms(x_replay)

            # get the student logits for this batch using the current model

            student_feature_replay = torch.cat(
                [
                    self.backbone(
                        x_replay[i].unsqueeze(0), stage="test", test_task_id=tid.item()
                    )[0]
                    for i, tid in enumerate(task_labels_replay)
                ]
            )
            student_logits_replay = torch.cat(
                [
                    self.heads(student_feature_replay[i].unsqueeze(0), task_id=tid)
                    for i, tid in enumerate(task_labels_replay)
                ]
            )
            with torch.no_grad():  # stop updating the previous heads

                teacher_logits_replay = logits_replay

            loss_reg += self.distillation_reg(
                student_logits=student_logits_replay,
                teacher_logits=teacher_logits_replay,
            )

        # total loss. See Eq. (4) in Sec. 2.6 "Promoting Low Capacity Usage" in the [HAT paper](http://proceedings.mlr.press/v80/serra18a)
        loss = loss_cls + loss_reg

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
            "loss_reg": loss_reg,
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
        super().on_train_end()

        current_state_dict = self.backbone.state_dict()
        parameters_task_t_update = {}

        # compute the parameters update for the current task
        for layer_name, current_param_tensor in current_state_dict.items():
            parameters_task_t_update[layer_name] = (
                current_param_tensor - self.state_dict_task_start[layer_name]
            )

        # store the parameters update for the current task
        self.parameters_task_update[self.task_id] = parameters_task_t_update

    def construct_parameters_from_updates(self):
        r"""Construct the parameters of the model from parameters task updates."""
        updated_state_dict = deepcopy(self.original_backbone_state_dict)
        for param_update in self.parameters_task_update.values():
            for layer_name, param_tensor in param_update.items():
                updated_state_dict[layer_name] += param_tensor

        self.backbone.load_state_dict(updated_state_dict)
