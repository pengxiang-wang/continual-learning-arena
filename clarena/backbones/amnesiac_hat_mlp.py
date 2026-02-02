r"""
The submodule in `backbones` for the AmnesiacHAT masked MLP backbone network.
"""

__all__ = ["AmnesiacHATMLP"]

import logging

from torch import Tensor, nn

from clarena.backbones import MLP, AmnesiacHATBackbone, HATMaskMLP

# always get logger for built-in logging in each module
pylogger = logging.getLogger(__name__)


class AmnesiacHATMLP(AmnesiacHATBackbone, HATMaskMLP):
    """AmnesiacHAT masked multi-layer perceptron (MLP) backbone network for continual learning.

    This backbone network implements the AmnesiacHAT algorithm that extends HAT to support unlearning of specific tasks while retaining performance on other tasks. It uses task-specific attention masks to protect important neurons for each task, and maintains backup backbones for unlearning compensation.
    """

    original_backbone_class: type = MLP
    r"""The original backbone class used for the AmnesiacHAT backbone network."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int],
        output_dim: int,
        gate: str,
        activation_layer: nn.Module | None = nn.ReLU,
        batch_normalization: str | None = None,
        bias: bool = True,
        dropout: float | None = None,
        disable_unlearning: bool = False,
        **kwargs,
    ) -> None:
        r"""Construct and initialize the AmnesiacHAT MLP backbone network.

        **Args:**
        - **input_dim** (`int`): The input dimension. Any data need to be flattened before entering the MLP.
        - **hidden_dims** (`list[int]`): List of hidden layer dimensions. It can be an empty list, which means a single-layer MLP, and it can be as many layers as you want. Note that it doesn't include the last dimension, which we take as the output dimension.
        - **output_dim** (`int`): The output dimension that connects to CL output heads.
        - **gate** (`str`): The type of gate function turning real-valued task embeddings into attention masks; one of:
            - `sigmoid`: the sigmoid function.
        - **activation_layer** (`nn.Module` | `None`): Activation function of each layer (if not `None`). If `None`, this layer won't be used. Default `nn.ReLU`.
        - **batch_normalization** (`str` | `None`): How to use batch normalization after the fully connected layers; one of:
            - `None`: no batch normalization layers.
            - `shared`: use a single batch normalization layer for all tasks. Note that this can cause catastrophic forgetting.
            - `independent`: use independent batch normalization layers for each task.
        - **bias** (`bool`): Whether to use bias in the linear layer. Default `True`.
        - **dropout** (`float` | `None`): The probability for the dropout layer. If `None`, this layer won't be used. Default `None`.
        - **disable_unlearning** (`bool`): whether to disable unlearning. This is used in reference experiments following continual learning pipeline. Default is `False`.
        - **kwargs**: Reserved for multiple inheritance.
        """

        super().__init__(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            output_dim=output_dim,
            gate=gate,
            activation_layer=activation_layer,
            batch_normalization=batch_normalization,
            bias=bias,
            dropout=dropout,
            disable_unlearning=disable_unlearning,
            **kwargs,
        )

        if not disable_unlearning:
            # save these arguments for backup backbone initialization
            self.backup_backbone_kwargs: dict[str, object] = {
                "input_dim": input_dim,
                "hidden_dims": hidden_dims,
                "output_dim": output_dim,
                "activation_layer": activation_layer,
                "batch_normalization": batch_normalization,
                "bias": bias,
                "dropout": dropout,
            }
            r"""The initialization arguments for the backup MLP backbone network."""

    def _route_backup_layer_output_by_mask(
        self,
        layer_name: str,
        main_layer: nn.Module,
        backup_layer: nn.Module,
        input: Tensor,
        mask_backup_task: dict[str, Tensor],
        main_bn: nn.Module | None = None,
        backup_bn: nn.Module | None = None,
    ) -> Tensor:
        mask = mask_backup_task[layer_name]
        main_output = main_layer(input)
        backup_output = backup_layer(input)
        if main_bn is not None:
            main_output = main_bn(main_output)
        if backup_bn is not None:
            backup_output = backup_bn(backup_output)
        return mask * backup_output + (1 - mask) * main_output

    def forward(
        self,
        input: Tensor,
        stage: str,
        s_max: float | None = None,
        batch_idx: int | None = None,
        num_batches: int | None = None,
        test_task_id: int | None = None,
    ) -> tuple[Tensor, dict[str, Tensor], dict[str, Tensor]]:
        r"""The forward pass for data from task `self.task_id`. Task-specific masks for `self.task_id` are applied to units (neurons) in each fully connected layer. During training, the backup backbone masked by cumulative mask is trained in parallel.

        **Args:**
        - **input** (`Tensor`): The input tensor from data.
        - **stage** (`str`): The stage of the forward pass; one of:
            1. 'train_with_backup': training stage with training backup backbones for future unlearning backup compensation.
            2. 'train': training stage.
            3. 'validation': validation stage.
            4. 'test': testing stage.
            5. 'unlearning_test': unlearning testing stage.
        - **s_max** (`float`): The maximum scaling factor in the gate function. Doesn't apply to the testing stage. See Sec. 2.4 in the [HAT paper](http://proceedings.mlr.press/v80/serra18a).
        - **batch_idx** (`int` | `None`): The current batch index. Applies only to the training stage. For other stages, it is `None`.
        - **num_batches** (`int` | `None`): The total number of batches. Applies only to the training stage. For other stages, it is `None`.
        - **test_task_id** (`int` | `None`): The test task ID. Applies only to the testing stage. For other stages, it is `None`.

        **Returns:**
        - **output_feature** (`Tensor`): The output feature tensor to be passed into heads. This is the main target of backpropagation.
        - **output_backup_feature** (`Tensor`): The output feature tensor from the backup backbone masked by cumulative mask. This is the parallel target of backpropagation.
        - **mask** (`dict[str, Tensor]`): The mask for the current task. Keys (`str`) are the layer names and values (`Tensor`) are the mask tensors. The mask tensor has size (number of units, ).
        - **activations** (`dict[str, Tensor]`): The hidden features (after activation) in each weighted layer. Keys (`str`) are the weighted layer names and values (`Tensor`) are the hidden feature tensors. This is used for continual learning algorithms that need hidden features. Although the HAT algorithm does not need this, it is still provided for API consistency for other HAT-based algorithms that inherit this `forward()` method of the `HAT` class.
        """
        batch_size = input.size(0)
        activations = {}

        mask = self.get_mask(
            stage="train" if stage == "train_with_backup" else stage,
            s_max=s_max,
            batch_idx=batch_idx,
            num_batches=num_batches,
            test_task_id=test_task_id,
        )
        if self.batch_normalization:
            fc_bn = self.get_bn(
                stage="train" if stage == "train_with_backup" else stage,
                test_task_id=test_task_id,
            )

        # for each layer, backup backbones are trained in parallel during 'train_with_backup' stage. Operations are repeated in the same way as the main backbone.

        x = input.view(batch_size, -1)  # flatten before going through MLP
        if stage == "train_with_backup":
            # flatten for backup backbone
            x_backup = {
                backup_task_id: input.view(batch_size, -1)
                for backup_task_id in self.backup_task_ids
            }

        for layer_idx, layer_name in enumerate(self.weighted_layer_names):

            # fully-connected layer first
            fc_layer = self.fc[layer_idx]
            x = fc_layer(x)

            # batch normalization second
            fc_bn_layer = None
            if self.batch_normalization:
                fc_bn_layer = fc_bn[layer_idx]
                x = fc_bn_layer(x)

            # apply the mask to the parameters
            mask_layer = mask[f"fc/{layer_idx}"]
            x = x * mask_layer

            # activation function third
            if self.activation:
                fc_activation_layer = self.fc_activation[layer_idx]
                x = fc_activation_layer(x)
            activations[layer_name] = x  # store the hidden feature

            # dropout last
            if self.dropout:
                fc_dropout_layer = self.fc_dropout[layer_idx]
                x = fc_dropout_layer(x)

            if stage == "train_with_backup":
                # backup backbone forward after main backbone completes this layer
                for backup_task_id in self.backup_task_ids:
                    backup_backbone = self.backup_backbones[f"{backup_task_id}"]
                    backup_fc_layer = backup_backbone.fc[layer_idx]
                    mask_backup_task = self.masks[backup_task_id]
                    backup_fc_bn_layer = None
                    if self.batch_normalization and backup_backbone.batch_normalization:
                        backup_fc_bn_layer = backup_backbone.fc_bn[layer_idx]

                    x_b = x_backup[backup_task_id]
                    x_b = self._route_backup_layer_output_by_mask(
                        layer_name=layer_name,
                        main_layer=fc_layer,
                        backup_layer=backup_fc_layer,
                        input=x_b,
                        mask_backup_task=mask_backup_task,
                        main_bn=fc_bn_layer,
                        backup_bn=backup_fc_bn_layer,
                    )

                    x_b = x_b * mask_layer

                    if self.activation:
                        x_b = fc_activation_layer(x_b)

                    if self.dropout:
                        x_b = fc_dropout_layer(x_b)

                    x_backup[backup_task_id] = x_b

        output_feature = x

        if stage == "train_with_backup":
            output_backup_feature = x_backup
            return output_feature, output_backup_feature, mask, activations
        else:
            return output_feature, mask, activations
