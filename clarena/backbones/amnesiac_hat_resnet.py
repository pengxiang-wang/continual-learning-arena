r"""
The submodule in `backbones` for AmnesiacHAT masked ResNet backbone networks.
"""

__all__ = [
    "AmnesiacHATResNetBase",
    "AmnesiacHATResNet18",
    "AmnesiacHATResNet34",
    "AmnesiacHATResNet50",
    "AmnesiacHATResNet101",
    "AmnesiacHATResNet152",
]

import logging

import torchvision
from torch import Tensor, nn
from torch.nn import ModuleDict

from clarena.backbones.base import AmnesiacHATBackbone
from clarena.backbones.constants import HATMASKRESNET18_STATE_DICT_MAPPING
from clarena.backbones.hat_mask_resnet import (
    HATMaskResNetBase,
    HATMaskResNetBlockLarge,
    HATMaskResNetBlockSmall,
)

# always get logger for built-in logging in each module
pylogger = logging.getLogger(__name__)


class AmnesiacHATResNetBase(AmnesiacHATBackbone, HATMaskResNetBase):
    r"""AmnesiacHAT masked ResNet backbone network."""

    def __init__(
        self,
        input_channels: int,
        building_block_type: HATMaskResNetBlockSmall | HATMaskResNetBlockLarge,
        building_block_nums: tuple[int, int, int, int],
        building_block_preceding_output_channels: tuple[int, int, int, int],
        building_block_input_channels: tuple[int, int, int, int],
        output_dim: int,
        gate: str,
        activation_layer: nn.Module | None = nn.ReLU,
        batch_normalization: str | None = None,
        bias: bool = False,
        disable_unlearning: bool = False,
        **kwargs,
    ) -> None:
        r"""Construct and initialize the AmnesiacHAT masked ResNet backbone network.

        **Args:**
        - **input_channels** (`int`): the number of channels of input. Image data are kept channels when going in ResNet. Note that convolutional networks require number of input channels instead of dimension.
        - **building_block_type** (`HATMaskResNetBlockSmall` | `HATMaskResNetBlockLarge`): the type of building block used in the ResNet.
        - **building_block_nums** (`tuple[int, int, int, int]`): the number of building blocks in the 2-5 convolutional layer correspondingly.
        - **building_block_preceding_output_channels** (`tuple[int, int, int, int]`): the number of channels of preceding output of each building block in the 2-5 convolutional layer correspondingly.
        - **building_block_input_channels** (`tuple[int, int, int, int]`): the number of channels of input of each building block in the 2-5 convolutional layer correspondingly.
        - **output_dim** (`int`): the output dimension after flattening at last which connects to CL output heads.
        - **gate** (`str`): the type of gate function turning the real value task embeddings into attention masks; one of:
            - `sigmoid`: the sigmoid function.
        - **activation_layer** (`nn.Module`): activation function of each layer (if not `None`). Default `nn.ReLU`.
        - **batch_normalization** (`str` | `None`): How to use batch normalization after the fully connected layers; one of:
            - `None`: no batch normalization layers.
            - `shared`: use a single batch normalization layer for all tasks. Note that this can cause catastrophic forgetting.
            - `independent`: use independent batch normalization layers for each task.
        - **bias** (`bool`): whether to use bias in the convolutional layer. Default `False`.
        - **disable_unlearning** (`bool`): whether to disable unlearning. This is used in reference experiments following continual learning pipeline. Default is `False`.
        - **kwargs**: Reserved for multiple inheritance.
        """
        super().__init__(
            input_channels=input_channels,
            building_block_type=building_block_type,
            building_block_nums=building_block_nums,
            building_block_preceding_output_channels=building_block_preceding_output_channels,
            building_block_input_channels=building_block_input_channels,
            output_dim=output_dim,
            gate=gate,
            activation_layer=activation_layer,
            batch_normalization=batch_normalization,
            bias=bias,
            **kwargs,
        )

        self.disable_unlearning: bool = disable_unlearning
        r"""Whether to disable unlearning. This is used in reference experiments following continual learning pipeline."""

        if not disable_unlearning:
            self.input_channels: int = input_channels
            r"""The input channels of the AmnesiacHATResNet backbone network."""
            self.building_block_type = building_block_type
            r"""The building block type of the AmnesiacHATResNet backbone network."""
            self.building_block_nums = building_block_nums
            r"""The number of building blocks in each ResNet layer."""
            self.building_block_preceding_output_channels = (
                building_block_preceding_output_channels
            )
            r"""The preceding output channels for each ResNet layer."""
            self.building_block_input_channels = building_block_input_channels
            r"""The input channels for each ResNet layer."""
            self.output_dim: int = output_dim
            r"""The output dimension of the AmnesiacHATResNet backbone network."""
            self.activation_layer: nn.Module | None = activation_layer
            r"""The activation layer of the AmnesiacHATResNet backbone network."""
            self.batch_normalization: str | None = batch_normalization
            r"""The way to use batch normalization after the convolutional layers."""
            self.bias: bool = bias
            r"""Whether to use bias in the convolutional layers of the AmnesiacHATResNet backbone network."""

    def initialize_backup_backbone(
        self,
        unlearnable_task_ids: list[int],
    ) -> None:
        r"""Initialize the backup backbone network for the current task.

        **Args:**
        - **unlearnable_task_ids** (`list[int]`): The list of unlearnable task IDs at current task `self.task_id`.
        """
        backup_task_ids = [
            tid for tid in unlearnable_task_ids if tid != self.task_id
        ]  # exclude current task, as we don't need backup backbone for current task

        backup_block_type = (
            ResNetBlockSmall
            if self.building_block_type == HATMaskResNetBlockSmall
            else ResNetBlockLarge
        )

        self.backup_backbones = ModuleDict(
            {
                f"{unlearnable_task_id}": ResNetBase(
                    input_channels=self.input_channels,
                    building_block_type=backup_block_type,
                    building_block_nums=self.building_block_nums,
                    building_block_preceding_output_channels=self.building_block_preceding_output_channels,
                    building_block_input_channels=self.building_block_input_channels,
                    output_dim=self.output_dim,
                    activation_layer=self.activation_layer,
                    batch_normalization=self.batch_normalization,
                    bias=self.bias,
                )
                for unlearnable_task_id in backup_task_ids
            }
        )

        self.unlearnable_task_ids = backup_task_ids

    def _mix_output(
        self,
        main_layer: nn.Module | None,
        backup_layer: nn.Module | None,
        input: Tensor,
        mask: Tensor,
    ) -> Tensor:
        main_output = input if main_layer is None else main_layer(input)
        backup_output = input if backup_layer is None else backup_layer(input)
        mask_broadcast = mask.view(1, -1, 1, 1)
        return mask_broadcast * backup_output + (1 - mask_broadcast) * main_output

    def _forward_block_small(
        self,
        block: HATMaskResNetBlockSmall,
        backup_blocks: dict[int, nn.Module] | None,
        x: Tensor,
        mask: dict[str, Tensor],
        x_backup: dict[int, Tensor] | None,
    ) -> tuple[Tensor, dict[int, Tensor] | None, dict[str, Tensor]]:
        activations = {}

        mask_1 = mask[block.full_1st_layer_name].view(1, -1, 1, 1)
        mask_2 = mask[block.full_2nd_layer_name].view(1, -1, 1, 1)

        identity = (
            block.identity_downsample(x) if block.identity_downsample is not None else x
        )

        x = block.conv1(x)
        x = x * mask_1
        if block.activation:
            x = block.conv_activation1(x)
        activations[block.full_1st_layer_name] = x

        x = block.conv2(x)
        x = x + identity
        x = x * mask_2
        if block.activation:
            x = block.conv_activation2(x)
        activations[block.full_2nd_layer_name] = x

        if x_backup is not None and backup_blocks is not None:
            for unlearnable_task_id in self.unlearnable_task_ids:
                backup_block = backup_blocks[unlearnable_task_id]
                x_b = x_backup[unlearnable_task_id]
                x_b_input = x_b

                mask_backup = self.masks[unlearnable_task_id]
                mask_b1 = mask_backup[block.full_1st_layer_name]
                mask_b2 = mask_backup[block.full_2nd_layer_name]

                x_b = self._mix_output(block.conv1, backup_block.conv1, x_b, mask_b1)
                x_b = x_b * mask_1
                if block.activation:
                    x_b = block.conv_activation1(x_b)

                x_b = self._mix_output(block.conv2, backup_block.conv2, x_b, mask_b2)
                identity_b = self._mix_output(
                    block.identity_downsample,
                    backup_block.identity_downsample,
                    x_b_input,
                    mask_b2,
                )
                x_b = x_b + identity_b
                x_b = x_b * mask_2
                if block.activation:
                    x_b = block.conv_activation2(x_b)

                x_backup[unlearnable_task_id] = x_b

        return x, x_backup, activations

    def _forward_block_large(
        self,
        block: HATMaskResNetBlockLarge,
        backup_blocks: dict[int, nn.Module] | None,
        x: Tensor,
        mask: dict[str, Tensor],
        x_backup: dict[int, Tensor] | None,
    ) -> tuple[Tensor, dict[int, Tensor] | None, dict[str, Tensor]]:
        activations = {}

        mask_1 = mask[block.full_1st_layer_name].view(1, -1, 1, 1)
        mask_2 = mask[block.full_2nd_layer_name].view(1, -1, 1, 1)
        mask_3 = mask[block.full_3rd_layer_name].view(1, -1, 1, 1)

        identity = (
            block.identity_downsample(x) if block.identity_downsample is not None else x
        )

        x = block.conv1(x)
        x = x * mask_1
        if block.activation:
            x = block.conv_activation1(x)
        activations[block.full_1st_layer_name] = x

        x = block.conv2(x)
        x = x * mask_2
        if block.activation:
            x = block.conv_activation2(x)
        activations[block.full_2nd_layer_name] = x

        x = block.conv3(x)
        x = x + identity
        x = x * mask_3
        if block.activation:
            x = block.conv_activation3(x)
        activations[block.full_3rd_layer_name] = x

        if x_backup is not None and backup_blocks is not None:
            for unlearnable_task_id in self.unlearnable_task_ids:
                backup_block = backup_blocks[unlearnable_task_id]
                x_b = x_backup[unlearnable_task_id]
                x_b_input = x_b

                mask_backup = self.masks[unlearnable_task_id]
                mask_b1 = mask_backup[block.full_1st_layer_name]
                mask_b2 = mask_backup[block.full_2nd_layer_name]
                mask_b3 = mask_backup[block.full_3rd_layer_name]

                x_b = self._mix_output(block.conv1, backup_block.conv1, x_b, mask_b1)
                x_b = x_b * mask_1
                if block.activation:
                    x_b = block.conv_activation1(x_b)

                x_b = self._mix_output(block.conv2, backup_block.conv2, x_b, mask_b2)
                x_b = x_b * mask_2
                if block.activation:
                    x_b = block.conv_activation2(x_b)

                x_b = self._mix_output(block.conv3, backup_block.conv3, x_b, mask_b3)
                identity_b = self._mix_output(
                    block.identity_downsample,
                    backup_block.identity_downsample,
                    x_b_input,
                    mask_b3,
                )
                x_b = x_b + identity_b
                x_b = x_b * mask_3
                if block.activation:
                    x_b = block.conv_activation3(x_b)

                x_backup[unlearnable_task_id] = x_b

        return x, x_backup, activations

    def forward(
        self,
        input: Tensor,
        stage: str,
        s_max: float | None = None,
        batch_idx: int | None = None,
        num_batches: int | None = None,
        test_task_id: int | None = None,
    ) -> tuple[Tensor, dict[str, Tensor], dict[str, Tensor]]:
        r"""The forward pass for data from task `self.task_id`.

        **Args:**
        - **input** (`Tensor`): The input tensor from data.
        - **stage** (`str`): The stage of the forward pass; one of:
            1. 'train': training stage.
            2. 'validation': validation stage.
            3. 'test': testing stage.
            4. 'unlearning_test': unlearning testing stage.
        - **s_max** (`float` | `None`): The maximum scaling factor in the gate function. Doesn't apply to the testing stage.
        - **batch_idx** (`int` | `None`): The current batch index. Applies only to the training stage.
        - **num_batches** (`int` | `None`): The total number of batches. Applies only to the training stage.
        - **test_task_id** (`int` | `None`): The test task ID. Applies only to the testing stage.

        **Returns:**
        - **output_feature** (`Tensor`): The output feature tensor to be passed into heads.
        - **output_backup_feature** (`Tensor`): The output feature tensor from the backup backbone masked by cumulative mask. Applies only during training when unlearning is enabled.
        - **mask** (`dict[str, Tensor]`): The mask for the current task. Keys are layer names and values are the mask tensors.
        - **activations** (`dict[str, Tensor]`): The hidden features (after activation) in each weighted layer.
        """
        batch_size = input.size(0)
        activations = {}

        mask = self.get_mask(
            stage=stage,
            s_max=s_max,
            batch_idx=batch_idx,
            num_batches=num_batches,
            test_task_id=test_task_id,
        )

        train_with_backup = stage == "train" and not self.disable_unlearning
        use_backup = train_with_backup and self.task_id > 1

        x_backup: dict[int, Tensor] | None = {} if train_with_backup else None
        if use_backup:
            x_backup = {
                unlearnable_task_id: input
                for unlearnable_task_id in self.unlearnable_task_ids
            }

        x = input

        mask_conv1 = mask["conv1"].view(1, -1, 1, 1)

        x = self.conv1(x)
        if use_backup:
            for unlearnable_task_id in self.unlearnable_task_ids:
                backup_backbone = self.backup_backbones[f"{unlearnable_task_id}"]
                mask_backup = self.masks[unlearnable_task_id]["conv1"]
                x_backup[unlearnable_task_id] = self._mix_output(
                    self.conv1,
                    backup_backbone.conv1,
                    x_backup[unlearnable_task_id],
                    mask_backup,
                )

        x = x * mask_conv1
        if use_backup:
            for unlearnable_task_id in self.unlearnable_task_ids:
                x_backup[unlearnable_task_id] = (
                    x_backup[unlearnable_task_id] * mask_conv1
                )
        if self.activation:
            x = self.conv_activation1(x)
            if use_backup:
                for unlearnable_task_id in self.unlearnable_task_ids:
                    x_backup[unlearnable_task_id] = self.conv_activation1(
                        x_backup[unlearnable_task_id]
                    )
        activations["conv1"] = x

        x = self.maxpool(x)
        if use_backup:
            for unlearnable_task_id in self.unlearnable_task_ids:
                x_backup[unlearnable_task_id] = self.maxpool(
                    x_backup[unlearnable_task_id]
                )

        for layer_name in ("conv2x", "conv3x", "conv4x", "conv5x"):
            layer = getattr(self, layer_name)
            for block_idx, block in enumerate(layer):
                backup_blocks = None
                if use_backup:
                    backup_blocks = {
                        unlearnable_task_id: getattr(
                            self.backup_backbones[f"{unlearnable_task_id}"], layer_name
                        )[block_idx]
                        for unlearnable_task_id in self.unlearnable_task_ids
                    }

                if isinstance(block, HATMaskResNetBlockSmall):
                    x, x_backup, activations_block = self._forward_block_small(
                        block=block,
                        backup_blocks=backup_blocks,
                        x=x,
                        mask=mask,
                        x_backup=x_backup,
                    )
                else:
                    x, x_backup, activations_block = self._forward_block_large(
                        block=block,
                        backup_blocks=backup_blocks,
                        x=x,
                        mask=mask,
                        x_backup=x_backup,
                    )

                activations.update(activations_block)

        x = self.avepool(x)
        output_feature = x.view(batch_size, -1)

        if train_with_backup:
            output_backup_feature = {}
            for unlearnable_task_id, x_b in x_backup.items():
                x_b = self.avepool(x_b)
                output_backup_feature[unlearnable_task_id] = x_b.view(batch_size, -1)
            return output_feature, output_backup_feature, mask, activations

        return output_feature, mask, activations


class AmnesiacHATResNet18(AmnesiacHATResNetBase):
    r"""AmnesiacHAT masked ResNet-18 backbone network."""

    def __init__(
        self,
        input_channels: int,
        output_dim: int,
        gate: str,
        activation_layer: nn.Module | None = nn.ReLU,
        batch_normalization: str | None = None,
        bias: bool = False,
        pretrained_weights: str | None = None,
        disable_unlearning: bool = False,
        **kwargs,
    ) -> None:
        r"""Construct and initialize the AmnesiacHAT masked ResNet-18 backbone network."""
        super().__init__(
            input_channels=input_channels,
            building_block_type=HATMaskResNetBlockSmall,
            building_block_nums=(2, 2, 2, 2),
            building_block_preceding_output_channels=(64, 64, 128, 256),
            building_block_input_channels=(64, 128, 256, 512),
            output_dim=output_dim,
            gate=gate,
            activation_layer=activation_layer,
            batch_normalization=batch_normalization,
            bias=bias,
            disable_unlearning=disable_unlearning,
            **kwargs,
        )

        if pretrained_weights is not None:
            torchvision_resnet18_state_dict = torchvision.models.resnet18(
                weights=pretrained_weights
            ).state_dict()

            state_dict_converted = {}
            for key, value in torchvision_resnet18_state_dict.items():
                if HATMASKRESNET18_STATE_DICT_MAPPING[key] is not None:
                    state_dict_converted[HATMASKRESNET18_STATE_DICT_MAPPING[key]] = (
                        value
                    )

            self.load_state_dict(state_dict_converted, strict=False)


class AmnesiacHATResNet34(AmnesiacHATResNetBase):
    r"""AmnesiacHAT masked ResNet-34 backbone network."""

    def __init__(
        self,
        input_channels: int,
        output_dim: int,
        gate: str,
        activation_layer: nn.Module | None = nn.ReLU,
        batch_normalization: str | None = None,
        bias: bool = False,
        disable_unlearning: bool = False,
        **kwargs,
    ) -> None:
        r"""Construct and initialize the AmnesiacHAT masked ResNet-34 backbone network."""
        super().__init__(
            input_channels=input_channels,
            building_block_type=HATMaskResNetBlockSmall,
            building_block_nums=(3, 4, 6, 3),
            building_block_preceding_output_channels=(64, 64, 128, 256),
            building_block_input_channels=(64, 128, 256, 512),
            output_dim=output_dim,
            gate=gate,
            activation_layer=activation_layer,
            batch_normalization=batch_normalization,
            bias=bias,
            disable_unlearning=disable_unlearning,
            **kwargs,
        )


class AmnesiacHATResNet50(AmnesiacHATResNetBase):
    r"""AmnesiacHAT masked ResNet-50 backbone network."""

    def __init__(
        self,
        input_channels: int,
        output_dim: int,
        gate: str,
        activation_layer: nn.Module | None = nn.ReLU,
        batch_normalization: str | None = None,
        bias: bool = False,
        disable_unlearning: bool = False,
        **kwargs,
    ) -> None:
        r"""Construct and initialize the AmnesiacHAT masked ResNet-50 backbone network."""
        super().__init__(
            input_channels=input_channels,
            building_block_type=HATMaskResNetBlockLarge,
            building_block_nums=(3, 4, 6, 3),
            building_block_preceding_output_channels=(64, 256, 512, 1024),
            building_block_input_channels=(64, 128, 256, 512),
            output_dim=output_dim,
            gate=gate,
            activation_layer=activation_layer,
            batch_normalization=batch_normalization,
            bias=bias,
            disable_unlearning=disable_unlearning,
            **kwargs,
        )


class AmnesiacHATResNet101(AmnesiacHATResNetBase):
    r"""AmnesiacHAT masked ResNet-101 backbone network."""

    def __init__(
        self,
        input_channels: int,
        output_dim: int,
        gate: str,
        activation_layer: nn.Module | None = nn.ReLU,
        batch_normalization: str | None = None,
        bias: bool = False,
        disable_unlearning: bool = False,
        **kwargs,
    ) -> None:
        r"""Construct and initialize the AmnesiacHAT masked ResNet-101 backbone network."""
        super().__init__(
            input_channels=input_channels,
            building_block_type=HATMaskResNetBlockLarge,
            building_block_nums=(3, 4, 23, 3),
            building_block_preceding_output_channels=(64, 256, 512, 1024),
            building_block_input_channels=(64, 128, 256, 512),
            output_dim=output_dim,
            gate=gate,
            activation_layer=activation_layer,
            batch_normalization=batch_normalization,
            bias=bias,
            disable_unlearning=disable_unlearning,
            **kwargs,
        )


class AmnesiacHATResNet152(AmnesiacHATResNetBase):
    r"""AmnesiacHAT masked ResNet-152 backbone network."""

    def __init__(
        self,
        input_channels: int,
        output_dim: int,
        gate: str,
        activation_layer: nn.Module | None = nn.ReLU,
        batch_normalization: str | None = None,
        bias: bool = False,
        disable_unlearning: bool = False,
        **kwargs,
    ) -> None:
        r"""Construct and initialize the AmnesiacHAT masked ResNet-152 backbone network."""
        super().__init__(
            input_channels=input_channels,
            building_block_type=HATMaskResNetBlockLarge,
            building_block_nums=(3, 8, 36, 3),
            building_block_preceding_output_channels=(64, 256, 512, 1024),
            building_block_input_channels=(64, 128, 256, 512),
            output_dim=output_dim,
            gate=gate,
            activation_layer=activation_layer,
            batch_normalization=batch_normalization,
            bias=bias,
            disable_unlearning=disable_unlearning,
            **kwargs,
        )
