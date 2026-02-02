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

from clarena.backbones.base import AmnesiacHATBackbone
from clarena.backbones.constants import HATMASKRESNET18_STATE_DICT_MAPPING
from clarena.backbones.hat_mask_resnet import (
    HATMaskResNetBase,
    HATMaskResNetBlockLarge,
    HATMaskResNetBlockSmall,
)
from clarena.backbones.resnet import ResNetBase, ResNetBlockLarge, ResNetBlockSmall

# always get logger for built-in logging in each module
pylogger = logging.getLogger(__name__)


class AmnesiacHATResNetBase(AmnesiacHATBackbone, HATMaskResNetBase):
    r"""The base class of AmnesiacHAT masked [residual network (ResNet)](https://www.cv-foundation.org/openaccess/content_cvpr_2016/html/He_Deep_Residual_Learning_CVPR_2016_paper.html).

    [HAT (Hard Attention to the Task, 2018)](http://proceedings.mlr.press/v80/serra18a) is an architecture-based continual learning approach that uses learnable hard attention masks to select the task-specific parameters. AmnesiacHAT extends HAT to support unlearning of specific tasks while retaining performance on other tasks by training parallel backup backbones for unlearning compensation.

    ResNet is a convolutional network architecture, which has 1st convolutional parameter layer and a maxpooling layer, connecting to 4 convolutional layers which contains multiple convolutional parameter layer. Each layer of the 4 are constructed from basic building blocks which are either small (`ResNetBlockSmall`) or large (`ResNetBlockLarge`). Each building block contains several convolutional parameter layers. The building blocks are connected by a skip connection which is a direct connection from the input of the block to the output of the block, and this is why it's called residual (find "shortcut connections" in the paper for more details). After the 5th convolutional layer, there are average pooling layer and a fully connected layer which connects to the CL output heads.

    Mask is applied to the units which are output channels in each weighted convolutional layer. The mask is generated from the neuron-wise task embedding and gate function. For AmnesiacHAT, backup backbones are trained in parallel for future unlearning compensation.
    """

    original_backbone_class: type = ResNetBase
    r"""The original backbone class used for the AmnesiacHAT backbone network."""

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
        r"""Construct and initialize the AmnesiacHAT masked ResNet backbone network with task embedding.

        **Args:**
        - **input_channels** (`int`): the number of channels of input. Image data are kept channels when going in ResNet. Note that convolutional networks require number of input channels instead of dimension.
        - **building_block_type** (`HATMaskResNetBlockSmall` | `HATMaskResNetBlockLarge`): the type of building block used in the ResNet.
        - **building_block_nums** (`tuple[int, int, int, int]`): the number of building blocks in the 2-5 convolutional layer correspondingly.
        - **building_block_preceding_output_channels** (`tuple[int, int, int, int]`): the number of channels of preceding output of each building block in the 2-5 convolutional layer correspondingly.
        - **building_block_input_channels** (`tuple[int, int, int, int]`): the number of channels of input of each building block in the 2-5 convolutional layer correspondingly.
        - **output_dim** (`int`): the output dimension after flattening at last which connects to CL output heads. Although this is not determined by us but the architecture built before the flattening layer, we still need to provide this to construct the heads.
        - **gate** (`str`): the type of gate function turning the real value task embeddings into attention masks; one of:
            - `sigmoid`: the sigmoid function.
        - **activation_layer** (`nn.Module`): activation function of each layer (if not `None`). Default `nn.ReLU`.
        - **batch_normalization** (`str` | `None`): How to use batch normalization after the convolutional layers; one of:
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
            disable_unlearning=disable_unlearning,
            **kwargs,
        )

        if not disable_unlearning:
            backup_block_type = (
                ResNetBlockSmall
                if building_block_type == HATMaskResNetBlockSmall
                else ResNetBlockLarge
            )
            backup_batch_normalization = (
                batch_normalization
                if isinstance(batch_normalization, bool)
                else batch_normalization in ("shared", "independent")
            )
            self.backup_backbone_kwargs: dict[str, object] = {
                "input_channels": input_channels,
                "building_block_type": backup_block_type,
                "building_block_nums": building_block_nums,
                "building_block_preceding_output_channels": building_block_preceding_output_channels,
                "building_block_input_channels": building_block_input_channels,
                "output_dim": output_dim,
                "activation_layer": activation_layer,
                "batch_normalization": backup_batch_normalization,
                "bias": bias,
            }
            r"""The initialization arguments for the backup ResNet backbone network."""

    def _route_backup_layer_output_by_mask(
        self,
        layer_name: str,
        main_layer: nn.Module | None,
        backup_layer: nn.Module | None,
        input: Tensor,
        mask_backup_task: dict[str, Tensor],
        main_bn: nn.Module | None = None,
        backup_bn: nn.Module | None = None,
    ) -> Tensor:
        r"""Route the output of the backup layer according to the mask of the backup task.

        **Args:**
        - **layer_name** (`str`): The name of the layer.
        - **main_layer** (`nn.Module` | `None`): The main backbone layer. If `None`, use identity.
        - **backup_layer** (`nn.Module` | `None`): The backup backbone layer. If `None`, use identity.
        - **input** (`Tensor`): The input tensor to the layer.
        - **mask_backup_task** (`dict[str, Tensor]`): The mask of the backup task. Keys are layer names and values are mask tensors.
        - **main_bn** (`nn.Module` | `None`): The main backbone batch normalization layer. If `None`, no BN is applied.
        - **backup_bn** (`nn.Module` | `None`): The backup backbone batch normalization layer. If `None`, no BN is applied.

        **Returns:**
        - **output** (`Tensor`): The routed output tensor, mixing main and backup layer outputs by the backup task mask.
        """
        mask = mask_backup_task[layer_name]
        main_output = input if main_layer is None else main_layer(input)
        backup_output = input if backup_layer is None else backup_layer(input)
        if main_bn is not None:
            main_output = main_bn(main_output)
        if backup_bn is not None:
            backup_output = backup_bn(backup_output)
        mask_broadcast = mask.view(1, -1, 1, 1)
        return mask_broadcast * backup_output + (1 - mask_broadcast) * main_output


    def _forward_block_small(
        self,
        block: HATMaskResNetBlockSmall,
        backup_blocks: dict[int, nn.Module] | None,
        x: Tensor,
        mask: dict[str, Tensor],
        x_backup: dict[int, Tensor] | None,
        stage: str,
        test_task_id: int | None,
    ) -> tuple[Tensor, dict[int, Tensor] | None, dict[str, Tensor]]:
        r"""Forward pass for a small ResNet block (ResNet-18/34 style).

        **Args:**
        - **block** (`HATMaskResNetBlockSmall`): The target building block.
        - **backup_blocks** (`dict[int, nn.Module]` | `None`): Backup blocks for each backup task. Only used in `train_with_backup`.
        - **x** (`Tensor`): The input feature maps.
        - **mask** (`dict[str, Tensor]`): The current task mask. Keys are layer names and values are mask tensors.
        - **x_backup** (`dict[int, Tensor]` | `None`): Backup feature maps for each backup task. Only used in `train_with_backup`.
        - **stage** (`str`): The stage of the forward pass.
        - **test_task_id** (`int` | `None`): The test task ID. Applies only to the testing stage.

        **Returns:**
        - **output_feature** (`Tensor`): The output feature maps of the block.
        - **output_backup_feature** (`dict[int, Tensor]` | `None`): The backup feature maps after the block.
        - **activations** (`dict[str, Tensor]`): The hidden features (after activation) in each weighted layer within the block.
        """
        activations = {}

        mask_1 = mask[block.full_1st_layer_name].view(1, -1, 1, 1)
        mask_2 = mask[block.full_2nd_layer_name].view(1, -1, 1, 1)
        stage_for_bn = "test" if stage in ("test", "unlearning_test") else "train"
        if block.batch_normalization:
            conv_bn1, conv_bn2 = block.get_bn(
                stage=stage_for_bn, test_task_id=test_task_id
            )

        identity = (
            block.identity_downsample(x) if block.identity_downsample is not None else x
        )

        x = block.conv1(x)
        if block.batch_normalization:
            x = conv_bn1(x)
        x = x * mask_1
        if block.activation:
            x = block.conv_activation1(x)
        activations[block.full_1st_layer_name] = x

        x = block.conv2(x)
        if block.batch_normalization:
            x = conv_bn2(x)
        x = x + identity
        x = x * mask_2
        if block.activation:
            x = block.conv_activation2(x)
        activations[block.full_2nd_layer_name] = x

        if x_backup is not None and backup_blocks is not None:
            for backup_task_id in self.backup_task_ids:
                backup_block = backup_blocks[backup_task_id]
                x_b = x_backup[backup_task_id]
                x_b_input = x_b

                mask_backup_task = self.masks[backup_task_id]
                backup_bn1 = (
                    backup_block.conv_bn1
                    if backup_block.batch_normalization
                    else None
                )
                backup_bn2 = (
                    backup_block.conv_bn2
                    if backup_block.batch_normalization
                    else None
                )

                x_b = self._route_backup_layer_output_by_mask(
                    layer_name=block.full_1st_layer_name,
                    main_layer=block.conv1,
                    backup_layer=backup_block.conv1,
                    input=x_b,
                    mask_backup_task=mask_backup_task,
                    main_bn=conv_bn1 if block.batch_normalization else None,
                    backup_bn=backup_bn1,
                )
                x_b = x_b * mask_1
                if block.activation:
                    x_b = block.conv_activation1(x_b)

                x_b = self._route_backup_layer_output_by_mask(
                    layer_name=block.full_2nd_layer_name,
                    main_layer=block.conv2,
                    backup_layer=backup_block.conv2,
                    input=x_b,
                    mask_backup_task=mask_backup_task,
                    main_bn=conv_bn2 if block.batch_normalization else None,
                    backup_bn=backup_bn2,
                )
                identity_b = self._route_backup_layer_output_by_mask(
                    layer_name=block.full_2nd_layer_name,
                    main_layer=block.identity_downsample,
                    backup_layer=backup_block.identity_downsample,
                    input=x_b_input,
                    mask_backup_task=mask_backup_task,
                )
                x_b = x_b + identity_b
                x_b = x_b * mask_2
                if block.activation:
                    x_b = block.conv_activation2(x_b)

                x_backup[backup_task_id] = x_b

        return x, x_backup, activations


    def _forward_block_large(
        self,
        block: HATMaskResNetBlockLarge,
        backup_blocks: dict[int, nn.Module] | None,
        x: Tensor,
        mask: dict[str, Tensor],
        x_backup: dict[int, Tensor] | None,
        stage: str,
        test_task_id: int | None,
    ) -> tuple[Tensor, dict[int, Tensor] | None, dict[str, Tensor]]:
        r"""Forward pass for a large ResNet block (ResNet-50/101/152 style).

        **Args:**
        - **block** (`HATMaskResNetBlockLarge`): The target building block.
        - **backup_blocks** (`dict[int, nn.Module]` | `None`): Backup blocks for each backup task. Only used in `train_with_backup`.
        - **x** (`Tensor`): The input feature maps.
        - **mask** (`dict[str, Tensor]`): The current task mask. Keys are layer names and values are mask tensors.
        - **x_backup** (`dict[int, Tensor]` | `None`): Backup feature maps for each backup task. Only used in `train_with_backup`.
        - **stage** (`str`): The stage of the forward pass.
        - **test_task_id** (`int` | `None`): The test task ID. Applies only to the testing stage.

        **Returns:**
        - **output_feature** (`Tensor`): The output feature maps of the block.
        - **output_backup_feature** (`dict[int, Tensor]` | `None`): The backup feature maps after the block.
        - **activations** (`dict[str, Tensor]`): The hidden features (after activation) in each weighted layer within the block.
        """
        activations = {}

        mask_1 = mask[block.full_1st_layer_name].view(1, -1, 1, 1)
        mask_2 = mask[block.full_2nd_layer_name].view(1, -1, 1, 1)
        mask_3 = mask[block.full_3rd_layer_name].view(1, -1, 1, 1)
        stage_for_bn = "test" if stage in ("test", "unlearning_test") else "train"
        if block.batch_normalization:
            conv_bn1, conv_bn2, conv_bn3 = block.get_bn(
                stage=stage_for_bn, test_task_id=test_task_id
            )

        identity = (
            block.identity_downsample(x) if block.identity_downsample is not None else x
        )

        x = block.conv1(x)
        if block.batch_normalization:
            x = conv_bn1(x)
        x = x * mask_1
        if block.activation:
            x = block.conv_activation1(x)
        activations[block.full_1st_layer_name] = x

        x = block.conv2(x)
        if block.batch_normalization:
            x = conv_bn2(x)
        x = x * mask_2
        if block.activation:
            x = block.conv_activation2(x)
        activations[block.full_2nd_layer_name] = x

        x = block.conv3(x)
        if block.batch_normalization:
            x = conv_bn3(x)
        x = x + identity
        x = x * mask_3
        if block.activation:
            x = block.conv_activation3(x)
        activations[block.full_3rd_layer_name] = x

        if x_backup is not None and backup_blocks is not None:
            for backup_task_id in self.backup_task_ids:
                backup_block = backup_blocks[backup_task_id]
                x_b = x_backup[backup_task_id]
                x_b_input = x_b

                mask_backup_task = self.masks[backup_task_id]
                backup_bn1 = (
                    backup_block.conv_bn1
                    if backup_block.batch_normalization
                    else None
                )
                backup_bn2 = (
                    backup_block.conv_bn2
                    if backup_block.batch_normalization
                    else None
                )
                backup_bn3 = (
                    backup_block.conv_bn3
                    if backup_block.batch_normalization
                    else None
                )

                x_b = self._route_backup_layer_output_by_mask(
                    layer_name=block.full_1st_layer_name,
                    main_layer=block.conv1,
                    backup_layer=backup_block.conv1,
                    input=x_b,
                    mask_backup_task=mask_backup_task,
                    main_bn=conv_bn1 if block.batch_normalization else None,
                    backup_bn=backup_bn1,
                )
                x_b = x_b * mask_1
                if block.activation:
                    x_b = block.conv_activation1(x_b)

                x_b = self._route_backup_layer_output_by_mask(
                    layer_name=block.full_2nd_layer_name,
                    main_layer=block.conv2,
                    backup_layer=backup_block.conv2,
                    input=x_b,
                    mask_backup_task=mask_backup_task,
                    main_bn=conv_bn2 if block.batch_normalization else None,
                    backup_bn=backup_bn2,
                )
                x_b = x_b * mask_2
                if block.activation:
                    x_b = block.conv_activation2(x_b)

                x_b = self._route_backup_layer_output_by_mask(
                    layer_name=block.full_3rd_layer_name,
                    main_layer=block.conv3,
                    backup_layer=backup_block.conv3,
                    input=x_b,
                    mask_backup_task=mask_backup_task,
                    main_bn=conv_bn3 if block.batch_normalization else None,
                    backup_bn=backup_bn3,
                )
                identity_b = self._route_backup_layer_output_by_mask(
                    layer_name=block.full_3rd_layer_name,
                    main_layer=block.identity_downsample,
                    backup_layer=backup_block.identity_downsample,
                    input=x_b_input,
                    mask_backup_task=mask_backup_task,
                )
                x_b = x_b + identity_b
                x_b = x_b * mask_3
                if block.activation:
                    x_b = block.conv_activation3(x_b)

                x_backup[backup_task_id] = x_b

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
            1. 'train_with_backup': training stage with training backup backbones for future unlearning backup compensation.
            2. 'train': training stage.
            3. 'validation': validation stage.
            4. 'test': testing stage.
            5. 'unlearning_test': unlearning testing stage.
        - **s_max** (`float` | `None`): The maximum scaling factor in the gate function. Doesn't apply to the testing stage.
        - **batch_idx** (`int` | `None`): The current batch index. Applies only to the training stage.
        - **num_batches** (`int` | `None`): The total number of batches. Applies only to the training stage.
        - **test_task_id** (`int` | `None`): The test task ID. Applies only to the testing stage.

        **Returns:**
        - **output_feature** (`Tensor`): The output feature tensor to be passed into heads. This is the main target of backpropagation.
        - **output_backup_feature** (`Tensor`): The output feature tensor from the backup backbone masked by cumulative mask. Applies only during training when unlearning is enabled.
        - **mask** (`dict[str, Tensor]`): The mask for the current task. Keys are layer names and values are the mask tensors.
        - **activations** (`dict[str, Tensor]`): The hidden features (after activation) in each weighted layer. This is used for continual learning algorithms that need hidden features.
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
        stage_for_bn = "test" if stage in ("test", "unlearning_test") else "train"
        if self.batch_normalization:
            test_task_id_for_bn = self.task_id if test_task_id is None else test_task_id
            if self.batch_normalization == "independent" and stage_for_bn == "test":
                conv_bn1 = self.conv_bn1s[f"{test_task_id_for_bn}"]
            else:
                conv_bn1 = self.conv_bn1

        x = input
        x_backup = None
        if stage == "train_with_backup":
            x_backup = {
                backup_task_id: input for backup_task_id in self.backup_task_ids
            }

        mask_conv1 = mask["conv1"].view(1, -1, 1, 1)

        x = self.conv1(x)
        if self.batch_normalization:
            x = conv_bn1(x)
        if stage == "train_with_backup":
            for backup_task_id in self.backup_task_ids:
                backup_backbone = self.backup_backbones[f"{backup_task_id}"]
                mask_backup_task = self.masks[backup_task_id]
                backup_bn1 = (
                    backup_backbone.conv_bn1
                    if backup_backbone.batch_normalization
                    else None
                )
                x_backup[backup_task_id] = self._route_backup_layer_output_by_mask(
                    layer_name="conv1",
                    main_layer=self.conv1,
                    backup_layer=backup_backbone.conv1,
                    input=x_backup[backup_task_id],
                    mask_backup_task=mask_backup_task,
                    main_bn=conv_bn1 if self.batch_normalization else None,
                    backup_bn=backup_bn1,
                )

        x = x * mask_conv1
        if stage == "train_with_backup":
            for backup_task_id in self.backup_task_ids:
                x_backup[backup_task_id] = x_backup[backup_task_id] * mask_conv1
        if self.activation:
            x = self.conv_activation1(x)
            if stage == "train_with_backup":
                for backup_task_id in self.backup_task_ids:
                    x_backup[backup_task_id] = self.conv_activation1(
                        x_backup[backup_task_id]
                    )
        activations["conv1"] = x

        x = self.maxpool(x)
        if stage == "train_with_backup":
            for backup_task_id in self.backup_task_ids:
                x_backup[backup_task_id] = self.maxpool(x_backup[backup_task_id])

        for layer_name in ("conv2x", "conv3x", "conv4x", "conv5x"):
            layer = getattr(self, layer_name)
            for block_idx, block in enumerate(layer):
                backup_blocks = None
                if stage == "train_with_backup":
                    backup_blocks = {
                        backup_task_id: getattr(
                            self.backup_backbones[f"{backup_task_id}"], layer_name
                        )[block_idx]
                        for backup_task_id in self.backup_task_ids
                    }

                if isinstance(block, HATMaskResNetBlockSmall):
                    x, x_backup, activations_block = self._forward_block_small(
                        block=block,
                        backup_blocks=backup_blocks,
                        x=x,
                        mask=mask,
                        x_backup=x_backup,
                        stage=stage,
                        test_task_id=test_task_id,
                    )
                else:
                    x, x_backup, activations_block = self._forward_block_large(
                        block=block,
                        backup_blocks=backup_blocks,
                        x=x,
                        mask=mask,
                        x_backup=x_backup,
                        stage=stage,
                        test_task_id=test_task_id,
                    )

                activations.update(activations_block)

        x = self.avepool(x)
        output_feature = x.view(batch_size, -1)

        if stage == "train_with_backup":
            output_backup_feature = {}
            for backup_task_id, x_b in x_backup.items():
                x_b = self.avepool(x_b)
                output_backup_feature[backup_task_id] = x_b.view(batch_size, -1)
            return output_feature, output_backup_feature, mask, activations

        return output_feature, mask, activations


class AmnesiacHATResNet18(AmnesiacHATResNetBase):
    r"""AmnesiacHAT masked ResNet-18 backbone network.

    [HAT (Hard Attention to the Task, 2018)](http://proceedings.mlr.press/v80/serra18a) is an architecture-based continual learning approach that uses learnable hard attention masks to select the task-specific parameters. AmnesiacHAT extends HAT to support unlearning of specific tasks while retaining performance on other tasks by training parallel backup backbones for unlearning compensation.

    ResNet-18 is a smaller architecture proposed in the [original ResNet paper](https://www.cv-foundation.org/openaccess/content_cvpr_2016/html/He_Deep_Residual_Learning_CVPR_2016_paper.html). It consists of 18 weight convolutional layers in total. See Table 1 in the paper for details.

    Mask is applied to the units which are output channels in each weighted convolutional layer. The mask is generated from the neuron-wise task embedding and gate function.
    """

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
        r"""Construct and initialize the ResNet-18 backbone network with task embedding. Batch normalization can be shared or independent per task.

        **Args:**
        - **input_channels** (`int`): the number of channels of input of this building block. Note that convolutional networks require number of input channels instead of dimension.
        - **output_dim** (`int`): the output dimension after flattening at last which connects to CL output heads. Although this is not determined by us but the architecture built before the flattening layer, we still need to provide this to construct the heads.
        - **gate** (`str`): the type of gate function turning the real value task embeddings into attention masks; one of:
            - `sigmoid`: the sigmoid function.
        - **activation_layer** (`nn.Module`): activation function of each layer (if not `None`), if `None` this layer won't be used. Default `nn.ReLU`.
        - **batch_normalization** (`str` | `None`): How to use batch normalization after the convolutional layers; one of:
            - `None`: no batch normalization layers.
            - `shared`: use a single batch normalization layer for all tasks. Note that this can cause catastrophic forgetting.
            - `independent`: use independent batch normalization layers for each task.
        - **bias** (`bool`): whether to use bias in the convolutional layer. Default `False`.
        - **pretrained_weights** (`str`): the name of pretrained weights to be loaded. See [TorchVision docs](https://pytorch.org/vision/main/models.html). If `None`, no pretrained weights are loaded. Default `None`.
        - **disable_unlearning** (`bool`): whether to disable unlearning. This is used in reference experiments following continual learning pipeline. Default is `False`.
        - **kwargs**: Reserved for multiple inheritance.
        """
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
    r"""AmnesiacHAT masked ResNet-34 backbone network.

    [HAT (Hard Attention to the Task, 2018)](http://proceedings.mlr.press/v80/serra18a) is an architecture-based continual learning approach that uses learnable hard attention masks to select the task-specific parameters. AmnesiacHAT extends HAT to support unlearning of specific tasks while retaining performance on other tasks by training parallel backup backbones for unlearning compensation.

    ResNet-34 is a smaller architecture proposed in the [original ResNet paper](https://www.cv-foundation.org/openaccess/content_cvpr_2016/html/He_Deep_Residual_Learning_CVPR_2016_paper.html). It consists of 34 weight convolutional layers in total. See Table 1 in the paper for details.

    Mask is applied to the units which are output channels in each weighted convolutional layer. The mask is generated from the neuron-wise task embedding and gate function.
    """

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
        r"""Construct and initialize the ResNet-34 backbone network with task embedding. Batch normalization can be shared or independent per task.

        **Args:**
        - **input_channels** (`int`): the number of channels of input of this building block. Note that convolutional networks require number of input channels instead of dimension.
        - **output_dim** (`int`): the output dimension after flattening at last which connects to CL output heads. Although this is not determined by us but the architecture built before the flattening layer, we still need to provide this to construct the heads.
        - **gate** (`str`): the type of gate function turning the real value task embeddings into attention masks; one of:
            - `sigmoid`: the sigmoid function.
        - **activation_layer** (`nn.Module`): activation function of each layer (if not `None`), if `None` this layer won't be used. Default `nn.ReLU`.
        - **batch_normalization** (`str` | `None`): How to use batch normalization after the convolutional layers; one of:
            - `None`: no batch normalization layers.
            - `shared`: use a single batch normalization layer for all tasks. Note that this can cause catastrophic forgetting.
            - `independent`: use independent batch normalization layers for each task.
        - **bias** (`bool`): whether to use bias in the convolutional layer. Default `False`.
        - **disable_unlearning** (`bool`): whether to disable unlearning. This is used in reference experiments following continual learning pipeline. Default is `False`.
        - **kwargs**: Reserved for multiple inheritance.
        """
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
    r"""AmnesiacHAT masked ResNet-50 backbone network.

    [HAT (Hard Attention to the Task, 2018)](http://proceedings.mlr.press/v80/serra18a) is an architecture-based continual learning approach that uses learnable hard attention masks to select the task-specific parameters. AmnesiacHAT extends HAT to support unlearning of specific tasks while retaining performance on other tasks by training parallel backup backbones for unlearning compensation.

    ResNet-50 is a larger architecture proposed in the [original ResNet paper](https://www.cv-foundation.org/openaccess/content_cvpr_2016/html/He_Deep_Residual_Learning_CVPR_2016_paper.html). It consists of 50 weight convolutional layers in total. See Table 1 in the paper for details.

    Mask is applied to the units which are output channels in each weighted convolutional layer. The mask is generated from the neuron-wise task embedding and gate function.
    """

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
        r"""Construct and initialize the ResNet-50 backbone network with task embedding. Batch normalization can be shared or independent per task.

        **Args:**
        - **input_channels** (`int`): the number of channels of input of this building block. Note that convolutional networks require number of input channels instead of dimension.
        - **output_dim** (`int`): the output dimension after flattening at last which connects to CL output heads. Although this is not determined by us but the architecture built before the flattening layer, we still need to provide this to construct the heads.
        - **gate** (`str`): the type of gate function turning the real value task embeddings into attention masks; one of:
            - `sigmoid`: the sigmoid function.
        - **activation_layer** (`nn.Module`): activation function of each layer (if not `None`), if `None` this layer won't be used. Default `nn.ReLU`.
        - **batch_normalization** (`str` | `None`): How to use batch normalization after the convolutional layers; one of:
            - `None`: no batch normalization layers.
            - `shared`: use a single batch normalization layer for all tasks. Note that this can cause catastrophic forgetting.
            - `independent`: use independent batch normalization layers for each task.
        - **bias** (`bool`): whether to use bias in the convolutional layer. Default `False`.
        - **disable_unlearning** (`bool`): whether to disable unlearning. This is used in reference experiments following continual learning pipeline. Default is `False`.
        - **kwargs**: Reserved for multiple inheritance.
        """
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
    r"""AmnesiacHAT masked ResNet-101 backbone network.

    [HAT (Hard Attention to the Task, 2018)](http://proceedings.mlr.press/v80/serra18a) is an architecture-based continual learning approach that uses learnable hard attention masks to select the task-specific parameters. AmnesiacHAT extends HAT to support unlearning of specific tasks while retaining performance on other tasks by training parallel backup backbones for unlearning compensation.

    ResNet-101 is a larger architecture proposed in the [original ResNet paper](https://www.cv-foundation.org/openaccess/content_cvpr_2016/html/He_Deep_Residual_Learning_CVPR_2016_paper.html). It consists of 101 weight convolutional layers in total. See Table 1 in the paper for details.

    Mask is applied to the units which are output channels in each weighted convolutional layer. The mask is generated from the neuron-wise task embedding and gate function.
    """

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
        r"""Construct and initialize the ResNet-101 backbone network with task embedding. Batch normalization can be shared or independent per task.

        **Args:**
        - **input_channels** (`int`): the number of channels of input of this building block. Note that convolutional networks require number of input channels instead of dimension.
        - **output_dim** (`int`): the output dimension after flattening at last which connects to CL output heads. Although this is not determined by us but the architecture built before the flattening layer, we still need to provide this to construct the heads.
        - **gate** (`str`): the type of gate function turning the real value task embeddings into attention masks; one of:
            - `sigmoid`: the sigmoid function.
        - **activation_layer** (`nn.Module`): activation function of each layer (if not `None`), if `None` this layer won't be used. Default `nn.ReLU`.
        - **batch_normalization** (`str` | `None`): How to use batch normalization after the convolutional layers; one of:
            - `None`: no batch normalization layers.
            - `shared`: use a single batch normalization layer for all tasks. Note that this can cause catastrophic forgetting.
            - `independent`: use independent batch normalization layers for each task.
        - **bias** (`bool`): whether to use bias in the convolutional layer. Default `False`.
        - **disable_unlearning** (`bool`): whether to disable unlearning. This is used in reference experiments following continual learning pipeline. Default is `False`.
        - **kwargs**: Reserved for multiple inheritance.
        """
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
    r"""AmnesiacHAT masked ResNet-152 backbone network.

    [HAT (Hard Attention to the Task, 2018)](http://proceedings.mlr.press/v80/serra18a) is an architecture-based continual learning approach that uses learnable hard attention masks to select the task-specific parameters. AmnesiacHAT extends HAT to support unlearning of specific tasks while retaining performance on other tasks by training parallel backup backbones for unlearning compensation.

    ResNet-152 is the largest architecture proposed in the [original ResNet paper](https://www.cv-foundation.org/openaccess/content_cvpr_2016/html/He_Deep_Residual_Learning_CVPR_2016_paper.html). It consists of 152 weight convolutional layers in total. See Table 1 in the paper for details.

    Mask is applied to the units which are output channels in each weighted convolutional layer. The mask is generated from the neuron-wise task embedding and gate function.
    """

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
        r"""Construct and initialize the ResNet-152 backbone network with task embedding. Batch normalization can be shared or independent per task.

        **Args:**
        - **input_channels** (`int`): the number of channels of input of this building block. Note that convolutional networks require number of input channels instead of dimension.
        - **output_dim** (`int`): the output dimension after flattening at last which connects to CL output heads. Although this is not determined by us but the architecture built before the flattening layer, we still need to provide this to construct the heads.
        - **gate** (`str`): the type of gate function turning the real value task embeddings into attention masks; one of:
            - `sigmoid`: the sigmoid function.
        - **activation_layer** (`nn.Module`): activation function of each layer (if not `None`), if `None` this layer won't be used. Default `nn.ReLU`.
        - **batch_normalization** (`str` | `None`): How to use batch normalization after the convolutional layers; one of:
            - `None`: no batch normalization layers.
            - `shared`: use a single batch normalization layer for all tasks. Note that this can cause catastrophic forgetting.
            - `independent`: use independent batch normalization layers for each task.
        - **bias** (`bool`): whether to use bias in the convolutional layer. Default `False`.
        - **disable_unlearning** (`bool`): whether to disable unlearning. This is used in reference experiments following continual learning pipeline. Default is `False`.
        - **kwargs**: Reserved for multiple inheritance.
        """
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
