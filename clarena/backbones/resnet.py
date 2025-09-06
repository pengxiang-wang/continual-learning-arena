r"""
The submodule in `backbones` for ResNet backbone network. It includes multiple versions of ResNet, including the basic ResNet, the continual learning ResNet, and the [HAT](http://proceedings.mlr.press/v80/serra18a) masked ResNet.
"""

__all__ = [
    "ResNetBlockSmall",
    "ResNetBlockLarge",
    "ResNetBase",
    "ResNet18",
    "ResNet34",
    "ResNet50",
    "ResNet101",
    "ResNet152",
    "CLResNet18",
    "CLResNet34",
    "CLResNet50",
    "CLResNet101",
    "CLResNet152",
    "HATMaskResNetBlockSmall",
    "HATMaskResNetBlockLarge",
    "HATMaskResNetBase",
    "HATMaskResNet18",
    "HATMaskResNet34",
    "HATMaskResNet50",
    "HATMaskResNet101",
    "HATMaskResNet152",
]


import logging

import torchvision
from torch import Tensor, nn

from clarena.backbones import Backbone, CLBackbone, HATMaskBackbone
from clarena.backbones.constants import (
    HATMASKRESNET18_STATE_DICT_MAPPING,
    RESNET18_STATE_DICT_MAPPING,
)

# always get logger for built-in logging in each module
pylogger = logging.getLogger(__name__)


class ResNetBlockSmall(Backbone):
    r"""The smaller building block for ResNet-18/34.

    It consists of 2 weight convolutional layers, each followed by an activation function. See Table 1 or Figure 5 (left) in the [original ResNet paper](https://www.cv-foundation.org/openaccess/content_cvpr_2016/html/He_Deep_Residual_Learning_CVPR_2016_paper.html).
    """

    def __init__(
        self,
        outer_layer_name: str,
        block_idx: int,
        preceding_output_channels: int,
        input_channels: int,
        overall_stride: int,
        activation_layer: nn.Module | None = nn.ReLU,
        batch_normalization: bool = True,
        bias: bool = False,
        **kwargs,
    ) -> None:
        r"""Construct and initialize the smaller building block.

        **Args:**
        - **outer_layer_name** (`str`): pass the name of the multi-building-block layer that contains this building block to construct the full name of each weighted convolutional layer.
        - **block_idx** (`int`): the index of the building blocks in the multi-building-block layer to construct the full name of each weighted convolutional layer.
        - **preceding_output_channels** (`int`): the number of channels of preceding output of this particular building block.
        - **input_channels** (`int`): the number of channels of input of this building block.
        - **overall_stride** (`int`): the overall stride of this building block. This stride is performed at 2nd (last) convolutional layer where the 1st convolutional layer remain stride of 1.
        - **activation_layer** (`nn.Module`): activation function of each layer (if not `None`), if `None` this layer won't be used. Default `nn.ReLU`.
        - **batch_normalization** (`bool`): whether to use batch normalization after convolutional layers. Default `True`, same as what the [original ResNet paper](https://www.cv-foundation.org/openaccess/content_cvpr_2016/html/He_Deep_Residual_Learning_CVPR_2016_paper.html) does.
        - **bias** (`bool`): whether to use bias in the convolutional layer. Default `False`, because batch normalization are doing the similar thing with bias.
        - **kwargs**: Reserved for multiple inheritance.
        """
        super().__init__(output_dim=None)

        self.batch_normalization: bool = batch_normalization
        r"""Whether to use batch normalization after convolutional layers."""
        self.activation: bool = activation_layer is not None
        r"""Whether to use activation function after convolutional layers."""

        self.full_1st_layer_name = f"{outer_layer_name}/{block_idx}/conv1"
        r"""Format and store full name of the 1st weighted convolutional layer. """
        self.full_2nd_layer_name = f"{outer_layer_name}/{block_idx}/conv2"
        r"""Format and store full name of the 2nd weighted convolutional layer. """

        # construct the 1st weighted convolutional layer and attached layers (batchnorm, activation, etc)
        layer_input_channels = preceding_output_channels  # the input channels of the 1st convolutional layer, which receive the output channels of the preceding module
        layer_output_channels = (
            input_channels  # the output channels of the 1st convolutional layer
        )
        self.conv1 = nn.Conv2d(
            in_channels=layer_input_channels,
            out_channels=layer_output_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=bias,
        )  # construct the 1st weight convolutional layer of the smaller building block. Overall stride is not performed here
        r"""The 1st weight convolutional layer of the smaller building block. """
        self.weighted_layer_names.append(
            self.full_1st_layer_name
        )  # update the weighted layer names
        if self.batch_normalization:
            self.conv_bn1 = nn.BatchNorm2d(
                num_features=layer_output_channels
            )  # construct the batch normalization layer
            r"""The batch normalization (`nn.BatchNorm2d`) layer after the 1st weighted convolutional layer. """
        if self.activation:
            self.conv_activation1 = activation_layer()  # construct the activation layer
            r"""The activation layer after the 1st weighted convolutional layer. """

        # construct the 2nd weighted convolutional layer and attached layers (batchnorm, activation, etc)
        layer_input_channels = input_channels  # the input channels of the 2nd convolutional layer, which is `input_channels`, the same as the output channels of the 1st convolutional layer
        layer_output_channels = (
            input_channels * 1
        )  # the output channels of the 2nd convolutional layer, which is the same as the input channels (without expansion)
        self.conv2 = nn.Conv2d(
            in_channels=layer_input_channels,
            out_channels=layer_output_channels,
            kernel_size=3,
            stride=overall_stride,
            padding=1,
            bias=bias,
        )  # construct the 2nd weight convolutional layer of the smaller building block. Overall stride is performed here
        r"""The 2nd weight convolutional layer of the smaller building block. """
        self.weighted_layer_names.append(
            self.full_2nd_layer_name
        )  # update the weighted layer names
        if batch_normalization:
            self.conv_bn2 = nn.BatchNorm2d(
                num_features=layer_output_channels
            )  # construct the batch normalization layer
            r"""The batch normalization (`nn.BatchNorm2d`) layer after the 2nd weighted convolutional layer. """
        if self.activation:
            self.conv_activation2 = activation_layer()  # construct the activation layer
            r"""The activation layer after the 2nd weighted convolutional layer. """

        self.identity_downsample: nn.Module = (
            nn.Conv2d(
                in_channels=preceding_output_channels,
                out_channels=input_channels,
                kernel_size=1,
                stride=overall_stride,
                bias=False,
            )
            if preceding_output_channels != input_channels or overall_stride != 1
            else None
        )  # construct the identity downsample function
        r"""The convolutional layer for downsampling identity in the shortcut connection if the dimension of identity from input doesn't match the output's. This case only happens when the number of input channels doesn't equal to the number of preceding output channels or a layer with stride > 1 exists. """

    def forward(self, input: Tensor) -> tuple[Tensor, dict[str, Tensor]]:
        r"""The forward pass for data.

        **Args:**
        - **input** (`Tensor`): the input feature maps.

        **Returns:**
        - **output_feature** (`Tensor`): the output feature maps.
        - **activations** (`dict[str, Tensor]`): the hidden features (after activation) in each weighted layer. Keys (`str`) are the weighted layer names and values (`Tensor`) are the hidden feature tensors. This is used for the continual learning algorithms that need to use the hidden features for various purposes.
        """
        activations = {}

        identity = (
            self.identity_downsample(input)
            if self.identity_downsample is not None
            else input
        )  # remember the identity of input for the shortcut connection. Perform downsampling if its dimension doesn't match the output's

        x = input
        x = self.conv1(x)
        if self.batch_normalization:
            x = self.conv_bn1(
                x
            )  # batch normalization can be before or after activation. We put it before activation here
        if self.activation:
            x = self.conv_activation1(x)
        activations[self.full_1st_layer_name] = x  # store the hidden feature

        x = self.conv2(x)
        if self.batch_normalization:
            x = self.conv_bn2(
                x
            )  # batch normalization can be before or after activation. We put it before activation here

        x = x + identity
        if self.activation:
            x = self.conv_activation2(x)  # activation after the shortcut connection
        activations[self.full_2nd_layer_name] = x  # store the hidden feature

        output_feature = x

        return output_feature, activations


class ResNetBlockLarge(Backbone):
    r"""The larger building block for ResNet-50/101/152. It is referred to "bottleneck" building block in the paper.

    It consists of 3 weight convolutional layers, each followed by an activation function. See Table 1 or Figure 5 (right) in the [original ResNet paper](https://www.cv-foundation.org/openaccess/content_cvpr_2016/html/He_Deep_Residual_Learning_CVPR_2016_paper.html).
    """

    def __init__(
        self,
        outer_layer_name: str,
        block_idx: int,
        preceding_output_channels: int,
        input_channels: int,
        overall_stride: int,
        activation_layer: nn.Module | None = nn.ReLU,
        batch_normalization: bool = True,
        bias: bool = False,
        **kwargs,
    ) -> None:
        r"""Construct and initialize the larger building block.

        **Args:**
        - **outer_layer_name** (`str`): pass the name of the multi-building-block layer that contains this building block to construct the full name of each weighted convolutional layer.
        - **block_idx** (`int`): the index of the building blocks in the multi-building-block layer to construct the full name of each weighted convolutional layer.
        - **preceding_output_channels** (`int`): the number of channels of preceding output of this particular building block.
        - **input_channels** (`int`): the number of channels of input of this building block.
        - **overall_stride** (`int`): the overall stride of this building block. This stride is performed at 2nd (middle) convolutional layer where 1st and 3rd convolutional layers remain stride of 1.
        - **activation_layer** (`nn.Module`): activation function of each layer (if not `None`), if `None` this layer won't be used. Default `nn.ReLU`.
        - **batch_normalization** (`bool`): whether to use batch normalization after convolutional layers. Default `True`, same as what the [original ResNet paper](https://www.cv-foundation.org/openaccess/content_cvpr_2016/html/He_Deep_Residual_Learning_CVPR_2016_paper.html) does.
        - **bias** (`bool`): whether to use bias in the convolutional layer. Default `False`, because batch normalization are doing the similar thing with bias.
        - **kwargs**: Reserved for multiple inheritance.
        """
        super().__init__(output_dim=None)

        self.batch_normalization: bool = batch_normalization
        r"""Whether to use batch normalization after convolutional layers."""
        self.activation: bool = activation_layer is not None
        r"""Whether to use activation function after convolutional layers."""

        self.full_1st_layer_name = f"{outer_layer_name}/{block_idx}/conv1"
        r"""Format and store full name of the 1st weighted convolutional layer. """
        self.full_2nd_layer_name = f"{outer_layer_name}/{block_idx}/conv2"
        r"""Format and store full name of the 2nd weighted convolutional layer. """
        self.full_3rd_layer_name = f"{outer_layer_name}/{block_idx}/conv3"
        r"""Format and store full name of the 3rd weighted convolutional layer. """

        # construct the 1st weighted convolutional layer and attached layers (batchnorm, activation, etc)
        layer_input_channels = preceding_output_channels  # the input channels of the 1st convolutional layer, which receive the output channels of the preceding module
        layer_output_channels = (
            input_channels  # the output channels of the 1st convolutional layer
        )
        self.conv1 = nn.Conv2d(
            in_channels=layer_input_channels,
            out_channels=layer_output_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=bias,
        )  # construct the 1st weight convolutional layer of the larger building block. Overall stride is not performed here
        r"""The 1st weight convolutional layer of the larger building block. """
        self.weighted_layer_names.append(
            self.full_1st_layer_name
        )  # update the weighted layer names
        if self.batch_normalization:
            self.conv_bn1 = nn.BatchNorm2d(
                num_features=layer_output_channels
            )  # construct the batch normalization layer
            r"""The batch normalization (`nn.BatchNorm2d`) layer after the 1st weighted convolutional layer. """
        if self.activation:
            self.conv_activation1 = activation_layer()  # construct the activation layer
            r"""The activation layer after the 1st weighted convolutional layer. """

        # construct the 2nd weighted convolutional layer and attached layers (batchnorm, activation, etc)
        layer_input_channels = input_channels  # the input channels of the 2nd convolutional layer, which is `input_channels`, the same as the output channels of the 1st convolutional layer
        layer_output_channels = (
            input_channels
            * 1  # the output channels of the 2nd convolutional layer, which is the same as the input channels (without expansion)
        )
        self.conv2 = nn.Conv2d(
            in_channels=layer_input_channels,
            out_channels=layer_output_channels,
            kernel_size=3,
            stride=overall_stride,
            padding=1,
            bias=bias,
        )  # construct the 2nd weight convolutional layer of the larger building block. Overall stride is performed here
        r"""The 2nd weight convolutional layer of the larger building block. """
        self.weighted_layer_names.append(
            self.full_2nd_layer_name
        )  # update the weighted layer names
        if self.batch_normalization:
            self.conv_bn2 = nn.BatchNorm2d(
                num_features=layer_output_channels
            )  # construct the batch normalization layer
            r"""The batch normalization (`nn.BatchNorm2d`) layer after the 2nd weighted convolutional layer. """
        if self.activation:
            self.conv_activation2 = activation_layer()  # construct the activation layer
            r"""The activation layer after the 2nd weighted convolutional layer. """

        # construct the 3rd weighted convolutional layer and attached layers (batchnorm, activation, etc)
        layer_input_channels = (
            input_channels * 1
        )  # the input channels of the 3rd (final) convolutional layer, same as output of 2nd layer
        layer_output_channels = (
            input_channels * 4  # the output channels of the 3rd layer (4x expansion)
        )
        self.conv3 = nn.Conv2d(
            in_channels=layer_input_channels,
            out_channels=layer_output_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=bias,
        )  # construct the 3rd weight convolutional layer of the larger building block. Overall stride is not performed here
        r"""The 3rd weight convolutional layer of the larger building block. """
        self.weighted_layer_names.append(
            self.full_3rd_layer_name
        )  # update the weighted layer names
        if batch_normalization:
            self.conv_bn3 = nn.BatchNorm2d(
                num_features=layer_output_channels
            )  # construct the batch normalization layer
            r"""The batch normalization (`nn.BatchNorm2d`) layer after the 3rd weighted convolutional layer. """
        if self.activation:
            self.conv_activation3 = activation_layer()  # construct the activation layer
            r"""The activation layer after the 3rd weighted convolutional layer. """

        self.identity_downsample: nn.Module = (
            nn.Conv2d(
                in_channels=preceding_output_channels,
                out_channels=input_channels * 4,
                kernel_size=1,
                stride=overall_stride,
                bias=False,
            )
            if preceding_output_channels != input_channels * 4 or overall_stride != 1
            else None
        )
        r"""The convolutional layer for downsampling identity in the shortcut connection if the dimension of identity from input doesn't match the output's. This case only happens when the number of input channels doesn't equal to the number of preceding output channels or a layer with stride > 1 exists. """

    def forward(self, input: Tensor) -> tuple[Tensor, dict[str, Tensor]]:
        r"""The forward pass for data.

        **Args:**
        - **input** (`Tensor`): the input feature maps.

        **Returns:**
        - **output_feature** (`Tensor`): the output feature maps.
        - **activations** (`dict[str, Tensor]`): the hidden features (after activation) in each weighted layer. Keys (`str`) are the weighted layer names and values (`Tensor`) are the hidden feature tensors. This is used for the continual learning algorithms that need to use the hidden features for various purposes.
        """
        activations = {}

        identity = (
            self.identity_downsample(input)
            if self.identity_downsample is not None
            else input
        )  # remember the identity of input for the shortcut connection. Perform downsampling if its dimension doesn't match the output's

        x = input
        x = self.conv1(x)
        if self.batch_normalization:
            x = self.conv_bn1(
                x
            )  # batch normalization can be before or after activation. We put it before activation here
        if self.activation:
            x = self.conv_activation1(x)
        activations[self.full_1st_layer_name] = x  # store the hidden feature

        x = self.conv2(x)
        if self.batch_normalization:
            x = self.conv_bn2(
                x
            )  # batch normalization can be before or after activation. We put it before activation here
        if self.activation:
            x = self.conv_activation2(x)
        activations[self.full_2nd_layer_name] = x  # store the hidden feature

        x = self.conv3(x)
        if self.batch_normalization:
            x = self.conv_bn3(
                x
            )  # batch normalization can be before or after activation. We put it before activation here

        x = x + identity
        if self.activation:
            x = self.conv_activation3(x)  # activation after the shortcut connection
        activations[self.full_3rd_layer_name] = x  # store the hidden feature

        output_feature = x

        return output_feature, activations


class ResNetBase(Backbone):
    r"""The base class of [residual network (ResNet)](https://www.cv-foundation.org/openaccess/content_cvpr_2016/html/He_Deep_Residual_Learning_CVPR_2016_paper.html).

    ResNet is a convolutional network architecture, which has 1st convolutional parameter layer and a maxpooling layer, connecting to 4 convolutional layers which contains multiple convolutional parameter layer. Each layer of the 4 are constructed from basic building blocks which are either small (`ResNetBlockSmall`) or large (`ResNetBlockLarge`). Each building block contains several convolutional parameter layers. The building blocks are connected by a skip connection which is a direct connection from the input of the block to the output of the block, and this is why it's called residual (find "shortcut connections" in the paper for more details). After the 5th convolutional layer, there are average pooling layer and a fully connected layer which connects to the CL output heads.
    """

    def __init__(
        self,
        input_channels: int,
        building_block_type: ResNetBlockSmall | ResNetBlockLarge,
        building_block_nums: tuple[int, int, int, int],
        building_block_preceding_output_channels: tuple[int, int, int, int],
        building_block_input_channels: tuple[int, int, int, int],
        output_dim: int,
        activation_layer: nn.Module | None = nn.ReLU,
        batch_normalization: bool = True,
        bias: bool = False,
        **kwargs,
    ) -> None:
        r"""Construct and initialize the ResNet backbone network.

        **Args:**
        - **input_channels** (`int`): the number of channels of input. Image data are kept channels when going in ResNet. Note that convolutional networks require number of input channels instead of dimension.
        - **building_block_type** (`ResNetBlockSmall` | `ResNetBlockLarge`): the type of building block used in the ResNet.
        - **building_block_nums** (`tuple[int, int, int, int]`): the number of building blocks in the 2-5 convolutional layer correspondingly.
        - **building_block_preceding_output_channels** (`tuple[int, int, int, int]`): the number of channels of preceding output of each building block in the 2-5 convolutional layer correspondingly.
        - **building_block_input_channels** (`tuple[int, int, int, int]`): the number of channels of input of each building block in the 2-5 convolutional layer correspondingly.
        - **output_dim** (`int`): the output dimension after flattening at last which connects to CL output heads. Although this is not determined by us but the architecture built before the flattening layer, we still need to provide this to construct the heads.
        - **activation_layer** (`nn.Module`): activation function of each layer (if not `None`), if `None` this layer won't be used. Default `nn.ReLU`.
        - **batch_normalization** (`bool`): whether to use batch normalization after convolutional layers. Default `True`, same as what the [original ResNet paper](https://www.cv-foundation.org/openaccess/content_cvpr_2016/html/He_Deep_Residual_Learning_CVPR_2016_paper.html) does.
        - **bias** (`bool`): whether to use bias in the convolutional layer. Default `False`, because batch normalization are doing the similar thing with bias.
        - **kwargs**: Reserved for multiple inheritance.
        """
        super().__init__(output_dim=output_dim)

        self.batch_normalization: bool = batch_normalization
        r"""Whether to use batch normalization after convolutional layers."""
        self.activation: bool = activation_layer is not None
        r"""Whether to use activation function after convolutional layers."""

        # construct the 1st weighted convolutional layer and attached layers (batchnorm, activation, etc)
        layer_input_channels = input_channels  # the input channels of the 1st convolutional layer, which receive the input of the entire network
        layer_output_channels = 64  # the output channels of the 1st convolutional layer
        self.conv1 = nn.Conv2d(
            in_channels=layer_input_channels,
            out_channels=layer_output_channels,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=bias,
        )  # construct the 1st weight convolutional layer of the entire ResNet
        r"""The 1st weight convolutional layer of the entire ResNet. It  is always with fixed kernel size 7x7, stride 2, and padding 3. """
        self.weighted_layer_names.append("conv1")  # collect the layer name to be masked
        if self.batch_normalization:
            self.conv_bn1 = nn.BatchNorm2d(
                num_features=layer_output_channels
            )  # construct the batch normalization layer
            r"""The batch normalization (`nn.BatchNorm2d`) layer after the 1st weighted convolutional layer. """
        if self.activation:
            self.conv_activation1 = activation_layer()  # construct the activation layer
            r"""The activation layer after the 1st weighted convolutional layer. """

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  #
        r"""The max pooling layer which is laid in between 1st and 2nd convolutional layers with kernel size 3x3, stride 2. """

        # construct the 2nd convolutional layer with multiple blocks, and its attached layers (batchnorm, activation, etc)
        self.conv2x = self._multiple_blocks(
            layer_name="conv2x",
            building_block_type=building_block_type,
            building_block_num=building_block_nums[0],
            preceding_output_channels=building_block_preceding_output_channels[0],
            input_channels=building_block_input_channels[0],
            overall_stride=1,  # the overall stride of the 2nd convolutional layer should be 1, as the preceding maxpooling layer has stride 2, which already made 112x112 -> 56x56. See Table 2 in the [original ResNet paper](https://www.cv-foundation.org/openaccess/content_cvpr_2016/html/He_Deep_Residual_Learning_CVPR_2016_paper.html) for details.
            activation_layer=activation_layer,
            batch_normalization=batch_normalization,
            bias=bias,
        )
        r"""The 2nd convolutional layer of the ResNet, which contains multiple blocks. """

        # construct the 3rd convolutional layer with multiple blocks, and its attached layers (batchnorm, activation, etc)
        self.conv3x = self._multiple_blocks(
            layer_name="conv3x",
            building_block_type=building_block_type,
            building_block_num=building_block_nums[1],
            preceding_output_channels=building_block_preceding_output_channels[1],
            input_channels=building_block_input_channels[1],
            overall_stride=2,  # the overall stride of the 3rd convolutional layer should be 2, making 56x56 -> 28x28. See Table 2 in the [original ResNet paper](https://www.cv-foundation.org/openaccess/content_cvpr_2016/html/He_Deep_Residual_Learning_CVPR_2016_paper.html) for details.
            activation_layer=activation_layer,
            batch_normalization=batch_normalization,
            bias=bias,
        )
        r"""The 3rd convolutional layer of the ResNet, which contains multiple blocks. """

        # construct the 4th convolutional layer with multiple blocks, and its attached layers (batchnorm, activation, etc)
        self.conv4x = self._multiple_blocks(
            layer_name="conv4x",
            building_block_type=building_block_type,
            building_block_num=building_block_nums[2],
            preceding_output_channels=building_block_preceding_output_channels[2],
            input_channels=building_block_input_channels[2],
            overall_stride=2,  # the overall stride of the 4th convolutional layer should be 2, making 28x28 -> 14x14. See Table 2 in the [original ResNet paper](https://www.cv-foundation.org/openaccess/content_cvpr_2016/html/He_Deep_Residual_Learning_CVPR_2016_paper.html) for details.
            activation_layer=activation_layer,
            batch_normalization=batch_normalization,
            bias=bias,
        )
        r"""The 4th convolutional layer of the ResNet, which contains multiple blocks. """

        # construct the 5th convolutional layer with multiple blocks, and its attached layers (batchnorm, activation, etc)
        self.conv5x = self._multiple_blocks(
            layer_name="conv5x",
            building_block_type=building_block_type,
            building_block_num=building_block_nums[3],
            preceding_output_channels=building_block_preceding_output_channels[3],
            input_channels=building_block_input_channels[3],
            overall_stride=2,  # the overall stride of the 2nd convolutional layer should be 2, making 14x14 -> 7x7. See Table 2 in the [original ResNet paper](https://www.cv-foundation.org/openaccess/content_cvpr_2016/html/He_Deep_Residual_Learning_CVPR_2016_paper.html) for details.
            activation_layer=activation_layer,
            batch_normalization=batch_normalization,
            bias=bias,
        )
        r"""The 5th convolutional layer of the ResNet, which contains multiple blocks. """

        self.avepool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        r"""The average pooling layer which is laid after the convolutional layers and before feature maps are flattened. """

    def _multiple_blocks(
        self,
        layer_name: str,
        building_block_type: ResNetBlockSmall | ResNetBlockLarge,
        building_block_num: int,
        preceding_output_channels: int,
        input_channels: int,
        overall_stride: int,
        activation_layer: nn.Module | None = nn.ReLU,
        batch_normalization: bool = True,
        bias: bool = False,
    ) -> nn.Sequential:
        r"""Construct a layer consisting of multiple building blocks. It's used to construct the 2-5 convolutional layers of the ResNet.

        The "shortcut connections" are performed between the input and output of each building block:
        1. If the input and output of the building block have exactly the same dimensions (including number of channels and size), add the input to the output.
        2. If the input and output of the building block have different dimensions (including number of channels and size), add the input to the output after a convolutional layer to make the dimensions match.

        **Args:**
        - **layer_name** (`str`): pass the name of this multi-building-block layer to construct the full name of each weighted convolutional layer.
        - **building_block_type** (`ResNetBlockSmall` | `ResNetBlockLarge`): the type of the building block.
        - **building_block_num** (`int`): the number of building blocks in this multi-building-block layer.
        - **preceding_output_channels** (`int`): the number of channels of preceding output of this entire multi-building-block layer.
        - **input_channels** (`int`): the number of channels of input of this multi-building-block layer.
        - **overall_stride** (`int`): the overall stride of the building blocks. This stride is performed at the 1st building block where other building blocks remain their own overall stride of 1. Inside that building block, this stride is performed at certain convolutional layer in the building block where other convolutional layers remain stride of 1:
            - For `ResNetBlockSmall`, it performs at the 2nd (last) layer.
            - For `ResNetBlockLarge`, it performs at the 2nd (middle) layer.
        - **activation_layer** (`nn.Module`): activation function of each layer (if not `None`), if `None` this layer won't be used. Default `nn.ReLU`.
        - **batch_normalization** (`bool`): whether to use batch normalization after convolutional layers. Default `True`, same as what the [original ResNet paper](https://www.cv-foundation.org/openaccess/content_cvpr_2016/html/He_Deep_Residual_Learning_CVPR_2016_paper.html) does.
        - **bias** (`bool`): whether to use bias in the convolutional layer. Default `False`, because batch normalization are doing the similar thing with bias.

        **Returns:**
        - **layer** (`nn.Sequential`): the constructed layer consisting of multiple building blocks.
        """

        layer = []

        for block_idx in range(building_block_num):
            layer.append(
                building_block_type(
                    outer_layer_name=layer_name,
                    block_idx=block_idx,
                    preceding_output_channels=(
                        preceding_output_channels
                        if block_idx == 0
                        else (
                            input_channels
                            if building_block_type == ResNetBlockSmall
                            else input_channels * 4
                        )
                    ),  # if it's the 1st block in this multi-building-block layer, it should be the number of channels of the preceding output of this entire multi-building-block layer. Otherwise, it should be the number of channels from last building block where the number of channels is 4 times of the input channels for `ResNetBlockLarge` than `ResNetBlockSmall`.
                    input_channels=input_channels,
                    overall_stride=(
                        overall_stride if block_idx == 0 else 1
                    ),  # only perform the overall stride at the 1st block in this multi-building-block layer
                    activation_layer=activation_layer,
                    batch_normalization=batch_normalization,
                    bias=bias,
                )
            )

            self.weighted_layer_names += layer[
                -1
            ].weighted_layer_names  # collect the weighted layer names in the blocks and sync to the weighted layer names list in the outer network

        return nn.Sequential(*layer)

    def forward(
        self, input: Tensor, stage: str = None
    ) -> tuple[Tensor, dict[str, Tensor]]:
        r"""The forward pass for data. It is the same for all tasks.

        **Args:**
        - **input** (`Tensor`): the input tensor from data.

        **Returns:**
        - **output_feature** (`Tensor`): the output feature tensor to be passed into heads. This is the main target of backpropagation.
        - **activations** (`dict[str, Tensor]`): the hidden features (after activation) in each weighted layer. Keys (`str`) are the weighted layer names and values (`Tensor`) are the hidden feature tensors. This is used for the continual learning algorithms that need to use the hidden features for various purposes.
        """
        batch_size = input.size(0)
        activations = {}

        x = input

        x = self.conv1(x)
        if self.batch_normalization:
            x = self.conv_bn1(x)
        if self.activation:
            x = self.conv_activation1(x)
        activations["conv1"] = x

        x = self.maxpool(x)

        for block in self.conv2x:
            x, activations_block = block(x)
            activations.update(activations_block)  # store the hidden feature
        for block in self.conv3x:
            x, activations_block = block(x)
            activations.update(activations_block)  # store the hidden feature
        for block in self.conv4x:
            x, activations_block = block(x)
            activations.update(activations_block)  # store the hidden feature
        for block in self.conv5x:
            x, activations_block = block(x)
            activations.update(activations_block)  # store the hidden feature

        x = self.avepool(x)

        output_feature = x.view(batch_size, -1)  # flatten before going through heads

        return output_feature, activations


class ResNet18(ResNetBase):
    r"""ResNet-18 backbone network.

    This is a smaller architecture proposed in the [original ResNet paper](https://www.cv-foundation.org/openaccess/content_cvpr_2016/html/He_Deep_Residual_Learning_CVPR_2016_paper.html). It consists of 18 weight convolutional layers in total. See Table 1 in the paper for details.
    """

    def __init__(
        self,
        input_channels: int,
        output_dim: int,
        activation_layer: nn.Module | None = nn.ReLU,
        batch_normalization: bool = True,
        bias: bool = False,
        pretrained_weights: str | None = None,
        **kwargs,
    ) -> None:
        r"""Construct and initialize the ResNet-18 backbone network.

        **Args:**
        - **input_channels** (`int`): the number of channels of input of this building block. Note that convolutional networks require number of input channels instead of dimension.
        - **output_dim** (`int`): the output dimension after flattening at last which connects to CL output heads. Although this is not determined by us but the architecture built before the flattening layer, we still need to provide this to construct the heads.
        - **activation_layer** (`nn.Module`): activation function of each layer (if not `None`), if `None` this layer won't be used. Default `nn.ReLU`.
        - **batch_normalization** (`bool`): whether to use batch normalization after convolutional layers. Default `True`, same as what the [original ResNet paper](https://www.cv-foundation.org/openaccess/content_cvpr_2016/html/He_Deep_Residual_Learning_CVPR_2016_paper.html) does.
        - **bias** (`bool`): whether to use bias in the convolutional layer. Default `False`, because batch normalization are doing the similar thing with bias.
        - **pretrained_weights** (`str`): the name of pretrained weights to be loaded. See [TorchVision docs](https://pytorch.org/vision/main/models.html). If `None`, no pretrained weights are loaded. Default `None`.
        - **kwargs**: Reserved for multiple inheritance.
        """
        super().__init__(
            input_channels=input_channels,
            building_block_type=ResNetBlockSmall,  # use the smaller building block for ResNet-18
            building_block_nums=(2, 2, 2, 2),
            building_block_preceding_output_channels=(64, 64, 128, 256),
            building_block_input_channels=(64, 128, 256, 512),
            output_dim=output_dim,
            activation_layer=activation_layer,
            batch_normalization=batch_normalization,
            bias=bias,
        )

        if pretrained_weights is not None:
            # load the pretrained weights from TorchVision
            torchvision_resnet18_state_dict = torchvision.models.resnet18(
                weights=pretrained_weights
            ).state_dict()

            # mapping from torchvision resnet18 state dict to our ResNet18 state dict
            state_dict_converted = {}
            for key, value in torchvision_resnet18_state_dict.items():
                if RESNET18_STATE_DICT_MAPPING[key] is not None:
                    state_dict_converted[RESNET18_STATE_DICT_MAPPING[key]] = value

            self.load_state_dict(state_dict_converted, strict=False)


class ResNet34(ResNetBase):
    r"""ResNet-34 backbone network.

    This is a smaller architecture proposed in the [original ResNet paper](https://www.cv-foundation.org/openaccess/content_cvpr_2016/html/He_Deep_Residual_Learning_CVPR_2016_paper.html). It consists of 34 weight convolutional layers in total. See Table 1 in the paper for details.
    """

    def __init__(
        self,
        input_channels: int,
        output_dim: int,
        activation_layer: nn.Module | None = nn.ReLU,
        batch_normalization: bool = True,
        bias: bool = False,
        **kwargs,
    ) -> None:
        r"""Construct and initialize the ResNet-34 backbone network.

        **Args:**
        - **input_channels** (`int`): the number of channels of input of this building block. Note that convolutional networks require number of input channels instead of dimension.
        - **output_dim** (`int`): the output dimension after flattening at last which connects to CL output heads. Although this is not determined by us but the architecture built before the flattening layer, we still need to provide this to construct the heads.
        - **activation_layer** (`nn.Module`): activation function of each layer (if not `None`), if `None` this layer won't be used. Default `nn.ReLU`.
        - **batch_normalization** (`bool`): whether to use batch normalization after convolutional layers. Default `True`, same as what the [original ResNet paper](https://www.cv-foundation.org/openaccess/content_cvpr_2016/html/He_Deep_Residual_Learning_CVPR_2016_paper.html) does.
        - **bias** (`bool`): whether to use bias in the convolutional layer. Default `False`, because batch normalization are doing the similar thing with bias.
        - **kwargs**: Reserved for multiple inheritance.
        """
        super().__init__(
            input_channels=input_channels,
            building_block_type=ResNetBlockSmall,  # use the smaller building block for ResNet-34
            building_block_nums=(3, 4, 6, 3),
            building_block_preceding_output_channels=(64, 64, 128, 256),
            building_block_input_channels=(64, 128, 256, 512),
            output_dim=output_dim,
            activation_layer=activation_layer,
            batch_normalization=batch_normalization,
            bias=bias,
        )


class ResNet50(ResNetBase):
    r"""ResNet-50 backbone network.

    This is a larger architecture proposed in the [original ResNet paper](https://www.cv-foundation.org/openaccess/content_cvpr_2016/html/He_Deep_Residual_Learning_CVPR_2016_paper.html). It consists of 50 weight convolutional layers in total. See Table 1 in the paper for details.
    """

    def __init__(
        self,
        input_channels: int,
        output_dim: int,
        activation_layer: nn.Module | None = nn.ReLU,
        batch_normalization: bool = True,
        bias: bool = False,
        **kwargs,
    ) -> None:
        r"""Construct and initialize the ResNet-50 backbone network.

        **Args:**
        - **input_channels** (`int`): the number of channels of input of this building block. Note that convolutional networks require number of input channels instead of dimension.
        - **output_dim** (`int`): the output dimension after flattening at last which connects to CL output heads. Although this is not determined by us but the architecture built before the flattening layer, we still need to provide this to construct the heads.
        - **activation_layer** (`nn.Module`): activation function of each layer (if not `None`), if `None` this layer won't be used. Default `nn.ReLU`.
        - **batch_normalization** (`bool`): whether to use batch normalization after convolutional layers. Default `True`, same as what the [original ResNet paper](https://www.cv-foundation.org/openaccess/content_cvpr_2016/html/He_Deep_Residual_Learning_CVPR_2016_paper.html) does.
        - **bias** (`bool`): whether to use bias in the convolutional layer. Default `False`, because batch normalization are doing the similar thing with bias.
        - **kwargs**: Reserved for multiple inheritance.
        """
        super().__init__(
            input_channels=input_channels,
            building_block_type=ResNetBlockLarge,  # use the larger building block for ResNet-50
            building_block_nums=(3, 4, 6, 3),
            building_block_preceding_output_channels=(64, 256, 512, 1024),
            building_block_input_channels=(64, 128, 256, 512),
            output_dim=output_dim,
            activation_layer=activation_layer,
            batch_normalization=batch_normalization,
            bias=bias,
        )


class ResNet101(ResNetBase):
    r"""ResNet-101 backbone network.

    This is a larger architecture proposed in the [original ResNet paper](https://www.cv-foundation.org/openaccess/content_cvpr_2016/html/He_Deep_Residual_Learning_CVPR_2016_paper.html). It consists of 101 weight convolutional layers in total. See Table 1 in the paper for details.
    """

    def __init__(
        self,
        input_channels: int,
        output_dim: int,
        activation_layer: nn.Module | None = nn.ReLU,
        batch_normalization: bool = True,
        bias: bool = False,
        **kwargs,
    ) -> None:
        r"""Construct and initialize the ResNet-101 backbone network.

        **Args:**
        - **input_channels** (`int`): the number of channels of input of this building block. Note that convolutional networks require number of input channels instead of dimension.
        - **output_dim** (`int`): the output dimension after flattening at last which connects to CL output heads. Although this is not determined by us but the architecture built before the flattening layer, we still need to provide this to construct the heads.
        - **activation_layer** (`nn.Module`): activation function of each layer (if not `None`), if `None` this layer won't be used. Default `nn.ReLU`.
        - **batch_normalization** (`bool`): whether to use batch normalization after the weight convolutional layers. Default `True`, same as what the [original ResNet paper](https://www.cv-foundation.org/openaccess/content_cvpr_2016/html/He_Deep_Residual_Learning_CVPR_2016_paper.html) does.
        - **bias** (`bool`): whether to use bias in the convolutional layer. Default `False`, because batch normalization are doing the similar thing with bias.
        - **kwargs**: Reserved for multiple inheritance.
        """
        super().__init__(
            input_channels=input_channels,
            building_block_type=ResNetBlockLarge,  # use the larger building block for ResNet-101
            building_block_nums=(3, 4, 23, 3),
            building_block_preceding_output_channels=(64, 256, 512, 1024),
            building_block_input_channels=(64, 128, 256, 512),
            output_dim=output_dim,
            activation_layer=activation_layer,
            batch_normalization=batch_normalization,
            bias=bias,
        )


class ResNet152(ResNetBase):
    r"""ResNet-152 backbone network.

    This is the largest architecture proposed in the [original ResNet paper](https://www.cv-foundation.org/openaccess/content_cvpr_2016/html/He_Deep_Residual_Learning_CVPR_2016_paper.html). It consists of 152 weight convolutional layers in total. See Table 1 in the paper for details.
    """

    def __init__(
        self,
        input_channels: int,
        output_dim: int,
        activation_layer: nn.Module | None = nn.ReLU,
        batch_normalization: bool = True,
        bias: bool = False,
        **kwargs,
    ) -> None:
        r"""Construct and initialize the ResNet-152 backbone network.

        **Args:**
        - **input_channels** (`int`): the number of channels of input of this building block. Note that convolutional networks require number of input channels instead of dimension.
        - **output_dim** (`int`): the output dimension after flattening at last which connects to CL output heads. Although this is not determined by us but the architecture built before the flattening layer, we still need to provide this to construct the heads.
        - **activation_layer** (`nn.Module`): activation function of each layer (if not `None`), if `None` this layer won't be used. Default `nn.ReLU`.
        - **batch_normalization** (`bool`): whether to use batch normalization after the weight convolutional layers. Default `True`, same as what the [original ResNet paper](https://www.cv-foundation.org/openaccess/content_cvpr_2016/html/He_Deep_Residual_Learning_CVPR_2016_paper.html) does.
        - **bias** (`bool`): whether to use bias in the convolutional layer. Default `False`, because batch normalization are doing the similar thing with bias.
        - **kwargs**: Reserved for multiple inheritance.
        """
        super().__init__(
            input_channels=input_channels,
            building_block_type=ResNetBlockLarge,  # use the larger building block for ResNet-152
            building_block_nums=(3, 8, 36, 3),
            building_block_preceding_output_channels=(64, 256, 512, 1024),
            building_block_input_channels=(64, 128, 256, 512),
            output_dim=output_dim,
            activation_layer=activation_layer,
            batch_normalization=batch_normalization,
            bias=bias,
        )


class CLResNet18(CLBackbone, ResNet18):
    r"""The ResNet-18 backbone network for continual learning.

    This is a smaller architecture proposed in the [original ResNet paper](https://www.cv-foundation.org/openaccess/content_cvpr_2016/html/He_Deep_Residual_Learning_CVPR_2016_paper.html). It consists of 18 weight convolutional layers in total. See Table 1 in the paper for details.
    """

    def __init__(
        self,
        input_channels: int,
        output_dim: int,
        activation_layer: nn.Module | None = nn.ReLU,
        batch_normalization: bool = True,
        bias: bool = False,
        pretrained_weights: str | None = None,
        **kwargs,
    ) -> None:
        r"""Construct and initialize the ResNet-18 backbone network for continual learning.

        **Args:**
        - **input_channels** (`int`): the number of channels of input of this building block. Note that convolutional networks require number of input channels instead of dimension.
        - **output_dim** (`int`): the output dimension after flattening at last which connects to CL output heads. Although this is not determined by us but the architecture built before the flattening layer, we still need to provide this to construct the heads.
        - **activation_layer** (`nn.Module`): activation function of each layer (if not `None`), if `None` this layer won't be used. Default `nn.ReLU`.
        - **batch_normalization** (`bool`): whether to use batch normalization after the weight convolutional layers. Default `True`, same as what the [original ResNet paper](https://www.cv-foundation.org/openaccess/content_cvpr_2016/html/He_Deep_Residual_Learning_CVPR_2016_paper.html) does.
        - **bias** (`bool`): whether to use bias in the convolutional layer. Default `False`, because batch normalization are doing the similar thing with bias.
        - **pretrained_weights** (`str`): the name of pretrained weights to be loaded. See [TorchVision docs](https://pytorch.org/vision/main/models.html). If `None`, no pretrained weights are loaded. Default `None`.
        - **kwargs**: Reserved for multiple inheritance.
        """
        super().__init__(
            input_channels=input_channels,
            output_dim=output_dim,
            activation_layer=activation_layer,
            batch_normalization=batch_normalization,
            bias=bias,
            pretrained_weights=pretrained_weights,
        )


class CLResNet34(CLBackbone, ResNet34):
    r"""The ResNet-34 backbone network for continual learning.

    This is a smaller architecture proposed in the [original ResNet paper](https://www.cv-foundation.org/openaccess/content_cvpr_2016/html/He_Deep_Residual_Learning_CVPR_2016_paper.html). It consists of 34 weight convolutional layers in total. See Table 1 in the paper for details.
    """

    def __init__(
        self,
        input_channels: int,
        output_dim: int,
        activation_layer: nn.Module | None = nn.ReLU,
        batch_normalization: bool = True,
        bias: bool = False,
        pretrained_weights: str | None = None,
        **kwargs,
    ) -> None:
        r"""Construct and initialize the ResNet-34 backbone network for continual learning.

        **Args:**
        - **input_channels** (`int`): the number of channels of input of this building block. Note that convolutional networks require number of input channels instead of dimension.
        - **output_dim** (`int`): the output dimension after flattening at last which connects to CL output heads. Although this is not determined by us but the architecture built before the flattening layer, we still need to provide this to construct the heads.
        - **activation_layer** (`nn.Module`): activation function of each layer (if not `None`), if `None` this layer won't be used. Default `nn.ReLU`.
        - **batch_normalization** (`bool`): whether to use batch normalization after the weight convolutional layers. Default `True`, same as what the [original ResNet paper](https://www.cv-foundation.org/openaccess/content_cvpr_2016/html/He_Deep_Residual_Learning_CVPR_2016_paper.html) does.
        - **bias** (`bool`): whether to use bias in the convolutional layer. Default `False`, because batch normalization are doing the similar thing with bias.
        - **pretrained_weights** (`str`): the name of pretrained weights to be loaded. See [TorchVision docs](https://pytorch.org/vision/main/models.html). If `None`, no pretrained weights are loaded. Default `None`.
        - **kwargs**: Reserved for multiple inheritance.
        """
        super().__init__(
            input_channels=input_channels,
            output_dim=output_dim,
            activation_layer=activation_layer,
            batch_normalization=batch_normalization,
            bias=bias,
            pretrained_weights=pretrained_weights,
        )


class CLResNet50(CLBackbone, ResNet50):
    r"""The ResNet-50 backbone network for continual learning.

    This is a smaller architecture proposed in the [original ResNet paper](https://www.cv-foundation.org/openaccess/content_cvpr_2016/html/He_Deep_Residual_Learning_CVPR_2016_paper.html). It consists of 50 weight convolutional layers in total. See Table 1 in the paper for details.
    """

    def __init__(
        self,
        input_channels: int,
        output_dim: int,
        activation_layer: nn.Module | None = nn.ReLU,
        batch_normalization: bool = True,
        bias: bool = False,
        pretrained_weights: str | None = None,
        **kwargs,
    ) -> None:
        r"""Construct and initialize the ResNet-50 backbone network for continual learning.

        **Args:**
        - **input_channels** (`int`): the number of channels of input of this building block. Note that convolutional networks require number of input channels instead of dimension.
        - **output_dim** (`int`): the output dimension after flattening at last which connects to CL output heads. Although this is not determined by us but the architecture built before the flattening layer, we still need to provide this to construct the heads.
        - **activation_layer** (`nn.Module`): activation function of each layer (if not `None`), if `None` this layer won't be used. Default `nn.ReLU`.
        - **batch_normalization** (`bool`): whether to use batch normalization after the weight convolutional layers. Default `True`, same as what the [original ResNet paper](https://www.cv-foundation.org/openaccess/content_cvpr_2016/html/He_Deep_Residual_Learning_CVPR_2016_paper.html) does.
        - **bias** (`bool`): whether to use bias in the convolutional layer. Default `False`, because batch normalization are doing the similar thing with bias.
        - **pretrained_weights** (`str`): the name of pretrained weights to be loaded. See [TorchVision docs](https://pytorch.org/vision/main/models.html). If `None`, no pretrained weights are loaded. Default `None`.
        - **kwargs**: Reserved for multiple inheritance.
        """
        super().__init__(
            input_channels=input_channels,
            output_dim=output_dim,
            activation_layer=activation_layer,
            batch_normalization=batch_normalization,
            bias=bias,
            pretrained_weights=pretrained_weights,
        )


class CLResNet101(CLBackbone, ResNet101):
    r"""The ResNet-101 backbone network for continual learning.

    This is a larger architecture proposed in the [original ResNet paper](https://www.cv-foundation.org/openaccess/content_cvpr_2016/html/He_Deep_Residual_Learning_CVPR_2016_paper.html). It consists of 101 weight convolutional layers in total. See Table 1 in the paper for details.
    """

    def __init__(
        self,
        input_channels: int,
        output_dim: int,
        activation_layer: nn.Module | None = nn.ReLU,
        batch_normalization: bool = True,
        bias: bool = False,
        pretrained_weights: str | None = None,
        **kwargs,
    ) -> None:
        r"""Construct and initialize the ResNet-101 backbone network for continual learning.

        **Args:**
        - **input_channels** (`int`): the number of channels of input of this building block. Note that convolutional networks require number of input channels instead of dimension.
        - **output_dim** (`int`): the output dimension after flattening at last which connects to CL output heads. Although this is not determined by us but the architecture built before the flattening layer, we still need to provide this to construct the heads.
        - **activation_layer** (`nn.Module`): activation function of each layer (if not `None`), if `None` this layer won't be used. Default `nn.ReLU`.
        - **batch_normalization** (`bool`): whether to use batch normalization after the weight convolutional layers. Default `True`, same as what the [original ResNet paper](https://www.cv-foundation.org/openaccess/content_cvpr_2016/html/He_Deep_Residual_Learning_CVPR_2016_paper.html) does.
        - **bias** (`bool`): whether to use bias in the convolutional layer. Default `False`, because batch normalization are doing the similar thing with bias.
        - **pretrained_weights** (`str`): the name of pretrained weights to be loaded. See [TorchVision docs](https://pytorch.org/vision/main/models.html). If `None`, no pretrained weights are loaded. Default `None`.
        - **kwargs**: Reserved for multiple inheritance.
        """
        super().__init__(
            input_channels=input_channels,
            output_dim=output_dim,
            activation_layer=activation_layer,
            batch_normalization=batch_normalization,
            bias=bias,
            pretrained_weights=pretrained_weights,
        )


class CLResNet152(CLBackbone, ResNet152):
    r"""The ResNet-152 backbone network for continual learning.

    This is a larger architecture proposed in the [original ResNet paper](https://www.cv-foundation.org/openaccess/content_cvpr_2016/html/He_Deep_Residual_Learning_CVPR_2016_paper.html). It consists of 152 weight convolutional layers in total. See Table 1 in the paper for details.
    """

    def __init__(
        self,
        input_channels: int,
        output_dim: int,
        activation_layer: nn.Module | None = nn.ReLU,
        batch_normalization: bool = True,
        bias: bool = False,
        pretrained_weights: str | None = None,
        **kwargs,
    ) -> None:
        r"""Construct and initialize the ResNet-152 backbone network for continual learning.

        **Args:**
        - **input_channels** (`int`): the number of channels of input of this building block. Note that convolutional networks require number of input channels instead of dimension.
        - **output_dim** (`int`): the output dimension after flattening at last which connects to CL output heads. Although this is not determined by us but the architecture built before the flattening layer, we still need to provide this to construct the heads.
        - **activation_layer** (`nn.Module`): activation function of each layer (if not `None`), if `None` this layer won't be used. Default `nn.ReLU`.
        - **batch_normalization** (`bool`): whether to use batch normalization after the weight convolutional layers. Default `True`, same as what the [original ResNet paper](https://www.cv-foundation.org/openaccess/content_cvpr_2016/html/He_Deep_Residual_Learning_CVPR_2016_paper.html) does.
        - **bias** (`bool`): whether to use bias in the convolutional layer. Default `False`, because batch normalization are doing the similar thing with bias.
        - **pretrained_weights** (`str`): the name of pretrained weights to be loaded. See [TorchVision docs](https://pytorch.org/vision/main/models.html). If `None`, no pretrained weights are loaded. Default `None`.
        - **kwargs**: Reserved for multiple inheritance.
        """
        super().__init__(
            input_channels=input_channels,
            output_dim=output_dim,
            activation_layer=activation_layer,
            batch_normalization=batch_normalization,
            bias=bias,
            pretrained_weights=pretrained_weights,
        )


class HATMaskResNetBlockSmall(HATMaskBackbone, ResNetBlockSmall):
    r"""The smaller building block for HAT masked ResNet-18/34.

    It consists of 2 weight convolutional layers, each followed by an activation function. See Table 1 or Figure 5 (left) in the [original ResNet paper](https://www.cv-foundation.org/openaccess/content_cvpr_2016/html/He_Deep_Residual_Learning_CVPR_2016_paper.html).

    Mask is applied to the units which are output channels in each weighted convolutional layer. The mask is generated from the neuron-wise task embedding and gate function.
    """

    def __init__(
        self,
        outer_layer_name: str,
        block_idx: int,
        preceding_output_channels: int,
        input_channels: int,
        overall_stride: int,
        gate: str,
        activation_layer: nn.Module | None = nn.ReLU,
        batch_normalization: str | None = None,
        bias: bool = False,
        **kwargs,
    ) -> None:
        r"""Construct and initialize the smaller building block with task embedding.

        **Args:**
        - **outer_layer_name** (`str`): pass the name of the multi-building-block layer that contains this building block to construct the full name of each weighted convolutional layer.
        - **block_idx** (`int`): the index of the building blocks in the multi-building-block layer to construct the full name of each weighted convolutional layer.
        - **preceding_output_channels** (`int`): the number of channels of preceding output of this particular building block.
        - **input_channels** (`int`): the number of channels of input of this building block.
        - **overall_stride** (`int`): the overall stride of this building block. This stride is performed at 2nd (last) convolutional layer where the 1st convolutional layer remain stride of 1.
        - **gate** (`str`): the type of gate function turning the real value task embeddings into attention masks; one of:
            - `sigmoid`: the sigmoid function.
        - **activation_layer** (`nn.Module`): activation function of each layer (if not `None`), if `None` this layer won't be used. Default `nn.ReLU`.
        - **bias** (`bool`): whether to use bias in the convolutional layer. Default `False`.
        - **batch_normalization** (`str` | `None`): the way to use batch normalization after the fully-connected layers; one of:
            - `None`: no batch normalization layers.
            - `shared`: use a single batch normalization layer for all tasks. Note that this can cause catastrophic forgetting.
            - `independent`: use independent batch normalization layers for each task.
        - **bias** (`bool`): whether to use bias in the convolutional layer. Default `False`, because batch normalization are doing the similar thing with bias.
        - **kwargs**: Reserved for multiple inheritance.
        """
        super().__init__(
            output_dim=None,
            gate=gate,
            outer_layer_name=outer_layer_name,
            block_idx=block_idx,
            preceding_output_channels=preceding_output_channels,
            input_channels=input_channels,
            overall_stride=overall_stride,
            activation_layer=activation_layer,
            batch_normalization=(
                True if batch_normalization == "shared" else False
            ),  # we construct batch normalization layers separately in HATMaskBackbone
            bias=bias,
        )

        # construct the task embedding over the 1st weighted convolutional layer. It is channel-wise
        layer_output_channels = (
            input_channels  # the output channels of the 1st convolutional layer
        )
        self.task_embedding_t[self.full_1st_layer_name] = nn.Embedding(
            num_embeddings=1, embedding_dim=layer_output_channels
        )

        # construct the task embedding over the 2nd weighted convolutional layer. It is channel-wise
        layer_output_channels = (
            input_channels * 1
        )  # the output channels of the 2nd convolutional layer, which is the same as the input channels (without expansion)
        self.task_embedding_t[self.full_2nd_layer_name] = nn.Embedding(
            num_embeddings=1, embedding_dim=layer_output_channels
        )

        # construct the batch normalization layers if needed
        if batch_normalization == "independent":
            self.conv_bn1s: nn.ModuleDict = nn.ModuleDict()  # initially no heads
            r"""Independent batch normalization layers are stored independently in a `ModuleDict`. Keys are task IDs and values are the corresponding batch normalization layers for the `nn.Linear`. We use `ModuleDict` rather than `dict` to make sure `LightningModule` can track these model parameters for the purpose of, such as automatically to device, recorded in model summaries.
            
            Note that the task IDs must be string type in order to let `LightningModule` identify this part of the model. """
            self.conv_bn1: nn.BatchNorm2d | None = None
            r"""The batch normalization layer after the 1st weighted convolutional layer. It is created when the task arrives and stored in `self.conv_bn1s`."""
            self.conv_bn2s: nn.ModuleDict = nn.ModuleDict()  # initially no heads
            r"""Independent batch normalization layers are stored independently in a `ModuleDict`. Keys are task IDs and values are the corresponding batch normalization layers for the `nn.Linear`. We use `ModuleDict` rather than `dict` to make sure `LightningModule` can track these model parameters for the purpose of, such as automatically to device, recorded in model summaries.
            
            Note that the task IDs must be string type in order to let `LightningModule` identify this part of the model. """
            self.conv_bn2: nn.BatchNorm2d | None = None
            r"""The batch normalization layer after the 2nd weighted convolutional layer. It is created when the task arrives and stored in `self.conv_bn2s`."""

    def setup_task_id(self, task_id: int) -> None:
        r"""Setup about task `task_id`. This must be done before `forward()` method is called.

        **Args:**
        - **task_id** (`int`): the target task ID.
        """
        HATMaskBackbone.setup_task_id(self, task_id=task_id)

        if self.batch_normalization == "independent":
            if self.task_id not in self.fc_bns.keys():

                # construct the batch normalization layer 1st after weighted convolutional layer
                layer_output_channels = (
                    self.input_channels  # the output channels of the 1st convolutional layer
                )

                # construct the batch normalization layer
                self.conv_bn1 = nn.BatchNorm2d(num_features=layer_output_channels)

                self.fc_bns[f"{self.task_id}"] = self.fc_bn

    def forward(
        self,
        input: Tensor,
        stage: str,
        s_max: float | None = None,
        batch_idx: int | None = None,
        num_batches: int | None = None,
        test_task_id: int | None = None,
    ) -> tuple[Tensor, dict[str, Tensor], dict[str, Tensor]]:
        r"""The forward pass for data from task `self.task_id`. Task-specific mask for `self.task_id` are applied to the units which are channels in each weighted convolutional layer.

        **Args:**
        - **input** (`Tensor`): The input tensor from data.
        - **stage** (`str`): the stage of the forward pass; one of:
            1. 'train': training stage.
            2. 'validation': validation stage.
            3. 'test': testing stage.
        - **s_max** (`float` | `None`): the maximum scaling factor in the gate function. Doesn't apply to testing stage. See chapter 2.4 "Hard Attention Training" in the [HAT paper](http://proceedings.mlr.press/v80/serra18a).
        - **batch_idx** (`int` | `None`): the current batch index. Applies only to training stage. For other stages, it is default `None`.
        - **num_batches** (`int` | `None`): the total number of batches. Applies only to training stage. For other stages, it is default `None`.
        - **test_task_id** (`dict[str, Tensor]` | `None`): the test task ID. Applies only to testing stage. For other stages, it is default `None`.

        **Returns:**
        - **output_feature** (`Tensor`): the output feature maps.
        - **mask** (`dict[str, Tensor]`): the mask for the current task. Key (`str`) is layer name, value (`Tensor`) is the mask tensor. The mask tensor has size (number of units, ).
        - **activations** (`dict[str, Tensor]`): the hidden features (after activation) in each weighted layer. Keys (`str`) are the weighted layer names and values (`Tensor`) are the hidden feature tensors. This is used for the continual learning algorithms that need to use the hidden features for various purposes. Although HAT algorithm does not need this, it is still provided for API consistence for other HAT-based algorithms inherited this `forward()` method of `HAT` class.
        """
        activations = {}

        # get the mask for the current task from the task embedding in this stage
        mask = self.get_mask(
            stage=stage,
            s_max=s_max,
            batch_idx=batch_idx,
            num_batches=num_batches,
            test_task_id=test_task_id,
        )

        identity = (
            self.identity_downsample(input)
            if self.identity_downsample is not None
            else input
        )  # remember the identity of input for the shortcut connection. Perform downsampling if its dimension doesn't match the output's

        x = input
        x = self.conv1(x)  # weighted convolutional layer first
        x = x * (
            mask[self.full_1st_layer_name].view(1, -1, 1, 1)
        )  # apply the mask to the 1st convolutional layer. Broadcast the dimension of mask to match the input
        if self.activation:
            x = self.conv_activation1(x)  # activation function third
        activations[self.full_1st_layer_name] = x  # store the hidden feature

        x = self.conv2(x)  # weighted convolutional layer first
        x = x + identity
        x = x * (
            mask[self.full_2nd_layer_name].view(1, -1, 1, 1)
        )  # apply the mask to the 2nd convolutional layer after the shortcut connection. Broadcast the dimension of mask to match the input
        if self.activation:
            x = self.conv_activation2(x)  # activation after the shortcut connection
        activations[self.full_2nd_layer_name] = x  # store the hidden feature

        output_feature = x

        return output_feature, mask, activations


class HATMaskResNetBlockLarge(HATMaskBackbone, ResNetBlockLarge):
    r"""The larger building block for ResNet-50/101/152. It is referred to "bottleneck" building block in the ResNet paper.

    It consists of 3 weight convolutional layers, each followed by an activation function. See Table 1 or Figure 5 (right) in the [original ResNet paper](https://www.cv-foundation.org/openaccess/content_cvpr_2016/html/He_Deep_Residual_Learning_CVPR_2016_paper.html).

    Mask is applied to the units which are output channels in each weighted convolutional layer. The mask is generated from the neuron-wise task embedding and gate function.
    """

    def __init__(
        self,
        outer_layer_name: str,
        block_idx: int,
        preceding_output_channels: int,
        input_channels: int,
        overall_stride: int,
        gate: str,
        activation_layer: nn.Module | None = nn.ReLU,
        bias: bool = False,
        **kwargs,
    ) -> None:
        r"""Construct and initialize the larger building block with task embedding.

        **Args:**
        - **outer_layer_name** (`str`): pass the name of the multi-building-block layer that contains this building block to construct the full name of each weighted convolutional layer.
        - **block_idx** (`int`): the index of the building blocks in the multi-building-block layer to construct the full name of each weighted convolutional layer.
        - **preceding_output_channels** (`int`): the number of channels of preceding output of this particular building block.
        - **input_channels** (`int`): the number of channels of input of this building block.
        - **overall_stride** (`int`): the overall stride of this building block. This stride is performed at 2nd (middle) convolutional layer where 1st and 3rd convolutional layers remain stride of 1.
        - **gate** (`str`): the type of gate function turning the real value task embeddings into attention masks; one of:
            - `sigmoid`: the sigmoid function.
        - **activation_layer** (`nn.Module`): activation function of each layer (if not `None`), if `None` this layer won't be used. Default `nn.ReLU`.
        - **bias** (`bool`): whether to use bias in the convolutional layer. Default `False`.
        - **kwargs**: Reserved for multiple inheritance.
        """
        super().__init__(
            output_dim=None,
            gate=gate,
            outer_layer_name=outer_layer_name,
            block_idx=block_idx,
            preceding_output_channels=preceding_output_channels,
            input_channels=input_channels,
            overall_stride=overall_stride,
            activation_layer=activation_layer,
            batch_normalization=False,
            bias=bias,
        )

        # construct the task embedding over the 1st weighted convolutional layer. It is channel-wise
        layer_output_channels = (
            input_channels  # the output channels of the 1st convolutional layer
        )
        self.task_embedding_t[self.full_1st_layer_name] = nn.Embedding(
            num_embeddings=1, embedding_dim=layer_output_channels
        )

        # construct the task embedding over the 2nd weighted convolutional layer. It is channel-wise
        layer_output_channels = (
            input_channels * 1
        )  # the output channels of the 2nd convolutional layer, which is the same as the input channels (without expansion)
        self.task_embedding_t[self.full_2nd_layer_name] = nn.Embedding(
            num_embeddings=1, embedding_dim=layer_output_channels
        )

        # construct the task embedding over the 3rd weighted convolutional layer. It is channel-wise
        layer_output_channels = (
            input_channels * 4  # the output channels of the 3rd convolutional layer
        )
        self.task_embedding_t[self.full_3rd_layer_name] = nn.Embedding(
            num_embeddings=1, embedding_dim=layer_output_channels
        )

    def forward(
        self,
        input: Tensor,
        stage: str,
        s_max: float | None = None,
        batch_idx: int | None = None,
        num_batches: int | None = None,
        test_task_id: int | None = None,
    ) -> tuple[Tensor, dict[str, Tensor], dict[str, Tensor]]:
        r"""The forward pass for data from task `self.task_id`. Task-specific mask for `self.task_id` are applied to the units which are channels in each weighted convolutional layer.

        **Args:**
        - **input** (`Tensor`): The input tensor from data.
        - **stage** (`str`): the stage of the forward pass; one of:
            1. 'train': training stage.
            2. 'validation': validation stage.
            3. 'test': testing stage.
        - **s_max** (`float` | `None`): the maximum scaling factor in the gate function. Doesn't apply to testing stage. See chapter 2.4 "Hard Attention Training" in the [HAT paper](http://proceedings.mlr.press/v80/serra18a).
        - **batch_idx** (`int` | `None`): the current batch index. Applies only to training stage. For other stages, it is default `None`.
        - **num_batches** (`int` | `None`): the total number of batches. Applies only to training stage. For other stages, it is default `None`.
        - **test_task_id** (`dict[str, Tensor]` | `None`): the test task ID. Applies only to testing stage. For other stages, it is default `None`.

        **Returns:**
        - **output_feature** (`Tensor`): the output feature maps.
        - **mask** (`dict[str, Tensor]`): the mask for the current task. Key (`str`) is layer name, value (`Tensor`) is the mask tensor. The mask tensor has size (number of units, ).
        - **activations** (`dict[str, Tensor]`): the hidden features (after activation) in each weighted layer. Keys (`str`) are the weighted layer names and values (`Tensor`) are the hidden feature tensors. This is used for the continual learning algorithms that need to use the hidden features for various purposes. Although HAT algorithm does not need this, it is still provided for API consistence for other HAT-based algorithms inherited this `forward()` method of `HAT` class.
        """
        activations = {}

        # get the mask for the current task from the task embedding in this stage
        mask = self.get_mask(
            stage=stage,
            s_max=s_max,
            batch_idx=batch_idx,
            num_batches=num_batches,
            test_task_id=test_task_id,
        )

        identity = (
            self.identity_downsample(input)
            if self.identity_downsample is not None
            else input
        )  # remember the identity of input for the shortcut connection. Perform downsampling if its dimension doesn't match the output's

        x = input
        x = self.conv1(x)  # weighted convolutional layer first
        x = x * (
            mask[self.full_1st_layer_name].view(1, -1, 1, 1)
        )  # apply the mask to the 1st convolutional layer. Broadcast the dimension of mask to match the input
        if self.activation:
            x = self.conv_activation1(x)  # activation function third
        activations[self.full_1st_layer_name] = x  # store the hidden feature

        x = self.conv2(x)  # weighted convolutional layer first
        x = x * (
            mask[self.full_2nd_layer_name].view(1, -1, 1, 1)
        )  # apply the mask to the 2nd convolutional layer. Broadcast the dimension of mask to match the input
        if self.activation:
            x = self.conv_activation2(x)  # activation function third
        activations[self.full_2nd_layer_name] = x  # store the hidden feature

        x = self.conv3(x)  # weighted convolutional layer first
        x = x + identity
        x = x * (
            mask[self.full_3rd_layer_name].view(1, -1, 1, 1)
        )  # apply the mask to the 3rd convolutional layer after the shortcut connection. Broadcast the dimension of mask to match the input
        if self.activation:
            x = self.conv_activation3(x)  # activation after the shortcut connection
        activations[self.full_3rd_layer_name] = x  # store the hidden feature

        output_feature = x

        return output_feature, mask, activations


class HATMaskResNetBase(HATMaskBackbone, ResNetBase):
    r"""The base class of HAT masked [residual network (ResNet)](https://www.cv-foundation.org/openaccess/content_cvpr_2016/html/He_Deep_Residual_Learning_CVPR_2016_paper.html).

    [HAT (Hard Attention to the Task, 2018)](http://proceedings.mlr.press/v80/serra18a) is an architecture-based continual learning approach that uses learnable hard attention masks to select the task-specific parameters.

    ResNet is a convolutional network architecture, which has 1st convolutional parameter layer and a maxpooling layer, connecting to 4 convolutional layers which contains multiple convolutional parameter layer. Each layer of the 4 are constructed from basic building blocks which are either small (`ResNetBlockSmall`) or large (`ResNetBlockLarge`). Each building block contains several convolutional parameter layers. The building blocks are connected by a skip connection which is a direct connection from the input of the block to the output of the block, and this is why it's called residual (find "shortcut connections" in the paper for more details). After the 5th convolutional layer, there are average pooling layer and a fully connected layer which connects to the CL output heads.

    Mask is applied to the units which are output channels in each weighted convolutional layer. The mask is generated from the neuron-wise task embedding and gate function.
    """

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
        bias: bool = False,
        **kwargs,
    ) -> None:
        r"""Construct and initialize the HAT masked ResNet backbone network with task embedding. Note that batch normalization is incompatible with HAT mechanism.

        **Args:**
        - **input_channels** (`int`): the number of channels of input. Image data are kept channels when going in ResNet. Note that convolutional networks require number of input channels instead of dimension.
        - **building_block_type** (`HATMaskResNetBlockSmall` | `HATMaskResNetBlockLarge`): the type of building block used in the ResNet.
        - **building_block_nums** (`tuple[int, int, int, int]`): the number of building blocks in the 2-5 convolutional layer correspondingly.
        - **building_block_preceding_output_channels** (`tuple[int, int, int, int]`): the number of channels of preceding output of each building block in the 2-5 convolutional layer correspondingly.
        - **building_block_input_channels** (`tuple[int, int, int, int]`): the number of channels of input of each building block in the 2-5 convolutional layer correspondingly.
        - **output_dim** (`int`): the output dimension after flattening at last which connects to CL output heads. Although this is not determined by us but the architecture built before the flattening layer, we still need to provide this to construct the heads.
        - **gate** (`str`): the type of gate function turning the real value task embeddings into attention masks; one of:
            - `sigmoid`: the sigmoid function.
        - **activation_layer** (`nn.Module`): activation function of each layer (if not `None`), if `None` this layer won't be used. Default `nn.ReLU`.
        - **bias** (`bool`): whether to use bias in the convolutional layer. Default `False`, because batch normalization are doing the similar thing with bias.
        - **kwargs**: Reserved for multiple inheritance.
        """
        # init from both inherited classes
        super().__init__(
            gate=gate,
            input_channels=input_channels,
            building_block_type=building_block_type,
            building_block_nums=building_block_nums,
            building_block_preceding_output_channels=building_block_preceding_output_channels,
            building_block_input_channels=building_block_input_channels,
            output_dim=output_dim,
            activation_layer=activation_layer,
            batch_normalization=False,  # batch normalization is incompatible with HAT mechanism
            bias=bias,
        )

        self.update_multiple_blocks_task_embedding()

        # construct the task embedding over the 1st weighted convolutional layers. It is channel-wise
        layer_output_channels = 64  # the output channels of the 1st convolutional layer
        self.task_embedding_t["conv1"] = nn.Embedding(
            num_embeddings=1, embedding_dim=layer_output_channels
        )

    def _multiple_blocks(
        self,
        layer_name: str,
        building_block_type: HATMaskResNetBlockSmall | HATMaskResNetBlockLarge,
        building_block_num: int,
        preceding_output_channels: int,
        input_channels: int,
        overall_stride: int,
        activation_layer: nn.Module | None = nn.ReLU,
        batch_normalization: bool = False,
        bias: bool = False,
    ) -> None:
        r"""Construct a layer consisting of multiple building blocks with task embedding. It's used to construct the 2-5 convolutional layers of the HAT masked ResNet.

        The "shortcut connections" are performed between the input and output of each building block:
        1. If the input and output of the building block have exactly the same dimensions (including number of channels and size), add the input to the output.
        2. If the input and output of the building block have different dimensions (including number of channels and size), add the input to the output after a convolutional layer to make the dimensions match.

        **Args:**
        - **layer_name** (`str`): pass the name of this multi-building-block layer to construct the full name of each weighted convolutional layer.
        - **building_block_type** (`HATMaskResNetBlockSmall` | `HATMaskResNetBlockLarge`): the type of the building block.
        - **building_block_num** (`int`): the number of building blocks in this multi-building-block layer.
        - **preceding_output_channels** (`int`): the number of channels of preceding output of this entire multi-building-block layer.
        - **input_channels** (`int`): the number of channels of input of this multi-building-block layer.
        - **overall_stride** (`int`): the overall stride of the building blocks. This stride is performed at the 1st building block where other building blocks remain their own overall stride of 1. Inside that building block, this stride is performed at certain convolutional layer in the building block where other convolutional layers remain stride of 1:
            - For `ResNetBlockSmall`, it performs at the 2nd (last) layer.
            - For `ResNetBlockLarge`, it performs at the 2nd (middle) layer.
        - **activation_layer** (`nn.Module`): activation function of each layer (if not `None`), if `None` this layer won't be used. Default `nn.ReLU`.
        - **batch_normalization** (`bool`): whether to use batch normalization after the weight convolutional layers. In HATMaskResNet, batch normalization is incompatible with HAT mechanism and shoule be always set `False`. We include this argument for compatibility with the original ResNet API.
        - **bias** (`bool`): whether to use bias in the convolutional layer. Default `False`.

        **Returns:**
        - **layer** (`nn.Sequential`): the constructed layer consisting of multiple building blocks.
        """

        layer = []

        for block_idx in range(building_block_num):
            layer.append(
                building_block_type(
                    outer_layer_name=layer_name,
                    block_idx=block_idx,
                    preceding_output_channels=(
                        preceding_output_channels
                        if block_idx == 0
                        else (
                            input_channels
                            if building_block_type == HATMaskResNetBlockSmall
                            else input_channels * 4
                        )
                    ),  # if it's the 1st block in this multi-building-block layer, it should be the number of channels of the preceding output of this entire multi-building-block layer. Otherwise, it should be the number of channels from last building block where the number of channels is 4 times of the input channels for `ResNetBlockLarge` than `ResNetBlockSmall`.
                    input_channels=input_channels,
                    overall_stride=(
                        overall_stride if block_idx == 0 else 1
                    ),  # only perform the overall stride at the 1st block in this multi-building-block layer
                    gate=self.gate,
                    # no batch normalization in HAT masked blocks
                    activation_layer=activation_layer,
                    bias=bias,
                )
            )

            self.weighted_layer_names += layer[
                -1
            ].weighted_layer_names  # collect the weighted layer names in the blocks and sync to the weighted layer names list in the outer network

        return nn.Sequential(*layer)

    def update_multiple_blocks_task_embedding(self) -> None:
        r"""Collect the task embeddings in the multiple building blocks (2-5 convolutional layers) and sync to the weighted layer names list in the outer network.

        This should only be called explicitly after the `__init__()` method, just because task embedding as `nn.Module` instance was wiped out at the beginning of it.
        """
        for block in self.conv2x:
            self.task_embedding_t.update(block.task_embedding_t)
        for block in self.conv3x:
            self.task_embedding_t.update(block.task_embedding_t)
        for block in self.conv4x:
            self.task_embedding_t.update(block.task_embedding_t)
        for block in self.conv5x:
            self.task_embedding_t.update(block.task_embedding_t)

    def forward(
        self,
        input: Tensor,
        stage: str,
        s_max: float | None = None,
        batch_idx: int | None = None,
        num_batches: int | None = None,
        test_task_id: int | None = None,
    ) -> tuple[Tensor, dict[str, Tensor], dict[str, Tensor]]:
        r"""The forward pass for data from task `self.task_id`. Task-specific mask for `self.task_id` are applied to the units which are channels in each weighted convolutional layer.

        **Args:**
        - **input** (`Tensor`): the input tensor from data.
        - **stage** (`str`): the stage of the forward pass; one of:
            1. 'train': training stage.
            2. 'validation': validation stage.
            3. 'test': testing stage.
        - **s_max** (`float` | `None`): the maximum scaling factor in the gate function. Doesn't apply to testing stage. See chapter 2.4 "Hard Attention Training" in the [HAT paper](http://proceedings.mlr.press/v80/serra18a).
        - **batch_idx** (`int` | `None`): the current batch index. Applies only to training stage. For other stages, it is default `None`.
        - **num_batches** (`int` | `None`): the total number of batches. Applies only to training stage. For other stages, it is default `None`.
        - **test_task_id** (`dict[str, Tensor]` | `None`): the test task ID. Applies only to testing stage. For other stages, it is default `None`.

        **Returns:**
        - **output_feature** (`Tensor`): the output feature tensor to be passed to the heads.
        - **mask** (`dict[str, Tensor]`): the mask for the current task. Key (`str`) is layer name, value (`Tensor`) is the mask tensor. The mask tensor has size (number of units, ).
        - **activations** (`dict[str, Tensor]`): the hidden features (after activation) in each weighted layer. Keys (`str`) are the weighted layer names and values (`Tensor`) are the hidden feature tensors. This is used for the continual learning algorithms that need to use the hidden features for various purposes. Although HAT algorithm does not need this, it is still provided for API consistence for other HAT-based algorithms inherited this `forward()` method of `HAT` class.
        """
        batch_size = input.size(0)
        activations = {}

        # get the mask for the current task from the task embedding in this stage
        mask = self.get_mask(
            stage=stage,
            s_max=s_max,
            batch_idx=batch_idx,
            num_batches=num_batches,
            test_task_id=test_task_id,
        )

        x = input

        x = self.conv1(x)

        x = x * (
            mask["conv1"].view(1, -1, 1, 1)
        )  # apply the mask to the 1st convolutional layer. Broadcast the dimension of mask to match the input
        if self.activation:
            x = self.conv_activation1(x)
        activations["conv1"] = x

        x = self.maxpool(x)

        for block in self.conv2x:
            x, _, activations_block = block(
                x,
                stage=stage,
                s_max=s_max,
                batch_idx=batch_idx,
                num_batches=num_batches,
                test_task_id=test_task_id,
            )
            activations.update(activations_block)  # store the hidden feature
        for block in self.conv3x:
            x, _, activations_block = block(
                x,
                stage=stage,
                s_max=s_max,
                batch_idx=batch_idx,
                num_batches=num_batches,
                test_task_id=test_task_id,
            )
            activations.update(activations_block)  # store the hidden feature
        for block in self.conv4x:
            x, _, activations_block = block(
                x,
                stage=stage,
                s_max=s_max,
                batch_idx=batch_idx,
                num_batches=num_batches,
                test_task_id=test_task_id,
            )
            activations.update(activations_block)  # store the hidden feature
        for block in self.conv5x:
            x, _, activations_block = block(
                x,
                stage=stage,
                s_max=s_max,
                batch_idx=batch_idx,
                num_batches=num_batches,
                test_task_id=test_task_id,
            )
            activations.update(activations_block)  # store the hidden feature

        x = self.avepool(x)

        output_feature = x.view(batch_size, -1)  # flatten before going through heads

        return output_feature, mask, activations


class HATMaskResNet18(HATMaskResNetBase):
    r"""HAT masked ResNet-18 backbone network.

    [HAT (Hard Attention to the Task, 2018)](http://proceedings.mlr.press/v80/serra18a) is an architecture-based continual learning approach that uses learnable hard attention masks to select the task-specific parameters.

    ResNet-18 is a smaller architecture proposed in the [original ResNet paper](https://www.cv-foundation.org/openaccess/content_cvpr_2016/html/He_Deep_Residual_Learning_CVPR_2016_paper.html). It consists of 18 weight convolutional layers in total. See Table 1 in the paper for details.

    Mask is applied to the units which are output channels in each weighted convolutional layer. The mask is generated from the neuron-wise task embedding and gate function.
    """

    def __init__(
        self,
        input_channels: int,
        output_dim: int,
        gate: str,
        activation_layer: nn.Module | None = nn.ReLU,
        bias: bool = False,
        pretrained_weights: str | None = None,
        **kwargs,
    ) -> None:
        r"""Construct and initialize the ResNet-18 backbone network with task embedding. Note that batch normalization is incompatible with HAT mechanism.

        **Args:**
        - **input_channels** (`int`): the number of channels of input of this building block. Note that convolutional networks require number of input channels instead of dimension.
        - **output_dim** (`int`): the output dimension after flattening at last which connects to CL output heads. Although this is not determined by us but the architecture built before the flattening layer, we still need to provide this to construct the heads.
        - **gate** (`str`): the type of gate function turning the real value task embeddings into attention masks; one of:
            - `sigmoid`: the sigmoid function.
        - **activation_layer** (`nn.Module`): activation function of each layer (if not `None`), if `None` this layer won't be used. Default `nn.ReLU`.
        - **bias** (`bool`): whether to use bias in the convolutional layer. Default `False`.
        - **pretrained_weights** (`str`): the name of pretrained weights to be loaded. See [TorchVision docs](https://pytorch.org/vision/main/models.html). If `None`, no pretrained weights are loaded. Default `None`.
        - **kwargs**: Reserved for multiple inheritance.
        """
        super().__init__(
            input_channels=input_channels,
            building_block_type=HATMaskResNetBlockSmall,  # use the smaller building block for ResNet-18
            building_block_nums=(2, 2, 2, 2),
            building_block_preceding_output_channels=(64, 64, 128, 256),
            building_block_input_channels=(64, 128, 256, 512),
            output_dim=output_dim,
            gate=gate,
            activation_layer=activation_layer,
            bias=bias,
        )

        if pretrained_weights is not None:
            # load the pretrained weights from TorchVision
            torchvision_resnet18_state_dict = torchvision.models.resnet18(
                weights=pretrained_weights
            ).state_dict()

            # mapping from torchvision resnet18 state dict to our HATMaskResNet18 state dict
            state_dict_converted = {}
            for key, value in torchvision_resnet18_state_dict.items():
                if HATMASKRESNET18_STATE_DICT_MAPPING[key] is not None:
                    state_dict_converted[HATMASKRESNET18_STATE_DICT_MAPPING[key]] = (
                        value
                    )

            self.load_state_dict(state_dict_converted, strict=False)


class HATMaskResNet34(HATMaskResNetBase):
    r"""HAT masked ResNet-34 backbone network.

    [HAT (Hard Attention to the Task, 2018)](http://proceedings.mlr.press/v80/serra18a) is an architecture-based continual learning approach that uses learnable hard attention masks to select the task-specific parameters.

    ResNet-34 is a smaller architecture proposed in the [original ResNet paper](https://www.cv-foundation.org/openaccess/content_cvpr_2016/html/He_Deep_Residual_Learning_CVPR_2016_paper.html). It consists of 34 weight convolutional layers in total. See Table 1 in the paper for details.

    Mask is applied to the units which are output channels in each weighted convolutional layer. The mask is generated from the neuron-wise task embedding and gate function.
    """

    def __init__(
        self,
        input_channels: int,
        output_dim: int,
        gate: str,
        activation_layer: nn.Module | None = nn.ReLU,
        bias: bool = False,
        **kwargs,
    ) -> None:
        r"""Construct and initialize the ResNet-34 backbone network with task embedding. Note that batch normalization is incompatible with HAT mechanism.

        **Args:**
        - **input_channels** (`int`): the number of channels of input of this building block. Note that convolutional networks require number of input channels instead of dimension.
        - **output_dim** (`int`): the output dimension after flattening at last which connects to CL output heads. Although this is not determined by us but the architecture built before the flattening layer, we still need to provide this to construct the heads.
        - **gate** (`str`): the type of gate function turning the real value task embeddings into attention masks; one of:
            - `sigmoid`: the sigmoid function.
        - **activation_layer** (`nn.Module`): activation function of each layer (if not `None`), if `None` this layer won't be used. Default `nn.ReLU`.
        - **bias** (`bool`): whether to use bias in the convolutional layer. Default `False`.
        - **kwargs**: Reserved for multiple inheritance.
        """
        super().__init__(
            input_channels=input_channels,
            building_block_type=HATMaskResNetBlockSmall,  # use the smaller building block for ResNet-34
            building_block_nums=(3, 4, 6, 3),
            building_block_preceding_output_channels=(64, 64, 128, 256),
            building_block_input_channels=(64, 128, 256, 512),
            output_dim=output_dim,
            gate=gate,
            activation_layer=activation_layer,
            bias=bias,
        )


class HATMaskResNet50(HATMaskResNetBase):
    r"""HAT masked ResNet-50 backbone network.

    [HAT (Hard Attention to the Task, 2018)](http://proceedings.mlr.press/v80/serra18a) is an architecture-based continual learning approach that uses learnable hard attention masks to select the task-specific parameters.

    ResNet-50 is a larger architecture proposed in the [original ResNet paper](https://www.cv-foundation.org/openaccess/content_cvpr_2016/html/He_Deep_Residual_Learning_CVPR_2016_paper.html). It consists of 50 weight convolutional layers in total. See Table 1 in the paper for details.

    Mask is applied to the units which are output channels in each weighted convolutional layer. The mask is generated from the neuron-wise task embedding and gate function.
    """

    def __init__(
        self,
        input_channels: int,
        output_dim: int,
        gate: str,
        activation_layer: nn.Module | None = nn.ReLU,
        bias: bool = False,
        **kwargs,
    ) -> None:
        r"""Construct and initialize the ResNet-50 backbone network with task embedding. Note that batch normalization is incompatible with HAT mechanism.

        **Args:**
        - **input_channels** (`int`): the number of channels of input of this building block. Note that convolutional networks require number of input channels instead of dimension.
        - **output_dim** (`int`): the output dimension after flattening at last which connects to CL output heads. Although this is not determined by us but the architecture built before the flattening layer, we still need to provide this to construct the heads.
        - **gate** (`str`): the type of gate function turning the real value task embeddings into attention masks; one of:
            - `sigmoid`: the sigmoid function.
        - **activation_layer** (`nn.Module`): activation function of each layer (if not `None`), if `None` this layer won't be used. Default `nn.ReLU`.
        - **bias** (`bool`): whether to use bias in the convolutional layer. Default `False`.
        - **kwargs**: Reserved for multiple inheritance.
        """
        super().__init__(
            input_channels=input_channels,
            building_block_type=HATMaskResNetBlockLarge,  # use the smaller building block for ResNet-50
            building_block_nums=(3, 4, 6, 3),
            building_block_preceding_output_channels=(64, 256, 512, 1024),
            building_block_input_channels=(64, 128, 256, 512),
            output_dim=output_dim,
            gate=gate,
            activation_layer=activation_layer,
            bias=bias,
        )


class HATMaskResNet101(HATMaskResNetBase):
    r"""HAT masked ResNet-101 backbone network.

    [HAT (Hard Attention to the Task, 2018)](http://proceedings.mlr.press/v80/serra18a) is an architecture-based continual learning approach that uses learnable hard attention masks to select the task-specific parameters.

    ResNet-101 is a larger architecture proposed in the [original ResNet paper](https://www.cv-foundation.org/openaccess/content_cvpr_2016/html/He_Deep_Residual_Learning_CVPR_2016_paper.html). It consists of 101 weight convolutional layers in total. See Table 1 in the paper for details.

    Mask is applied to the units which are output channels in each weighted convolutional layer. The mask is generated from the neuron-wise task embedding and gate function.
    """

    def __init__(
        self,
        input_channels: int,
        output_dim: int,
        gate: str,
        activation_layer: nn.Module | None = nn.ReLU,
        bias: bool = False,
        **kwargs,
    ) -> None:
        r"""Construct and initialize the ResNet-101 backbone network with task embedding. Note that batch normalization is incompatible with HAT mechanism.

        **Args:**
        - **input_channels** (`int`): the number of channels of input of this building block. Note that convolutional networks require number of input channels instead of dimension.
        - **output_dim** (`int`): the output dimension after flattening at last which connects to CL output heads. Although this is not determined by us but the architecture built before the flattening layer, we still need to provide this to construct the heads.
        - **gate** (`str`): the type of gate function turning the real value task embeddings into attention masks; one of:
            - `sigmoid`: the sigmoid function.
        - **activation_layer** (`nn.Module`): activation function of each layer (if not `None`), if `None` this layer won't be used. Default `nn.ReLU`.
        - **bias** (`bool`): whether to use bias in the convolutional layer. Default `False`.
        - **kwargs**: Reserved for multiple inheritance.
        """
        super().__init__(
            input_channels=input_channels,
            building_block_type=HATMaskResNetBlockLarge,  # use the smaller building block for ResNet-18
            building_block_nums=(3, 4, 23, 3),
            building_block_preceding_output_channels=(64, 256, 512, 1024),
            building_block_input_channels=(64, 128, 256, 512),
            output_dim=output_dim,
            gate=gate,
            activation_layer=activation_layer,
            bias=bias,
        )


class HATMaskResNet152(HATMaskResNetBase):
    r"""HAT masked ResNet-152 backbone network.

    [HAT (Hard Attention to the Task, 2018)](http://proceedings.mlr.press/v80/serra18a) is an architecture-based continual learning approach that uses learnable hard attention masks to select the task-specific parameters.

    ResNet-152 is the largest architecture proposed in the [original ResNet paper](https://www.cv-foundation.org/openaccess/content_cvpr_2016/html/He_Deep_Residual_Learning_CVPR_2016_paper.html). It consists of 152 weight convolutional layers in total. See Table 1 in the paper for details.

    Mask is applied to the units which are output channels in each weighted convolutional layer. The mask is generated from the neuron-wise task embedding and gate function.
    """

    def __init__(
        self,
        input_channels: int,
        output_dim: int,
        gate: str,
        activation_layer: nn.Module | None = nn.ReLU,
        bias: bool = False,
        **kwargs,
    ) -> None:
        r"""Construct and initialize the ResNet-152 backbone network with task embedding. Note that batch normalization is incompatible with HAT mechanism.

        **Args:**
        - **input_channels** (`int`): the number of channels of input of this building block. Note that convolutional networks require number of input channels instead of dimension.
        - **output_dim** (`int`): the output dimension after flattening at last which connects to CL output heads. Although this is not determined by us but the architecture built before the flattening layer, we still need to provide this to construct the heads.
        - **gate** (`str`): the type of gate function turning the real value task embeddings into attention masks; one of:
            - `sigmoid`: the sigmoid function.
        - **activation_layer** (`nn.Module`): activation function of each layer (if not `None`), if `None` this layer won't be used. Default `nn.ReLU`.
        - **bias** (`bool`): whether to use bias in the convolutional layer. Default `False`.
        - **kwargs**: Reserved for multiple inheritance.
        """
        super().__init__(
            input_channels=input_channels,
            building_block_type=HATMaskResNetBlockLarge,  # use the smaller building block for ResNet-152
            building_block_nums=(3, 8, 36, 3),
            building_block_preceding_output_channels=(64, 256, 512, 1024),
            building_block_input_channels=(64, 128, 256, 512),
            output_dim=output_dim,
            gate=gate,
            activation_layer=activation_layer,
            bias=bias,
        )
