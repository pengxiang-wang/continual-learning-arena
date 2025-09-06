"""The submodule in `utils` for data transforms."""

__all__ = [
    "ClassMapping",
    "Permute",
    "insert_permute_in_compose",
    "min_max_normalize",
    "js_div",
]

import logging

import torch
from torch import Tensor, nn
from torchvision import transforms

# always get logger for built-in logging in each module
pylogger = logging.getLogger(__name__)


class ClassMapping:
    r"""Class mapping to dataset labels. Used as a PyTorch target Transform."""

    def __init__(self, class_map: dict[str | int, int]) -> None:
        r"""
        **Args:**
        - **cl_class_map** (`dict[str | int, int]`): the class map.
        """
        self.class_map = class_map

    def __call__(self, target: torch.Tensor) -> torch.Tensor:
        r"""The class mapping transform to dataset labels. It is defined as a callable object like a PyTorch Transform.

        **Args:**
        - **target** (`Tensor`): the target tensor.

        **Returns:**
        - **transformed_target** (`Tensor`): the transformed target tensor.
        """

        target = int(
            target
        )  # convert to int if it is a tensor to avoid keyerror in map
        return self.class_map[target]


class Permute:
    r"""Permutation operation to image. Used to construct permuted CL dataset.

    Used as a PyTorch Dataset Transform.
    """

    def __init__(
        self,
        num_channels: int,
        img_size: torch.Size,
        mode: str = "first_channel_only",
        seed: int | None = None,
    ) -> None:
        r"""Initialize the Permute transform object. The permutation order is constructed in the initialization to save runtime.

        **Args:**
        - **num_channels** (`int`): the number of channels in the image.
        - **img_size** (`torch.Size`): the size of the image to be permuted.
        - **mode** (`str`): the mode of permutation, shouble be one of the following:
            - 'all': permute all pixels.
            - 'by_channel': permute channel by channel separately. All channels are applied the same permutation order.
            - 'first_channel_only': permute only the first channel.
        - **seed** (`int` or `None`): seed for permutation operation. If None, the permutation will use a default seed from PyTorch generator.
        """
        self.mode = mode
        r"""The mode of permutation."""

        # get generator for permutation
        torch_generator = torch.Generator()
        if seed:
            torch_generator.manual_seed(seed)

        # calculate the number of pixels from the image size
        if self.mode == "all":
            num_pixels = num_channels * img_size[0] * img_size[1]
        elif self.mode == "by_channel" or "first_channel_only":
            num_pixels = img_size[0] * img_size[1]

        self.permute: torch.Tensor = torch.randperm(
            num_pixels, generator=torch_generator
        )
        r"""The permutation order, a `Tensor` permuted from [1,2, ..., `num_pixels`] with the given seed. It is the core element of permutation operation."""

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        r"""The permutation operation to image is defined as a callable object like a PyTorch Transform.

        **Args:**
        - **img** (`Tensor`): image to be permuted. Must match the size of `img_size` in the initialization.

        **Returns:**
        - **img_permuted** (`Tensor`): the permuted image.
        """

        if self.mode == "all":

            img_flat = img.view(
                -1
            )  # flatten the whole image to 1d so that it can be applied 1d permuted order
            img_flat_permuted = img_flat[self.permute]  # conduct permutation operation
            img_permuted = img_flat_permuted.view(
                img.size()
            )  # return to the original image shape
            return img_permuted

        if self.mode == "by_channel":

            permuted_channels = []
            for i in range(img.size(0)):
                # act on every channel
                channel_flat = img[i].view(
                    -1
                )  # flatten the channel to 1d so that it can be applied 1d permuted order
                channel_flat_permuted = channel_flat[
                    self.permute
                ]  # conduct permutation operation
                channel_permuted = channel_flat_permuted.view(
                    img[0].size()
                )  # return to the original channel shape
                permuted_channels.append(channel_permuted)
            img_permuted = torch.stack(
                permuted_channels
            )  # stack the permuted channels to restore the image
            return img_permuted

        if self.mode == "first_channel_only":

            first_channel_flat = img[0].view(
                -1
            )  # flatten the first channel to 1d so that it can be applied 1d permuted order
            first_channel_flat_permuted = first_channel_flat[
                self.permute
            ]  # conduct permutation operation
            first_channel_permuted = first_channel_flat_permuted.view(
                img[0].size()
            )  # return to the original channel shape

            img_permuted = img.clone()
            img_permuted[0] = first_channel_permuted

            return img_permuted


def insert_permute_in_compose(compose: transforms.Compose, permute_transform: Permute):
    r"""Insert `permute_transform` in a `compose` (`transforms.Compose`)."""

    last_insert_index = -1

    for index, transform in enumerate(compose.transforms):
        if transform.__class__ in [
            transforms.Grayscale,
            transforms.ToTensor,
            transforms.Resize,
        ]:
            last_insert_index = index  # insert after this one

    if last_insert_index >= 0:
        # insert permute after last detected transform
        new_list = (
            compose.transforms[:last_insert_index]
            + [permute_transform]
            + compose.transforms[last_insert_index:]
        )
    else:
        # None of repeat/to_tensor/resize found → insert at start
        new_list = [permute_transform] + compose.transforms

    return transforms.Compose(new_list)


def min_max_normalize(
    tensor: Tensor, dim: int | None = None, epsilon: float = 1e-8
) -> Tensor:
    r"""Normalize the tensor using min-max normalization.

    **Args:**
    - **tensor** (`Tensor`): the input tensor to normalize.
    - **dim** (`int` | `None`): the dimension to normalize along. If `None`, normalize the whole tensor.
    - **epsilon** (`float`): the epsilon value to avoid division by zero.

    **Returns:**
    - **tensor** (`Tensor`): the normalized tensor.
    """
    min_val = (
        tensor.min(dim=dim, keepdim=True).values if dim is not None else tensor.min()
    )
    max_val = (
        tensor.max(dim=dim, keepdim=True).values if dim is not None else tensor.max()
    )

    return (tensor - min_val) / (max_val - min_val + epsilon)


def js_div(
    input: Tensor,
    target: Tensor,
    size_average: bool | None = None,
    reduce: bool | None = None,
):
    r"""Jensen-Shannon divergence between two probability distributions."""

    eps = 1e-8
    input_safe = input.clamp(min=eps)
    target_safe = target.clamp(min=eps)

    m_safe = 0.5 * (input_safe + target_safe)
    # m_safe = m.clamp(min=eps)

    kl_input = nn.functional.kl_div(
        input_safe.log(),
        m_safe,
        size_average=size_average,
        reduce=reduce,
        reduction="mean",
    )

    kl_target = nn.functional.kl_div(
        target_safe.log(),
        m_safe,
        size_average=size_average,
        reduce=reduce,
        reduction="mean",
    )

    js = 0.5 * (kl_input + kl_target)

    return js
