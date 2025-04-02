"""The submodule in `utils` for transforming tensors."""

__all__ = ["min_max_normalise", "js_div"]


from torch import Tensor, nn


def min_max_normalise(
    tensor: Tensor, dim: int | None = None, epsilon: float = 1e-8
) -> Tensor:
    r"""Normalise the tensor using min-max normalisation.

    **Args:**
    - **tensor** (`Tensor`): the input tensor to normalise.
    - **dim** (`int` | `None`): the dimension to normalise along. If `None`, normalise the whole tensor.
    - **epsilon** (`float`): the epsilon value to avoid division by zero.

    **Returns:**
    - **tensor** (`Tensor`): the normalised tensor.
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
    reduction: str = "mean",
):
    m = 0.5 * (input + target)

    kl_input = nn.functional.kl_div(
        input.log(),
        m,
        size_average=size_average,
        reduce=reduce,
        reduction=reduction,
    )
    kl_target = nn.functional.kl_div(
        target.log(),
        m,
        size_average=size_average,
        reduce=reduce,
        reduction=reduction,
    )

    js = 0.5 * (kl_input + kl_target)

    return js
