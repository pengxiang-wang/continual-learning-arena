"""The submodule in `utils` for transforming tensors."""

__all__ = ["min_max_normalise"]


from torch import Tensor


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
