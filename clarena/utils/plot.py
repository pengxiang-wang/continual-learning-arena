"""The submodule in `utils` for plotting utils."""

__all__ = ["plot_mask"]


from torch import Tensor


def plot_hat_mask(mask: dict[str, Tensor]) -> None:
    """Plot mask in [HAT (Hard Attention to the Task)](http://proceedings.mlr.press/v80/serra18a)) algorithm.

    **Args:**
    - **mask** (`dict[str, Tensor]`): the hard attention (whose values are 0 or 1) mask. Key (`str`) is layer name, value (`Tensor`) is the mask tensor. 
    """
    
    