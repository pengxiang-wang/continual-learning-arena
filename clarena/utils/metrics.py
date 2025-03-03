"""The submodule in `utils` for plotting utils."""

__all__ = ["MeanMetricBatch", "HATNetworkCapacity"]

from typing import Any

import torch
from torch import Tensor
from torchmetrics.aggregation import BaseAggregator


class MeanMetricBatch(BaseAggregator):
    r"""A TorchMetrics metric to calculate the mean of metrics across data batches.

    This is used for accumulated metrics in deep learning. See [here](https://pengxiang-wang.com/posts/continual-learning-metrics.html#nte-accumulate) for more details.
    """

    def __init__(self, nan_strategy: str | float = "error", **kwargs: Any) -> None:
        r"""Initialise the metric. Add state variables."""
        BaseAggregator.__init__(
            self,
            "sum",
            torch.tensor(0.0, dtype=torch.get_default_dtype()),
            nan_strategy,
            state_name="sum",
            **kwargs,
        )

        self.sum: Tensor
        r"""State variable created by `super().__init__()` to store the sum of the metric values till this batch."""

        self.add_state(
            "num",
            default=torch.tensor(0, dtype=torch.get_default_dtype()),
            dist_reduce_fx="sum",
        )
        self.num: Tensor
        r"""State variable created by `add_state()` to store the number of the data till this batch."""

    def update(self, value: torch.Tensor, batch_size: int) -> None:
        r"""Update and accumulate the sum of metric value and num of the data till this batch from the batch.

        **Args:**
        - **val** (`torch.Tensor`): the metric value of the batch to update the sum.
        - **batch_size** (`int`): the value to update the num, which is the batch size.
        """

        value = torch.as_tensor(value, dtype=self.dtype, device=self.device)
        batch_size = torch.as_tensor(batch_size, dtype=self.dtype, device=self.device)

        self.sum += value * batch_size
        self.num += batch_size

    def compute(self) -> Tensor:
        r"""Compute this mean metric value till this batch.

        **Returns:**
        - **mean** (`Tensor`): the calculated mean result.
        """
        return self.sum / self.num


class HATNetworkCapacity(BaseAggregator):
    r"""A torchmetrics metric to calculate the network capacity of HAT (Hard Attention to the Task) algorithm.

    Network capacity is defined as the average adjustment rate over all paramaters. See chapter 4.1 in [AdaHAT paper](https://link.springer.com/chapter/10.1007/978-3-031-70352-2_9).
    """

    def __init__(self, nan_strategy: str | float = "error", **kwargs: Any) -> None:
        r"""Initialise the HAT network capacity metric. Add state variables."""
        BaseAggregator.__init__(
            self,
            "sum",
            torch.tensor(0.0, dtype=torch.get_default_dtype()),
            nan_strategy,
            state_name="sum_adjustment_rate",
            **kwargs,
        )
        self.sum_adjustment_rate: Tensor
        r"""State variable created by `add_state()` to store the sum of the adjustment rate values till this layer."""

        self.add_state("num_params", default=torch.tensor(0), dist_reduce_fx="sum")
        self.num_params: Tensor
        r"""State variable created by `add_state()` to store the number of the parameters till this layer."""

    def update(
        self, adjustment_rate_weight: Tensor, adjustment_rate_bias: Tensor
    ) -> None:
        r"""Update and accumulate the sum of adjustment rate values till this layer from the layer.

        **Args:**
        - **adjustment_rate_weight** (`Tensor`): the adjustment rate values of the weight matrix of the layer.
        - **adjustment_rate_bias** (`Tensor`): the adjustment rate values of the bias vector of the layer.
        """
        adjustment_rate_weight = torch.as_tensor(
            adjustment_rate_weight, dtype=self.dtype, device=self.device
        )
        adjustment_rate_bias = torch.as_tensor(
            adjustment_rate_bias, dtype=self.dtype, device=self.device
        )

        self.sum_adjustment_rate += (
            adjustment_rate_weight.sum() + adjustment_rate_bias.sum()
        )
        self.num_params += adjustment_rate_weight.numel() + adjustment_rate_bias.numel()

    def compute(self) -> Tensor:
        r"""Compute this HAT network capacity till this layer.

        **Returns:**
        - **network_capacity** (`Tensor`): the calculated network capacity result.
        """

        return self.sum_adjustment_rate / self.num_params
