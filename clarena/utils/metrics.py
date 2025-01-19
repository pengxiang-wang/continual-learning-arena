"""The submodule in `utils` for plotting utils."""

__all__ = ["MeanMetricBatch", "HATNetworkCapacity"]

import torch
from torch import Tensor
from torchmetrics import Metric


class MeanMetricBatch(Metric):
    """A torchmetrics metric to calculate the mean of certain metrics across data batches."""

    def __init__(self) -> None:
        """Initialise the Mean Metric Batch. Add state variables."""
        super().__init__()

        self.add_state("sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.sum: Tensor
        """State variable created by `add_state()` to store the sum of the metric values till this batch."""

        self.add_state("num", default=torch.tensor(0), dist_reduce_fx="sum")
        self.num: Tensor
        """State variable created by `add_state()` to store the number of the data till this batch."""

    def update(self, val: torch.Tensor, batch_size: int) -> None:
        """Update and accumulate the sum of metric value and num of the data till this batch from the batch.

        **Args:**
        - **val** (`torch.Tensor`): the metric value of the batch to update the sum.
        - **batch_size** (`int`): the value to update the num, which is the batch size.
        """
        self.sum += val * batch_size
        self.num += batch_size

    def compute(self) -> Tensor:
        """Compute this mean metric value till this batch.
        
        **Returns:**
        - **mean** (`Tensor`): the calculated mean result.
        """
        return self.sum.float() / self.num


class HATNetworkCapacity(Metric):
    """A torchmetrics metric to calculate the network capacity of HAT (Hard Attention to the Task) algorithm. 
    
    Network capacity is defined as the average adjustment rate over all paramaters. See chapter 4.1 in [AdaHAT paper](https://link.springer.com/chapter/10.1007/978-3-031-70352-2_9). 
    """
    
    def __init__(self) -> None:
        """Initialise the HAT network capacity metric. Add state variables."""
        super().__init__()

        self.add_state("sum_adjustment_rate", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.sum_adjustment_rate: Tensor
        """State variable created by `add_state()` to store the sum of the adjustment rate values till this layer."""
        
        self.add_state("num_params", default=torch.tensor(0), dist_reduce_fx="sum")
        self.num_params: Tensor
        """State variable created by `add_state()` to store the number of the parameters till this layer."""
        
    def update(self, adjustment_rate_weight: Tensor, adjustment_rate_bias: Tensor) -> None:
        """Update and accumulate the sum of adjustment rate values till this layer from the layer.
        
        **Args:**
        - **adjustment_rate_weight** (`Tensor`): the adjustment rate values of the weight matrix of the layer.
        - **adjustment_rate_bias** (`Tensor`): the adjustment rate values of the bias vector of the layer.
        """
        self.sum_adjustment_rate += adjustment_rate_weight.sum() + adjustment_rate_bias.sum()
        self.num_params += adjustment_rate_weight.numel() + adjustment_rate_bias.numel()
        
    def compute(self) -> Tensor:
        """Compute this HAT network capacity till this layer.
        
        **Returns:**
        - **network_capacity** (`Tensor`): the calculated network capacity result.
        """
        
        return self.sum_adjustment_rate.float() / self.num_params
