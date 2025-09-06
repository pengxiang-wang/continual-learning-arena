r"""

# Metrics

This submodule provides the **metric callbacks** in CLArena, which control each metric's computation, logging and visualization process.

Here are the base classes for metric callbacks, which inherit from PyTorch Lightning `Callback`:

- `MetricCallback`: the base class for all metric callbacks.

Please note that this is an API documentation. Please refer to the main documentation pages for more information about how to configure and implement metric callbacks:

- [**Configure Metrics**](https://pengxiang-wang.com/projects/continual-learning-arena/docs/components/metrics)
- [**Implement Custom Callback**](https://pengxiang-wang.com/projects/continual-learning-arena/docs/custom-implementation/callback)
- [**A Summary of Continual Learning Metrics**](https://pengxiang-wang.com/posts/continual-learning-metrics)

"""

from .base import MetricCallback

from .cl_acc import CLAccuracy
from .cl_loss import CLLoss
from .cul_dd import CULDistributionDistance
from .cul_ad import CULAccuracyDifference
from .hat_adjustment_rate import HATAdjustmentRate
from .hat_network_capacity import HATNetworkCapacity
from .hat_masks import HATMasks


from .mtl_acc import MTLAccuracy
from .mtl_loss import MTLLoss

from .stl_acc import STLAccuracy
from .stl_loss import STLLoss

__all__ = [
    "MetricCallback",
    "cl_acc",
    "cl_loss",
    "cul_dd",
    "cul_ad",
    "hat_adjustment_rate",
    "hat_network_capacity",
    "hat_masks",
    "mtl_acc",
    "mtl_loss",
    "stl_acc",
    "stl_loss",
]
