r"""

# Metrics

This submodule provides the **metrics** in CLArena. This includes:

- Callbacks that control each metric's calculation, logging and visualization process. They are implemented as subclasses of `MetricCallback`.
- Custom metrics that can be used in continual learning experiments. They are implemented as classes in `torchmetrics`.

Please note that this is an API documentation. Please refer to the main documentation pages for more information about the metrics and how to configure and implement them:

- [**Configure Metrics**](https://pengxiang-wang.com/projects/continual-learning-arena/docs/configure-your-experiment/metrics)
- [**Implement Your Metrics**](https://pengxiang-wang.com/projects/continual-learning-arena/docs/implement-your-cl-modules/metrics)
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
    "CLAccuracy",
    "CLLoss",
    "CULDistributionDistance",
    "CULAccuracyDifference",
    "HATAdjustmentRate",
    "HATNetworkCapacity",
    "HATMasks",
    "MTLAccuracy",
    "MTLLoss",
    "STLAccuracy",
    "STLLoss",
]
