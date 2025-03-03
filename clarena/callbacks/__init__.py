r"""

# Callbacks

This submodule provides **callbacks** that can be used in CLArena. 

Please note that this is an API documantation. Please refer to the main documentation pages for more information about the callbacks and how to configure and implement them:

- [**Configure Callbacks**](https://pengxiang-wang.com/projects/continual-learning-arena/docs/configure-your-experiment/callbacks)
- [**Implement Your Callbacks**](https://pengxiang-wang.com/projects/continual-learning-arena/docs/implement-your-cl-modules/callback)

The callbacks are implemented as subclasses of `lightning.Callback`.

"""

from .cl_metrics import CLMetricsCallback
from .cl_rich_progress_bar import CLRichProgressBar
from .hat_metrics import HATMetricsCallback
from .pylogger import PyloggerCallback
from .save_first_batch_images import SaveFirstBatchImagesCallback

__all__ = [
    "cl_rich_progress_bar",
    "save_first_batch_images",
    "cl_metrics",
    "pylogger",
    "hat_metrics",
]
