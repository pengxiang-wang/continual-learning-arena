r"""

# Callbacks

This submodule provides **callbacks** (other than metric callbacks) that can be used in CLArena.

The callbacks inherit from `lightning.Callback`.

Please note that this is an API documantation. Please refer to the main documentation pages for more information about how to configure and implement callbacks:

- [**Configure Callbacks**](https://pengxiang-wang.com/projects/continual-learning-arena/docs/components/callbacks)
- [**Implement Custom Callback**](https://pengxiang-wang.com/projects/continual-learning-arena/docs/custom-implementation/callback)


"""

from .cl_rich_progress_bar import CLRichProgressBar
from .pylogger import CLPylogger, CULPylogger, MTLPylogger, STLPylogger
from .save_first_batch_images import SaveFirstBatchImages
from .save_models import SaveModels

__all__ = [
    "cl_rich_progress_bar",
    "save_first_batch_images",
    "save_models",
    "pylogger",
]
