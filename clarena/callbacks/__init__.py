r"""

# Callbacks

This submodule provides **callbacks** (other than metric callbacks) that can be used in CLArena.

The callbacks inherit from `lightning.Callback`.

Please note that this is an API documantation. Please refer to the main documentation pages for more information about how to configure and implement callbacks:

- [**Configure Callbacks (CL)**](https://pengxiang-wang.com/projects/continual-learning-arena/docs/continual-learning/configure-main-experiment/callbacks)
- [**Configure Callbacks (CUL)**](https://pengxiang-wang.com/projects/continual-learning-arena/docs/continual-unlearning/configure-main-experiment/callbacks)
- [**Configure Callbacks (MTL)**](https://pengxiang-wang.com/projects/continual-learning-arena/docs/multi-task-learning/configure-main-experiment/callbacks)
- [**Configure Callbacks (STL)**](https://pengxiang-wang.com/projects/continual-learning-arena/docs/single-task-learning/configure-main-experiment/callbacks)
- [**Implement Your Callbacks**](https://pengxiang-wang.com/projects/continual-learning-arena/docs/implement-your-cl-modules/callback)


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
