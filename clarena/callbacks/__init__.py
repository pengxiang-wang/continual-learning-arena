"""

# Callbacks

This submodule provides **callbacks** that can be used in CLArena. 

Please note that this is an API documantation. Please refer to the main documentation page for more information about the callbacks and how to use and customize them:

- **Configure your callbacks:** [https://pengxiang-wang.com/projects/continual-learning-arena/docs/configure-your-experiments/callbacks](https://pengxiang-wang.com/projects/continual-learning-arena/docs/configure-your-experiments/callbacks)
- **Implement your callbacks:** [https://pengxiang-wang.com/projects/continual-learning-arena/docs/implement-your-cl-modules/callbacks](https://pengxiang-wang.com/projects/continual-learning-arena/docs/implement-your-cl-modules/callbacks)


"""

from .cl_rich_progress_bar import CLRichProgressBar
from .image_show import ImageShowCallback
from .metrics import MetricsCallback
from .pylogger import PyloggerCallback

__all__ = []
