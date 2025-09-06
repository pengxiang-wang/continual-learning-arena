r"""The submodule in `utils` of miscellaneous utilities."""

__all__ = ["str_to_class"]

import importlib
import logging

# always get logger for built-in logging in each module
pylogger = logging.getLogger(__name__)


def str_to_class(class_path: str) -> type:
    r"""Convert a string to a class.

    **Args:**
    - **class_path** (`str`): the string of the class path, e.g. `torchvision.datasets.MNIST`.

    **Returns:**
    - **cls** (`type`): the class object.
    """
    module_name, class_name = class_path.rsplit(".", 1)
    module = importlib.import_module(module_name)
    cls = getattr(module, class_name)
    return cls
