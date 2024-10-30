"""
The submodule in `callbacks` for `CLRichProgressBar`.
"""

__all__ = ["CLRichProgressBar"]

import logging

from lightning.pytorch.callbacks import RichProgressBar

# always get logger for built-in logging in each module
pylogger = logging.getLogger(__name__)


class CLRichProgressBar(RichProgressBar):
    """Customised Rich Progress Bar for continual learning."""

    def get_metrics(self, *args, **kwargs):
        """Filter out the version number from the metrics displayed in the progress bar."""
        items = super().get_metrics(*args, **kwargs)
        items.pop("v_num", None)  # Remove the version number entry
        return items
