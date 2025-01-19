"""

# Utilities

This submodule provides utilities that are used in CLArena, which includes:

- **Metrics**: define metrics particularly for CL.
- **Save**: for saving results to files.
- **Plot**: for plotting complex figures.


"""

from .metrics import HATNetworkCapacity, MeanMetricBatch
from .plot import (
    plot_acc_matrix_from_csv,
    plot_ave_acc_from_csv,
    plot_ave_loss_cls_from_csv,
    plot_hat_mask,
    plot_loss_cls_matrix_from_csv,
)
from .save import save_acc_to_csv, save_loss_cls_to_csv

__all__ = ["metrics", "save", "plot"]
