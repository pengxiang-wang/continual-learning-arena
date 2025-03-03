"""

# Utilities

This submodule provides utilities that are used in CLArena, which includes:

- **Metrics**: define metrics particularly for CL.
- **Save**: for saving results to files.
- **Plot**: for plotting figures.


"""

from .cfg import preprocess_config
from .metrics import HATNetworkCapacity, MeanMetricBatch
from .plot import (
    plot_test_ave_acc_curve_from_csv,
    plot_test_ave_loss_cls_curve_from_csv,
    plot_hat_mask,
    plot_test_acc_matrix_from_csv,
    plot_test_loss_cls_matrix_from_csv,
)
from .save import update_test_acc_to_csv, update_test_loss_cls_to_csv
from .transforms import min_max_normalise

__all__ = ["cfg", "metrics", "save", "plot", "transforms"]
