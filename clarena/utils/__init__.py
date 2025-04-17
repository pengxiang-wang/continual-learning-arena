"""

# Utilities

This submodule provides utilities that are used in CLArena, which includes:

- **Cfg**: configuration file for CLArena.
- **Metrics**: define metrics particularly for CL.
- **Save**: for saving results to files.
- **Plot**: for plotting figures.
- **Transforms**: for data transforms.
- **Misc**: miscellaneous functions.

"""

from .cfg import preprocess_config
from .metrics import HATNetworkCapacity, MeanMetricBatch
from .plot import (
    plot_test_ave_acc_curve_from_csv,
    plot_test_ave_loss_cls_curve_from_csv,
    plot_hat_mask,
    plot_hat_adjustment_rate,
    plot_unlearning_test_distance_from_csv,
    plot_test_acc_matrix_from_csv,
    plot_test_loss_cls_matrix_from_csv,
)
from .save import update_test_acc_to_csv, update_test_loss_cls_to_csv
from .transforms import min_max_normalise, js_div
from .misc import str_to_class

__all__ = ["cfg", "metrics", "save", "plot", "transforms", "misc"]
