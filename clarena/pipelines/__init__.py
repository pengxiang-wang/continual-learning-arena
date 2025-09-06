r"""

# Experiment and Evaluation Pipelines

This submodule provides **experiment and evaluation pipeline classes** that manage the index configs and process of experiments and evaluations.

We provide experiment pipeline classes:

- `CLMainExperiment`: Continual Learning Main Experiment.
- `CULMainExperiment`: Continual Unlearning Main Experiment.
- `MTLExperiment`: Multi-Task Learning Experiment.
- `STLExperiment`: Single-Task Learning Experiment.

And evaluation pipeline classes:

- `CLMainEvaluation`: Continual Learning Main Evaluation.
- `CLFullEvaluation`: Continual Learning Full Evaluation.
- `CULFullEvaluation`: Continual Unlearning Full Evaluation.
- `MTLEvaluation`: Multi-Task Learning Evaluation.
- `STLEvaluation`: Single-Task Learning Evaluation.

Please note that this is an API documantation. Please refer to the main documentation for more information about experiments and evaluations:

- [**Experiments and Evaluations**](https://pengxiang-wang.com/projects/continual-learning-arena/docs/experiments-and-evaluations)

"""

# experiments
from .cl_main_expr import CLMainExperiment
from .cul_main_expr import CULMainExperiment
from .mtl_expr import MTLExperiment
from .stl_expr import STLExperiment

# evaluations
from .cl_main_eval import CLMainEvaluation
from .cl_full_eval import CLFullEvaluation
from .cul_full_eval import CULFullEvaluation
from .mtl_eval import MTLEvaluation
from .stl_eval import STLEvaluation

__all__ = [
    "cl_main_expr",
    "cul_main_expr",
    "mtl_expr",
    "stl_expr",
    "cl_main_eval",
    "cl_full_eval",
    "cul_full_eval",
    "mtl_eval",
    "stl_eval",
]
