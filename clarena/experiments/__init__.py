r"""

# Experiments

This submodule provides **experiment classes** that manage the overall configs and process of different experiments.

We provide:

- `CLMainTrainExperiment`: continual learning main experiment.
- `CLMainEvalExperiment`: evaluating trained continual learning main experiment.
- `CLEvalExperiment`: full evaluating trained continual learning experiment.
- `CULMainTrainExperiment`: continual unlearning main experiment.
- `CULEvalExperiment`: full evaluating trained continual unlearning experiment.
- `MTLTrainExperiment`: multi-task learning experiment.
- `MTLEvalExperiment`: evaluating trained multi-task learning experiment.
- `STLTrainExperiment`: single-task learning experiment.
- `STLEvalExperiment`: evaluating trained single-task learning experiment.

The `clarena` commands are binded to one or multiple experiments. For example, `clarena train clmain` corresponds to experiment `CLMainTrain`; `clarena full cul` correspond to multiple experiments including `CULMainTrain`, some reference experiments that use `CLMainTrain`, and `CULEval`.

Please note that this is an API documantation. Please refer to the main documentation pages for more information about the `clarena` commands:

- [**Full Usage of `clarena` Command**](https://pengxiang-wang.com/projects/continual-learning-arena/FAQs)
"""

from .clmain_train import CLMainTrain
from .clmain_eval import CLMainEval
from .cl_full_metrics_calculation import CLFullMetricsCalculation
from .culmain_train import CULMainTrain
from .cul_full_eval import CULFullEval
from .mtl_train import MTLTrain
from .mtl_eval import MTLEval
from .stl_train import STLTrain
from .stl_eval import STLEval
