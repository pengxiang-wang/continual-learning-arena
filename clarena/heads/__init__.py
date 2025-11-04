r"""

# Output Heads

This submodule provides the **output heads** in CLArena.

There are two types of continual learning / unlearning heads in CLArena: `HeadsTIL`, `HeadsCIL` and `HeadDIL`, corresponding to three CL paradigms respectively: Task-Incremental Learning (TIL), Class-Incremental Learning (CIL) and  Domain-Incremental Learning (DIL). For Multi-Task Learning (MTL), we have `HeadsMTL` which is a collection of independent heads for each task.

Please note that this is an API documantation. Please refer to the main documentation pages for more information about the heads.

- [**Configure CL Paradigm in Experiment Index Config**](https://pengxiang-wang.com/projects/continual-learning-arena/docs/configure-your-experiment/experiment-index-config)
- [**A Beginners' Guide to Continual Learning (Multi-head Classifier)](https://pengxiang-wang.com/posts/continual-learning-beginners-guide#sec-CL-classification)

"""

from .heads_cil import HeadsCIL
from .head_dil import HeadDIL
from .heads_til import HeadsTIL

from .heads_mtl import HeadsMTL

from .head_stl import HeadSTL

__all__ = ["HeadsTIL", "HeadsCIL", "HeadDIL", "HeadsMTL", "HeadSTL"]
