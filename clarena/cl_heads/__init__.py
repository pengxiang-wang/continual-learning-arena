r"""

# Continual Learning Heads

This submodule provides the **continual learning heads** in CLArena. 

There are two types of heads in CLArena: `HeadsTIL` and `HeadsCIL`, corresponding to two CL paradigms respectively: Task-Incremental Learning (TIL) and Class-Incremental Learning (CIL). 

Please note that this is an API documantation. Please refer to the main documentation pages for more information about the heads.

- [**Configure CL Paradigm in Experiment Index Config**](https://pengxiang-wang.com/projects/continual-learning-arena/docs/configure-your-experiment/experiment-index-config)
- [**A Beginners' Guide to Continual Learning (Multi-head Classifier)](https://pengxiang-wang.com/posts/continual-learning-beginners-guide#sec-CL-classification)

"""

from .heads_cil import HeadsCIL
from .heads_til import HeadsTIL

__all__ = ["HeadsTIL", "HeadsCIL"]
