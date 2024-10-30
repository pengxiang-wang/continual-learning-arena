"""

# Continual Learning Algorithms

This submodule provides the **continual learning algorithms** in CLArena. 

Please note that this is an API documantation. Please refer to the main documentation page for more information about the backbone networks and how to use and customize them:

- **Configure your CL algorithm:** [https://pengxiang-wang.com/projects/continual-learning-arena/docs/configure-your-experiments/cl-algorithm](https://pengxiang-wang.com/projects/continual-learning-arena/docs/configure-your-experiments/cl-algorithm)
- **Implement your CL algorithm:** [https://pengxiang-wang.com/projects/continual-learning-arena/docs/implement-your-cl-modules/cl-algorithm](https://pengxiang-wang.com/projects/continual-learning-arena/docs/implement-your-cl-modules/cl-algorithm)
- **A beginners' guide to continual learning (CL algorithm):** [https://pengxiang-wang.com/posts/continual-learning-beginners-guide#methodology](https://pengxiang-wang.com/posts/continual-learning-beginners-guide#methodology)

"""

from .base import CLAlgorithm
from .finetuning import Finetuning

__all__ = ["CLAlgorithm", "finetuning"]
