r"""

# Continual Learning Regularisers

This submodule provides the **regularizers** which are added to the loss function of relative continual learning algorithms in CLArena. It could promote forgetting preventing which is the major mechanism in regularisation-based approaches, or for other purposes.

Please note that this is an API documantation. Please refer to the main documentation page for more information about the regularizers: 

- **Implement your regularisers in CL algorithms:** [https://pengxiang-wang.com/projects/continual-learning-arena/docs/implement-your-cl-modules/cl-algorithm](https://pengxiang-wang.com/projects/continual-learning-arena/docs/implement-your-cl-modules/cl-algorithm#regularisers)
- **A beginners' guide to continual learning (regularisation-based approaches):** [https://pengxiang-wang.com/posts/continual-learning-beginners-guide#methodology](https://pengxiang-wang.com/posts/continual-learning-beginners-guide#regularisation-based-approaches)

"""

from .distillation import DistillationReg
from .hat_mask_sparsity import HATMaskSparsityReg
from .parameter_change import ParameterChangeReg

__all__ = ["distillation", "hat_mask_sparsity", "parameter_change"]
