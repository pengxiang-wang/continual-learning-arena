r"""

# Continual Learning Regularisers

This submodule provides the **regularisers** which are added to the loss function of corresponding continual learning algorithms in CLArena. It can promote forgetting preventing which is the major mechanism in regularisation-based approaches, or for other purposes.

Please note that this is an API documantation. Please refer to the main documentation pages for more information about the regularisers: 

- [**Implement your regularisers in CL algorithms**](https://pengxiang-wang.com/projects/continual-learning-arena/docs/implement-your-cl-modules/cl-algorithm#sec-regularisers)
- [**A Beginners' Guide to Continual Learning (Regularisation-based Approaches)**](https://pengxiang-wang.com/posts/continual-learning-beginners-guide#sec-regularisation-based-approaches)


The regularisers are implemented as subclasses of `nn.Module`.

"""

from .distillation import DistillationReg
from .hat_mask_sparsity import HATMaskSparsityReg
from .parameter_change import ParameterChangeReg

__all__ = ["distillation", "hat_mask_sparsity", "parameter_change"]
