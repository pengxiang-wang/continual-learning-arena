r"""

# Continual Learning Regularizers

This submodule provides the **regularizers** which are added to the loss function of corresponding continual learning algorithms. It can promote forgetting preventing which is the major mechanism in regularization-based approaches, or for other purposes.

The regularizers inherit from `nn.Module`.

Please note that this is an API documantation. Please refer to the main documentation pages for more information about the regularizers:

- [**Implement custom regularizers in CL algorithms**](https://pengxiang-wang.com/projects/continual-learning-arena/docs/custom-implementation/cl-algorithm#sec-regularizers)
- [**A Beginners' Guide to Continual Learning (Regularization-based Approaches)**](https://pengxiang-wang.com/posts/continual-learning-beginners-guide#sec-regularisation-based-approaches)



"""

from .distillation import DistillationReg
from .hat_mask_sparsity import HATMaskSparsityReg
from .parameter_change import ParameterChangeReg

__all__ = ["distillation", "hat_mask_sparsity", "parameter_change"]
