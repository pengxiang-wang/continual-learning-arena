r"""

# Continual Unlearning Algorithms

This submodule provides the **continual unlearning algorithms** in CLArena.

The algorithms are implemented as subclasses of `UnlearningAlgorithm`.

Please note that this is an API documantation. Please refer to the main documentation pages for more information about the backbone networks and how to configure and implement them:

- [**Configure Unlearning Algorithm**]()
- [**Implement Your Unlearning Algorithm Class**]()



"""

from .base import CULAlgorithm
from .independent_unlearn import IndependentUnlearn
from .amnesiac_hat_unlearn import AmnesiacHATUnlearn


__all__ = ["CULAlgorithm", "IndependentUnlearn", "AmnesiacHATUnlearn"]
