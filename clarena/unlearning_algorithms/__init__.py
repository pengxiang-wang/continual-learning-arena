r"""

# Continual Unlearning Algorithms

This submodule provides the **continual unlearning algorithms** in CLArena.

Please note that this is an API documantation. Please refer to the main documentation pages for more information about the backbone networks and how to configure and implement them:

- [**Configure Unlearning Algorithm**]()
- [**Implement Your Unlearning Algorithm Class**]()


The algorithms are implemented as subclasses of `UnlearningAlgorithm`.

"""

from .base import UnlearningAlgorithm
from .independent_unlearn import IndependentUnlearn


__all__ = ["UnlearningAlgorithm", "IndependentUnlearn"]
