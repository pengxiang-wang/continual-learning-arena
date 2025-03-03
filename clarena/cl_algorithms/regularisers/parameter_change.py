r"""The submodule in `regularisers` for parameter change regularisation."""

__all__ = ["ParameterChangeReg"]

import torch
from torch import Tensor, nn


class ParameterChangeReg(nn.Module):
    r"""Parameter change regulariser.

    $$R(\theta) = \text{factor} * \sum_i w_i \|\theta_i - \theta^\star_i\|^p$$

    It promotes the target set of parameters $\theta = {\theta_i}_i$ not changing too much from another set of parameters $\theta^\star = {\theta^\star_i}_i$. The parameter distance here is $L^p$ distance. The regularisation can be parameter-wise weighted, i.e. $w_i$ in the formula.

    It is used in:
    - [L2 Regularisation algorithm](https://www.pnas.org/doi/10.1073/pnas.1611835114): as a L2 regulariser for the current task parameters to prevent them from changing too much from the previous task parameters.
    - [EWC (Elastic Weight Consolidation) algorithm](https://www.pnas.org/doi/10.1073/pnas.1611835114): as a weighted L2 regulariser for the current task parameters to prevent them from changing too much from the previous task parameters. The regularisation weights are parameter importance measure calculated from fisher information. See equation 3 in the [EWC paper](https://www.pnas.org/doi/10.1073/pnas.1611835114).
    """

    def __init__(
        self,
        factor: float,
        p_norm: float,
    ) -> None:
        r"""Initialise the regulariser.

        **Args:**
        - **factor** (`float`): the regularisation factor. Note that it is $\frac{\lambda}{2}$ rather than $\lambda$ in the [EWC paper](https://www.pnas.org/doi/10.1073/pnas.1611835114).
        - **p_norm** (`float`): the norm of the distance, should be a positive float.
        """
        nn.Module.__init__(self)

        self.factor = factor
        """Store the regularisation factor for parameter change."""
        self.p_norm = p_norm
        """Store the the norm of the distance of two set of parameters. """

    def forward(
        self,
        target_model: nn.Module,
        ref_model: nn.Module,
        weights: dict[str, Tensor],
    ) -> Tensor:
        r"""Calculate the regularisation loss.

        **Args:**
        - **target_model** (`nn.Module`): the model of the target parameters. In EWC, it's the model of current training task.
        - **ref_model** (`nn.Module`): the reference model that you want target model parameters to prevent changing from. The reference model must have the same structure as the target model. In EWC, it's the model of one of the previous tasks.
        - **weights** (`dict[str, Tensor]`): the regularisation weight for each parameter. Keys are parameter names and values are the weight tensors. The weight tensors must match the shape of model parameters. In EWC, it's the importance measure of each parameter, calculated from fisher information thing.

        **Returns:**
        - **reg** (`Tensor`): the parameter change regularisation value.
        """
        reg = 0.0

        for (param_name, target_param), (param_name, ref_param) in zip(
            target_model.named_parameters(), ref_model.named_parameters()
        ):
            weight = weights[param_name]
            reg += torch.sum(weight * (target_param - ref_param).norm(p=self.p_norm))

        return self.factor * reg
