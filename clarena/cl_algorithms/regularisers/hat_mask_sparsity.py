r"""The submodule in `regularisers` for HAT mask sparsity regularisation`."""

__all__ = ["HATMaskSparsityReg"]

from torch import Tensor, nn


class HATMaskSparsityReg(nn.Module):
    r"""Mask Sparsity Regulariser of HAT (Hard Attention to the Task).

    It promotes the low capacity usage that is reflected by occupation of masks in the parameter space.

    See chapter 2.6 "Promoting Low Capacity Usage" in [HAT paper](http://proceedings.mlr.press/v80/serra18a).
    """

    def __init__(
        self,
        factor: float,
        mode: str = "original",
    ) -> None:
        r"""Initialise the regulariser.

        **Args:**
        - **factor** (`float`): the regularisation factor.
        - **mode** (`str`): the mode of mask sparsity regularisation, should be one of the following:
            1. 'original' (default): the original mask sparsity regularisation in HAT paper.
            2. 'cross': the cross version mask sparsity regularisation.
        """

        super().__init__()

        self.factor = factor
        """Store the regularisation factor for mask sparsity."""
        self.mode = mode
        """Store the mode of mask sparsity regularisation."""

    def forward(
        self, mask: dict[str, Tensor], previous_cumulative_mask: dict[str, Tensor]
    ) -> Tensor:
        r"""Calculate the mask sparsity regularisation loss.

        **Args:**
        - **mask** (`dict[str, Tensor]`): the mask for the current task.
        - **previous_cumulative_mask** (`dict[str, Tensor]`): the cumulative mask for the previous tasks.

        **Returns:**
        - **reg** (`Tensor`): the mask sparsity regularisation loss.
        """

        if self.mode == "original":
            return self.original_reg(mask, previous_cumulative_mask)
        elif self.mode == "cross":
            return self.cross_reg(mask, previous_cumulative_mask)

    def original_reg(
        self, mask: dict[str, Tensor], previous_cumulative_mask: dict[str, Tensor]
    ) -> Tensor:
        r"""Calculate the original mask sparsity regularisation loss in HAT paper.

        See chapter 2.6 "Promoting Low Capacity Usage" in [HAT paper](http://proceedings.mlr.press/v80/serra18a).

        **Args:**
        - **mask** (`dict[str, Tensor]`): the mask for the current task. The $\mathrm{A}^t$ in the paper.
        - **previous_cumulative_mask** (`dict[str, Tensor]`): the cumulative mask for the previous tasks. The $\mathrm{A}^{<t}$ in the paper.

        **Returns:**
        - **reg** (`Tensor`): the original mask sparsity regularisation loss.
        """

        count_available = 0  # number of units available for the new task
        count_new_task_occupation_in_available = (
            0  # number of units occupied by the new task in the available units
        )

        network_sparsity = {}

        for layer_name in mask.keys():
            # statistics through all layers
            available = (
                1 - previous_cumulative_mask[layer_name]
            ).sum()  # count the number of units available for the new task

            new_task_occupation_in_available = (
                mask[layer_name] * (1 - previous_cumulative_mask[layer_name])
            ).sum()
            # count the number of units occupied by the new task in the available units

            # add to statistics
            count_available += available
            count_new_task_occupation_in_available += new_task_occupation_in_available
            network_sparsity[layer_name] = (
                (new_task_occupation_in_available / available) if available > 10 else 0
            )

        # the mask sparsity regularisation minimises the ratio of the number of units occupied by the new task to the number of units available for the new task. The regularisizer is to let HAT allocates more units from previous tasks to the new task rather than using available units.
        reg = (
            count_new_task_occupation_in_available / count_available
            if count_available
            > 10  # to avoid division by a very small number, which makes the regularisation less meaningful
            else 0
        )

        return self.factor * reg, network_sparsity

    def cross_reg(
        self, mask: dict[str, Tensor], previous_cumulative_mask: dict[str, Tensor]
    ) -> Tensor:
        r"""Calculate the cross mask sparsity regularisation loss. This is an attempting improvement by me to the original regularisation, which not only considers the sparsity in available units but also the density in the units occupied by previous tasks.

        **Args:**
        - **mask** (`dict[str, Tensor]`): the mask for the current task. The $\mathrm{A}^t$ in the paper.
        - **previous_cumulative_mask** (`dict[str, Tensor]`): the cumulative mask for the previous tasks. The $\mathrm{A}^{<t}$ in the paper.

        **Returns:**
        - **reg** (`Tensor`): the cross mask sparsity regularisation loss.
        """

        count_previous = 0  # number of units occupied by the previous tasks
        count_new_task_occupation_in_previous = (
            0  # number of units occupied by the new task in the previous tasks
        )

        network_sparsity_2 = {}

        for layer_name in mask.keys():
            # statistics through all layers
            previous = previous_cumulative_mask[
                layer_name
            ].sum()  # count the number of units occupied by the previous tasks
            new_task_occupation_in_previous = (
                mask[layer_name] * previous_cumulative_mask[layer_name].sum()
            )  # count the number of units occupied by the new task in the previous tasks

            # add to statistics
            count_previous += previous
            count_new_task_occupation_in_previous += new_task_occupation_in_previous
            network_sparsity_2[layer_name] = (
                (new_task_occupation_in_previous / previous) if previous > 10 else 0
            )

        # the mask sparsity regularisation maximises the ratio of the number of units occupied by the new task to the number of units occupied by the previous tasks. The regularisizer is to let HAT allocates more units from previous tasks to the new task rather than using available units.
        reg2 = (
            1 - count_new_task_occupation_in_previous / count_previous
            if count_previous
            > 10  # to avoid division by a very small number, which makes the regularisation less meaningful
            else 0
        )

        reg1, network_sparsity_1 = self.original_reg(mask, previous_cumulative_mask)

        reg = (
            reg1 + reg2
        ) / 2  # our cross regularisation is the average of the original and the regularisation proposed above

        network_sparsity = {}
        for layer_name in mask.keys():
            # merge the two network sparsity statistics
            network_sparsity[layer_name] = (
                network_sparsity_1[layer_name] + network_sparsity_2[layer_name]
            ) / 2

        return self.factor * reg, network_sparsity
