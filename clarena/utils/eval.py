r"""The submodule in `utils` for evaluation utilities."""

import logging

from lightning import LightningModule
from torch import Tensor, nn
from torch.utils.data import DataLoader

from clarena.cl_algorithms import CLAlgorithm
from clarena.cul_algorithms import CULAlgorithm

# always get logger for built-in logging in each module
pylogger = logging.getLogger(__name__)


class CULEvaluation(LightningModule):
    r"""Full evaluation module for continual unlearning."""

    def __init__(
        self,
        main_model: CULAlgorithm,
        refretrain_model: CLAlgorithm,
        reforiginal_model: CLAlgorithm,
        dd_eval_task_ids: list[int],
        ad_eval_task_ids: list[int],
    ):
        r"""
        **Args:**
        - **main_model** (`CULAlgorithm`): the main model to evaluate.
        - **refretrain_model** (`CLAlgorithm`): the reference retrain model to evaluate against.
        - **reforiginal_model** (`CLAlgorithm`): the reference original model that has been trained on all tasks.
        - **dd_eval_task_ids** (`list[int]`): the list of task IDs to evaluate the DD on.
        - **ad_eval_task_ids** (`list[int]`): the list of task IDs to evaluate the accuracy difference on.
        """
        super().__init__()

        self.criterion = nn.CrossEntropyLoss()
        r"""The loss function bewteen the output logits and the target labels. Default is cross-entropy loss."""

        self.main_model = main_model
        r"""The main model for evaluation."""
        self.refretrain_model = refretrain_model
        r"""The reference retrain model for evaluation."""
        self.reforiginal_model = reforiginal_model
        r"""The reference original model for evaluation."""

        self.dd_eval_task_ids: list[int] = dd_eval_task_ids
        r"""The task IDs to evaluate the DD on. """
        self.ad_eval_task_ids: list[int] = ad_eval_task_ids
        r"""The task IDs to evaluate the AD on. """

    def get_test_task_id_from_dataloader_idx(self, dataloader_idx: int) -> int:
        r"""Get the test task ID from the dataloader index.

        **Args:**
        - **dataloader_idx** (`int`): the dataloader index.

        **Returns:**
        - **test_task_id** (`int`): the test task ID.
        """
        dataset_test = self.trainer.datamodule.dataset_test
        test_task_id = list(dataset_test.keys())[dataloader_idx]
        return test_task_id

    def test_step(
        self, batch: DataLoader, batch_idx: int, dataloader_idx: int = 0
    ) -> dict[str, Tensor]:
        r"""Test step for current task `self.task_id`, which tests for all seen tasks indexed by `dataloader_idx`.

        **Args:**
        - **batch** (`Any`): a batch of test data.
        - **dataloader_idx** (`int`): the task ID of seen tasks to be tested. A default value of 0 is given otherwise the LightningModule will raise a `RuntimeError`.

        **Returns:**
        - **outputs** (`dict[str, Tensor]`): a dictionary contains loss and other metrics from this test step. Keys (`str`) are the metrics names, and values (`Tensor`) are the metrics.
        """
        test_task_id = self.get_test_task_id_from_dataloader_idx(dataloader_idx)

        x, y = batch

        # get the aggregated backbone output (instead of logits)
        agg_out_main = self.main_model.aggregated_backbone_output(x)
        agg_out_ref = self.refretrain_model.aggregated_backbone_output(x)

        logits_main, activations_main = self.main_model.forward(
            x, stage="test", task_id=test_task_id
        )  # use the corresponding head to test (instead of the current task `self.task_id`)
        loss_cls_main = self.criterion(logits_main, y)
        acc_main = (logits_main.argmax(dim=1) == y).float().mean()

        logits_full, activations_full = self.reforiginal_model.forward(
            x, stage="test", task_id=test_task_id
        )  # use the corresponding head to test (instead of the current task `self.task_id`)
        loss_cls_full = self.criterion(logits_full, y)
        acc_full = (logits_full.argmax(dim=1) == y).float().mean()
        # Return metrics for lightning loggers callback to handle at `on_test_batch_end()`

        # calculate the difference in accuracy between the main model and the full model
        acc_diff = acc_main - acc_full

        return {
            "agg_out_main": agg_out_main,
            "agg_out_ref": agg_out_ref,
            "acc_diff": acc_diff,
        }


# print("Unlearning JS divergence results:", eval_module.results)


# def compute_unlearning_metrics(output_dir: str) -> None:
#     r"""Compute the unlearning metrics for the continual unlearning experiment and save the results to the `results/` in the output directory.

#     **Args:**
#     - **output_dir** (`str`): the output directory path of the continual unlearning experiment. This directory must contain a `unlearning_ref` directory, which contains the output directory of the unlearning reference experiment.
#     """

#     # initialize unlearning test metrics for unlearned tasks

#     for unlearned_task_id in unlearned_task_ids:
#         # test on the unlearned task

#         test_dataloader = datamodule.test_dataloader()[
#             f"{unlearned_task_id}"
#         ]  # get the test data

#         # set the model to evaluation mode
#         model.to("cpu")
#         model.eval()
#         model_unlearning_test_reference.eval()

#         for batch in test_dataloader:
#             # unlearning test step
#             x, _ = batch
#             batch_size = len(batch)

#             with torch.no_grad():

#                 # get the aggregated backbone output (instead of logits)
#                 aggregated_backbone_output = model.aggregated_backbone_output(x)
#                 aggregated_backbone_output_unlearning_test_reference = (
#                     model_unlearning_test_reference.aggregated_backbone_output(x)
#                 )

#                 # calculate the Jensen-Shannon divergence as distribution distance
#                 js = js_div(
#                     aggregated_backbone_output,
#                     aggregated_backbone_output_unlearning_test_reference,
#                 )

#             print("js", js)
#             print("js", js)
#             print("js", js)
#             print("js", js)
#             print("js", js)
#             print("js", js)
