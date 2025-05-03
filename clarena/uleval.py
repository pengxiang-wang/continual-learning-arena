r"""Entrance to evaluate the unlearning performance of a continual unlearning experiment.

This module only evaluates the unlearning performance from the already executed continual unlearning experiment and its reference experiment. It is a standalone script run independently from both experiments. For the original unlearning experiment, please go to the `culrun.py` module. For the reference experiment, please go to the `culrunref.py` module.
"""

import argparse
import os
from argparse import Namespace

import hydra
from omegaconf import DictConfig


def add_and_parse_args() -> tuple[Namespace, list[str]]:
    r"""Add and parse command line arguments that are not handled by Hydra.

    **Returns:**
    - **known_args** (`argparse.Namespace`): parsed arguments, including `--cul-output-dir`.
    - **unknown_args** (`list[str]`): unknown arguments that are handled by Hydra.
    """
    parser = argparse.ArgumentParser(
        usage="uleval [-h] [--cul-output-dir CUL_OUTPUT_DIR]",
        description="Evaluates the unlearning performance of a continual unlearning experiment. ",
    )
    parser.add_argument(
        "--cul-output-dir",
        type=str,
        help="The output directory path of continual unlearning experiment.",
    )

    return (
        parser.parse_known_args()
    )  # use parse_known_args() instead of parse_args() to enable Hydra overrides


def uleval() -> None:
    r"""The main entrance to evaluate the unlearning performance of a continual unlearning experiment."""

    # parse the arguments
    args, _ = add_and_parse_args()

    # get the absolute path of the continual unlearning experiment output directory
    cul_output_dir = os.path.abspath(args.cul_output_dir)

    # def compute_unlearning_metrics(output_dir: str) -> None:
    #     r"""Compute the unlearning metrics for the continual unlearning experiment and save the results to the `results/` in the output directory.

    #     **Args:**
    #     - **output_dir** (`str`): the output directory path of the continual unlearning experiment. This directory must contain a `unlearning_ref` directory, which contains the output directory of the unlearning reference experiment.
    #     """
    #     pylogger.info("Evaluating unlearning metrics...")

    #     # initialise unlearning test metrics for unlearned tasks
    #     distribution_distance_unlearning_test = {
    #         f"{task_id}": MeanMetricBatch() for task_id in unlearned_task_ids
    #     }

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

    #             # update the accumulated metrics in order to calculate the metrics of the epoch
    #             self.distribution_distance_unlearning_test[f"{unlearned_task_id}"].update(
    #                 js,
    #                 batch_size,
    #             )

    #     save.update_unlearning_test_distance_to_csv(
    #         unlearning_test_after_task_id=unlearning_test_after_task_id,
    #         distance_metric=self.distribution_distance_unlearning_test,
    #         csv_path=self.unlearning_test_distance_csv_path,
    #     )
    #     plot.plot_unlearning_test_distance_from_csv(
    #         csv_path=self.unlearning_test_distance_csv_path,
    #         plot_path=self.unlearning_test_distance_plot_path,
    #     )
