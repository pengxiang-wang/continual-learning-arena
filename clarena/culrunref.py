r"""Entrance to run the reference experiment of a continual unlearning experiment.

The unlearning reference experiment is used to evaluate the unlearning performance of a continual unlearning experiment. It is a standalone script run independently from the continual unlearning experiment. For the original unlearning experiment, please go to the `culrun.py` module.
"""

import argparse
import os
from argparse import Namespace

import hydra
from omegaconf import DictConfig

from clarena.base import (
    CLExperiment,  # It was supposed to be `from clarena import CLExperiment`, but it is not working when pip install -e . is used.
)
from clarena.utils import construct_unlearning_ref_config


def add_and_parse_args() -> tuple[Namespace, list[str]]:
    r"""Add and parse command line arguments that are not handled by Hydra.

    **Returns:**
    - **known_args** (`argparse.Namespace`): parsed arguments, including `--cul-output-dir`.
    - **unknown_args** (`list[str]`): unknown arguments that are handled by Hydra.
    """
    parser = argparse.ArgumentParser(
        usage="culrunref [-h] [--cul-output-dir CUL_OUTPUT_DIR]",
        description="Run the reference experiment of a continual unlearning experiment. ",
    )
    parser.add_argument(
        "--cul-output-dir",
        type=str,
        help="The output directory path of original continual unlearning experiment.",
    )

    return (
        parser.parse_known_args()
    )  # use parse_known_args() instead of parse_args() to enable Hydra overrides


def culrunref() -> None:
    r"""The main entrance to run the reference experiment of a continual unlearning experiment."""

    # parse the arguments
    args, _ = add_and_parse_args()

    # get the absolute path of the original continual unlearning experiment output directory
    cul_output_dir = os.path.abspath(args.cul_output_dir)

    # get the absolute path of the config directory, which is the .hydra directory in the output directory
    hydra_config_dir = os.path.join(cul_output_dir, ".hydra")

    # the config YAML file name, which is always "config.yaml"
    hydra_config_name = "config.yaml"

    def main(cfg: DictConfig) -> None:
        r"""A placeholder for the main codes. This function should be decorated by `hydra.main()`.

        **Args:**
        - **cfg** (`DictConfig`): the configuration for the experiment.
        """

        # construct the reference experiment config from the original continual unlearning experiment config
        ulref_cfg = construct_unlearning_ref_config(cfg)

        # construct the experiment
        expr = CLExperiment(ulref_cfg)

        # execute the reference experiment
        expr.run()

    # Use the specified config path for Hydra config
    hydra_decorated_main = hydra.main(
        version_base="1.3", config_path=hydra_config_dir, config_name=hydra_config_name
    )(main)

    hydra_decorated_main()
