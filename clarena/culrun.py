r"""Entrance to run a continual unlearning experiment.

Continual unlearning is a paradigm that injects unlearning requests into the continual learning process. Unlearning requests must be specified in the configuration.

The model is evaluated in two parts:

1. Continual learning performance on the tasks that are not unlearned.
2. Unlearning performance on the tasks that are unlearned.

Note that the second part cannot be done in this module run as it involves training a reference model (on not unlearned tasks) to compare with the unlearned model. For the unlearning performance, please go to the `culrunref.py` and `uleval.py` module.
"""

import argparse
import os
from argparse import Namespace

import hydra
from omegaconf import DictConfig

from clarena.base import (
    CULExperiment,  # It was supposed to be `from clarena import CULExperiment`, but it is not working when pip install -e . is used.
)
from clarena.utils import preprocess_config


def add_and_parse_args() -> tuple[Namespace, list[str]]:
    r"""Add and parse command line arguments that are not handled by Hydra.

    **Returns:**
    - **known_args** (`argparse.Namespace`): parsed arguments, including `--config-dir` and `--entrance`.
    - **unknown_args** (`list[str]`): unknown arguments that are handled by Hydra, for configuration overrides.
    """
    parser = argparse.ArgumentParser(
        usage="culrun [-h] [--config-dir CONFIG_DIR] [--entrance ENTRANCE] -- [experiment=EXPERIMENT_NAME] [(overrides)...]",
        description="Run a continual unlearning experiment. ",
        epilog="After them, you must specifiy your experiment through [experiment=EXPERIMENT_NAME]. The EXPERIMENT_NAME should be the name of the experiment YAML file in your configs/experiment directory. You can also add other overrides to the experiment configuration.",
    )
    parser.add_argument(
        "--config-dir",
        type=str,
        default="configs/",
        help="The directory path to your configs. Default: 'configs/'.",
    )
    parser.add_argument(
        "--entrance",
        type=str,
        default="entrance.yaml",
        help="The entrance YAML file name in your configs. Default: 'entrance.yaml'.",
    )
    return (
        parser.parse_known_args()
    )  # use parse_known_args() instead of parse_args() to enable Hydra overrides


def culrun() -> None:
    r"""The main entrance for running a continual unlearning experiment."""

    # parse the arguments
    args, _ = add_and_parse_args()

    # get the absolute path of the config directory
    hydra_config_dir = os.path.abspath(args.config_dir)

    # get the entrance YAML file name
    hydra_config_name = args.entrance

    def main(cfg: DictConfig) -> None:
        r"""A placeholder for the main codes. This function should be decorated by `hydra.main()`.

        **Args:**
        - **cfg** (`DictConfig`): the configuration for the experiment.
        """

        # preprocess the configuration before constructing the experiment
        preprocess_config(cfg)

        # construct the experiment
        expr = CULExperiment(cfg)

        # execute the experiment
        expr.run()

    # Use the specified config path for Hydra config
    hydra_decorated_main = hydra.main(
        version_base="1.3", config_path=hydra_config_dir, config_name=hydra_config_name
    )(main)

    hydra_decorated_main()
