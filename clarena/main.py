"""Main module containing entrances for the clarena package."""

import argparse
import os

from hydra import compose, initialize_config_dir

from clarena import CLExperiment


def cltrain():
    """Cli entrance for training continual learning models."""

    parser = argparse.ArgumentParser(description="Run a continual learning experiment.")
    parser.add_argument("experiment", type=str, help="The experiment to run.")
    parser.add_argument(
        "--config-dir",
        type=str,
        default="configs/",
        help="The directory path to your configs. Not applicable to example configs.",
    )
    parser.add_argument(
        "--entrance",
        type=str,
        default="entrance",
        help="The entrance YAML file name in your configs. Not applicable to example configs.",
    )

    args = parser.parse_args()
    config_dir = args.config_dir
    entrance = args.entrance
    experiment = args.experiment

    if not os.path.isabs(config_dir):
        config_dir = os.path.join(os.getcwd(), config_dir)
    initialize_config_dir(version_base="1.3", config_dir=config_dir)

    cfg = compose(
        config_name=entrance,
        overrides=[f"experiment={experiment}"],
        return_hydra_config=True,
    )
    expr = CLExperiment(cfg)
    expr.fit()
