"""The module to test full experiments."""

import os

import pytest
from hydra import compose, initialize

from clarena.base import CLExperiment

experiment_list = [
    os.path.splitext(f)[0]
    for f in os.listdir("example_configs/experiment")
    if os.path.isfile(os.path.join("example_configs/experiment", f))
]


@pytest.mark.parametrize(
    "experiment",
    experiment_list,
)
def test_experiments(experiment: str):
    """Test all experiments in the experiment configs directory."""
    initialize(version_base="1.3", config_path="example_configs/")

    cfg = compose(
        config_name="entrance",
        overrides=[f"experiment={experiment}"],
        return_hydra_config=True,
    )

    # faster necessary testing
    cfg.trainer.max_epochs = 1
    cfg.num_tasks = 2

    expr = CLExperiment(cfg)
    expr.fit()
