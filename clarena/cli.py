r"""Entrance for run all `clarena` commands."""

import logging
import os
from datetime import datetime

import hydra
from omegaconf import DictConfig

from clarena.pipelines import (
    CLFullEvaluation,
    CLMainEvaluation,
    CLMainExperiment,
    CULFullEvaluation,
    CULMainExperiment,
    MTLEvaluation,
    MTLExperiment,
    STLEvaluation,
    STLExperiment,
)
from clarena.utils.cfg import preprocess_config

# always get logger for built-in logging in each module
pylogger = logging.getLogger(__name__)


@hydra.main(
    config_path=os.path.join(
        os.getcwd(), "example_configs"
    ),  # construct absolute path so that it can be run from anywhere
    config_name="entrance.yaml",
    version_base="1.3",
)
def clarena(cfg: DictConfig) -> None:
    r"""The main entrance for running the continual learning arena.

    **Args:**
    - **cfg**: (DictConfig) The entire Hydra config.
    """
    if not cfg.get("pipeline"):
        raise ValueError("No pipeline specified in the config.")

    if cfg.pipeline == "CL_MAIN_EXPR":
        pylogger.info("Running: continual learning main experiment...")
        cfg = preprocess_config(cfg, type="CL_MAIN_EXPR")
        pipeline = CLMainExperiment(cfg)
        pipeline.run()

    elif cfg.pipeline == "CL_MAIN_EVAL":
        pylogger.info("Running: continual learning main evaluation...")
        cfg = preprocess_config(cfg, type="CL_MAIN_EVAL")
        pipeline = CLMainEvaluation(cfg)
        pipeline.run()

    elif cfg.pipeline == "CL_REF_JOINT_EXPR":
        pylogger.info(
            "Running: reference joint learning experiment (continual learning)..."
        )
        refjoint_cfg = preprocess_config(cfg, type="CL_REF_JOINT_EXPR")
        pipeline = MTLExperiment(refjoint_cfg)
        pipeline.run()

    elif cfg.pipeline == "CL_REF_INDEPENDENT_EXPR":
        pylogger.info(
            "Running: reference independent learning experiment (continual learning)..."
        )
        refindependent_cfg = preprocess_config(cfg, type="CL_REF_INDEPENDENT_EXPR")
        pipeline = CLMainExperiment(refindependent_cfg)
        pipeline.run()

    elif cfg.pipeline == "CL_REF_RANDOM_EXPR":
        pylogger.info(
            "Running: reference random learning experiment (continual learning)..."
        )
        refrandom_cfg = preprocess_config(cfg, type="CL_REF_RANDOM_EXPR")
        pipeline = CLMainExperiment(refrandom_cfg)
        pipeline.run()

    elif cfg.pipeline == "CL_FULL_EVAL":
        pylogger.info("Running: continual learning full evaluation...")
        cfg = preprocess_config(cfg, type="CL_FULL_EVAL")
        pipeline = CLFullEvaluation(cfg)
        pipeline.run()

    elif cfg.pipeline == "CL_FULL_EXPR":
        pylogger.info("Running: continual learning full experiment...")

        unified_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        cfg.misc.timestamp = unified_timestamp  # override unified timestamp

        pylogger.info(
            "[Full Exeperiment] Running: continual learning main experiment..."
        )
        main_cfg = preprocess_config(cfg, type="CL_MAIN_EXPR")
        pipeline = CLMainExperiment(main_cfg)
        pipeline.run()

        pylogger.info(
            "[Full Exeperiment] Running: reference joint learning experiment (continual learning)..."
        )
        if not cfg.get("refjoint_acc_csv_path"):
            refjoint_cfg = preprocess_config(cfg, type="CL_REF_JOINT_EXPR")
            pipeline = MTLExperiment(refjoint_cfg)
            pipeline.run()
        else:
            pylogger.info(
                "`refjoint_acc_csv_path` argument present. Skip reference joint learning experiment (continual learning)."
            )

        pylogger.info(
            "[Full Exeperiment] Running: reference independent learning experiment (continual learning)..."
        )
        if not cfg.get("refindependent_acc_csv_path"):
            refindependent_cfg = preprocess_config(cfg, type="CL_REF_INDEPENDENT_EXPR")
            pipeline = CLMainExperiment(refindependent_cfg)
            pipeline.run()
        else:
            pylogger.info(
                "`refindependent_acc_csv_path` argument present. Skip reference independent learning experiment (continual learning)."
            )

        pylogger.info(
            "[Full Exeperiment] Running: reference random learning experiment (continual learning)..."
        )
        if not cfg.get("refrandom_acc_csv_path"):
            refrandom_cfg = preprocess_config(cfg, type="CL_REF_RANDOM_EXPR")
            pipeline = CLMainExperiment(refrandom_cfg)
            pipeline.run()
        else:
            pylogger.info(
                "`refrandom_acc_csv_path` argument present. Skip reference random learning experiment (continual learning)."
            )

        pylogger.info(
            "[Full Exeperiment] Running: continual learning full evaluation..."
        )
        fulleval_cfg = preprocess_config(cfg, type="CL_FULL_EVAL_ATTACHED")
        pipeline = CLFullEvaluation(fulleval_cfg)
        pipeline.run()

    elif cfg.pipeline == "CUL_MAIN_EXPR":
        pylogger.info("Running: continual unlearning main experiment...")
        cfg = preprocess_config(cfg, type="CUL_MAIN_EXPR")
        pipeline = CULMainExperiment(cfg)
        pipeline.run()

    elif cfg.pipeline == "CUL_MAIN_EVAL":
        pylogger.info("Running: continual learning main evaluation...")
        cfg = preprocess_config(cfg, type="CUL_MAIN_EVAL")
        pipeline = CLMainEvaluation(cfg)
        pipeline.run()

    elif cfg.pipeline == "CUL_REF_RETRAIN_EXPR":
        pylogger.info("Running: reference retrain experiment (continual unlearning)...")
        refretrain_cfg = preprocess_config(cfg, type="CUL_REF_RETRAIN_EXPR")
        pipeline = CLMainExperiment(refretrain_cfg)
        pipeline.run()

    elif cfg.pipeline == "CUL_REF_ORIGINAL_EXPR":
        pylogger.info(
            "Running: reference original experiment (continual unlearning)..."
        )
        reforiginal_cfg = preprocess_config(cfg, type="CUL_REF_ORIGINAL_EXPR")
        pipeline = CLMainExperiment(reforiginal_cfg)
        pipeline.run()

    elif cfg.pipeline == "CUL_FULL_EVAL":
        pylogger.info("Running: continual unlearning full evaluation...")
        cfg = preprocess_config(cfg, type="CUL_FULL_EVAL")
        pipeline = CULFullEvaluation(cfg)
        pipeline.run()

    elif cfg.pipeline == "CUL_FULL_EXPR":
        pylogger.info("Running: continual unlearning full experiment...")

        unified_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        cfg.misc.timestamp = unified_timestamp  # override unified timestamp

        pylogger.info(
            "[Full Exeperiment] Running: continual unlearning main experiment..."
        )
        main_cfg = preprocess_config(cfg, type="CUL_MAIN_EXPR")
        pipeline = CULMainExperiment(main_cfg)
        pipeline.run()

        pylogger.info(
            "[Full Exeperiment] Running: reference retrain experiment (continual unlearning)..."
        )
        if not cfg.get("refretrain_model_path"):
            refretrain_cfg = preprocess_config(cfg, type="CUL_REF_RETRAIN_EXPR")
            pipeline = CLMainExperiment(refretrain_cfg)
            pipeline.run()
        else:
            pylogger.info(
                "`refretrain_model_path` argument present. Skip reference retrain experiment (continual unlearning)."
            )

        pylogger.info(
            "[Full Exeperiment] Running: reference original experiment (continual unlearning)..."
        )
        if not cfg.get("reforiginal_model_path"):
            reforiginal_cfg = preprocess_config(cfg, type="CUL_REF_ORIGINAL_EXPR")
            pipeline = CLMainExperiment(reforiginal_cfg)
            pipeline.run()
        else:
            pylogger.info(
                "`reforiginal_model_path` argument present. Skip reference original experiment (continual unlearning)."
            )

        pylogger.info(
            "[Full Exeperiment] Running: continual unlearning full evaluation..."
        )
        fulleval_cfg = preprocess_config(cfg, type="CUL_FULL_EVAL_ATTACHED")
        pipeline = CULFullEvaluation(fulleval_cfg)
        pipeline.run()

    elif cfg.pipeline == "MTL_EXPR":
        pylogger.info("Running: multi-task learning experiment...")
        cfg = preprocess_config(cfg, type="MTL_EXPR")
        pipeline = MTLExperiment(cfg)
        pipeline.run()

    elif cfg.pipeline == "MTL_EVAL":
        pylogger.info("Running: multi-task learning evaluation...")
        cfg = preprocess_config(cfg, type="MTL_EVAL")
        pipeline = MTLEvaluation(cfg)
        pipeline.run()

    elif cfg.pipeline == "STL_EXPR":
        pylogger.info("Running: single-task learning experiment...")
        cfg = preprocess_config(cfg, type="STL_EXPR")
        pipeline = STLExperiment(cfg)
        pipeline.run()

    elif cfg.pipeline == "STL_EVAL":
        pylogger.info("Running: single-task learning evaluation...")
        cfg = preprocess_config(cfg, type="STL_EVAL")
        pipeline = STLEvaluation(cfg)
        pipeline.run()
