r"""Entrance to run all CLArena commands."""

import os
import subprocess
from datetime import datetime

import typer
from hydra import compose, initialize
from omegaconf import DictConfig

from clarena.experiments import (
    CLFullMetricsCalculation,
    CLMainEval,
    CLMainTrain,
    CULFullEval,
    CULMainTrain,
    MTLEval,
    MTLTrain,
    STLEval,
    STLTrain,
)
from clarena.utils.cfg import preprocess_config

# outer app for all commands, which is binded to command `clarena` in pyproject.toml
app = typer.Typer()

# sub-app for command `clarena train`
train_app = typer.Typer()
app.add_typer(train_app, name="train")

# sub-app for command `clarena eval`
eval_app = typer.Typer()
app.add_typer(eval_app, name="eval")

# sub-app for command `clarena full`
full_app = typer.Typer()
app.add_typer(full_app, name="full")


def load_cfg(overrides: list[str] | None = None) -> DictConfig:
    """Helper function to load Hydra config dynamically with overrides."""

    # Get the relative path from cli.py location to the configs directory
    # in the current working directory
    current_dir = os.getcwd()
    cli_dir = os.path.dirname(os.path.abspath(__file__))
    relative_config_path = os.path.relpath(
        os.path.join(current_dir, "example_configs"), cli_dir
    )

    with initialize(config_path=relative_config_path, version_base="1.3"):
        cfg = compose(config_name="entrance.yaml", overrides=overrides or [])
    return cfg


# sub-app for command `clarena train clmain`
@train_app.command("clmain")
def clmain_train(overrides: list[str] = typer.Argument(None)):
    r"""The main entrance for continual learning main experiment.

    **Args:**
    - **overrides**: List of Hydra config overrides from CLI.
    """
    cfg = load_cfg(overrides)
    preprocess_config(cfg, expr_type="clmain_train")
    expr = CLMainTrain(cfg)
    expr.run()


# sub-app for command `clarena eval clmain`
@eval_app.command("clmain")
def clmain_eval(overrides: list[str] = typer.Argument(None)):
    r"""The main entrance for evaluating trained continual learning main experiment.

    **Args:**
    - **overrides**: List of Hydra config overrides from CLI.
    """
    cfg = load_cfg(overrides)
    preprocess_config(cfg, expr_type="clmain_eval")
    expr = CLMainEval(cfg)
    expr.run()


# sub-app for command `clarena train clrefjl`
@train_app.command("clrefjl")
def clrefjl_train(overrides: list[str] = typer.Argument(None)):
    r"""The main entrance for joint learning as a reference experiment of continual learning.

    **Args:**
    - **overrides**: List of Hydra config overrides from CLI.
    """
    cfg = load_cfg(overrides)
    preprocess_config(cfg, expr_type="clrefjl_train")
    expr = MTLTrain(cfg)
    expr.run()


# sub-app for command `clarena train clrefil`
@train_app.command("clrefil")
def clrefil_train(overrides: list[str] = typer.Argument(None)):
    r"""The main entrance for joint learning as a reference experiment of continual learning.

    **Args:**
    - **overrides**: List of Hydra config overrides from CLI.
    """
    cfg = load_cfg(overrides)
    preprocess_config(cfg, expr_type="clrefil_train")
    expr = CLMainTrain(cfg)
    expr.run()


# sub-app for command `clarena train clrefrandom`
@train_app.command("clrefrandom")
def clrefrandom_train(overrides: list[str] = typer.Argument(None)):
    r"""The main entrance for random stratified as a reference experiment of continual learning.

    **Args:**
    - **overrides**: List of Hydra config overrides from CLI.
    """
    cfg = load_cfg(overrides)
    preprocess_config(cfg, expr_type="clrefrandom_train")
    expr = CLMainTrain(cfg)
    expr.run()


# sub-app for command `clarena eval cl`
@eval_app.command("cl")
def cl_eval(overrides: list[str] = typer.Argument(None)):
    r"""The main entrance for full evaluating trained trained continual learning experiment.

    **Args:**
    - **overrides**: List of Hydra config overrides from CLI.
    """

    cfg = load_cfg(overrides)
    preprocess_config(cfg, expr_type="cl_eval")
    expr = CLFullMetricsCalculation(cfg)
    expr.run()


# sub-app for command `clarena eval clattached`
@eval_app.command("clattached")
def cl_eval_attached(overrides: list[str] = typer.Argument(None)):
    r"""The main entrance for full evaluating trained trained continual learning experiment (attached to full experiment).

    **Args:**
    - **overrides**: List of Hydra config overrides from CLI.
    """

    cfg = load_cfg(overrides)
    cfg = preprocess_config(cfg, expr_type="cl_eval_attached")
    expr = CLFullMetricsCalculation(cfg)
    expr.run()


# sub-app for command `clarena full cl`
@full_app.command("cl")
def cl_full(overrides: list[str] = typer.Argument(None)):
    r"""The main entrance for full continual learning experiment.

    **Args:**
    - **overrides**: List of Hydra config overrides from CLI.
    """
    args = overrides or []
    print(args)

    unified_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    args.append(f"misc.timestamp={unified_timestamp}")  # override timestamp

    print("[CL] Running: clarena train clmain")
    subprocess.run(["clarena", "train", "clmain"] + args, check=True)

    # Check if any argument contains 'refjl_acc_csv_path='
    if any(arg.startswith("+refjl_acc_csv_path=") for arg in args):
        print("[CL] refjl_acc_csv_path argument present. Skipping clrefjl training.")
    else:
        print("[CL] Running: clarena train clrefjl")
        subprocess.run(["clarena", "train", "clrefjl"] + args, check=True)

    # Check if any argument contains 'refjl_acc_csv_path='
    if any(arg.startswith("+refil_acc_csv_path=") for arg in args):
        print("[CL] refil_acc_csv_path argument present. Skipping clrefjl training.")
    else:
        print("[CL] Running: clarena train clrefil")
        subprocess.run(["clarena", "train", "clrefil"] + args, check=True)

    if any(arg.startswith("+refrandom_acc_csv_path=") for arg in args):
        print(
            "[CL] refrandom_model_path argument present. Skipping clrefrandom training."
        )
    else:
        print("[CL] Running: clarena train clrefrandom")
        subprocess.run(["clarena", "train", "clrefrandom"] + args, check=True)

    print("[CL] Running: clarena eval clattached")
    subprocess.run(["clarena", "eval", "clattached"] + args, check=True)


# sub-app for command `clarena train culmain`
@train_app.command("culmain")
def culmain_train(overrides: list[str] = typer.Argument(None)):
    r"""The main entrance for continual unlearning main experiment.

    **Args:**
    - **overrides**: List of Hydra config overrides from CLI.
    """
    cfg = load_cfg(overrides)
    preprocess_config(cfg, expr_type="culmain_train")
    expr = CULMainTrain(cfg)
    expr.run()


# sub-app for command `clarena train culref`
@train_app.command("culrefretrain")
def culref_train(overrides: list[str] = typer.Argument(None)):
    r"""The main entrance for the retraining reference experiment of continual unlearning.

    **Args:**
    - **overrides**: List of Hydra config overrides from CLI.
    """
    cfg = load_cfg(overrides)
    preprocess_config(cfg, expr_type="culrefretrain_train")
    expr = CLMainTrain(cfg)
    expr.run()


# sub-app for command `clarena train culreffull`
@train_app.command("culreforiginal")
def culreffull_train(overrides: list[str] = typer.Argument(None)):
    r"""The main entrance for the reference original experiment of continual unlearning.

    **Args:**
    - **overrides**: List of Hydra config overrides from CLI.
    """
    cfg = load_cfg(overrides)
    preprocess_config(cfg, expr_type="culreforiginal_train")
    expr = CLMainTrain(cfg)
    expr.run()


# sub-app for command `clarena eval cul`
@eval_app.command("cul")
def cul_eval_attached(overrides: list[str] = typer.Argument(None)):
    r"""The main entrance for full evaluating trained continual unlearning experiment.

    **Args:**
    - **overrides**: List of Hydra config overrides from CLI.
    """
    cfg = load_cfg(overrides)
    preprocess_config(cfg, expr_type="cul_eval")
    expr = CULFullEval(cfg)
    expr.run()


# sub-app for command `clarena eval culattached`
@eval_app.command("culattached")
def cul_eval(overrides: list[str] = typer.Argument(None)):
    r"""The main entrance for full evaluating trained continual unlearning experiment.

    **Args:**
    - **overrides**: List of Hydra config overrides from CLI.
    """
    cfg = load_cfg(overrides)
    preprocess_config(cfg, expr_type="cul_eval_attached")
    expr = CULFullEval(cfg)
    expr.run()


# sub-app for command `clarena full cul`
@full_app.command("cul")
def cul_full(overrides: list[str] = typer.Argument(None)):
    r"""The main entrance for full continual unlearning experiment.

    **Args:**
    - **overrides**: List of Hydra config overrides from CLI.
    """
    args = overrides or []

    unified_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    args.append(f"misc.timestamp={unified_timestamp}")  # override timestamp

    print("[CUL] Running: clarena train culmain")
    subprocess.run(["clarena", "train", "culmain"] + args, check=True)

    # Check if any argument contains '+refretrain_model_path='
    if any(arg.startswith("+refretrain_model_path=") for arg in args):
        print(
            "[CUL] refretrain_model_path argument present. Skipping culrefretrain training."
        )
    else:
        print("[CUL] Running: clarena train culrefretrain")
        subprocess.run(["clarena", "train", "culrefretrain"] + args, check=True)

    # Check if any argument contains 'reforiginal_model_path='
    if any(arg.startswith("+reforiginal_model_path=") for arg in args):
        print(
            "[CUL] reforiginal_model_path argument present. Skipping culreforiginal training."
        )
    else:
        print("[CUL] Running: clarena train culreforiginal")
        subprocess.run(["clarena", "train", "culreforiginal"] + args, check=True)

    print("[CL] Running: clarena eval culattached")
    subprocess.run(["clarena", "eval", "culattached"] + args, check=True)


# sub-app for command `clarena train mtl`
@train_app.command("mtl")
def mtl_train(overrides: list[str] = typer.Argument(None)):
    r"""The main entrance for multi-task learning experiment.

    **Args:**
    - **overrides**: List of Hydra config overrides from CLI.
    """
    cfg = load_cfg(overrides)
    preprocess_config(cfg, expr_type="mtl_train")
    expr = MTLTrain(cfg)
    expr.run()


# sub-app for command `clarena eval mtl`
@eval_app.command("mtl")
def mtl_eval(overrides: list[str] = typer.Argument(None)):
    r"""The main entrance for evaluating trained multi-task learning experiment.

    **Args:**
    - **overrides**: List of Hydra config overrides from CLI.
    """
    cfg = load_cfg(overrides)
    preprocess_config(cfg, expr_type="mtl")
    expr = MTLEval(cfg)
    expr.run()


# sub-app for command `clarena train stl`
@train_app.command("stl")
def stl_train(overrides: list[str] = typer.Argument(None)):
    r"""The main entrance for single-task learning experiment.

    **Args:**
    - **overrides**: List of Hydra config overrides from CLI.
    """
    cfg = load_cfg(overrides)
    preprocess_config(cfg, expr_type="stl_train")
    expr = STLTrain(cfg)
    expr.run()


# sub-app for command `clarena eval stl`
@eval_app.command("stl")
def stl_eval(overrides: list[str] = typer.Argument(None)):
    r"""The main entrance for evaluating trained single-task learning experiment.

    **Args:**
    - **overrides**: List of Hydra config overrides from CLI.
    """
    cfg = load_cfg(overrides)
    preprocess_config(cfg, expr_type="stl_eval")
    expr = STLEval(cfg)
    expr.run()


def main():
    r"""The main entrance for running the continual learning arena."""
    app()
