r"""
The submodule in `pipelines` for continual learning full evaluation.
"""

__all__ = ["CLFullEvaluation"]


import logging
import os

import pandas as pd
from matplotlib import pyplot as plt
from omegaconf import DictConfig

# always get logger for built-in logging in each module
pylogger = logging.getLogger(__name__)


class CLFullEvaluation:
    r"""The base class for continual learning full evaluation."""

    def __init__(self, cfg: DictConfig) -> None:
        r"""
        **Args:**
        - **cfg** (`DictConfig`): the config dict for the continual learning full evaluation.
        """
        self.cfg: DictConfig = cfg
        r"""The complete config dict."""

        CLFullEvaluation.sanity_check(self)

        self.eval_tasks: list[int] = (
            cfg.eval_tasks
            if isinstance(cfg.eval_tasks, list)
            else list(range(1, cfg.eval_tasks + 1))
        )
        r"""The list of task IDs to evaluate."""

        self.main_acc_csv_path: str = cfg.main_acc_csv_path
        r"""The path to the main experiment accuracy CSV file."""
        if cfg.get("refjoint_acc_csv_path"):
            self.refjoint_acc_csv_path: str = cfg.refjoint_acc_csv_path
            r"""The path to the reference joint learning experiment accuracy CSV file."""
        if cfg.get("refindependent_acc_csv_path"):
            self.refindependent_acc_csv_path: str = cfg.refindependent_acc_csv_path
            r"""The path to the reference independent learning experiment accuracy CSV file."""
        if cfg.get("refrandom_acc_csv_path"):
            self.refrandom_acc_csv_path: str = cfg.refrandom_acc_csv_path
            r"""The path to the reference random learning experiment accuracy CSV file."""

        self.output_dir: str = cfg.output_dir
        r"""The folder for storing the evaluation results."""

        self.bwt_save_dir: str = cfg.bwt_save_dir
        r"""The folder storing the BWT metric results."""
        if cfg.get("fwt_save_dir"):
            self.fwt_save_dir: str | None = cfg.fwt_save_dir
            r"""The folder storing the FWT metric results."""
        if cfg.get("fr_save_dir"):
            self.fr_save_dir: str | None = cfg.fr_save_dir
            r"""The folder storing the FR metric results."""

        self.bwt_csv_path: str = os.path.join(self.bwt_save_dir, cfg.bwt_csv_name)
        r"""The file path to store the BWT metrics as CSV file."""
        if cfg.get("fwt_csv_name"):
            self.fwt_csv_path: str | None = os.path.join(
                self.fwt_save_dir, cfg.fwt_csv_name
            )
            r"""The file path to store the FWT metrics as CSV file."""
        if cfg.get("fr_csv_name"):
            self.fr_csv_path: str | None = os.path.join(
                self.fr_save_dir, cfg.fr_csv_name
            )
            r"""The file path to store the FR metrics as CSV file."""

        if cfg.get("bwt_plot_name"):
            self.bwt_plot_path: str | None = os.path.join(
                self.bwt_save_dir, cfg.bwt_plot_name
            )
            r"""The file path to store the BWT metrics as plot figure."""
        if cfg.get("fwt_plot_name"):
            self.fwt_plot_path: str | None = os.path.join(
                self.fwt_save_dir, cfg.fwt_plot_name
            )
            r"""The file path to store the FWT metrics as plot figure."""

    def sanity_check(self) -> None:
        r"""Sanity check for config."""

        # check required config fields
        required_config_fields = [
            "pipeline",
            "eval_tasks",
            "main_acc_csv_path",
            "output_dir",
            "bwt_save_dir",
            "bwt_csv_name",
        ]
        for field in required_config_fields:
            if not self.cfg.get(field):
                raise KeyError(
                    f"Field `{field}` is required in the experiment index config."
                )

        # warn if any reference experiment result is not provided
        if not self.cfg.get("refindependent_acc_csv_path"):
            pylogger.warning(
                "`refindependent_acc_csv_path` not provided. Forward Transfer (FWT) cannot be evaluated."
            )

        if not self.cfg.get("refjoint_acc_csv_path"):
            pylogger.warning(
                "`refjoint_acc_csv_path` not provided. Forgetting Rate (FR) cannot be evaluated."
            )

        if not self.cfg.get("refrandom_acc_csv_path"):
            pylogger.warning(
                "`refrandom_acc_csv_path` not provided. Forgetting Rate (FR) cannot be evaluated."
            )

    def run(self) -> None:
        r"""The main method to run the continual learning full evaluation."""

        # BWT
        self.evaluate_and_save_bwt_to_csv(
            main_acc_csv_path=self.main_acc_csv_path,
            eval_tasks=self.eval_tasks,
            save_path=self.bwt_csv_path,
        )
        if self.bwt_plot_path:
            self.plot_bwt_curve_from_csv(
                bwt_csv_path=self.bwt_csv_path, save_path=self.bwt_plot_path
            )

        # FWT
        self.evaluate_and_save_fwt_to_csv(
            main_acc_csv_path=self.main_acc_csv_path,
            refindependent_acc_csv_path=self.refindependent_acc_csv_path,
            eval_tasks=self.eval_tasks,
            save_path=self.fwt_csv_path,
        )
        if self.fwt_plot_path:
            self.plot_fwt_curve_from_csv(
                fwt_csv_path=self.fwt_csv_path, save_path=self.fwt_plot_path
            )

        # FR
        self.evaluate_and_save_fr_to_csv(
            main_acc_csv_path=self.main_acc_csv_path,
            refjoint_acc_csv_path=self.refjoint_acc_csv_path,
            refrandom_acc_csv_path=self.refrandom_acc_csv_path,
            eval_tasks=self.eval_tasks,
            save_path=self.fr_csv_path,
        )

    def evaluate_and_save_bwt_to_csv(
        self, main_acc_csv_path: str, eval_tasks: list[int], save_path: str
    ) -> None:
        """Evaluate the backward transfer (BWT) from the main experiment accuracy CSV file and save it to a CSV file.

        **Args:**
        - **main_acc_csv_path** (`str`): the path to the main experiment accuracy CSV file.
        - **eval_tasks** (`list[int]`): the list of tasks to evaluate BWT.
        - **save_path** (`str`): the path to save the BWT CSV file.
        """
        # delete task 1 which is the first task and has no BWT
        eval_tasks = eval_tasks[2:]

        acc_main_df = pd.read_csv(main_acc_csv_path)

        bwt_data: list[dict[str, float | int]] = []
        for N in eval_tasks:  # skip the first task. BWT cannot be defined for it
            bwt_N = 0.0
            for t in range(1, N):
                a_t_N = float(
                    acc_main_df.loc[
                        acc_main_df["after_training_task"] == N,
                        f"test_on_task_{t}",
                    ].iloc[0]
                )
                a_t_t = float(
                    acc_main_df.loc[
                        acc_main_df["after_training_task"] == t,
                        f"test_on_task_{t}",
                    ].iloc[0]
                )
                bwt_N += a_t_N - a_t_t
            bwt_N /= N - 1
            bwt_data.append({"after_training_task": N, "BWT": float(bwt_N)})

        bwt_df = pd.DataFrame(bwt_data)
        bwt_df.to_csv(save_path, index=False)
        pylogger.info("Saved BWT to %s.", save_path)

    def evaluate_and_save_fwt_to_csv(
        self,
        main_acc_csv_path: str,
        refindependent_acc_csv_path: str,
        eval_tasks: list[int],
        save_path: str,
    ) -> None:
        """Evaluate the forward transfer (FWT) from the main experiment accuracy CSV file and reference independenet learning experiment accuracy CSV file, and save it to a CSV file.

        **Args:**
        - **main_acc_csv_path** (`str`): the path to the main experiment accuracy CSV file.
        - **refindependent_acc_csv_path** (`str`): the path to the reference independent learning accuracy CSV file.
        - **eval_tasks** (`list[int]`): the list of tasks to evaluate FWT.
        - **save_path** (`str`): the path to save the FWT CSV file.
        """
        # delete task 1 which is the first task and has no FWT
        eval_tasks = [task for task in eval_tasks if task != 1]

        acc_main_df = pd.read_csv(main_acc_csv_path)
        acc_refil_df = pd.read_csv(refindependent_acc_csv_path)

        fwt_data: list[dict[str, float | int]] = []
        for N in eval_tasks:  # skip the first task. FWT cannot be defined for it
            fwt_N = 0.0
            for t in range(2, N + 1):
                a_t_I = float(
                    acc_refil_df.loc[
                        acc_refil_df["after_training_task"] == t,
                        f"test_on_task_{t}",
                    ].iloc[0]
                )
                a_t_t = float(
                    acc_main_df.loc[
                        acc_main_df["after_training_task"] == t,
                        f"test_on_task_{t}",
                    ].iloc[0]
                )
                fwt_N += a_t_t - a_t_I
            fwt_N /= N - 1
            fwt_data.append({"after_training_task": N, "FWT": float(fwt_N)})

        fwt_df = pd.DataFrame(fwt_data)
        fwt_df.to_csv(save_path, index=False)
        pylogger.info("Saved FWT to %s.", save_path)

    def evaluate_and_save_fr_to_csv(
        self,
        main_acc_csv_path: str,
        refjoint_acc_csv_path: str,
        refrandom_acc_csv_path: str,
        eval_tasks: list[int],
        save_path: str,
    ) -> dict[str, float]:
        """evaluate the forgetting rate (FR) from the main experiment accuracy CSV file and save it to a CSV file.

        **Args:**
        - **main_acc_csv_path** (`str`): the path to the main experiment accuracy CSV file.
        - **refjoint_acc_csv_path** (`str`): the path to the reference joint learning accuracy CSV file.
        - **refrandom_acc_csv_path** (`str`): the path to the reference random learning experiment accuracy CSV file.
        - **eval_tasks** (`list[int]`): the list of tasks to evaluate FR.
        - **save_path** (`str`): the path to save the FR CSV file.
        """

        acc_main_df = pd.read_csv(main_acc_csv_path)
        acc_refjl_df = pd.read_csv(refjoint_acc_csv_path)
        acc_refrandom_df = pd.read_csv(refrandom_acc_csv_path)

        fr_data = []
        N = eval_tasks[
            -1
        ]  # under our framework, we can only compute FR for the last task where the joint learning of all tasks is conducted for reference
        fr_N = 0.0
        for t in eval_tasks:
            a_t_N = acc_main_df.loc[
                acc_main_df["after_training_task"] == N, f"test_on_task_{t}"
            ]
            a_t_N_J = acc_refjl_df.loc[
                0, f"test_on_task_{t}"
            ]  # joint learning of all tasks
            a_t_R = acc_refrandom_df.loc[
                acc_main_df["after_training_task"] == N, f"test_on_task_{t}"
            ]
            fr_N += (a_t_N - a_t_R) / (a_t_N_J - a_t_R)
        fr_N /= N
        fr_N -= 1
        fr_data.append({"after_training_task": N, "FR": float(fr_N)})

        fr_df = pd.DataFrame([{"after_training_task": N, "FR": float(fr_N)}])
        fr_df.to_csv(save_path, index=False)
        pylogger.info("Saved FWT to %s.", save_path)

    def plot_bwt_curve_from_csv(self, bwt_csv_path: str, save_path: str) -> None:
        """Plot the backward transfer (BWT) barchart from saved CSV file and save the plot.

        **Args:**
        - **bwt_csv_path** (`str`): the path to the CSV file where the `evaluate_and_save_bwt_to_csv()` method saved the BWT metric.
        - **save_path** (`str`): the path to save plot.
        """
        data = pd.read_csv(bwt_csv_path)
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(data["after_training_task"], data["BWT"], marker="o")
        ax.set_xlabel("Task ID")
        ax.set_ylabel("Backward Transfer (BWT)")
        ax.set_title("Backward Transfer (BWT) Plot")
        fig.savefig(save_path)
        plt.close(fig)

    def plot_fwt_curve_from_csv(self, fwt_csv_path: str, save_path: str) -> None:
        """Plot the forward transfer (FWT) barchart from saved CSV file and save the plot.

        **Args:**
        - **bwt_csv_path** (`str`): the path to the CSV file where the `evaluate_and_save_fwt_to_csv()` method saved the FWT metric.
        - **save_path** (`str`): the path to save plot.
        """
        data = pd.read_csv(fwt_csv_path)
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(data["after_training_task"], data["FWT"], marker="o")
        ax.set_xlabel("Task ID")
        ax.set_ylabel("Forward Transfer (FWT)")
        ax.set_title("Forward Transfer (FWT) Plot")
        fig.savefig(save_path)
        plt.close(fig)
