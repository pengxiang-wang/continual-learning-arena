r"""
The submodule in `experiments` for calculating full metrics of continual learning.
"""

__all__ = ["CLFullMetricsCalculation"]


import logging
import os

import pandas as pd
from matplotlib import pyplot as plt
from omegaconf import DictConfig

# always get logger for built-in logging in each module
pylogger = logging.getLogger(__name__)


class CLFullMetricsCalculation:
    r"""The base class for full calculating full metrics of continual learning.

    These metrics are beyond those can be evaluated and calculated after a single experiment, for example, accuracy and loss in CL main and other reference experiments. They include:

    - Backward Transfer (BWT): requires accuracy of CL main.
    - Forward Transfer (FWT): requires accuracy of CL main and reference independent learning.
    - Forgetting Rate (FR): requires accuracy of CL main, reference joint learning and reference random stratified model.
    """

    def __init__(self, cfg: DictConfig) -> None:
        r"""Initializes the `CLFullMetricsCalculationExperiment` with a configuration."""

        self.cfg: DictConfig = cfg
        r"""The complete config dict."""

        CLFullMetricsCalculation.sanity_check(self)

        self.calculate_tasks: list[int] = (
            cfg.calculate_tasks
            if isinstance(cfg.calculate_tasks, list)
            else list(range(1, cfg.calculate_tasks + 1))
        )
        r"""The list of tasks that the metrics is averaged on. Parsed from config and used in the metrics calculation loop. """

        self.main_acc_csv_path: str = cfg.main_acc_csv_path
        r"""The path of CL main accuracy CSV file to read. """
        if cfg.get("refjl_acc_csv_path"):
            self.refjl_acc_csv_path: str = cfg.refjl_acc_csv_path
            r"""The path of reference joint learning accuracy CSV file to read. """
        if cfg.get("refil_acc_csv_path"):
            self.refil_acc_csv_path: str = cfg.refil_acc_csv_path
            r"""The path of reference independent learning accuracy CSV file to read. """
        if cfg.get("refrandom_acc_csv_path"):
            self.refrandom_acc_csv_path: str = cfg.refrandom_acc_csv_path
            r"""The path of reference random stratified model accuracy CSV file to read. """

        self.output_dir: str = cfg.output_dir
        r"""The output directory to store the logs and checkpoints. Parsed from config and help any output operation to locate the correct directory."""

        self.bwt_save_dir: str = cfg.bwt_save_dir
        r"""The directory where data and figures of BWT metric will be saved. Better inside the output folder."""
        if cfg.get("fwt_save_dir"):
            self.fwt_save_dir: str | None = cfg.fwt_save_dir
            r"""The directory where data and figures of FWT metric will be saved. Better inside the output folder."""
        if cfg.get("fr_save_dir"):
            self.fr_save_dir: str | None = cfg.fr_save_dir
            r"""The directory where data and figures of FR metric will be saved. Better inside the output folder."""

        self.bwt_csv_path: str = os.path.join(self.bwt_save_dir, cfg.bwt_csv_name)
        r"""The path to save backward transfer (BWT) metrics CSV file."""
        if cfg.get("fwt_csv_name"):
            self.fwt_csv_path: str | None = os.path.join(
                self.fwt_save_dir, cfg.fwt_csv_name
            )
            r"""The path to save forward transfer (FWT) metrics CSV file."""
        if cfg.get("fr_csv_name"):
            self.fr_csv_path: str | None = os.path.join(
                self.fr_save_dir, cfg.fr_csv_name
            )
            r"""The path to save forgetting rate (FR) metrics CSV file."""

        if cfg.get("bwt_plot_name"):
            self.bwt_plot_path: str | None = os.path.join(
                self.bwt_save_dir, cfg.bwt_plot_name
            )
            r"""Store the path to save backward transfer (BWT) plot."""
        if cfg.get("fwt_plot_name"):
            self.fwt_plot_path: str | None = os.path.join(
                self.fwt_save_dir, cfg.fwt_plot_name
            )
            r"""The path to save forward transfer (FWT) plot."""
        if cfg.get("fr_plot_name"):
            self.fr_plot_path: str | None = os.path.join(
                self.fr_save_dir, cfg.fr_plot_name
            )
            r"""The path to save forgetting rate (FR) plot."""

    def sanity_check(self) -> None:
        r"""Check the sanity of the config dict `self.cfg`."""

        # check required config fields
        required_config_fields = [
            "calculate_tasks",
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
        if not self.cfg.get("refil_acc_csv_path"):
            pylogger.warning(
                "`refil_acc_csv_path` not provided. Forward Transfer (FWT) cannot be calculated."
            )

        if not self.cfg.get("refjl_acc_csv_path"):
            pylogger.warning(
                "`refjl_acc_csv_path` not provided. Forgetting Rate (FR) cannot be calculated."
            )

        if not self.cfg.get("refrandom_acc_csv_path"):
            pylogger.warning(
                "`refrandom_acc_csv_path` not provided. Forgetting Rate (FR) cannot be calculated."
            )

    def run(self) -> None:
        r"""The main method to run the continual learning full calculation."""

        # BWT
        self.calculate_and_save_bwt_to_csv(
            acc_main_csv_path=self.main_acc_csv_path,
            calculate_tasks=self.calculate_tasks,
            csv_path=self.bwt_csv_path,
        )
        if self.bwt_plot_path:
            self.plot_bwt_curve_from_csv(
                csv_path=self.bwt_csv_path, plot_path=self.bwt_plot_path
            )

        # FWT
        self.calculate_and_save_fwt_to_csv(
            self.main_acc_csv_path,
            self.refil_acc_csv_path,
            self.calculate_tasks,
            self.fwt_csv_path,
        )
        if self.fwt_plot_path:
            self.plot_fwt_curve_from_csv(
                csv_path=self.fwt_csv_path, plot_path=self.fwt_plot_path
            )

        # FR
        self.calculate_and_save_fr_to_csv(
            self.main_acc_csv_path,
            self.refjl_acc_csv_path,
            self.refrandom_acc_csv_path,
            self.calculate_tasks,
            self.fr_csv_path,
        )

    def calculate_and_save_bwt_to_csv(
        self, acc_main_csv_path: str, calculate_tasks: list[int], csv_path: str
    ) -> dict[str, float]:
        """Calculate the backward transfer (BWT) from the main accuracy CSV file and save it to a CSV file.

        **Args:**
        - **acc_main_csv_path** (`str`): the path to the main accuracy CSV file.
        - **calculate_tasks** (`list[int]`): the list of tasks to calculate the BWT.
        - **csv_path** (`str`): the path to save the BWT CSV file.
        """
        # delete task 1 which is the first task and has no BWT
        calculate_tasks = [task for task in calculate_tasks if task != 1]

        acc_main_df = pd.read_csv(acc_main_csv_path)

        bwt_data: list[dict[str, float | int]] = []
        for N in calculate_tasks:  # skip the first task. BWT cannot be defined for it
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
        bwt_df.to_csv(csv_path, index=False)
        print(f"Saved BWT to {csv_path}")

    def calculate_and_save_fwt_to_csv(
        self,
        acc_main_csv_path: str,
        acc_refil_csv_path: str,
        calculate_tasks: list[int],
        csv_path: str,
    ) -> dict[str, float]:
        """Calculate the forward transfer (FWT) from the main accuracy CSV file and save it to a CSV file.

        **Args:**
        - **acc_main_csv_path** (`str`): the path to the main accuracy CSV file.
        - **acc_refil_csv_path** (`str`): the path to the reference independent learning accuracy CSV file.
        - **calculate_tasks** (`list[int]`): the list of tasks to calculate the FWT.
        - **csv_path** (`str`): the path to save the FWT CSV file.
        """
        # delete task 1 which is the first task and has no FWT
        calculate_tasks = [task for task in calculate_tasks if task != 1]

        acc_main_df = pd.read_csv(acc_main_csv_path)
        acc_refil_df = pd.read_csv(acc_refil_csv_path)

        fwt_data: list[dict[str, float | int]] = []
        for N in calculate_tasks:  # skip the first task. FWT cannot be defined for it
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
        fwt_df.to_csv(csv_path, index=False)
        print(f"Saved FWT to {csv_path}")

    def calculate_and_save_fr_to_csv(
        self,
        acc_main_csv_path: str,
        acc_refjl_csv_path: str,
        acc_refrandom_csv_path: str,
        calculate_tasks: list[int],
        csv_path: str,
    ) -> dict[str, float]:
        """Calculate the forgetting rate (FR) from the main accuracy CSV file and save it to a CSV file.

        **Args:**
        - **acc_main_csv_path** (`str`): the path to the main accuracy CSV file.
        - **acc_refjl_csv_path** (`str`): the path to the reference joint learning accuracy CSV file.
        - **acc_refrandom_csv_path** (`str`): the path to the reference random stratified model accuracy CSV file.
        - **calculate_tasks** (`list[int]`): the list of tasks to calculate the FR.
        - **csv_path** (`str`): the path to save the FR CSV file.
        """
        # delete task 1 which is the first task and has no FR
        calculate_tasks = [task for task in calculate_tasks if task != 1]

        acc_main_df = pd.read_csv(acc_main_csv_path)
        acc_refjl_df = pd.read_csv(acc_refjl_csv_path)
        acc_refrandom_df = pd.read_csv(acc_refrandom_csv_path)

        fr_data = []
        N = calculate_tasks[
            -1
        ]  # under our framework, we can only compute FR for the last task where the joint learning of all tasks is conducted for reference
        fr_N = 0.0
        for t in calculate_tasks:
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
        fr_df.to_csv(csv_path, index=False)
        print(f"Saved FR to {csv_path}")

    def plot_bwt_curve_from_csv(self, csv_path: str, plot_path: str) -> None:
        """Plot the backward transfer (BWT) curve from saved CSV file and save the plot to the designated directory.

        **Args:**
        - **csv_path** (`str`): the path to the CSV file where the `utils.save_bwt_to_csv()` saved the BWT metric.
        - **plot_path** (`str`): the path to save plot. Better same as the output directory of the experiment. E.g. './outputs/expr_name/1970-01-01_00-00-00/results/bwt.png'.
        """
        data = pd.read_csv(csv_path)
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(data["after_training_task"], data["BWT"], marker="o")
        ax.set_xlabel("Task ID")
        ax.set_ylabel("Backward Transfer (BWT)")
        ax.set_title("Backward Transfer (BWT) Curve")
        fig.savefig(plot_path)
        plt.close(fig)

    def plot_fwt_curve_from_csv(self, csv_path: str, plot_path: str) -> None:
        """Plot the forward transfer (FWT) curve from saved CSV file and save the plot to the designated directory.

        **Args:**
        - **csv_path** (`str`): the path to the CSV file where the `utils.save_fwt_to_csv()` saved the FWT metric.
        - **plot_path** (`str`): the path to save plot. Better same as the output directory of the experiment. E.g. './outputs/expr_name/1970-01-01_00-00-00/results/fwt.png'.
        """
        data = pd.read_csv(csv_path)
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(data["after_training_task"], data["FWT"], marker="o")
        ax.set_xlabel("Task ID")
        ax.set_ylabel("Forward Transfer (FWT)")
        ax.set_title("Forward Transfer (FWT) Curve")
        fig.savefig(plot_path)
        plt.close(fig)
