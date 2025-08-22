r"""The submodule in `utils` with tools related to configs."""

__all__ = [
    "preprocess_config",
    "cfg_to_tree",
    "save_tree_to_file",
    "construct_unlearning_ref_config",
]

import logging
import os
from copy import deepcopy

import rich
from omegaconf import DictConfig, ListConfig, OmegaConf
from rich.syntax import Syntax
from rich.tree import Tree

# always get logger for built-in logging in each module
pylogger = logging.getLogger(__name__)


def preprocess_config(cfg: DictConfig, expr_type: str) -> None:
    r"""Preprocess the configuration before constructing experiment, which include:

    1. Convert the `DictConfig` to a Rich `Tree`.
    2. Print the Rich `Tree`.
    3. Save the Rich `Tree` to a file.

    **Args:**
    - **cfg** (`DictConfig`): the config dict to preprocess.
    - **expr_type** (`str`): the type of the experiment, should be one of the following:
        1. 'clmain_train': continual learning main experiment.
        2. 'clmain_eval': evaluating trained continual learning main experiment.
        3. 'clrefjl_train': joint learning as a reference experiment of continual learning.
        4. 'clrefil_train': independent learning as a reference experiment of continual learning.
        5. 'clrefrandom_train': random stratified model as a reference experiment of continual learning.
        6. 'cl_eval': full evaluating trained continual learning experiment.
        7. 'culmain_train': continual unlearning main experiment.
        8. 'culref_train': the reference experiment of continual unlearning.
        9. 'culreffull_train': the reference full experiment of continual unlearning.
        10. 'cul_eval': full evaluating trained continual unlearning experiment.
        12. 'mtl_train': multi-task learning experiment.
        13. 'mtl_eval': evaluating trained multi-task learning experiment.
        14. 'stl_train': single-task learning experiment.
        15. 'stl_eval': evaluating trained single-task learning experiment.

    **Returns:**
    - **cfg** (`DictConfig`): the preprocessed config dict.
    """
    OmegaConf.set_struct(cfg, False)  # enable editing

    if expr_type in [
        "clmain_train",
        "clmain_eval",
        "cl_eval",
        "culmain_train",
        "cul_eval",
        "mtl_train",
        "mtl_eval",
        "stl_train",
        "stl_eval",
    ]:
        pass

    if expr_type == "clrefjl_train":

        # set the output directory under the CL main experiment output directory
        cfg.output_dir = os.path.join(cfg.output_dir, "clrefjl")

        # set the CL paradigm to None, since this is a joint learning experiment
        del cfg.cl_paradigm

        # set the eval tasks to the train tasks
        cfg.eval_tasks = cfg.train_tasks

        # set the eval after tasks to None, since this is a joint learning experiment
        del cfg.eval_after_tasks

        # delete the cl_algorithm, since this is a joint learning experiment
        del cfg.cl_algorithm

        # add the mtl_algorithm to the config
        cfg.mtl_algorithm = {"_target_": "clarena.mtl_algorithms.JointLearning"}

        # revise metrics
        # Remove CLAccuracy and CLLoss metrics entirely, and add MTL metrics
        # Use a new list to avoid modifying the list while iterating
        new_metrics = []
        for metric in cfg.metrics:
            target = metric.get("_target_")
            if target == "clarena.metrics.CLAccuracy":
                new_metrics.append(
                    {
                        "_target_": "clarena.metrics.MTLAccuracy",
                        "save_dir": "${output_dir}/results/",
                        "test_acc_csv_name": "acc.csv",
                        "test_acc_plot_name": "acc.png",
                    }
                )
            elif target == "clarena.metrics.CLLoss":
                new_metrics.append(
                    {
                        "_target_": "clarena.metrics.MTLLoss",
                        "save_dir": "${output_dir}/results/",
                        "test_loss_cls_csv_name": "loss_cls.csv",
                        "test_loss_cls_plot_name": "loss_cls.png",
                    }
                )
            else:
                new_metrics.append(metric)
        cfg.metrics = new_metrics

        # revise callbacks
        for cb in cfg.callbacks:
            if cb.get("_target_") == "clarena.callbacks.CLPylogger":
                cb["_target_"] = "clarena.callbacks.MTLPylogger"

    elif expr_type == "clrefil_train":

        # set the output directory under the CL main experiment output directory
        cfg.output_dir = os.path.join(cfg.output_dir, "clrefil")

        # add the cl_algorithm to the config
        cfg.cl_algorithm = {"_target_": "clarena.cl_algorithms.Independent"}

    elif expr_type == "clrefrandom_train":

        # set the output directory under the CL main experiment output directory
        cfg.output_dir = os.path.join(cfg.output_dir, "clrefrandom")

        # add the cl_algorithm to the config
        cfg.cl_algorithm = {"_target_": "clarena.cl_algorithms.Random"}

    elif expr_type == "cl_eval_attached":

        calculate_tasks = cfg.train_tasks

        main_acc_csv_path = os.path.join(cfg.output_dir, "results", "acc.csv")

        if cfg.get("refjl_acc_csv_path"):
            refjl_acc_csv_path = cfg.refjl_acc_csv_path
        else:
            refjl_acc_csv_path = os.path.join(
                cfg.output_dir, "clrefjl", "results", "acc.csv"
            )

        if cfg.get("refil_acc_csv_path"):
            refil_acc_csv_path = cfg.refil_acc_csv_path
        else:
            refil_acc_csv_path = os.path.join(
                cfg.output_dir, "clrefil", "results", "acc.csv"
            )

        if cfg.get("refrandom_acc_csv_path"):
            refrandom_acc_csv_path = cfg.refrandom_acc_csv_path
        else:
            refrandom_acc_csv_path = os.path.join(
                cfg.output_dir, "clrefrandom", "results", "acc.csv"
            )

        output_dir = cfg.output_dir
        bwt_save_dir = os.path.join(output_dir, "results")
        bwt_csv_name = "bwt.csv"
        bwt_plot_name = "bwt.png"
        fwt_save_dir = os.path.join(output_dir, "results")
        fwt_csv_name = "fwt.csv"
        fwt_plot_name = "fwt.png"
        fr_save_dir = os.path.join(output_dir, "results")
        fr_csv_name = "fr.csv"
        fr_plot_name = "fr.png"
        misc_cfg = cfg.misc

        cfg = OmegaConf.create(
            {
                "calculate_tasks": calculate_tasks,
                "main_acc_csv_path": main_acc_csv_path,
                "refjl_acc_csv_path": refjl_acc_csv_path,
                "refil_acc_csv_path": refil_acc_csv_path,
                "refrandom_acc_csv_path": refrandom_acc_csv_path,
                "output_dir": output_dir,
                "bwt_save_dir": bwt_save_dir,
                "bwt_csv_name": bwt_csv_name,
                "bwt_plot_name": bwt_plot_name,
                "fwt_save_dir": fwt_save_dir,
                "fwt_csv_name": fwt_csv_name,
                "fwt_plot_name": fwt_plot_name,
                "fr_save_dir": fr_save_dir,
                "fr_csv_name": fr_csv_name,
                "fr_plot_name": fr_plot_name,
                "misc": misc_cfg,
            }
        )

    elif expr_type == "culrefretrain_train":

        # set the output directory under the main experiment output directory
        cfg.output_dir = os.path.join(cfg.output_dir, "culrefretrain")

        # skip the unlearning tasks specified in unlearning_requests

        train_tasks = (
            cfg.train_tasks
            if isinstance(cfg.train_tasks, ListConfig)
            else ListConfig(list(range(1, cfg.train_tasks + 1)))
        )
        for unlearning_task_ids in cfg.unlearning_requests.values():
            for unlearning_task_id in unlearning_task_ids:
                if unlearning_task_id in train_tasks:
                    train_tasks.remove(unlearning_task_id)

        cfg.train_tasks = train_tasks

        # delete the unlearning_algorithm, since this is a joint learning experiment
        del cfg.unlearning_algorithm

        # revise callbacks
        for cb in cfg.callbacks:
            if cb.get("_target_") == "clarena.callbacks.CULPylogger":
                cb["_target_"] = "clarena.callbacks.CLPylogger"

    elif expr_type == "culreforiginal_train":

        # set the output directory under the main experiment output directory
        cfg.output_dir = os.path.join(cfg.output_dir, "culreforiginal")

        # just do the CLExperiment using the unlearning experiment config will automatically ignore the unlearning process, which is exactly the full experiment

        # delete the unlearning_algorithm, since this is a joint learning experiment
        del cfg.unlearning_algorithm

        # revise callbacks
        for cb in cfg.callbacks:
            if cb.get("_target_") == "clarena.callbacks.CULPylogger":
                cb["_target_"] = "clarena.callbacks.CLPylogger"

    elif expr_type == "cul_eval_attached":

        cfg.dd_eval_tasks = cfg.train_tasks
        cfg.ad_eval_tasks = cfg.train_tasks

        cfg.main_model_path = os.path.join(
            cfg.output_dir, "saved_models", "cl_model.pth"
        )

        if not cfg.get("refretrain_model_path"):
            cfg.refretrain_model_path = os.path.join(
                cfg.output_dir, "culrefretrain", "saved_models", "cl_model.pth"
            )

        if not cfg.get("reforiginal_model_path"):
            cfg.reforiginal_model_path = os.path.join(
                cfg.output_dir, "culreforiginal", "saved_models", "cl_model.pth"
            )

        cfg.metrics = OmegaConf.create([
            {
            "_target_": "clarena.metrics.CULDistributionDistance",
            "save_dir": "${output_dir}/results/",
            "distribution_distance_type": "cosine",
            "distribution_distance_csv_name": "dd.csv",
            "distribution_distance_plot_name": "dd.png",
            },
            {
            "_target_": "clarena.metrics.CULAccuracyDifference",
            "save_dir": "${output_dir}/results/",
            "accuracy_difference_csv_name": "ad.csv",
            "accuracy_difference_plot_name": "ad.png",
            },
        ])

    else:
        pass  # only process the reference experiment config, not the others

    OmegaConf.set_struct(cfg, True)

    if cfg.get("misc"):
        if cfg.misc.get("config_tree"):
            # parse config used for config tree
            config_tree_cfg = cfg.misc.config_tree
            if_print = (
                config_tree_cfg.print
            )  # to avoid using `print` as a variable name, which is supposed to be a built-in function
            save = config_tree_cfg.save
            save_path = config_tree_cfg.save_path

            # convert config to tree
            tree = cfg_to_tree(cfg, config_tree_cfg)

            if if_print:
                rich.print(tree)  # print the tree
            if save:
                save_tree_to_file(tree, save_path)  # save the tree to file

    return cfg


def cfg_to_tree(cfg: DictConfig, config_tree_cfg: DictConfig) -> Tree:
    r"""Convert the configuration to a Rich `Tree`.

    **Args:**
    - **cfg** (`DictConfig`): the target config dict to be converted.
    - **config_tree_cfg** (`DictConfig`): the configuration for conversion of config tree.

    **Returns:**
    - **tree** (`Tree`): the Rich `Tree`.
    """
    # configs for tree
    style = config_tree_cfg.style
    guide_style = config_tree_cfg.guide_style
    fields_order = config_tree_cfg.fields_order

    # initialize the tree
    tree = rich.tree.Tree(label="CONFIG", style=style, guide_style=guide_style)

    queue = []

    # add fields from `fields_order` to queue
    for field in fields_order:
        if field in cfg:
            queue.append(field)
        else:
            pylogger.warning(
                "Field %s not found in config. Skipping %s config printing...",
                field,
                field,
            )

    # add all the other fields to queue (not specified in `fields_order`)
    for field in cfg:
        if field not in queue:
            queue.append(field)

    # generate config tree from queue
    for field in queue:
        branch = tree.add(field, style=style, guide_style=guide_style)
        field_cfg = cfg[field]
        branch_content = (
            OmegaConf.to_yaml(field_cfg, resolve=True)
            if isinstance(field_cfg, DictConfig)
            else str(field_cfg)
        )
        branch.add(Syntax(branch_content, "yaml"))

    return tree


def save_tree_to_file(tree: dict, save_path: str) -> None:
    """Save Rich `Tree` to a file.

    **Args:**
    - **tree** (`dict`): the Rich `Tree` to save.
    - **save_path** (`str`): the path to save the tree.
    """
    if not os.path.exists(save_path):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

    with open(save_path, "w") as file:
        rich.print(tree, file=file)


def construct_unlearning_ref_config(
    cul_cfg: DictConfig,
) -> DictConfig:
    r"""Construct the config for reference experiment to evaluate the unlearning performance, for the continual unlearning experiment whose config is given.

    **Args:**
    - **cul_cfg** (`DictConfig`): the config dict of the continual unlearning experiment to be evaluated.

    **Returns:**
    - **ulref_cfg** (`DictConfig`): the  constructed unlearning reference config.
    """

    ulref_cfg = deepcopy(cul_cfg)

    ulref_cfg.output_dir = os.path.join(
        ulref_cfg.output_dir, "unlearning_ref"
    )  # set the output directory for unlearning reference experiment

    OmegaConf.set_struct(ulref_cfg, False)
    ulref_cfg.skip_unlearning_tasks = (
        True  # skip the unlearning tasks specified in unlearning_requests
    )
    OmegaConf.set_struct(ulref_cfg, True)

    return ulref_cfg


def construct_cl_full_metrics_calculation_cfg(cfg: DictConfig) -> DictConfig:
    r"""Construct the config for CL full metrics calculation experiment from the continual learning experiment.

    **Args:**
    - **cfg** (`DictConfig`): the config dict of the continual learning experiment to calculate full metrics for.

    **Returns:**
    - **full_metrics_calculation_cfg** (`DictConfig`): the constructed CL full metrics calculation config.
    """

    full_metrics_calculation_cfg = DictConfig({})

    full_metrics_calculation_cfg.main_acc_csv_path = cfg.callbacks.acc.csv_path
    full_metrics_calculation_cfg.refjl_acc_csv_path = (
        cfg.output_dir,
        "clrefjl",
        "acc.csv",
    )
    full_metrics_calculation_cfg.refil_acc_csv_path = (
        cfg.output_dir,
        "clrefil",
        "acc.csv",
    )
    full_metrics_calculation_cfg.refrandom_acc_csv_path = (
        cfg.output_dir,
        "clrefrandom",
        "acc.csv",
    )

    return full_metrics_calculation_cfg
