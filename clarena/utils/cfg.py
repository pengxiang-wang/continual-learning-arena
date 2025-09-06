r"""The submodule in `utils` with tools related to configs."""

__all__ = [
    "preprocess_config",
    "cfg_to_tree",
    "save_tree_to_file",
]

import logging
import os
from copy import deepcopy
from typing import Any

import rich
from omegaconf import DictConfig, ListConfig, OmegaConf
from rich.syntax import Syntax
from rich.tree import Tree

# always get logger for built-in logging in each module
pylogger = logging.getLogger(__name__)


def preprocess_config(cfg: DictConfig, type: str) -> None:
    r"""Preprocess the configuration before constructing experiment, which include:

    1. Construct the config for pipelines that borrow from other config.
    2. Convert the `DictConfig` to a Rich `Tree`, print the Rich `Tree` and save the Rich `Tree` to a file.

    **Args:**
    - **cfg** (`DictConfig`): the config dict to preprocess.
    - **type** (`str`): the type of the pipeline; one of:
        1. 'CL_MAIN_EXPR': continual learning main experiment.
        2. 'CL_MAIN_EVAL': continual learning main evaluation.
        3. 'CL_REF_JOINT_EXPR': reference joint learning experiment (continual learning).
        4. 'CL_REF_INDEPENDENT_EXPR': reference independent learning experiment (continual learning).
        5. 'CL_REF_RANDOM_EXPR': reference random learning experiment (continual learning).
        6. 'CL_FULL_EVAL': continual learning full evaluation.
        7. 'CL_FULL_EVAL_ATTACHED': continual unlearning full evaluation (attached to continual learning full experiment).
        8. 'CUL_MAIN_EXPR': continual unlearning main experiment.
        9. 'CUL_MAIN_EVAL': continual unlearning main evaluation.
        10. 'CUL_REF_RETRAIN_EXPR': reference retrain learning experiment (continual unlearning).
        11. 'CUL_REF_ORIGINAL_EXPR': reference original learning experiment (contin
        12, 'CUL_FULL_EVAL': continual unlearning full evaluation.
        13. 'CUL_FULL_EVAL_ATTACHED': continual unlearning full evaluation (attached to continual unlearning full experiment).
        14. 'MTL_EXPR': multi-task learning experiment.
        15. 'MTL_EVAL': multi-task learning evaluation.
        16. 'STL_EXPR': single-task learning experiment.
        17. 'STL_EVAL': single-task learning evaluation.

    **Returns:**
    - **cfg** (`DictConfig`): the preprocessed config dict.
    """
    cfg = deepcopy(cfg)

    OmegaConf.set_struct(cfg, False)  # enable editing

    if type in [
        "CL_MAIN_EXPR",
        "CL_MAIN_EVAL",
        "CL_FULL_EVAL",
        "CUL_MAIN_EXPR",
        "CUL_MAIN_EVAL",
        "CUL_FULL_EVAL",
        "MTL_EXPR",
        "MTL_EVAL",
        "STL_EXPR",
        "STL_EVAL",
    ]:
        pass  # keep the config unchanged

    if type == "CL_REF_JOINT_EXPR":
        # construct the config for reference joint learning experiment (continual learning) from the config for continual learning main experiment

        # set the output directory under the CL main experiment output directory
        cfg.output_dir = os.path.join(cfg.output_dir, "refjoint")

        # set the CL paradigm to None, since this is a joint learning experiment
        del cfg.cl_paradigm

        # set the eval tasks to the train tasks
        cfg.eval_tasks = cfg.train_tasks

        # set the eval after tasks to None, since this is a joint learning experiment
        del cfg.eval_after_tasks

        cl_dataset_cfg = cfg.cl_dataset

        # add the mtl_dataset to the config
        cfg.mtl_dataset = {
            "_target_": "clarena.mtl_datasets.MTLDatasetFromCL",
            "cl_dataset": cl_dataset_cfg,
            "sampling_strategy": "mixed",
            "batch_size": (
                cl_dataset_cfg.batch_size
                if isinstance(cl_dataset_cfg.batch_size, int)
                else cl_dataset_cfg.batch_size[0]
            ),
        }

        # delete the cl_dataset, since this is a joint learning experiment
        del cfg.cl_dataset

        # delete the cl_algorithm, since this is a joint learning experiment
        del cfg.cl_algorithm

        # add the mtl_algorithm to the config
        cfg.mtl_algorithm = {"_target_": "clarena.mtl_algorithms.JointLearning"}

        # revise metrics
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
        for callback in cfg.callbacks:
            if callback.get("_target_") == "clarena.callbacks.CLPylogger":
                callback["_target_"] = "clarena.callbacks.MTLPylogger"

    elif type == "CL_REF_INDEPENDENT_EXPR":
        # construct the config for reference independent learning experiment (continual learning) from the config for continual learning main experiment

        # set the output directory under the CL main experiment output directory
        cfg.output_dir = os.path.join(cfg.output_dir, "refindependent")

        # change the cl_algorithm in the config
        cfg.cl_algorithm = {"_target_": "clarena.cl_algorithms.Independent"}

    elif type == "CL_REF_RANDOM_EXPR":
        # construct the config for reference random learning experiment (continual learning) from the config for continual learning main experiment

        # set the output directory under the CL main experiment output directory
        cfg.output_dir = os.path.join(cfg.output_dir, "refrandom")

        # change the cl_algorithm in the config
        cfg.cl_algorithm = {"_target_": "clarena.cl_algorithms.Random"}

    elif type == "CL_FULL_EVAL_ATTACHED":
        # construct the config for continual learning full evaluation from the config for continual learning main experiment

        eval_tasks = cfg.train_tasks

        main_acc_csv_path = os.path.join(cfg.output_dir, "results", "acc.csv")

        if cfg.get("refjoint_acc_csv_path"):
            refjoint_acc_csv_path = cfg.refjoint_acc_csv_path
        else:
            refjoint_acc_csv_path = os.path.join(
                cfg.output_dir, "refjoint", "results", "acc.csv"
            )

        if cfg.get("refindependent_acc_csv_path"):
            refindependent_acc_csv_path = cfg.refindependent_acc_csv_path
        else:
            refindependent_acc_csv_path = os.path.join(
                cfg.output_dir, "refindependent", "results", "acc.csv"
            )

        if cfg.get("refrandom_acc_csv_path"):
            refrandom_acc_csv_path = cfg.refrandom_acc_csv_path
        else:
            refrandom_acc_csv_path = os.path.join(
                cfg.output_dir, "refrandom", "results", "acc.csv"
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
        misc_cfg = cfg.misc

        cfg = OmegaConf.create(
            {
                "pipeline": "CL_FULL_EVAL",
                "eval_tasks": eval_tasks,
                "main_acc_csv_path": main_acc_csv_path,
                "refjoint_acc_csv_path": refjoint_acc_csv_path,
                "refindependent_acc_csv_path": refindependent_acc_csv_path,
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
                "misc": misc_cfg,
            }
        )

    elif type == "CUL_REF_RETRAIN_EXPR":
        # construct the config for reference retrain learning experiment (continual unlearning) from the config for continual unlearning main experiment

        # set the output directory under the main experiment output directory
        cfg.output_dir = os.path.join(cfg.output_dir, "refretrain")

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

        # delete the unlearning configs, since this is a continual learning experiment
        del cfg.cul_algorithm, cfg.unlearning_requests
        if cfg.get("permanent_mark"):
            del cfg.permanent_mark

        # revise callbacks
        for callback in cfg.callbacks:
            if callback.get("_target_") == "clarena.callbacks.CULPylogger":
                callback["_target_"] = "clarena.callbacks.CLPylogger"

    elif type == "CUL_REF_ORIGINAL_EXPR":
        # construct the config for reference original learning experiment (continual unlearning) from the config for continual unlearning main experiment

        # set the output directory under the main experiment output directory
        cfg.output_dir = os.path.join(cfg.output_dir, "reforiginal")

        # just do the CLExperiment using the unlearning experiment config will automatically ignore the unlearning process, which is exactly the full experiment

        # delete the unlearning configs, since this is a continual learning experiment
        del cfg.cul_algorithm, cfg.unlearning_requests
        if cfg.get("permanent_mark"):
            del cfg.permanent_mark

        # revise callbacks
        for callback in cfg.callbacks:
            if callback.get("_target_") == "clarena.callbacks.CULPylogger":
                callback["_target_"] = "clarena.callbacks.CLPylogger"

    elif type == "CUL_FULL_EVAL_ATTACHED":
        # construct the config for continual unlearning full evaluation from the config for continual unlearning main experiment

        dd_eval_tasks = cfg.train_tasks
        ad_eval_tasks = cfg.train_tasks
        global_seed = cfg.global_seed

        main_model_path = os.path.join(cfg.output_dir, "saved_models", "cl_model.pth")

        if cfg.get("refretrain_model_path"):
            refretrain_model_path = cfg.refretrain_model_path
        else:
            refretrain_model_path = os.path.join(
                cfg.output_dir, "refretrain", "saved_models", "cl_model.pth"
            )

        if cfg.get("reforiginal_model_path"):
            reforiginal_model_path = cfg.reforiginal_model_path
        else:
            reforiginal_model_path = os.path.join(
                cfg.output_dir, "reforiginal", "saved_models", "cl_model.pth"
            )

        cl_paradigm = cfg.cl_paradigm
        cl_dataset = cfg.cl_dataset
        trainer = cfg.trainer
        metrics = OmegaConf.create(
            [
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
            ]
        )
        callbacks = OmegaConf.create(
            [
                {
                    "_target_": "lightning.pytorch.callbacks.RichProgressBar",
                },
            ]
        )
        misc = cfg.misc
        output_dir = cfg.output_dir

        cfg = OmegaConf.create(
            {
                "pipeline": "CUL_FULL_EVAL",
                "dd_eval_tasks": dd_eval_tasks,
                "ad_eval_tasks": ad_eval_tasks,
                "global_seed": global_seed,
                "main_model_path": main_model_path,
                "refretrain_model_path": refretrain_model_path,
                "reforiginal_model_path": reforiginal_model_path,
                "cl_paradigm": cl_paradigm,
                "cl_dataset": cl_dataset,
                "trainer": trainer,
                "metrics": metrics,
                "callbacks": callbacks,
                "misc": misc,
                "output_dir": output_dir,
            }
        )

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


def select_hyperparameters_from_config(cfg: DictConfig, type: str) -> dict[str, Any]:
    r"""Select hyperparameters from the configuration based on the experiment type.

    **Args:**
    - **cfg** (`DictConfig`): the config dict to select hyperparameters from.
    - **type** (`str`): the type of the experiment; one of:
        1. 'CL_MAIN_EXPR': continual learning main experiment.
        2. 'CL_REF_JOINT_EXPR': reference joint learning experiment (continual learning).
        3. 'CL_REF_INDEPENDENT_EXPR': reference independent learning experiment (continual learning).
        4. 'CL_REF_RANDOM_EXPR': reference random learning experiment (continual learning).
        5. 'CL_FULL_EVAL_ATTACHED': continual unlearning full evaluation (attached to continual learning full experiment).
        6. 'CUL_MAIN_EXPR': continual unlearning main experiment.
        7. 'CUL_REF_RETRAIN_EXPR': reference retrain learning experiment (continual unlearning).
        8. 'CUL_REF_ORIGINAL_EXPR': reference original learning experiment (contin
        9. 'MTL_EXPR': multi-task learning experiment.
        10. 'STL_EXPR': single-task learning experiment.

    **Returns:**
    - **hyperparameters** (`dict[str, Any]`): the selected hyperparameters.
    """
    hparams = {}

    if cfg.get("cl_dataset"):
        hparams["batch_size"] = cfg.cl_dataset.batch_size
    elif cfg.get("mtl_dataset"):
        hparams["batch_size"] = cfg.mtl_dataset.batch_size
    elif cfg.get("stl_dataset"):
        hparams["batch_size"] = cfg.stl_dataset.batch_size

    # take backbone hyperparameters
    hparams["backbone"] = cfg.backbone.get("_target_")
    for k, v in cfg.backbone.items():
        if k != "_target_":
            hparams[f"backbone.{k}"] = v

    # take optimizer hyperparameters
    if isinstance(
        cfg.optimizer, ListConfig
    ):  # only apply to uniform optimizer, or it will be too messy
        hparams["optimizer"] = cfg.optimizer.get("_target_")
        for k, v in cfg.optimizer.items():
            if k != "_target_" and k != "_partial_":
                hparams[f"optimizer.{k}"] = v

    # take lr_scheduler hyperparameters
    if cfg.get("lr_scheduler"):
        if isinstance(
            cfg.lr_scheduler, ListConfig
        ):  # only apply to uniform lr_scheduler, or it will be too messy
            hparams["lr_scheduler"] = cfg.lr_scheduler.get("_target_")
            for k, v in cfg.lr_scheduler.items():
                if k != "_target_" and k != "_partial_":
                    hparams[f"lr_scheduler.{k}"] = v

    return hparams


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

    # initialize the tree
    tree = rich.tree.Tree(label="CONFIG", style=style, guide_style=guide_style)

    queue = []

    # add all fields to queue
    for field in cfg:
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
