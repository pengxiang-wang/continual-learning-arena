"""The submodule in `utils` with tools related to configs."""

__all__ = ["preprocess_config", "cfg_to_tree", "save_tree_to_file"]

import logging

import rich
from omegaconf import DictConfig, OmegaConf
from rich.syntax import Syntax
from rich.tree import Tree

# always get logger for built-in logging in each module
pylogger = logging.getLogger(__name__)


def preprocess_config(cfg: DictConfig) -> None:
    r"""Preprocess the configuration before constructing experiment, which may include:

    1. Convert the `DictConfig` to a Rich `Tree`.
    2. Print the Rich `Tree`.
    3. Save the Rich `Tree` to a file.

    **Args:**
    - **cfg** (`DictConfig`): the config dict to preprocess.
    """

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

    # initialise the tree
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
    with open(save_path, "w") as file:
        rich.print(tree, file=file)
