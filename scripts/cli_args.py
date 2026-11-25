# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg


DEFAULT_TASK = "Unitree-G1-29dof-Velocity-CE"


def add_terrain_arg(parser: argparse.ArgumentParser):
    parser.add_argument("--terrain", help="Terrain preset: walk, stairs, stairs-medium, complex, or stable.")


def add_rsl_rl_args(parser: argparse.ArgumentParser):
    """Add shared RSL-RL options for training and playback."""

    arg_group = parser.add_argument_group("rsl_rl", description="Arguments for RSL-RL agent.")

    arg_group.add_argument(
        "--experiment_name", type=str, default=None, help="Name of the experiment folder where logs will be stored."
    )
    arg_group.add_argument("--run_name", type=str, default=None, help="Run name suffix to the log directory.")

    arg_group.add_argument("--resume", action="store_true", default=None, help="Whether to resume from a checkpoint.")
    arg_group.add_argument(
        "--load_experiment_name",
        type=str,
        default=None,
        help="Name of the experiment folder to load a checkpoint from. Defaults to the current experiment.",
    )
    arg_group.add_argument("--load_run", type=str, default=None, help="Name of the run folder to resume from.")
    arg_group.add_argument("--checkpoint", type=str, default=None, help="Local path, Nucleus URL, or checkpoint name in a matching run.")

    arg_group.add_argument(
        "--logger", type=str, default=None, choices={"wandb", "tensorboard", "neptune"}, help="Logger module to use."
    )
    arg_group.add_argument(
        "--log_project_name", type=str, default=None, help="Name of the logging project when using wandb or neptune."
    )


def parse_rsl_rl_cfg(task_name: str, args_cli: argparse.Namespace) -> RslRlOnPolicyRunnerCfg:
    """Load the registered runner configuration and apply CLI overrides."""
    from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

    rslrl_cfg: RslRlOnPolicyRunnerCfg = load_cfg_from_registry(task_name, "rsl_rl_cfg_entry_point")
    return update_rsl_rl_cfg(rslrl_cfg, args_cli)


def update_rsl_rl_cfg(agent_cfg: RslRlOnPolicyRunnerCfg, args_cli: argparse.Namespace):
    """Apply explicitly supplied CLI values to the runner configuration."""

    if hasattr(args_cli, "seed") and args_cli.seed is not None:

        if args_cli.seed == -1:
            args_cli.seed = random.randint(0, 10000)
        agent_cfg.seed = args_cli.seed
    if args_cli.resume is not None:
        agent_cfg.resume = args_cli.resume
    if args_cli.load_run is not None:
        agent_cfg.load_run = args_cli.load_run
    if args_cli.checkpoint is not None:
        agent_cfg.load_checkpoint = args_cli.checkpoint
    if args_cli.run_name is not None:
        agent_cfg.run_name = args_cli.run_name
    if args_cli.logger is not None:
        agent_cfg.logger = args_cli.logger

    if agent_cfg.logger in {"wandb", "neptune"} and args_cli.log_project_name:
        agent_cfg.wandb_project = args_cli.log_project_name
        agent_cfg.neptune_project = args_cli.log_project_name

    if args_cli.experiment_name is not None:
        agent_cfg.experiment_name = args_cli.experiment_name
    if agent_cfg.experiment_name == "":
        task_name = args_cli.task
        agent_cfg.experiment_name = task_name.lower().replace("-", "_").removesuffix("_play")

    return agent_cfg


def resolve_checkpoint(agent_cfg: RslRlOnPolicyRunnerCfg, args_cli: argparse.Namespace) -> str:
    """Resolve an explicit checkpoint or Isaac Lab's run-directory search."""
    from isaaclab.utils.assets import retrieve_file_path
    from isaaclab_tasks.utils import get_checkpoint_path

    if args_cli.checkpoint:
        if "://" in args_cli.checkpoint:
            return retrieve_file_path(args_cli.checkpoint)
        checkpoint = Path(args_cli.checkpoint).expanduser()
        if checkpoint.is_file() or "/" in args_cli.checkpoint:
            return retrieve_file_path(str(checkpoint))

    experiment = args_cli.load_experiment_name or agent_cfg.experiment_name
    log_root = Path("logs", "rsl_rl", experiment).resolve()
    return get_checkpoint_path(str(log_root), agent_cfg.load_run, agent_cfg.load_checkpoint)
