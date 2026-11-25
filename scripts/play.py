# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Replay a Unitree G1 checkpoint in Isaac Sim."""

import argparse
import time
from importlib.metadata import version
from pathlib import Path

from isaaclab.app import AppLauncher

import cli_args

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", default=cli_args.DEFAULT_TASK, help="Gymnasium task ID.")
cli_args.add_terrain_arg(parser)
parser.add_argument("--num_envs", type=int, help="Number of simulated robots.")
parser.add_argument("--disable_fabric", action="store_true", help="Use USD I/O instead of Fabric.")
parser.add_argument("--real-time", action="store_true", help="Run in real time when possible.")
parser.add_argument("--max-steps", type=int, help="Exit after this many policy steps.")
parser.add_argument("--video", action="store_true", help="Record a video of the rollout.")
parser.add_argument("--video_length", type=int, default=200, help="Number of video steps.")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

if args_cli.video:
    args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch
from rsl_rl.runners import OnPolicyRunner

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab_rl.rsl_rl.utils import handle_deprecated_rsl_rl_cfg

from humanoid_dreamwaq import locomotion  # noqa: F401
from humanoid_dreamwaq.locomotion.velocity_env_cfg import set_terrain_preset
from humanoid_dreamwaq.utils.parser_cfg import parse_env_cfg


def main():
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
        entry_point_key="play_env_cfg_entry_point",
    )
    if args_cli.terrain:
        set_terrain_preset(env_cfg, args_cli.terrain, for_play=True)
    agent_cfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, version("rsl-rl-lib"))
    checkpoint = cli_args.resolve_checkpoint(agent_cfg, args_cli)
    print(f"[INFO] Loading model checkpoint from: {checkpoint}")

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    if args_cli.video:
        env = gym.wrappers.RecordVideo(
            env,
            video_folder=str(Path(checkpoint).parent / "videos" / "play"),
            step_trigger=lambda step: step == 0,
            video_length=args_cli.video_length,
            disable_logger=True,
        )
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    try:
        device = env.unwrapped.device
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=device)
        runner.load(checkpoint, map_location=device)
        policy = runner.get_inference_policy(device=device)
        obs = env.get_observations()
        max_steps = args_cli.max_steps
        if max_steps is None and args_cli.video:
            max_steps = args_cli.video_length

        steps = 0
        dt = env.unwrapped.step_dt
        while simulation_app.is_running() and (max_steps is None or steps < max_steps):
            start = time.monotonic()
            with torch.inference_mode():
                obs, _, _, _ = env.step(policy(obs))
            steps += 1
            if args_cli.real_time:
                time.sleep(max(0.0, dt - (time.monotonic() - start)))
        print(f"[INFO] Completed {steps} policy steps.")
    finally:
        env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
