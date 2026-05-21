# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
from importlib.metadata import version

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import time
import torch

from rsl_rl.runners import OnPolicyRunner

import isaaclab_tasks  # noqa: F401
from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
#from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper, export_policy_as_jit, export_policy_as_onnx
from isaaclab_tasks.utils import get_checkpoint_path

import unitree_rl_lab.tasks  # noqa: F401
from unitree_rl_lab.utils.parser_cfg import parse_env_cfg
import traceback


def main():
    """Play with RSL-RL agent."""
    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
        entry_point_key="play_env_cfg_entry_point",
    )
    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)
    import rsl_rl
    from isaaclab_rl.rsl_rl.utils import handle_deprecated_rsl_rl_cfg
    import importlib.metadata
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, importlib.metadata.version("rsl-rl-lib"))
    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    '''if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("rsl_rl", args_cli.task)
        if not resume_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return'''
    if args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    log_dir = os.path.dirname(resume_path)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    # load previously trained model
    if not hasattr(agent_cfg, "class_name") or agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        from rsl_rl.runners import DistillationRunner

        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    runner.load(resume_path)

    # obtain the trained policy for inference
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    # extract the neural network module
    # we do this in a try-except to maintain backwards compatibility.
    try:
    # rsl_rl >= 4.0.0 新版 API
        policy_nn = runner.alg.actor
    except AttributeError:
        try:
        # version 2.3
            policy_nn = runner.alg.policy
        except AttributeError:
        # version 2.2 and below
            policy_nn = runner.alg.actor_critic

    # extract the normalizer
    if hasattr(policy_nn, "actor_obs_normalizer"):
        normalizer = policy_nn.actor_obs_normalizer
    elif hasattr(policy_nn, "student_obs_normalizer"):
        normalizer = policy_nn.student_obs_normalizer
    else:
        normalizer = None

    # export policy to onnx/jit
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    try:
        export_policy_as_jit(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.pt")
        export_policy_as_onnx(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.onnx")
    except Exception as e:
        print(f"[WARNING] Export skipped: {e}")

    dt = env.unwrapped.step_dt

    # reset environment
    obs = env.get_observations()
    if version("rsl-rl-lib").startswith("2.3."):
        obs, _ = env.get_observations()
    timestep = 0
    # simulate environment
    while simulation_app.is_running():
        start_time = time.time()
        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            actions = policy(obs)
            # env stepping
            obs, _, _, _ = env.step(actions)
        if args_cli.video:
            timestep += 1
            # Exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break

        vel = env.unwrapped.scene["robot"].data.root_lin_vel_b
        actual_speed = torch.norm(vel[:, :2], dim=1)
        cmd = env.unwrapped.command_manager.get_command("base_velocity")
        cmd_speed = torch.norm(cmd[:, :2], dim=1)
        tracking_error = torch.abs(actual_speed - cmd_speed)

        materials = env.unwrapped.scene["robot"].root_physx_view.get_material_properties()
        ankle_ids, _ = env.unwrapped.scene["robot"].find_bodies(".*ankle_roll.*")
        static_friction = materials[:, ankle_ids, 0].mean(dim=1)

        terrain = env.unwrapped.scene.terrain
        levels = terrain.terrain_levels.float()

        num_envs = env.unwrapped.num_envs
        sensor = env.unwrapped.scene.sensors["height_scanner"]
        # 打印前几条射线的xy坐标，确认排列顺序
        print(sensor.data.ray_hits_w[0, :5, :2])   # 前5条
        print(sensor.data.ray_hits_w[0, 8:13, :2]) # 中间附近5条

        # 摩擦分区统计
        ice_envs    = (static_friction < 0.4).sum().item()
        normal_envs = ((static_friction >= 0.4) & (static_friction <= 0.8)).sum().item()
        rough_envs  = (static_friction > 0.8).sum().item()

        print("=" * 60)
        print(f"  速度跟踪")
        print(f"    命令速度:  {cmd_speed.mean():.3f} m/s")
        print(f"    实际速度:  {actual_speed.mean():.3f} m/s")
        print(f"    跟踪误差:  {tracking_error.mean():.3f} m/s  "
            f"(最大 {tracking_error.max():.3f})")
        print(f"cmd shape: {cmd.shape}, cmd[0]: {cmd[0]}")  # 看第一个env的原始值

        print(f"  地形难度")
        print(f"    平均等级:  {levels.mean():.2f} / {terrain.cfg.terrain_generator.num_rows - 1}")
        print(f"    最高等级:  {levels.max().item():.0f}  "
            f"最低等级: {levels.min().item():.0f}")

        print(f"  地面摩擦 (脚踝)")
        print(f"    范围:      [{static_friction.min():.3f}, {static_friction.max():.3f}]")
        print(f"    冰面(<0.4): {ice_envs:3d} envs | "
            f"正常: {normal_envs:3d} envs | "
            f"粗糙(>0.8): {rough_envs:3d} envs")

        print(f"  单环境详情 (前5个)")
        for i in range(min(5, num_envs)):
            mu  = static_friction[i].item()
            lvl = int(levels[i].item())
            spd = actual_speed[i].item()
            err = tracking_error[i].item()
            surface = "冰面" if mu < 0.4 else ("粗糙" if mu > 0.8 else "正常")
            print(f"    env{i}: 摩擦={mu:.3f}({surface})  "
                f"难度={lvl}  速度={spd:.3f}m/s  误差={err:.3f}")
        print("=" * 60)

        # time delay for real-time evaluation
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
