import gymnasium as gym

gym.register(
    id="Unitree-G1-29dof-Velocity",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg:RobotEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.velocity_env_cfg:RobotPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"humanoid_dreamwaq.locomotion.agents.rsl_rl_ppo_cfg:BasePPORunnerCfg",
    },
)

gym.register(
    id="Unitree-G1-29dof-Velocity-CE",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg:RobotEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.velocity_env_cfg:RobotPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"humanoid_dreamwaq.locomotion.agents.rsl_rl_cenet_cfg:CENetOnPolicyRunner",
    },
)

gym.register(
    id="Unitree-G1-29dof-Velocity-CE-Sym",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg:RobotSymEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.velocity_env_cfg:RobotSymPlayEnvCfg",
        "rsl_rl_cfg_entry_point": (
            "humanoid_dreamwaq.locomotion.agents.rsl_rl_cenet_sym_cfg:CENetSymmetryOnPolicyRunner"
        ),
    },
)

gym.register(
    id="Unitree-G1-29dof-Velocity-CE-Sym-Stairs",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg:RobotPureUpstairsEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.velocity_env_cfg:RobotPureUpstairsPlayEnvCfg",
        "rsl_rl_cfg_entry_point": (
            "humanoid_dreamwaq.locomotion.agents.rsl_rl_cenet_sym_cfg:CENetStairSymmetryOnPolicyRunner"
        ),
    },
)
