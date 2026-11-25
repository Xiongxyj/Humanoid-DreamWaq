from __future__ import annotations

import torch
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.managers import SceneEntityCfg


def apply_random_body_force(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    force_range: tuple[float, float] = (-40.0, 40.0),
) -> None:
    asset = env.scene[asset_cfg.name]

    force = torch.zeros(env.num_envs, 1, 3, device=env.device)
    force[:, 0, :2] = torch.empty(env.num_envs, 2, device=env.device).uniform_(
        force_range[0], force_range[1]
    )
    torque = torch.zeros_like(force)

    asset.set_external_force_and_torque(
        forces=force,
        torques=torque,
        body_ids=[0],
        env_ids=None,
    )

    env._applied_external_force = force[:, 0, :].clone()