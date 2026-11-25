from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def root_height_below_feet(
    env: ManagerBasedRLEnv,
    minimum_height: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    feet_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=".*ankle_roll.*"),
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces", body_names=".*ankle_roll.*"),
) -> torch.Tensor:
    """Terminate when root height is too low relative to contacted feet."""
    asset: RigidObject = env.scene[asset_cfg.name]
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]

    feet_z = asset.data.body_pos_w[:, feet_cfg.body_ids, 2]
    is_contact = sensor.data.current_contact_time[:, sensor_cfg.body_ids] > 0
    contact_float = is_contact.float()
    contact_count = contact_float.sum(dim=1)

    terrain_z = (feet_z * contact_float).sum(dim=1) / contact_count.clamp(min=1.0)
    relative_height = asset.data.root_pos_w[:, 2] - terrain_z
    terminated = relative_height < minimum_height
    terminated = terminated & (contact_count > 0)
    return terminated
