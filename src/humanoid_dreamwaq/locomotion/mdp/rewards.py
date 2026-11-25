from __future__ import annotations

import torch
from typing import TYPE_CHECKING

try:
    from isaaclab.utils.math import quat_apply_inverse
except ImportError:
    from isaaclab.utils.math import quat_rotate_inverse as quat_apply_inverse
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import quat_apply
from humanoid_dreamwaq.locomotion import mdp


if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def energy(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize the energy used by the robot's joints."""
    asset: Articulation = env.scene[asset_cfg.name]

    qvel = asset.data.joint_vel[:, asset_cfg.joint_ids]
    qfrc = asset.data.applied_torque[:, asset_cfg.joint_ids]
    return torch.sum(torch.abs(qvel) * torch.abs(qfrc), dim=-1)


def stand_still(
    env: ManagerBasedRLEnv, command_name: str = "base_velocity", asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Penalize joint displacement from the default pose under near-zero commands."""
    asset: Articulation = env.scene[asset_cfg.name]

    reward = torch.sum(torch.abs(asset.data.joint_pos - asset.data.default_joint_pos), dim=1)
    cmd_norm = torch.norm(env.command_manager.get_command(command_name), dim=1)
    return reward * (cmd_norm < 0.1)


def orientation_l2(
    env: ManagerBasedRLEnv, desired_gravity: list[float], asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Compute a squared penalty from normalized gravity alignment."""

    asset: RigidObject = env.scene[asset_cfg.name]

    desired_gravity = torch.tensor(desired_gravity, device=env.device)
    cos_dist = torch.sum(asset.data.projected_gravity_b * desired_gravity, dim=-1)
    normalized = 0.5 * cos_dist + 0.5
    return torch.square(normalized)


def upward(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize deviation of the vertical projected gravity component from one."""

    asset: RigidObject = env.scene[asset_cfg.name]
    reward = torch.square(1 - asset.data.projected_gravity_b[:, 2])
    return reward


def joint_position_penalty(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, stand_still_scale: float, velocity_threshold: float
) -> torch.Tensor:
    """Penalize joint position error from default on the articulation."""

    asset: Articulation = env.scene[asset_cfg.name]
    cmd = torch.linalg.norm(env.command_manager.get_command("base_velocity"), dim=1)
    body_vel = torch.linalg.norm(asset.data.root_lin_vel_b[:, :2], dim=1)
    reward = torch.linalg.norm((asset.data.joint_pos - asset.data.default_joint_pos), dim=1)
    return torch.where(torch.logical_or(cmd > 0.0, body_vel > velocity_threshold), reward, stand_still_scale * reward)

def base_height_above_terrain(
    env: ManagerBasedRLEnv,
    target_height: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    feet_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=".*ankle_roll.*"),
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces", body_names=".*ankle_roll.*"),
) -> torch.Tensor:
    """Penalize base height error relative to the mean height of contacting feet."""
    from isaaclab.sensors import ContactSensor
    asset: RigidObject = env.scene[asset_cfg.name]
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]

    feet_z = asset.data.body_pos_w[:, feet_cfg.body_ids, 2]

    is_contact = sensor.data.current_contact_time[:, sensor_cfg.body_ids] > 0
    contact_float = is_contact.float()
    contact_count_raw = contact_float.sum(dim=1, keepdim=True)
    contact_count = contact_count_raw.clamp(min=1)

    terrain_z = (feet_z * contact_float).sum(dim=1) / contact_count.squeeze(1)

    base_z = asset.data.root_pos_w[:, 2]
    relative_height = base_z - terrain_z
    penalty = torch.square(relative_height - target_height)
    penalty = penalty * (contact_count_raw.squeeze(1) > 0).float()
    return torch.nan_to_num(penalty, nan=0.0, posinf=0.0, neginf=0.0)


def feet_stumble(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:

    """Detect foot impacts with large lateral force relative to vertical force."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces_z = torch.abs(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2])
    forces_xy = torch.linalg.norm(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :2], dim=2)

    reward = torch.any(forces_xy > 4 * forces_z, dim=1).float()
    return reward


def feet_contact_orientation_l2(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg,
    command_name: str = "base_velocity",
) -> torch.Tensor:
    """Penalize contacted feet whose link frame is strongly tilted from horizontal."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    asset: RigidObject = env.scene[asset_cfg.name]

    is_contact = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids] > 0
    foot_quat_w = asset.data.body_quat_w[:, asset_cfg.body_ids]
    gravity_w = asset.data.GRAVITY_VEC_W.unsqueeze(1).expand(-1, len(asset_cfg.body_ids), -1)
    foot_gravity_b = quat_apply_inverse(foot_quat_w, gravity_w)
    foot_tilt = torch.sum(torch.square(foot_gravity_b[..., :2]), dim=-1)

    cmd_norm = torch.linalg.norm(env.command_manager.get_command(command_name), dim=1)
    reward = torch.sum(foot_tilt * is_contact.float(), dim=1) * (cmd_norm > 0.1).float()
    return torch.nan_to_num(reward, nan=0.0, posinf=0.0, neginf=0.0)


def foot_target_height_from_stance(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    sensor_cfg: SceneEntityCfg,
    contact_sensor_cfg: SceneEntityCfg,
    target_clearance: float = 0.12,
    stair_height_thresh: float = 0.04,
    command_name: str = "base_velocity"
) -> torch.Tensor:
    """Penalize swing-foot clearance error relative to the nearest raised terrain ahead of the stance foot."""
    asset: RigidObject = env.scene[asset_cfg.name]
    scanner = env.scene.sensors[sensor_cfg.name]
    contact_sensor: ContactSensor = env.scene.sensors[contact_sensor_cfg.name]

    is_contact = contact_sensor.data.current_contact_time[:, contact_sensor_cfg.body_ids] > 0
    is_swing = ~is_contact

    foot_pos_w = asset.data.body_pos_w[:, asset_cfg.body_ids, :]
    foot_z_world = foot_pos_w[..., 2]

    # Use the opposite foot as the stance reference for each swing foot.
    stance_idx = torch.tensor([1, 0], device=env.device)
    stance_foot_pos = foot_pos_w[:, stance_idx, :]

    ray_hits_w = scanner.data.ray_hits_w
    ray_hits_w = torch.nan_to_num(ray_hits_w, nan=0.0, posinf=10.0, neginf=-1.0)

    rel_pos_w = ray_hits_w.unsqueeze(1) - stance_foot_pos.unsqueeze(2)

    base_quat = asset.data.root_quat_w
    N = env.num_envs
    forward_vec_b = torch.tensor([1.0, 0.0, 0.0], device=env.device).view(1, 3).expand(N, 3)
    forward_vec_w = quat_apply(base_quat, forward_vec_b)

    forward_dist = torch.sum(rel_pos_w * forward_vec_w.view(N, 1, 1, 3), dim=-1)

    is_forward = forward_dist > 0.05

    rel_z = rel_pos_w[..., 2]
    is_stair = rel_z > stair_height_thresh

    valid_mask = is_forward & is_stair

    dist_xy = torch.norm(rel_pos_w[..., :2], dim=-1)

    dist_xy = torch.where(valid_mask, dist_xy, torch.full_like(dist_xy, float('inf')))

    min_idx = torch.argmin(dist_xy, dim=-1)

    ray_hits_z = ray_hits_w[..., 2]

    nearest_stair_z = torch.gather(ray_hits_z, 1, min_idx)

    # Fall back to stance height when no raised surface is detected ahead.
    no_stair_mask = torch.isinf(torch.gather(dist_xy, 2, min_idx.unsqueeze(-1)).squeeze(-1))
    target_terrain_z = torch.where(no_stair_mask, stance_foot_pos[..., 2], nearest_stair_z)

    target_z = target_terrain_z + target_clearance
    error = torch.abs(foot_z_world - target_z)
    normalized_error = torch.clamp(error / max(target_clearance, 1.0e-6), max=2.0)

    foot_vel_xy = torch.norm(
        asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2], dim=-1
    )
    vel_gate = torch.tanh(2.0 * foot_vel_xy)

    cmd_norm = torch.linalg.norm(env.command_manager.get_command(command_name), dim=1)
    moving = cmd_norm > 0.1

    active_swing = is_swing & moving.unsqueeze(1)
    active_count = active_swing.float().sum().clamp(min=1.0)
    if not hasattr(env, "_g1_gait_metrics"):
        env._g1_gait_metrics = {}
    env._g1_gait_metrics.update(
        {
            "foot_target_active_swing_frac": active_swing.float().mean().detach(),
            "foot_target_no_stair_frac": (no_stair_mask & active_swing).float().sum().detach() / active_count,
            "foot_target_left_no_stair_frac": (
                (no_stair_mask[:, 0] & active_swing[:, 0]).float().sum().detach()
                / active_swing[:, 0].float().sum().clamp(min=1.0)
            ),
            "foot_target_right_no_stair_frac": (
                (no_stair_mask[:, 1] & active_swing[:, 1]).float().sum().detach()
                / active_swing[:, 1].float().sum().clamp(min=1.0)
            ),
            "foot_target_height_mean": (target_terrain_z[active_swing].mean().detach() if active_swing.any() else torch.zeros((), device=env.device)),
            "foot_target_error_mean": (error[active_swing].mean().detach() if active_swing.any() else torch.zeros((), device=env.device)),
            "foot_target_error_norm_mean": (
                normalized_error[active_swing].mean().detach()
                if active_swing.any()
                else torch.zeros((), device=env.device)
            ),
        }
    )

    reward = torch.sum(normalized_error * is_swing.float() * vel_gate, dim=1)
    result = -reward * moving.float()

    return torch.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)

def foot_on_flat_terrain(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    scanner_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg,
    radius: float = 0.10,
    variance_scale: float = 100.0
) -> torch.Tensor:
    """Reward contact on locally flat ground while moving with feet at different elevations."""
    scanner = env.scene.sensors[scanner_cfg.name]
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    asset = env.scene[asset_cfg.name]

    is_contact = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids] > 0

    foot_pos_w = asset.data.body_pos_w[:, asset_cfg.body_ids, :]

    ray_hits_w = scanner.data.ray_hits_w
    ray_hits_w = torch.nan_to_num(ray_hits_w, nan=0.0, posinf=10.0, neginf=-1.0)

    foot_xy = foot_pos_w[..., :2].unsqueeze(2)
    ray_xy = ray_hits_w[..., :2].unsqueeze(1)
    dist_xy = torch.norm(foot_xy - ray_xy, dim=-1)

    in_radius_mask = dist_xy < radius

    ray_z = ray_hits_w[..., 2].unsqueeze(1)

    valid_counts = in_radius_mask.sum(dim=-1).float().clamp(min=1.0)

    masked_z = torch.where(in_radius_mask, ray_z, torch.zeros_like(ray_z))
    mean_z = masked_z.sum(dim=-1) / valid_counts

    sq_diff = torch.where(in_radius_mask, torch.square(ray_z - mean_z.unsqueeze(-1)), torch.zeros_like(ray_z))
    local_var = sq_diff.sum(dim=-1) / valid_counts

    # Treat missing nearby rays as a poor foothold.
    no_rays_mask = in_radius_mask.sum(dim=-1) == 0
    local_var = torch.where(no_rays_mask, torch.full_like(local_var, 0.1), local_var)

    feet_z = asset.data.body_pos_w[:, asset_cfg.body_ids, 2]
    z_diff = torch.abs(feet_z[:, 0] - feet_z[:, 1]).unsqueeze(1)

    actual_vel = torch.linalg.norm(asset.data.root_lin_vel_b[:, :2], dim=1)

    foot_reward = torch.exp(-variance_scale * local_var) * is_contact.float() * (z_diff > 0.04) * (actual_vel > 0.1).unsqueeze(1)

    return torch.nan_to_num(foot_reward.sum(dim=-1), nan=0.0, posinf=0.0, neginf=0.0)


def foot_clearance_reward(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, target_height: float, std: float, tanh_mult: float
) -> torch.Tensor:
    """Reward the swinging feet for clearing a specified height off the ground"""
    asset: RigidObject = env.scene[asset_cfg.name]
    foot_z_target_error = torch.square(asset.data.body_pos_w[:, asset_cfg.body_ids, 2] - target_height)
    foot_velocity_tanh = torch.tanh(tanh_mult * torch.norm(asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2], dim=2))
    reward = foot_z_target_error * foot_velocity_tanh
    return torch.exp(-torch.sum(reward, dim=1) / std)


def feet_too_near(
    env: ManagerBasedRLEnv, threshold: float = 0.2, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    feet_pos = asset.data.body_pos_w[:, asset_cfg.body_ids, :]
    distance = torch.norm(feet_pos[:, 0] - feet_pos[:, 1], dim=-1)
    return (threshold - distance).clamp(min=0)


def feet_contact_sagittal_separation(
    env: ManagerBasedRLEnv,
    target_gap: float,
    asset_cfg: SceneEntityCfg,
    sensor_cfg: SceneEntityCfg,
    command_name: str = "base_velocity",
) -> torch.Tensor:
    """Penalize moving double-support poses where both feet are side-by-side in base x."""
    asset: RigidObject = env.scene[asset_cfg.name]
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]

    foot_pos_w = asset.data.body_pos_w[:, asset_cfg.body_ids, :]
    foot_rel_w = foot_pos_w - asset.data.root_pos_w[:, None, :]
    root_quat_w = asset.data.root_quat_w[:, None, :].expand(-1, len(asset_cfg.body_ids), -1)
    foot_pos_b = quat_apply_inverse(root_quat_w.reshape(-1, 4), foot_rel_w.reshape(-1, 3)).reshape(
        env.num_envs, len(asset_cfg.body_ids), 3
    )

    sagittal_gap = torch.abs(foot_pos_b[:, 0, 0] - foot_pos_b[:, 1, 0])
    is_contact = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids] > 0
    both_contact = torch.all(is_contact, dim=1)
    command = env.command_manager.get_command(command_name)
    moving_forward = command[:, 0] > 0.1

    penalty = (target_gap - sagittal_gap).clamp(min=0.0)
    penalty = penalty * both_contact.float() * moving_forward.float()
    return torch.nan_to_num(penalty, nan=0.0, posinf=0.0, neginf=0.0)


def feet_touchdown_sagittal_gap(
    env: ManagerBasedRLEnv,
    target_gap: float,
    asset_cfg: SceneEntityCfg,
    sensor_cfg: SceneEntityCfg,
    command_name: str = "base_velocity",
    min_air_time: float = 0.06,
    dynamic_target_gap: bool = True,
    step_interval: float = 0.4,
    target_gap_scale: float = 1.0,
    min_target_gap: float = 0.06,
) -> torch.Tensor:
    """Penalize forward walking swing/touchdown feet that do not overtake the other foot in base x."""
    asset: RigidObject = env.scene[asset_cfg.name]
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]

    foot_pos_w = asset.data.body_pos_w[:, asset_cfg.body_ids, :]
    foot_rel_w = foot_pos_w - asset.data.root_pos_w[:, None, :]
    root_quat_w = asset.data.root_quat_w[:, None, :].expand(-1, len(asset_cfg.body_ids), -1)
    foot_pos_b = quat_apply_inverse(root_quat_w.reshape(-1, 4), foot_rel_w.reshape(-1, 3)).reshape(
        env.num_envs, len(asset_cfg.body_ids), 3
    )

    foot_x = foot_pos_b[..., 0]
    current_contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]
    current_air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]
    touchdown_window = env.step_dt + 1.0e-6
    is_touchdown = (current_contact_time > 0.0) & (current_contact_time <= touchdown_window)
    is_late_swing = current_air_time > min_air_time

    command = env.command_manager.get_command(command_name)
    moving_forward = command[:, 0] > 0.1
    if dynamic_target_gap:
        target_gap_env = torch.clamp(command[:, 0] * step_interval * target_gap_scale, min=min_target_gap, max=target_gap)
    else:
        target_gap_env = torch.full((env.num_envs,), target_gap, dtype=torch.float, device=env.device)
    target_gap_env = target_gap_env.clamp(min=1.0e-6)

    # The configured body IDs are ordered left, then right.
    left_overtake_gap = foot_x[:, 0] - foot_x[:, 1]
    right_overtake_gap = foot_x[:, 1] - foot_x[:, 0]
    left_active = (is_late_swing[:, 0] | is_touchdown[:, 0]) & moving_forward
    right_active = (is_late_swing[:, 1] | is_touchdown[:, 1]) & moving_forward
    left_penalty = torch.clamp((target_gap_env - left_overtake_gap) / target_gap_env, min=0.0, max=2.0)
    right_penalty = torch.clamp((target_gap_env - right_overtake_gap) / target_gap_env, min=0.0, max=2.0)
    left_penalty = left_penalty * left_active.float()
    right_penalty = right_penalty * right_active.float()
    penalty = (left_penalty + right_penalty) * moving_forward.float()

    left_touchdown = is_touchdown[:, 0] & moving_forward
    right_touchdown = is_touchdown[:, 1] & moving_forward
    left_count = left_touchdown.float().sum().clamp(min=1.0)
    right_count = right_touchdown.float().sum().clamp(min=1.0)
    touchdown_count = (left_touchdown | right_touchdown).float().sum().clamp(min=1.0)
    if not hasattr(env, "_g1_gait_metrics"):
        env._g1_gait_metrics = {}
    env._g1_gait_metrics.update(
        {
            "touchdown_left_frac": left_touchdown.float().sum().detach() / touchdown_count,
            "touchdown_right_frac": right_touchdown.float().sum().detach() / touchdown_count,
            "touchdown_left_overtake_frac": (
                (left_touchdown & (left_overtake_gap > target_gap_env)).float().sum().detach() / left_count
            ),
            "touchdown_right_overtake_frac": (
                (right_touchdown & (right_overtake_gap > target_gap_env)).float().sum().detach() / right_count
            ),
            "touchdown_left_gap_mean": (
                left_overtake_gap[left_touchdown].mean().detach()
                if left_touchdown.any()
                else torch.zeros((), device=env.device)
            ),
            "touchdown_right_gap_mean": (
                right_overtake_gap[right_touchdown].mean().detach()
                if right_touchdown.any()
                else torch.zeros((), device=env.device)
            ),
            "overtake_left_active_frac": left_active.float().mean().detach(),
            "overtake_right_active_frac": right_active.float().mean().detach(),
            "overtake_left_penalty_mean": (
                left_penalty[left_active].mean().detach()
                if left_active.any()
                else torch.zeros((), device=env.device)
            ),
            "overtake_right_penalty_mean": (
                right_penalty[right_active].mean().detach()
                if right_active.any()
                else torch.zeros((), device=env.device)
            ),
            "overtake_target_gap_mean": target_gap_env[moving_forward].mean().detach()
            if moving_forward.any()
            else torch.zeros((), device=env.device),
            "lead_left_frac": ((foot_x[:, 0] > foot_x[:, 1]) & moving_forward).float().mean().detach(),
            "lead_right_frac": ((foot_x[:, 1] > foot_x[:, 0]) & moving_forward).float().mean().detach(),
        }
    )
    return torch.nan_to_num(penalty, nan=0.0, posinf=0.0, neginf=0.0)


def feet_contact_without_cmd(
    env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, command_name: str = "base_velocity"
) -> torch.Tensor:
    """Reward foot contacts under near-zero commands."""

    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    is_contact = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids] > 0

    command_norm = torch.norm(env.command_manager.get_command(command_name), dim=1)
    reward = torch.sum(is_contact, dim=-1).float()
    return reward * (command_norm < 0.1)


def air_time_variance_penalty(
    env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, max_time: float = 0.5
) -> torch.Tensor:
    """Penalize left-right gaps in air and contact duration.

    The legacy name is retained for configuration compatibility.
    """

    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    if contact_sensor.cfg.track_air_time is False:
        raise RuntimeError("Activate ContactSensor's track_air_time!")

    # Include ongoing intervals so the penalty can grow before contact switches.
    air_time = torch.maximum(
        contact_sensor.data.current_air_time[:, sensor_cfg.body_ids],
        contact_sensor.data.last_air_time[:, sensor_cfg.body_ids],
    )
    contact_time = torch.maximum(
        contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids],
        contact_sensor.data.last_contact_time[:, sensor_cfg.body_ids],
    )
    air_time = torch.nan_to_num(torch.clip(air_time, max=max_time), nan=0.0, posinf=max_time, neginf=0.0)
    contact_time = torch.nan_to_num(
        torch.clip(contact_time, max=max_time), nan=0.0, posinf=max_time, neginf=0.0
    )
    air_gap = torch.max(air_time, dim=1).values - torch.min(air_time, dim=1).values
    contact_gap = torch.max(contact_time, dim=1).values - torch.min(contact_time, dim=1).values
    return air_gap + contact_gap


def feet_gait(
    env: ManagerBasedRLEnv,
    period: float,
    offset: list[float],
    sensor_cfg: SceneEntityCfg,
    threshold: float = 0.5,
    command_name=None,
) -> torch.Tensor:
    """Reward foot contacts that match the configured gait phase."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    is_contact = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids] > 0

    global_phase = ((env.episode_length_buf * env.step_dt) % period / period).unsqueeze(1)
    phases = []
    for offset_ in offset:
        phase = (global_phase + offset_) % 1.0
        phases.append(phase)
    leg_phase = torch.cat(phases, dim=-1)

    reward = torch.zeros(env.num_envs, dtype=torch.float, device=env.device)
    for i in range(len(sensor_cfg.body_ids)):
        is_stance = leg_phase[:, i] < threshold
        reward += ~(is_stance ^ is_contact[:, i])

    if command_name is not None:
        cmd_norm = torch.norm(env.command_manager.get_command(command_name), dim=1)
        reward *= cmd_norm > 0.1
    return reward


def joint_mirror(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, mirror_joints: list[list[str]]) -> torch.Tensor:

    """Penalize differences between configured pairs of joint positions."""
    asset: Articulation = env.scene[asset_cfg.name]
    if not hasattr(env, "joint_mirror_joints_cache") or env.joint_mirror_joints_cache is None:

        env.joint_mirror_joints_cache = [
            [asset.find_joints(joint_name) for joint_name in joint_pair] for joint_pair in mirror_joints
        ]
    reward = torch.zeros(env.num_envs, device=env.device)

    for joint_pair in env.joint_mirror_joints_cache:

        reward += torch.sum(
            torch.square(asset.data.joint_pos[:, joint_pair[0][0]] - asset.data.joint_pos[:, joint_pair[1][0]]),
            dim=-1,
        )
    reward *= 1 / len(mirror_joints) if len(mirror_joints) > 0 else 0
    return reward


def height_scan_safe(env, sensor_cfg, offset=0.5):
    """Replace nonfinite ray-cast observations with finite values."""
    raw = mdp.height_scan(env, sensor_cfg, offset)
    return torch.nan_to_num(raw, nan=0.0, posinf=1.5, neginf=-1.0)
