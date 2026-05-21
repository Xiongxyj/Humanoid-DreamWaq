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
from unitree_rl_lab.tasks.locomotion import mdp


if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

"""
Joint penalties.
"""


def energy(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize the energy used by the robot's joints."""
    asset: Articulation = env.scene[asset_cfg.name]

    qvel = asset.data.joint_vel[:, asset_cfg.joint_ids]
    qfrc = asset.data.applied_torque[:, asset_cfg.joint_ids]
    return torch.sum(torch.abs(qvel) * torch.abs(qfrc), dim=-1)


def stand_still(
    env: ManagerBasedRLEnv, command_name: str = "base_velocity", asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]

    reward = torch.sum(torch.abs(asset.data.joint_pos - asset.data.default_joint_pos), dim=1)
    cmd_norm = torch.norm(env.command_manager.get_command(command_name), dim=1)
    return reward * (cmd_norm < 0.1)


"""
Robot.
"""


def orientation_l2(
    env: ManagerBasedRLEnv, desired_gravity: list[float], asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward the agent for aligning its gravity with the desired gravity vector using L2 squared kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]

    desired_gravity = torch.tensor(desired_gravity, device=env.device)
    cos_dist = torch.sum(asset.data.projected_gravity_b * desired_gravity, dim=-1)  # cosine distance
    normalized = 0.5 * cos_dist + 0.5  # map from [-1, 1] to [0, 1]
    return torch.square(normalized)


def upward(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize z-axis base linear velocity using L2 squared kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    reward = torch.square(1 - asset.data.projected_gravity_b[:, 2])
    return reward


def joint_position_penalty(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, stand_still_scale: float, velocity_threshold: float
) -> torch.Tensor:
    """Penalize joint position error from default on the articulation."""
    # extract the used quantities (to enable type-hinting)
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
    from isaaclab.sensors import ContactSensor
    asset: RigidObject = env.scene[asset_cfg.name]
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]

    feet_z = asset.data.body_pos_w[:, feet_cfg.body_ids, 2]  # [N, 2]

    # 判断哪只脚在接触地面
    is_contact = sensor.data.current_contact_time[:, sensor_cfg.body_ids] > 0  # [N, 2]
    contact_float = is_contact.float()
    contact_count = contact_float.sum(dim=1, keepdim=True).clamp(min=1)

    # 只对接触地面的脚求均值
    terrain_z = (feet_z * contact_float).sum(dim=1) / contact_count.squeeze(1)

    base_z = asset.data.root_pos_w[:, 2]
    relative_height = base_z - terrain_z
    return torch.square(relative_height - target_height)

"""
Feet rewards.
"""


def feet_stumble(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces_z = torch.abs(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2])
    forces_xy = torch.linalg.norm(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :2], dim=2)
    # Penalize feet hitting vertical surfaces
    reward = torch.any(forces_xy > 4 * forces_z, dim=1).float()
    return reward


def foot_target_height_from_stance(
    env: ManagerBasedRLEnv, 
    asset_cfg: SceneEntityCfg, 
    sensor_cfg: SceneEntityCfg, 
    contact_sensor_cfg: SceneEntityCfg,
    target_clearance: float = 0.12, 
    stair_height_thresh: float = 0.04,
    command_name: str = "base_velocity"
) -> torch.Tensor:
    """
    基于支撑脚前方的第一个地形高度突变点，引导摆动脚的抬脚高度。
    纯向量化实现，无 for 循环。
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    scanner = env.scene.sensors[sensor_cfg.name]
    contact_sensor: ContactSensor = env.scene.sensors[contact_sensor_cfg.name]

    # [N, 2] 获取脚部状态
    is_contact = contact_sensor.data.current_contact_time[:, contact_sensor_cfg.body_ids] > 0
    is_swing = ~is_contact
    
    # 获取脚的世界坐标 [N, 2, 3]
    foot_pos_w = asset.data.body_pos_w[:, asset_cfg.body_ids, :]
    foot_z_world = foot_pos_w[..., 2]

    # 1. 交叉获取支撑脚的位置 (左脚摆动对应右脚支撑，反之亦然)
    # 假设 body_ids 中 0 为左脚，1 为右脚
    stance_idx = torch.tensor([1, 0], device=env.device)
    stance_foot_pos = foot_pos_w[:, stance_idx, :] # [N, 2, 3]

    # 2. 计算所有射线命中点相对于支撑脚的向量
    ray_hits_w = scanner.data.ray_hits_w # [N, 187, 3]
    ray_hits_w = torch.nan_to_num(ray_hits_w, nan=0.0, posinf=10.0, neginf=-1.0)
    # 扩展维度进行广播计算: ray[N, 1, 187, 3] - stance[N, 2, 1, 3] -> [N, 2, 187, 3]
    rel_pos_w = ray_hits_w.unsqueeze(1) - stance_foot_pos.unsqueeze(2)

    # 3. 计算机器人的全局前向向量
    base_quat = asset.data.root_quat_w # [N, 4]
    N = env.num_envs
    forward_vec_b = torch.tensor([1.0, 0.0, 0.0], device=env.device).view(1, 3).expand(N, 3)
    forward_vec_w = quat_apply(base_quat, forward_vec_b) # [N, 3]

    # 4. 投影相对位置到前向向量上，计算在前进方向上的距离
    # forward_dist: [N, 2, 187]
    forward_dist = torch.sum(rel_pos_w * forward_vec_w.view(N, 1, 1, 3), dim=-1)

    # 5. 构建掩码 (Masks)
    # 条件A：必须在支撑脚前方 (例如 > 0.05m)
    is_forward = forward_dist > 0.05 
    
    # 条件B：必须是台阶/波浪突变 (与支撑脚高度差的绝对值 > 阈值)
    # 这里用绝对值是为了同时兼容“上楼梯”和“下楼梯”
    rel_z = rel_pos_w[..., 2] # [N, 2, 187]
    is_stair = torch.abs(rel_z) > stair_height_thresh
    
    valid_mask = is_forward & is_stair # [N, 2, 187]

    # 6. 计算有效射线的水平距离 (或者直接用 forward_dist)，并屏蔽无效射线
    dist_xy = torch.norm(rel_pos_w[..., :2], dim=-1) # [N, 2, 187]
    # 将无效射线的距离设为无穷大
    dist_xy = torch.where(valid_mask, dist_xy, torch.full_like(dist_xy, float('inf')))

    # 7. 提取最近的有效台阶索引
    min_idx = torch.argmin(dist_xy, dim=-1) # [N, 2]

    # 8. 获取目标台阶的绝对高度
    ray_hits_z = ray_hits_w[..., 2] # [N, 187]
    # 利用 gather 根据 min_idx 提取高度，结果形状为 [N, 2]
    nearest_stair_z = torch.gather(ray_hits_z, 1, min_idx)

    # 9. Fallback 机制：如果前方全是平地（全被 masked 成 inf）
    # 此时 nearest_stair_z 会拿到索引 0 的值，这是不对的。我们将其高度设为支撑脚的高度。
    no_stair_mask = torch.isinf(torch.gather(dist_xy, 2, min_idx.unsqueeze(-1)).squeeze(-1)) # [N, 2]
    target_terrain_z = torch.where(no_stair_mask, stance_foot_pos[..., 2], nearest_stair_z)

    # 10. 计算最终奖励
    # 目标：摆动脚应高出即将踩踏的地形 target_clearance
    target_z = target_terrain_z + target_clearance
    error = torch.abs(foot_z_world - target_z)

    foot_vel_xy = torch.norm(
        asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2], dim=-1
    )
    vel_gate = torch.tanh(2.0 * foot_vel_xy)

    # 只有在摆动相，且有速度指令时才计算该奖励
    reward = torch.sum(error * is_swing.float() * vel_gate, dim=1)
    cmd_norm = torch.linalg.norm(env.command_manager.get_command(command_name), dim=1)

    result = -reward * (cmd_norm > 0.1).float()
    # 返回负值作为惩罚，或根据你的奖励函数配置调整符号
    return torch.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)

def foot_on_flat_terrain(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,   # contact_forces
    scanner_cfg: SceneEntityCfg,  # height_scanner
    asset_cfg: SceneEntityCfg,
    radius: float = 0.10,         # 局部考察半径 (8cm，能覆盖脚掌大小及周边)
    variance_scale: float = 100.0 # 指数缩放因子，方差极小，需要适当放大权重
) -> torch.Tensor:
    """
    精确奖励：支撑脚踩在局部平坦的地形上，惩罚踩在台阶边缘或不平整处。
    """
    scanner = env.scene.sensors[scanner_cfg.name]
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    asset = env.scene[asset_cfg.name]

    # [N, 2] 判断双脚是否接触地面
    is_contact = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids] > 0

    # [N, 2, 3] 获取双脚世界坐标
    foot_pos_w = asset.data.body_pos_w[:, asset_cfg.body_ids, :]
    
    # [N, 187, 3] 获取射线世界坐标
    ray_hits_w = scanner.data.ray_hits_w
    ray_hits_w = torch.nan_to_num(ray_hits_w, nan=0.0, posinf=10.0, neginf=-1.0)

    # 1. 计算射线到每只脚的 XY 距离
    foot_xy = foot_pos_w[..., :2].unsqueeze(2)    # [N, 2, 1, 2]
    ray_xy = ray_hits_w[..., :2].unsqueeze(1)     # [N, 1, 187, 2]
    dist_xy = torch.norm(foot_xy - ray_xy, dim=-1) # [N, 2, 187]

    # 2. 构建局部掩码：只取脚附近半径 radius 内的射线
    in_radius_mask = dist_xy < radius # [N, 2, 187]

    # 3. 计算局部区域的高程方差 (Masked Variance)
    ray_z = ray_hits_w[..., 2].unsqueeze(1) # [N, 1, 187]
    
    # 统计每只脚附近有多少条有效射线
    valid_counts = in_radius_mask.sum(dim=-1).float().clamp(min=1.0) # [N, 2]

    # 掩码均值
    masked_z = torch.where(in_radius_mask, ray_z, torch.zeros_like(ray_z))
    mean_z = masked_z.sum(dim=-1) / valid_counts # [N, 2]

    # 掩码方差
    sq_diff = torch.where(in_radius_mask, torch.square(ray_z - mean_z.unsqueeze(-1)), torch.zeros_like(ray_z))
    local_var = sq_diff.sum(dim=-1) / valid_counts # [N, 2]

    # 4. 极端情况处理：如果由于机器人跳太高、或者射线盲区，脚下没有射线，给一个默认的惩罚方差
    no_rays_mask = in_radius_mask.sum(dim=-1) == 0 # [N, 2]
    local_var = torch.where(no_rays_mask, torch.full_like(local_var, 0.1), local_var)

    #5.计算两脚高度差距，只有在走台阶时才奖励
    feet_z = asset.data.body_pos_w[:, asset_cfg.body_ids, 2]
    z_diff = torch.abs(feet_z[:, 0] - feet_z[:, 1]).unsqueeze(1)

    actual_vel = torch.linalg.norm(
    asset.data.root_lin_vel_b[:, :2], dim=1  # 实际 XY 线速度
)

    # 6. 计算奖励：方差越小，奖励越接近 1；仅对接触地面的脚生效
    # 注意：高度的方差通常是个很小的数字（比如 mm 级别平方），所以 variance_scale 需要给得大一点
    foot_reward = torch.exp(-variance_scale * local_var) * is_contact.float() * (z_diff>0.04)* (actual_vel > 0.1).unsqueeze(1)

    # [N] 将双脚的奖励相加
    return foot_reward.sum(dim=-1)


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


def feet_contact_without_cmd(
    env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, command_name: str = "base_velocity"
) -> torch.Tensor:
    """
    Reward for feet contact when the command is zero.
    """
    # asset: Articulation = env.scene[asset_cfg.name]
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    is_contact = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids] > 0

    command_norm = torch.norm(env.command_manager.get_command(command_name), dim=1)
    reward = torch.sum(is_contact, dim=-1).float()
    return reward * (command_norm < 0.1)


def air_time_variance_penalty(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize variance in the amount of time each foot spends in the air/on the ground relative to each other"""
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    if contact_sensor.cfg.track_air_time is False:
        raise RuntimeError("Activate ContactSensor's track_air_time!")
    # compute the reward
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    last_contact_time = contact_sensor.data.last_contact_time[:, sensor_cfg.body_ids]
    return torch.var(torch.clip(last_air_time, max=0.5), dim=1) + torch.var(
        torch.clip(last_contact_time, max=0.5), dim=1
    )


"""
Feet Gait rewards.
"""


def feet_gait(
    env: ManagerBasedRLEnv,
    period: float,
    offset: list[float],
    sensor_cfg: SceneEntityCfg,
    threshold: float = 0.5,
    command_name=None,
) -> torch.Tensor:
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


"""
Other rewards.
"""


def joint_mirror(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, mirror_joints: list[list[str]]) -> torch.Tensor:
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    if not hasattr(env, "joint_mirror_joints_cache") or env.joint_mirror_joints_cache is None:
        # Cache joint positions for all pairs
        env.joint_mirror_joints_cache = [
            [asset.find_joints(joint_name) for joint_name in joint_pair] for joint_pair in mirror_joints
        ]
    reward = torch.zeros(env.num_envs, device=env.device)
    # Iterate over all joint pairs
    for joint_pair in env.joint_mirror_joints_cache:
        # Calculate the difference for each pair and add to the total reward
        reward += torch.sum(
            torch.square(asset.data.joint_pos[:, joint_pair[0][0]] - asset.data.joint_pos[:, joint_pair[1][0]]),
            dim=-1,
        )
    reward *= 1 / len(mirror_joints) if len(mirror_joints) > 0 else 0
    return reward

# 在 mdp 或 rewards.py 里加一个包装函数
def height_scan_safe(env, sensor_cfg, offset=0.5):
    raw = mdp.height_scan(env, sensor_cfg, offset)
    return torch.nan_to_num(raw, nan=0.0, posinf=1.5, neginf=-1.0)