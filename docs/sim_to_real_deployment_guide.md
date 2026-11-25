# Sim-to-Real Deployment Guide

> 结项前的部署规划，记录真机输入约束和待验证步骤。项目结果的验证范围为 Isaac Sim。

本文档记录本项目从 IsaacLab 训练到 Unitree G1 真机部署的思路。重点不是“怎么一键部署”，而是解释为什么需要在能力训练后做一个更保守的部署阶段，以及每一步应该检查什么。

## 1. 当前项目的部署约束

本项目的 Actor 必须只使用真机能获得的本体感知输入：

- IMU 角速度 `base_ang_vel`
- IMU 投影重力 `projected_gravity`
- 速度命令 `velocity_commands`
- 关节位置 `joint_pos_rel`
- 关节速度 `joint_vel_rel`
- 上一帧动作 `last_action`

Actor 不能依赖 `height_scan`、真实 `base_lin_vel`、外力、地形类别等训练特权信息。Critic 和辅助监督可以在训练时使用这些信息，但最终导出的 Actor 必须只靠上述本体观测。

当前训练动作是关节位置目标偏移：

```text
target_joint_pos = default_joint_pos + action * 0.25
```

所以 raw action 不是直接关节角度，但它会被 `scale=0.25` 转成关节目标偏移。比如 `action=10` 对应大约 `2.5 rad` 的目标偏移。仿真中 actuator、关节限位和 PD 控制会进一步限制实际动作，但长期依赖很大的 raw action 对真机部署并不理想。

## 2. 能力训练和部署微调不是同一件事

当前复杂地形训练更像“能力上限训练”：

- 地形更难：较高台阶、窄台阶、波浪、粗糙地形。
- 命令更难：更高前进速度、更大的速度课程上限。
- 策略目标：尽可能学会通过复杂地形，即使动作幅度更大、步态更激进。

这类 checkpoint 很有价值，但不一定应该直接上真机。真机部署更关心：

- 动作是否平滑。
- 关节目标是否温和。
- 能耗和关节速度是否可控。
- 轻微观测误差、延迟、摩擦误差下是否还稳。
- 失败时是否可预测、可保护。

因此建议把训练分成两个后期阶段：

```text
能力上限训练 -> 部署保守 fine-tune -> sim2sim/真机分级测试
```

这个流程不是形式主义。对真机来说，它是很必要的安全收敛过程：仿真里“能走”的策略，可能用了真机上不希望出现的大关节目标、快速动作切换或边缘接触技巧。

## 3. 部署保守 fine-tune 是什么

部署保守 fine-tune 不是从零训练，而是从已经能走复杂地形的 checkpoint 继续训练，目标从“挑战最难地形”切换为“在目标部署场景里稳定、温和、可控”。

推荐思路：

- 保持 Actor 观测维度不变，确保 checkpoint 可以 resume。
- 不加入 Actor 无法部署的观测。
- 地形分布贴近真机目标，而不是一味追求最难。
- 速度命令降低到真机早期可接受范围。
- 增强动作平滑、能耗、关节速度、关节限位等约束。
- 保持足够 domain randomization，但不要让训练目标变成极限生存。

一个合理的部署微调目标不是 reward 最高，而是：

- 大多数 episode 能 timeout。
- 速度追踪稳定。
- action 不频繁撞 clip。
- 逐维 action 的 p99 不持续接近上限。
- `bad_orientation` 和 `base_height` termination 很低。
- play 中动作看起来没有抽搐、猛甩、突然大跨步。

## 4. 建议的阶段划分

### Phase C: 能力上限训练

目标：让机器人学会复杂地形能力。

可以使用：

- 较高楼梯，比如 `step_height_range` 到 `0.20`
- 较窄台阶，比如 `step_width=0.30`
- wave / rough 混合地形
- 更高速度课程，比如 `lin_vel_x` 上限到 `1.2`

判断重点：

- 是否能上台阶。
- 是否能在 rough/wave 上保持行走。
- `time_out` 是否仍高。
- action 是否只是少数腿部维度变大，而不是全局发散。

这一阶段允许动作更激进，因为目标是扩展能力边界。

### Phase D: 部署保守 fine-tune

目标：把能力 checkpoint 收敛成真机候选策略。

建议从 Phase C 的好 checkpoint resume，然后调整为：

- 命令范围降低，例如先回到 `lin_vel_x=(0.2, 0.8)` 或更保守。
- `ang_vel_z` 和 `lin_vel_y` 不要太大，早期真机以直行和小转向为主。
- 楼梯高度覆盖目标真机地形，但不要一直压最高难度。
- 台阶宽度不要只用最窄值，保留较宽台阶样本。
- 保留 rough/wave，但避免所有地形同时取最高难度。

奖励上可以更重视：

- `action_rate`
- `joint_vel`
- `joint_acc`
- `energy`
- `dof_pos_limits`
- 躯干姿态稳定
- 非脚部碰撞

但不要为了平滑把腿完全绑住。动作过小会让机器人无法抬脚上台阶。

### Phase E: Sim2Sim / 部署程序验证

目标：不再只相信 IsaacLab play，而是测试导出的策略和部署代码路径。

本项目已有相关链路：

- `train.py` 会导出 `params/deploy.yaml`
- `play.py` 会导出 `exported/policy.pt` 和 `exported/policy.onnx`
- `unitree_rl_lab/source/unitree_rl_lab/unitree_rl_lab/utils/export_deploy_cfg.py` 会导出：
  - `joint_ids_map`
  - `step_dt`
  - `stiffness`
  - `damping`
  - `default_joint_pos`
  - action scale / offset / joint ids
  - policy observation配置
- `unitree_rl_lab/README.md` 中已有 MuJoCo sim2sim 和 G1 deploy 编译说明。

这个阶段要确认：

- 训练 joint 顺序和 SDK joint 顺序一致。
- `joint_sdk_names` 映射正确。
- 部署端 observation 顺序、scale、clip、history length 和训练一致。
- `policy_dt=0.02` 和部署控制频率一致。
- action scale、default offset、PD stiffness/damping 与导出的 `deploy.yaml` 一致。
- JIT/ONNX 输出和 Python policy 输出一致。

### Phase F: 真机分级上线

真机不要直接从“复杂地形高速策略”开始。

建议顺序：

1. 离线推理测试：录一段真实传感器观测或构造静态观测，确认 policy 输出有限、无 NaN、无异常尖峰。
2. 架空/保护状态测试：机器人不承重或有保护架，检查关节方向、默认姿态、action scale。
3. 站立测试：零速度命令，确认不乱动、不持续积累大 action。
4. 平地低速直行：小速度命令，只测前进。
5. 小转向和小侧向：逐步加入 yaw 和 y 方向命令。
6. 简单地形：低台阶、软垫、轻微不平。
7. 目标复杂地形：在保护条件下逐渐接近训练目标。

每一步都要有急停、保护人员、限速和日志记录。

## 5. Action clip 应该怎么理解

`clip_actions` 不应该被当成课程难度或动作设计工具。它更像异常保护。

本地 IsaacLab RSL-RL wrapper 的事实是：

- `RslRlOnPolicyRunnerCfg.clip_actions` 默认是 `None`。
- `RslRlVecEnvWrapper.step()` 只有在 `clip_actions is not None` 时才会 clamp action。

我们曾经试过 `clip_actions=5.0`，结果 resume 后 reward 从 20+ 掉到约 12，说明 `5` 限制了合理动作。后来记录 action 分布发现：

- 全局 `action_abs_p99` 可能只有 4 左右。
- 但少数腿部维度的 p99 可以到 6-7。
- 少数关键动作 max 可以到 10-11。

所以全局 p99 低不代表所有关键动作都小。上台阶抬脚可能只发生在少数关节、少数相位、少数环境里，会被全局统计稀释。

当前较合理的理解：

- `clip_actions=20` 是异常保护，不是正常目标。
- 如果 `action_gt_20_frac` 长期接近 0，只是偶发 max 撞到 20，那么 clip 主要在拦异常尖峰。
- 如果某些维度的 `action_dim_XX_abs_p99` 长期接近 20，说明 clip 仍在限制正常策略。
- 如果部署 fine-tune 后 `action_abs_p99` 和逐维 p99 都降低，同时行为不退化，说明策略更适合真机。

## 6. 什么时候说明策略还没收敛

action 变大本身不一定坏。要结合这些指标判断：

更像正常适应：

- `time_out` 仍高。
- `bad_orientation` 不升高或逐渐下降。
- `track_lin_vel_xy` 和 `track_ang_vel_z` 逐渐恢复。
- action 变大集中在少数腿部维度。
- `action_rate` 没有巨大尖峰。
- play 中步态更高抬腿、更明确落脚，而不是抽搐。

更像没收敛或难度跨度过大：

- `bad_orientation` 持续升高。
- `time_out` 明显下降。
- 速度 tracking 变差且不恢复。
- 大 action 扩散到手臂、腰部或所有关节。
- `joint_vel`、`joint_acc`、`energy` 持续上升。
- play 中出现乱甩腿、脚尖乱刮、身体大幅摆动。

当前我们最近一次观察更接近“难度提升后策略正在适应”：大 action 主要集中在少数维度，`time_out` 仍高，未出现整体发散。

## 7. 真机候选 checkpoint 的建议门槛

以下不是硬标准，而是筛选部署候选的经验门槛：

- `time_out` 接近或高于 0.98。
- `bad_orientation` 长期低于 0.01-0.02。
- `base_height` termination 接近 0。
- `Episode_Reward/action_rate` 无万级尖峰。
- `Loss/ce_kl` 无突然爆炸。
- `Loss/value` 无巨大尖峰。
- 全局 `action_gt_20_frac` 接近 0。
- 逐维 `action_dim_XX_abs_p99` 不长期贴近 clip。
- play 中低速命令动作温和，高速命令动作合理但不过度。
- Sim2Sim 中部署程序路径能复现 IsaacLab play 的基本行为。

## 8. 本项目需要特别小心的点

### CENet / DreamWaQ 部署路径

当前 Actor 需要历史本体观测，并在内部从 history 估计 `v_pred` 和 latent `z`。部署端必须正确维护 history buffer。history 顺序、长度、scale、clip 任何一个错了，策略都会明显退化。

### `last_action`

`last_action` 是 Actor 输入的一部分。真机部署时必须保证：

- 部署端记录的是实际发送给策略环境接口的 action。
- action clip/scale 和训练一致。
- reset 或站立切换时 last_action buffer 不应残留旧策略大动作。

### 关节顺序

Unitree SDK joint 顺序由 `joint_sdk_names` 指定，训练资产内部 joint 顺序不一定相同。必须依赖导出的 `joint_ids_map`，不能手写猜测。

### 速度命令

真机早期不要使用训练中的最高命令范围。先用小速度命令验证：

```text
lin_vel_x: 0.1 -> 0.2 -> 0.4 -> 0.6
ang_vel_z: 0.0 -> small yaw
lin_vel_y: 先不用或很小
```

### 地形

真机早期不要直接上最高难度楼梯。先平地，再软垫/低障碍，再低台阶，再目标台阶。复杂地形能力 checkpoint 可以作为起点，但部署 fine-tune 应该贴近真实测试地形。

## 9. 我们后续应该怎么做

推荐下一步路线：

1. 继续当前能力上限训练，直到复杂地形 reward 和行为基本稳定。
2. 保存一个 best capability checkpoint。
3. 新建部署 fine-tune 配置，但保持 Actor 观测维度不变。
4. 从 best capability checkpoint resume。
5. 降低速度命令和极端地形占比，增强动作温和性约束。
6. 训练到低 termination、低 action spike、行为平滑。
7. 用 `play.py` 导出 `policy.pt` / `policy.onnx`。
8. 用 `deploy.yaml` 检查关节顺序、动作尺度、观测顺序。
9. 先 sim2sim，再真机分级上线。

这条路线的核心思想是：先让策略“会”，再让策略“稳”。能力上限训练给机器人技能，部署 fine-tune 把技能收敛成真机能接受的动作风格。
