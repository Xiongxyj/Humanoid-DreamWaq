# Unitree G1 训练经验记录

> 结项前的实验笔记，包含当时的诊断和调整建议。当前任务和命令见[项目 README](../README.md)。

本文档记录训练中观察到的现象、可能原因、已经尝试过的解决方式与结果。它是历史实验笔记，并非当前代码的完整变更记录。

## 1. 上楼梯阶段出现跛行

### 现象

- Phase A 平地/波浪走路已经收敛后，resume 到 Phase B 上楼梯训练。
- 机器人可以学到前进和抬脚，但步态出现明显左右不对称。
- 主要表现：右脚在地面/台阶上的停留时间明显长于左脚，像右脚长期作为主支撑脚，左脚负责短促补步或拖步。

### 已尝试

- 将 gait 周期从 `period=1.0` 调回 `period=0.8`。
- 将 gait 权重设置为 `weight=0.4`。
- 结果：跛行仍然存在，说明问题不只是 gait 周期过长。

### 当前判断

`period=1.0` 可能会放大跛行倾向，因为固定相位步态的单腿支撑时间变长，策略更容易找到“一只脚长期支撑，另一只脚补步”的局部最优。

但在 `period=0.8` 后仍然跛行，说明还需要检查奖励组合是否在鼓励左右不对称。

可疑点：

- `foot_on_flat` 对接触脚踩在平面上给正奖励，可能鼓励某一只脚长期稳定接触。
- `foot_clearance_stance_based` 主要约束摆动脚高度；如果某只脚长期不摆动，就能少吃这部分惩罚。
- `gait` 只奖励“是否符合固定相位接触”，但不直接惩罚左右脚实际接触时长差距。
- `gait` 当前若使用 `body_names=".*ankle_roll.*"`，左右脚顺序不是从配置语义上显式保证的。

## 2. 为什么左右脚名称最好显式写出

在 IsaacLab 里，`SceneEntityCfg(..., body_names=...)` 最终会解析成一组 `body_ids`。奖励函数里拿到的是张量列，例如：

```python
is_contact = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids] > 0
```

此时第 0 列、第 1 列只是 `body_ids` 的顺序，不天然带有“左脚”“右脚”的语义。

如果写：

```python
SceneEntityCfg("contact_forces", body_names=".*ankle_roll.*")
```

它大概率会匹配左右脚两个 body，但顺序来自资产内部 body 顺序和 regex 解析结果。这个顺序通常稳定，但代码层面没有明确表达“第 0 个就是左脚，第 1 个就是右脚”。

这在以下情况容易出问题：

- `gait` 里 `offset=[0.0, 0.5]` 假设第 0 条腿和第 1 条腿有固定左右语义。
- `foot_target_height_from_stance()` 内部用 `stance_idx = [1, 0]`，假设输入顺序是 `[left, right]`。
- 某些奖励已经显式使用 `["left_ankle_roll_link", "right_ankle_roll_link"]`，而另一些奖励仍使用 regex，可能造成不同奖励函数对“第 0 只脚”的理解不一致。

显式写成：

```python
SceneEntityCfg(
    "contact_forces",
    body_names=["left_ankle_roll_link", "right_ankle_roll_link"],
)
```

好处是：

- 奖励函数中的第 0 列稳定表示左脚，第 1 列稳定表示右脚。
- `offset=[0.0, 0.5]` 的相位分配有明确语义。
- `stance_idx=[1, 0]` 这种左右互换逻辑不会依赖 regex 的隐含顺序。
- 多个足端奖励之间对左右脚的理解保持一致，便于排查跛行。

这不意味着 regex 一定会错，也不意味着它一定导致右脚长时间支撑。它更像是一个容易隐藏问题的工程风险：一旦出现左右不对称，应该先把这种不确定性排除。

## 3. 针对跛行的建议排查顺序

### 第一步：统一左右脚顺序

将所有需要左右脚语义的奖励都改成显式 body names，尤其是 gait：

```python
"sensor_cfg": SceneEntityCfg(
    "contact_forces",
    body_names=["left_ankle_roll_link", "right_ankle_roll_link"],
)
```

这个修改风险低，不会改变 Actor 观测维度，也不影响 checkpoint 加载。

### 第二步：加入左右接触时长平衡惩罚

项目已有函数：

```python
mdp.air_time_variance_penalty
```

它会惩罚左右脚 air/contact 持续时间差距，适合处理“某一只脚接触时间明显更长”的问题。当前实现同时检查 `current_*` 和 `last_*`，避免某只脚正在长期支撑时必须等到离地后才产生惩罚。

注意：函数名仍叫 `air_time_variance_penalty`，但当前返回的是 `max(time)-min(time)`，对两只脚等价于 `abs(left-right)`，不是方差。这样数值比方差大得多，例如左右支撑时间差 `0.3s` 时，方差约为 `0.0225`，直接差值为 `0.3`。

已按较小权重启用：

```python
air_time_variance = RewTerm(
    func=mdp.air_time_variance_penalty,
    weight=-0.5,
    params={
        "sensor_cfg": SceneEntityCfg(
            "contact_forces",
            body_names=["left_ankle_roll_link", "right_ankle_roll_link"],
        ),
    },
)
```

不要一开始给太大权重，否则机器人可能为了追求左右时间一致而牺牲台阶落脚安全。

### 第三步：收紧 gait 的支撑比例

如果仍然右脚长时间支撑，可考虑：

```python
threshold=0.5
```

相比 `threshold=0.55`，它会减少期望支撑相比例，降低长期单脚支撑的吸引力。

### 第四步：移除 `foot_on_flat` 正奖励

如果右脚脚尖点地也能拿到 `foot_on_flat`，说明这个奖励没有限定足端姿态，只是在奖励“接触脚附近地形平整”。当前已暂停该奖励，避免脚尖点地被当成有效踩平面。

参考 `liuyufei-nubot/G1DWAQ_Lab` 的 G1 DWAQ 配置后，一个重要经验是：它没有使用类似 `foot_on_flat` 的支撑脚平面正奖励，而是更多使用坏接触惩罚，例如 `feet_stumble`、`feet_slide`、`feet_force`，并用 `feet_swing_height` 控制摆腿高度。

### 第五步：增加接触脚姿态惩罚

当前新增 `feet_contact_orientation`：只在有移动命令且脚接触时，惩罚足端 link 的 pitch/roll 偏离水平。这个奖励直接针对“脚尖点地停一下”的行为，而不是继续惩罚接触时长。

## 4. 训练时如何判断问题来自哪里

### 如果缩短 period 后明显改善

说明主要问题是步态节律太慢，单脚支撑窗口过长。可以继续在 `0.7-0.9` 之间扫 period。

### 如果 period 改回 0.8 仍跛行

说明主要问题不是周期本身，而是奖励组合或左右脚语义不一致。

优先检查：

- gait body 顺序是否显式。
- `foot_on_flat` 是否过强。
- 是否缺少左右接触时间平衡惩罚。

### 如果加了 air-time variance 后机器人不敢上楼梯

说明左右对称约束太强。应降低该项权重，例如从 `-0.5` 降到 `-0.2`，而不是直接删除楼梯脚部奖励。

### 如果步态对称但频繁踩台阶边缘

说明 gait/air-time/足姿态约束压过了落脚安全。应先降低 `feet_contact_orientation` 或 `air_time_variance` 权重，不要立即恢复 `foot_on_flat`。

## 5. 当前建议配置方向

优先尝试：

```python
gait = RewTerm(
    func=mdp.feet_gait,
    weight=0.4,
    params={
        "period": 0.8,
        "offset": [0.0, 0.5],
        "threshold": 0.5,
        "command_name": "base_velocity",
        "sensor_cfg": SceneEntityCfg(
            "contact_forces",
            body_names=["left_ankle_roll_link", "right_ankle_roll_link"],
        ),
    },
)
```

并加入：

```python
air_time_variance = RewTerm(
    func=mdp.air_time_variance_penalty,
    weight=-0.5,
    params={
        "sensor_cfg": SceneEntityCfg(
            "contact_forces",
            body_names=["left_ankle_roll_link", "right_ankle_roll_link"],
        ),
    },
)
```

当前已经暂停 `foot_on_flat`，并加入 `feet_stumble` 与 `feet_contact_orientation`，用于减少脚尖点地和台阶边缘坏接触。

## 6. 重要原则

- 不要只看总 reward，要看行为视频。很多不自然步态也能拿到高 reward。
- 先排除工程语义不确定性，再调权重。
- 一次只改少量参数，否则很难判断是哪一项起作用。
- 楼梯阶段 gait 不能太硬，否则会为了踩节拍而牺牲落脚安全。
- 左右对称不是最终目标，稳定上楼梯才是目标；但明显长期单脚支撑通常是局部最优，需要干预。

## 7. Phase C 复杂地形升级

当 Phase B 上楼梯已经比较稳定后，可以进入盲走复杂地形 fine-tune。当前采用 `BLIND_COMPLEX_CFG`：

- flat: 0.10
- wave: 0.20
- rough: 0.20
- stairs_up: 0.30
- stairs_down: 0.20

注意当前比例总和为 0.80；IsaacLab terrain generator 会按这些子地形 proportion 生成列分配，后续如果需要更均衡的列数，可以再把比例归一化或补充 flat/wave。

暂时不引入 stepping stones。原因是离散垫脚石更依赖视觉或明确的足端落点规划；在 Actor 不使用 height scan 的部署约束下，直接加入垫脚石可能让策略只能靠碰撞/接触试探，早期很容易死亡，也不太符合当前 CENet 依靠本体历史和接触后果估计地形上下文的训练节奏。

当前 Phase C 继续使用少量 `MeshInvertedPyramidStairsTerrainCfg`。这个地形的语义不是“普通下楼梯”，而是中心平台在底部、四周台阶向上；机器人从中心出生时首先面对的是放射状上楼梯。用户观察到增大平台宽度后，原地打转占比降低，说明平台中心宽度和 reset 初始位置很关键。因此 reset 已收回平台中心附近，避免直接出生在台阶上。

复杂地形阶段的命令范围略放宽，但仍以前进为主：

```python
lin_vel_x=(0.2, 0.6)
lin_vel_y=(-0.08, 0.08)
ang_vel_z=(-0.08, 0.08)
```

课程上限：

```python
lin_vel_x=(0.2, 0.9)
lin_vel_y=(-0.15, 0.15)
ang_vel_z=(-0.15, 0.15)
```

这一步优先只提高地形和命令难度，不同时大幅修改 reward。若出现退化，应先看是否是 rough/wave 导致姿态过硬、足端姿态惩罚过强，或 stairs_up 比例过高。

如果 play 中只在 inverted pyramid stairs 出现快速打转，优先把它当成特殊地形布局问题排查。当前 play 恢复最高难度课程和高速命令，用于压力测试；训练则通过平台中心 reset 降低冷启动直接踩台阶的概率。

## 8. 非平地终止条件

IsaacLab 原生 `root_height_below_minimum()` 检查的是 root 的世界坐标 z，并且源码注释说明它只适合 flat terrain。对于 inverted pyramid stairs 这类中心平台低于世界零平面的地形，机器人可能没有倒下，却因为世界 z 过低触发 termination。

当前 G1 配置已改用本项目自定义 `root_height_below_feet()`：

- 用接触脚的平均 z 作为局部地形高度参考。
- 只在 root 相对接触脚高度低于阈值时终止。
- 双脚都不接触时不触发，避免短暂腾空/跳步被误杀。

如果以后出现“没倒但走一两步突然 reset”，优先检查 termination manager，而不是先怀疑策略或地形 mesh。
