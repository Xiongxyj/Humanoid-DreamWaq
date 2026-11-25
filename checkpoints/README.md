# 回放权重

| 文件 | 迭代 | 来源 |
| --- | ---: | --- |
| `g1_dreamwaq_82800.pt` | 82,800 | 本地 `2026-06-04_00-12-55` 训练目录的起始保存。该 run 的 `agent.yaml` 记录从 `2026-06-03_10-11-29/model_82800.pt` 恢复，原始路径也出现在录屏日期附近的回放命令中。 |

SHA-256：`bc6b61a8e3a69da5a320c89dbd38ae1a98ff17970a68a04ad0508b571fb38a62`

这里保留的是与录屏最接近且可追溯的本地版本。录屏画面没有显示 checkpoint 路径或散列值，因此无法确认与录屏所用文件逐字节一致。此文件包含 Actor、Critic 和优化器状态，可用于当前 CENet 任务回放与续训。从仓库根目录运行：

```bash
python scripts/play.py --terrain complex --checkpoint checkpoints/g1_dreamwaq_82800.pt
```
