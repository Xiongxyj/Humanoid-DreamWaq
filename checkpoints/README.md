# Playback checkpoint / 回放权重

| File | Iteration | Source |
| --- | ---: | --- |
| `g1_dreamwaq_102000.pt` | 102,000 | `2026-06-04_00-12-55/model_102000.pt`, Phase D deployment-stability fine-tuning |

SHA-256: `859ce44d5331c99ed61dd67bcfdd8004a0e8e7116a963610c50ab02be7f6e132`

This checkpoint uses the historical CENet observation layout implemented in the released code. Playback on the stable terrain preset was verified by the project author. It is a later checkpoint than the June 3 demonstration recording, not a byte-identical copy of the recording's weights. It includes the actor, critic and optimizer states. No real-robot validation is claimed.

该权重来自 6 月 4 日 Phase D 稳定性微调阶段，沿用本仓库已发布的历史 CENet 观测布局。项目作者已确认 stable 地形回放正常。它并非 6 月 3 日录屏原始权重。包含 Actor、Critic 和优化器状态；尚未完成真机验证。

```bash
python scripts/play.py --terrain stable --checkpoint checkpoints/g1_dreamwaq_102000.pt
```
