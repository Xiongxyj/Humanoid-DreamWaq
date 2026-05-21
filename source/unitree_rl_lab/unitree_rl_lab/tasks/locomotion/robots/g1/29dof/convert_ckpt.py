import torch

old_ckpt_path = "/home/xiaowolaopo/arm/unitree_rl_lab/logs/rsl_rl/unitree_g1_29dof_velocity/2026-05-09_21-37-21/model_11900.pt"
# 生成的新权重路径
new_ckpt_path = "/home/xiaowolaopo/arm/unitree_rl_lab/logs/rsl_rl/unitree_g1_29dof_velocity/2026-05-09_21-37-21/model_11900_padded_v2.pt"

# 分别定义 Actor 和 Critic 的新维度
NEW_ACTOR_DIM = 1415
NEW_CRITIC_DIM = 1430

print(f"Loading original checkpoint: {old_ckpt_path}")
ckpt = torch.load(old_ckpt_path, map_location="cpu")

# 将对应的键和目标维度配对处理
process_targets = [
    ("actor_state_dict", NEW_ACTOR_DIM),
    ("critic_state_dict", NEW_CRITIC_DIM)
]

for key, target_dim in process_targets:
    if key in ckpt:
        state_dict = ckpt[key]
        old_weight = state_dict["mlp.0.weight"]
        out_features, in_features = old_weight.shape
        
        print(f"[{key}] Original shape: {old_weight.shape}")
        
        if in_features == target_dim:
            print(f"[{key}] Already matches the target dimension {target_dim}. Skipping.")
            continue
            
        # 创建新的全 0 权重矩阵
        new_weight = torch.zeros((out_features, target_dim), dtype=old_weight.dtype, device=old_weight.device)
        
        # 将旧的肌肉记忆复制到前半部分
        new_weight[:, :in_features] = old_weight
        
        # 写回字典
        state_dict["mlp.0.weight"] = new_weight
        print(f"[{key}] Successfully expanded to: {new_weight.shape}")

# 删除旧的优化器状态，防止 Adam 动量报错
if "optimizer_state_dict" in ckpt:
    del ckpt["optimizer_state_dict"]
    print("Deleted optimizer_state_dict.")

torch.save(ckpt, new_ckpt_path)
print(f"Successfully saved V2 padded checkpoint to: {new_ckpt_path}")