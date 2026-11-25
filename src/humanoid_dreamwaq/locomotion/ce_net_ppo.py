from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from tensordict import TensorDict

from rsl_rl.algorithms import PPO
from rsl_rl.env import VecEnv
from rsl_rl.models import MLPModel
from rsl_rl.storage import RolloutStorage
from rsl_rl.utils import resolve_callable, resolve_obs_groups
from rsl_rl.extensions import resolve_rnd_config, resolve_symmetry_config

from .ce_net_mlp_model import CENetMLPModel


class CENetPPO(PPO):
    """PPO with velocity supervision, history reconstruction, and latent KL regularization.

    The auxiliary losses and PPO objective share one optimizer step.
    """

    def __init__(
        self,
        actor: CENetMLPModel,
        critic: MLPModel,
        storage: RolloutStorage,
        beta_kl: float = 0.001,
        lambda_recon: float = 1.0,
        lambda_vel: float = 1.0,
        **kwargs,
    ) -> None:
        kwargs.pop("share_cnn_encoders", None)
        super().__init__(actor, critic, storage, **kwargs)
        self.beta_kl = beta_kl
        self.lambda_recon = lambda_recon
        self.lambda_vel = lambda_vel

    def update(self) -> dict[str, float]:
        """Update the policy and context estimator from one rollout."""
        if self.actor.is_recurrent or self.critic.is_recurrent:
            raise NotImplementedError("CENetPPO currently supports feed-forward actor/critic only.")

        mean_value_loss = 0.0
        mean_surrogate_loss = 0.0
        mean_entropy = 0.0
        mean_kl_loss = 0.0
        mean_recon_loss = 0.0
        mean_vel_loss = 0.0
        mean_symmetry_loss = 0.0 if self.symmetry else None

        st = self.storage
        batch_size = st.num_envs * st.num_transitions_per_env
        mini_batch_size = batch_size // self.num_mini_batches
        indices = torch.randperm(self.num_mini_batches * mini_batch_size, requires_grad=False, device=self.device)

        observations = st.observations.flatten(0, 1)
        actions = st.actions.flatten(0, 1)
        values = st.values.flatten(0, 1)
        returns = st.returns.flatten(0, 1)
        old_actions_log_prob = st.actions_log_prob.flatten(0, 1)
        advantages = st.advantages.flatten(0, 1)
        old_distribution_params = tuple(p.flatten(0, 1) for p in st.distribution_params)

        with torch.no_grad():
            abs_actions_by_dim = actions.detach().abs()
            abs_actions = abs_actions_by_dim.reshape(-1)
            action_abs_mean = abs_actions.mean().item()
            action_abs_max = abs_actions.max().item()
            action_abs_p95 = torch.quantile(abs_actions, 0.95).item()
            action_abs_p99 = torch.quantile(abs_actions, 0.99).item()
            action_gt_5_frac = (abs_actions > 5.0).float().mean().item()
            action_gt_10_frac = (abs_actions > 10.0).float().mean().item()
            action_gt_20_frac = (abs_actions > 20.0).float().mean().item()
            action_dim_abs_max = abs_actions_by_dim.max(dim=0).values
            action_dim_abs_p99 = torch.quantile(abs_actions_by_dim, 0.99, dim=0)

        # Reuse the current history slice at episode boundaries to avoid cross-episode targets.
        policy_single = st.observations["policy"][..., -self.actor.obs_single_dim:]
        next_single = policy_single.clone()
        next_single[:-1] = torch.where(st.dones[:-1].bool(), policy_single[:-1], policy_single[1:])
        next_single = next_single.flatten(0, 1).detach()

        for _ in range(self.num_learning_epochs):
            for i in range(self.num_mini_batches):
                start = i * mini_batch_size
                stop = (i + 1) * mini_batch_size
                batch_idx = indices[start:stop]
                obs_batch_ce = observations[batch_idx]
                actions_batch = actions[batch_idx]
                values_batch = values[batch_idx]
                returns_batch = returns[batch_idx]
                old_actions_log_prob_batch = old_actions_log_prob[batch_idx]
                advantages_batch = advantages[batch_idx]
                original_batch_size = batch_idx.shape[0]

                if self.normalize_advantage_per_mini_batch:
                    with torch.no_grad():
                        advantages_batch = (
                            (advantages_batch - advantages_batch.mean()) / (advantages_batch.std() + 1e-8)
                        )

                obs_batch = obs_batch_ce
                if self.symmetry and self.symmetry["use_data_augmentation"]:
                    data_augmentation_func = self.symmetry["data_augmentation_func"]
                    obs_batch, actions_batch = data_augmentation_func(
                        env=self.symmetry["_env"],
                        obs=obs_batch_ce,
                        actions=actions_batch,
                    )
                    num_aug = int(obs_batch.batch_size[0] / original_batch_size)
                    values_batch = values_batch.repeat(num_aug, 1)
                    returns_batch = returns_batch.repeat(num_aug, 1)
                    old_actions_log_prob_batch = old_actions_log_prob_batch.repeat(num_aug, 1)
                    advantages_batch = advantages_batch.repeat(num_aug, 1)

                self.actor(obs_batch, stochastic_output=True)
                actions_log_prob = self.actor.get_output_log_prob(actions_batch)
                value_batch = self.critic(obs_batch)
                distribution_params = tuple(p for p in self.actor.output_distribution_params)
                entropy = self.actor.output_entropy

                if self.desired_kl is not None and self.schedule == "adaptive":
                    with torch.inference_mode():
                        old_params = tuple(p[batch_idx][:original_batch_size] for p in old_distribution_params)
                        kl = self.actor.get_kl_divergence(old_params, tuple(p[:original_batch_size] for p in distribution_params))
                        kl_mean = torch.mean(kl)

                        if self.is_multi_gpu:
                            torch.distributed.all_reduce(kl_mean, op=torch.distributed.ReduceOp.SUM)
                            kl_mean /= self.gpu_world_size

                        if self.gpu_global_rank == 0:
                            if kl_mean > self.desired_kl * 2.0:
                                self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                            elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                                self.learning_rate = min(1e-2, self.learning_rate * 1.5)

                        if self.is_multi_gpu:
                            lr_tensor = torch.tensor(self.learning_rate, device=self.device)
                            torch.distributed.broadcast(lr_tensor, src=0)
                            self.learning_rate = lr_tensor.item()

                        for param_group in self.optimizer.param_groups:
                            param_group["lr"] = self.learning_rate

                ratio = torch.exp(actions_log_prob - torch.squeeze(old_actions_log_prob_batch))
                surrogate = -torch.squeeze(advantages_batch) * ratio
                surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(
                    ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
                )
                surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

                if self.use_clipped_value_loss:
                    value_clipped = values_batch + (value_batch - values_batch).clamp(
                        -self.clip_param, self.clip_param
                    )
                    value_losses = (value_batch - returns_batch).pow(2)
                    value_losses_clipped = (value_clipped - returns_batch).pow(2)
                    value_loss = torch.max(value_losses, value_losses_clipped).mean()
                else:
                    value_loss = (returns_batch - value_batch).pow(2).mean()

                # Train the context estimator on the original rollout, with simulated velocity as the target.
                history = torch.cat([obs_batch_ce[g] for g in self.actor.obs_groups], dim=-1)
                mu, sigma, v_pred = self.actor.encode_history(history)
                z = mu + sigma * torch.randn_like(sigma)
                pred_next = self.actor.decoder(z)

                kl_loss = 0.5 * (mu**2 + sigma**2 - (sigma**2).log() - 1).sum(-1).mean()
                recon_loss = F.mse_loss(pred_next, next_single[batch_idx])
                vel_loss = F.mse_loss(v_pred, obs_batch_ce["critic"][:, :3].detach())
                ce_loss = self.beta_kl * kl_loss + self.lambda_recon * recon_loss + self.lambda_vel * vel_loss

                loss = (
                    surrogate_loss
                    + self.value_loss_coef * value_loss
                    - self.entropy_coef * entropy.mean()
                    + ce_loss
                )

                # Use the mirrored action of the original observation as a fixed target.
                if self.symmetry and self.symmetry["use_mirror_loss"]:
                    data_augmentation_func = self.symmetry["data_augmentation_func"]
                    with torch.no_grad():
                        action_mean_orig = self.actor(obs_batch_ce.detach().clone())
                        _, actions_mean_symm = data_augmentation_func(
                            env=self.symmetry["_env"],
                            obs=None,
                            actions=action_mean_orig,
                        )
                        action_mean_target = actions_mean_symm[original_batch_size:]

                    obs_symm, _ = data_augmentation_func(
                        env=self.symmetry["_env"],
                        obs=obs_batch_ce.detach().clone(),
                        actions=None,
                    )
                    action_mean_symm = self.actor(obs_symm[original_batch_size:].detach().clone())
                    symmetry_loss = F.mse_loss(action_mean_symm, action_mean_target.detach())
                    loss = loss + self.symmetry["mirror_loss_coeff"] * symmetry_loss
                    mean_symmetry_loss += symmetry_loss.item()

                self.optimizer.zero_grad()
                loss.backward()

                if self.is_multi_gpu:
                    self.reduce_parameters()

                nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                self.optimizer.step()

                mean_value_loss += value_loss.item()
                mean_surrogate_loss += surrogate_loss.item()
                mean_entropy += entropy.mean().item()
                mean_kl_loss += kl_loss.item()
                mean_recon_loss += recon_loss.item()
                mean_vel_loss += vel_loss.item()

        num_updates = self.num_learning_epochs * self.num_mini_batches

        loss_dict = {
            "value": mean_value_loss / num_updates,
            "surrogate": mean_surrogate_loss / num_updates,
            "entropy": mean_entropy / num_updates,
            "ce_kl": mean_kl_loss / num_updates,
            "ce_recon": mean_recon_loss / num_updates,
            "ce_vel": mean_vel_loss / num_updates,
            "action_abs_mean": action_abs_mean,
            "action_abs_max": action_abs_max,
            "action_abs_p95": action_abs_p95,
            "action_abs_p99": action_abs_p99,
            "action_gt_5_frac": action_gt_5_frac,
            "action_gt_10_frac": action_gt_10_frac,
            "action_gt_20_frac": action_gt_20_frac,
        }
        if mean_symmetry_loss is not None:
            loss_dict["symmetry"] = mean_symmetry_loss / num_updates

        for action_idx in range(actions.shape[-1]):
            loss_dict[f"action_dim_{action_idx:02d}_abs_max"] = action_dim_abs_max[action_idx].item()
            loss_dict[f"action_dim_{action_idx:02d}_abs_p99"] = action_dim_abs_p99[action_idx].item()

        env = getattr(self, "_env", None)
        if env is None and self.symmetry:
            env = self.symmetry.get("_env", None)
        env = getattr(env, "unwrapped", env)
        gait_metrics = getattr(env, "_g1_gait_metrics", None)
        if gait_metrics:
            for metric_name, metric_value in gait_metrics.items():
                if torch.is_tensor(metric_value):
                    loss_dict[f"gait_{metric_name}"] = metric_value.detach().mean().item()
                else:
                    loss_dict[f"gait_{metric_name}"] = float(metric_value)

        self.storage.clear()
        return loss_dict

    @staticmethod
    def construct_algorithm(obs: TensorDict, env: VecEnv, cfg: dict, device: str) -> "CENetPPO":
        """Build the CENet runner components from the RSL-RL configuration."""
        alg_class: type[CENetPPO] = resolve_callable(cfg["algorithm"].pop("class_name"))
        actor_class: type[CENetMLPModel] = resolve_callable(cfg["actor"].pop("class_name"))
        critic_class: type[MLPModel] = resolve_callable(cfg["critic"].pop("class_name"))

        default_sets = ["actor", "critic"]
        cfg["obs_groups"] = resolve_obs_groups(obs, cfg["obs_groups"], default_sets)
        cfg["algorithm"] = resolve_rnd_config(cfg["algorithm"], obs, cfg["obs_groups"], env)
        cfg["algorithm"] = resolve_symmetry_config(cfg["algorithm"], env)

        beta_kl = cfg["algorithm"].pop("beta_kl", 0.001)
        lambda_recon = cfg["algorithm"].pop("lambda_recon", 1.0)
        lambda_vel = cfg["algorithm"].pop("lambda_vel", 1.0)

        actor = actor_class(obs, cfg["obs_groups"], "actor", env.num_actions, **cfg["actor"]).to(device)
        critic = critic_class(obs, cfg["obs_groups"], "critic", 1, **cfg["critic"]).to(device)
        storage = RolloutStorage("rl", env.num_envs, cfg["num_steps_per_env"], obs, [env.num_actions], device)

        alg = alg_class(
            actor, critic, storage,
            beta_kl=beta_kl,
            lambda_recon=lambda_recon,
            lambda_vel=lambda_vel,
            device=device,
            **cfg["algorithm"],
            multi_gpu_cfg=cfg["multi_gpu"],
        )
        alg._env = env
        return alg
