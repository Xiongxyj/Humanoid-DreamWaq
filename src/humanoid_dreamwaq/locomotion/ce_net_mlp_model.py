from __future__ import annotations

import torch
import torch.nn as nn
from tensordict import TensorDict

from rsl_rl.models import MLPModel
from rsl_rl.modules import MLP, HiddenState


class CENetMLPModel(MLPModel):
    """Actor with a shared encoder for proprioceptive history.

    The encoder estimates body velocity and a latent distribution. The actor uses the latent mean, and the decoder reconstructs the next history slice during training.
    """

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        obs_set: str,
        output_dim: int,
        obs_single_dim: int = 96,
        latent_dim: int = 16,
        encoder_hidden_dims: tuple[int, ...] = (256, 128),
        encoder_output_dim: int = 128,
        **kwargs,
    ) -> None:
        # The base initializer calls _get_latent_dim() to size the policy head.
        self.obs_single_dim = obs_single_dim
        self.latent_dim = latent_dim
        self.encoder_output_dim = encoder_output_dim

        super().__init__(obs, obs_groups, obs_set, output_dim, **kwargs)

        history_dim = self.obs_dim

        self.shared_encoder = MLP(history_dim, encoder_output_dim, list(encoder_hidden_dims))

        self.z_head = nn.Linear(encoder_output_dim, latent_dim * 2)

        self.vel_head = nn.Linear(encoder_output_dim, 3)

        self.decoder = MLP(latent_dim, obs_single_dim, [128])

        self._ce_mu: torch.Tensor | None = None
        self._ce_sigma: torch.Tensor | None = None
        self._ce_v: torch.Tensor | None = None

    def _get_latent_dim(self) -> int:
        """Return the width of the history slice, velocity estimate, and latent mean."""
        return self.obs_single_dim + 3 + self.latent_dim

    def encode_history(self, history: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return the latent mean, standard deviation, and body velocity estimate."""
        history = self.obs_normalizer(history)
        features = self.shared_encoder(history)

        z_params = self.z_head(features)
        mu, log_sigma = z_params.chunk(2, dim=-1)
        log_sigma = torch.clamp(log_sigma, min=-10.0, max=10.0)
        sigma = log_sigma.exp().clamp(min=1e-4)
        v_pred = self.vel_head(features)
        return mu, sigma, v_pred

    def get_latent(
        self,
        obs: TensorDict,
        masks: torch.Tensor | None = None,
        hidden_state: HiddenState = None,
    ) -> torch.Tensor:
        """Build the policy input from proprioceptive history and its encoding."""

        history = torch.cat([obs[g] for g in self.obs_groups], dim=-1)
        mu, sigma, v_pred = self.encode_history(history)

        self._ce_mu = mu
        self._ce_sigma = sigma
        self._ce_v = v_pred

        # Isaac Lab flattens each observation term separately. Saved weights expect this tail slice.
        history_tail = self.obs_normalizer(history)[:, -self.obs_single_dim:]
        return torch.cat([history_tail, v_pred, mu], dim=-1)
