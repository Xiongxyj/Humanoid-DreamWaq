from __future__ import annotations

import torch
from tensordict import TensorDict


_POLICY_FRAME_DIM = 96
_POLICY_HISTORY_LEN = 5
_HEIGHT_SCAN_SHAPE = (21, 33)
_HEIGHT_SCAN_DIM = _HEIGHT_SCAN_SHAPE[0] * _HEIGHT_SCAN_SHAPE[1]

_G1_MIRROR_INDEX = [
    6, 7, 8, 9, 10, 11,
    0, 1, 2, 3, 4, 5,
    12, 13, 14,
    22, 23, 24, 25, 26, 27, 28,
    15, 16, 17, 18, 19, 20, 21,
]
_G1_SIGN_FLIP = [
    1, 2, 5,
    7, 8, 11,
    12, 13,
    16, 17, 19, 21,
    23, 24, 26, 28,
]


@torch.no_grad()
def compute_symmetric_states(env, obs: TensorDict | None = None, actions: torch.Tensor | None = None):
    """Return original and mirrored observations/actions for symmetry experiments."""
    if obs is not None:
        batch_size = obs.batch_size[0]
        obs_aug = obs.repeat(2)
        obs_aug["policy"][:batch_size] = obs["policy"][:]
        obs_aug["policy"][batch_size:] = _mirror_policy(obs["policy"])
        if "critic" in obs:
            obs_aug["critic"][:batch_size] = obs["critic"][:]
            obs_aug["critic"][batch_size:] = _mirror_critic(obs["critic"])
    else:
        obs_aug = None

    if actions is not None:
        batch_size = actions.shape[0]
        actions_aug = torch.zeros(batch_size * 2, actions.shape[1], device=actions.device)
        actions_aug[:batch_size] = actions[:]
        actions_aug[batch_size:] = _mirror_actions(actions)
    else:
        actions_aug = None

    return obs_aug, actions_aug


def _mirror_policy(obs: torch.Tensor) -> torch.Tensor:
    obs = obs.clone()
    if obs.ndim == 2 and obs.shape[-1] == _POLICY_HISTORY_LEN * _POLICY_FRAME_DIM:
        # Legacy experiment assumes frame-major history, unlike Isaac Lab term-major flattening.
        obs = obs.reshape(obs.shape[0], _POLICY_HISTORY_LEN, _POLICY_FRAME_DIM)
        out = torch.zeros_like(obs)
        for i in range(_POLICY_HISTORY_LEN):
            out[:, i, :] = _mirror_single_frame(obs[:, i, :])
        return out.reshape(obs.shape[0], _POLICY_HISTORY_LEN * _POLICY_FRAME_DIM)
    return _mirror_single_frame(obs)


def _mirror_single_frame(frame: torch.Tensor) -> torch.Tensor:
    frame = frame.clone()
    device = frame.device
    frame[:, 0:3] = frame[:, 0:3] * torch.tensor([-1.0, 1.0, -1.0], device=device)
    frame[:, 3:6] = frame[:, 3:6] * torch.tensor([1.0, -1.0, 1.0], device=device)
    frame[:, 6:9] = frame[:, 6:9] * torch.tensor([1.0, -1.0, -1.0], device=device)
    frame[:, 9:38] = _mirror_joint_data(frame[:, 9:38])
    frame[:, 38:67] = _mirror_joint_data(frame[:, 38:67])
    frame[:, 67:96] = _mirror_joint_data(frame[:, 67:96])
    return frame


def _mirror_critic(obs: torch.Tensor) -> torch.Tensor:
    obs = obs.clone()
    device = obs.device
    obs[:, 0:3] = obs[:, 0:3] * torch.tensor([1.0, -1.0, 1.0], device=device)
    obs[:, 3:6] = obs[:, 3:6] * torch.tensor([-1.0, 1.0, -1.0], device=device)
    obs[:, 6:9] = obs[:, 6:9] * torch.tensor([1.0, -1.0, 1.0], device=device)
    obs[:, 9:12] = obs[:, 9:12] * torch.tensor([1.0, -1.0, -1.0], device=device)
    obs[:, 12:41] = _mirror_joint_data(obs[:, 12:41])
    obs[:, 41:70] = _mirror_joint_data(obs[:, 41:70])
    obs[:, 70:99] = _mirror_joint_data(obs[:, 70:99])
    if obs.shape[-1] >= 99 + _HEIGHT_SCAN_DIM + 3:
        height_start = 99
        height_end = height_start + _HEIGHT_SCAN_DIM
        obs[:, height_start:height_end] = (
            obs[:, height_start:height_end]
            .view(-1, *_HEIGHT_SCAN_SHAPE)
            .flip(dims=[1])
            .reshape(-1, _HEIGHT_SCAN_DIM)
        )
        obs[:, height_end:height_end + 3] = (
            obs[:, height_end:height_end + 3] * torch.tensor([1.0, -1.0, 1.0], device=device)
        )
    return obs


def _mirror_actions(actions: torch.Tensor) -> torch.Tensor:
    actions = actions.clone()
    actions = _mirror_joint_data(actions)
    return actions


def _mirror_joint_data(joint_data: torch.Tensor) -> torch.Tensor:
    joint_data = joint_data.clone()
    device = joint_data.device
    if joint_data.shape[-1] < 29:
        return joint_data

    mirror_index = torch.tensor(_G1_MIRROR_INDEX, device=device)
    joint_data_mirror = joint_data[..., mirror_index]
    sign_flip = torch.ones(joint_data.shape[-1], device=device)
    sign_flip[_G1_SIGN_FLIP] = -1.0
    return joint_data_mirror * sign_flip
