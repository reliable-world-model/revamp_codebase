"""Rewards for value learning: potential-based shaping plus sparse terminal reward.

The progress model (ViVa, initialized from Wan2.2-TI2V-5B) predicts a remaining
cost C(s): on successful trajectories it decreases linearly from 1 to 0; on
failed trajectories it follows the success profile until the annotated failure
onset, then rises sharply and settles at a terminal cost above the success
level. The shaping potential is Φ = -C and

    r_t = γ Φ(s_{t+1}) - Φ(s_t) + r_sparse,t,
    r_sparse,t = +1 at success, -1 at failure.
"""
from __future__ import annotations

from typing import Optional

import torch

from revamp.common.constants import DEFAULT_ALPHA, DEFAULT_BETA_FAIL, DEFAULT_BETA_SUCC, DEFAULT_GAMMA


def potential(cost: torch.Tensor, alpha: float = DEFAULT_ALPHA) -> torch.Tensor:
    """Φ(s) = -α · C(s)."""
    return -alpha * cost


def chunk_pbrs(
    cost_chunk: torch.Tensor,             # [B, H+1] remaining cost along the chunk: s_t..s_{t+H}
    alpha: float = DEFAULT_ALPHA,
    gamma: float = DEFAULT_GAMMA,
) -> torch.Tensor:
    """Per-step shaping γ Φ(s_{t+1}) - Φ(s_t) along an H-step chunk -> [B, H]."""
    phi = potential(cost_chunk, alpha)
    return gamma * phi[:, 1:] - phi[:, :-1]


def sparse_reward(
    success_mask: torch.Tensor,
    fail_mask: Optional[torch.Tensor] = None,
    beta_succ: float = DEFAULT_BETA_SUCC,
    beta_fail: float = DEFAULT_BETA_FAIL,
) -> torch.Tensor:
    """r_sparse = β_succ · 1[success] - β_fail · 1[failure] (β_succ = β_fail = 1)."""
    r = beta_succ * success_mask.float()
    if fail_mask is not None:
        r = r - beta_fail * fail_mask.float()
    return r


def chunk_step_rewards(
    cost_chunk: torch.Tensor,             # [B, H+1]
    success_mask_chunk: torch.Tensor,     # [B, H]
    fail_mask_chunk: Optional[torch.Tensor] = None,
    alpha: float = DEFAULT_ALPHA,
    gamma: float = DEFAULT_GAMMA,
    beta_succ: float = DEFAULT_BETA_SUCC,
    beta_fail: float = DEFAULT_BETA_FAIL,
) -> torch.Tensor:
    """Full per-step rewards along a chunk: shaping + sparse -> [B, H]."""
    return chunk_pbrs(cost_chunk, alpha, gamma) + sparse_reward(
        success_mask_chunk, fail_mask_chunk, beta_succ, beta_fail,
    )
