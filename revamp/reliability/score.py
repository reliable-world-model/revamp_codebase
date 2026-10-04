"""Combine the two reliability signals and feed the closed loop (Sec. 3.3-3.4).

- ``combined_reliability`` — the per-prediction score (Eq. 3):
      r = w_intr * r_intr + w_ext * r_ext
- ``gate_weight`` — the imagined-update gate (Eq. 4):
      w(s, ā) = exp(-λ * r(s, ā)),  λ = 1
  multiplying each sample's adjoint-matching loss.
- ``is_triggered`` — the discrete real-rollout trigger (Eq. 5):
      the imagined rollout stops at the first chunk with r > τ.
- ``calibrate`` — w_intr and w_ext scale each term to an in-distribution mean
  of 0.5; τ is the 95th percentile of r on held-out in-distribution
  predictions, pooled over tasks so that one τ is shared across tasks.
- ``ReliabilityScorer`` — evaluates both signals for a batch of predicted
  chunks.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
from typing import Dict, Optional

import torch

from revamp.reliability.action_faithfulness import (
    DepthPoseProbe,
    InverseDynamicsMLP,
    implied_action_residual,
)
from revamp.reliability.geometric import velocity_field_residual


@dataclass(frozen=True)
class ReliabilityCalibration:
    w_intr: float
    w_ext: float
    tau: float
    lam: float = 1.0
    k_probes: int = 4
    sigma_min: float = 0.2
    sigma_max: float = 0.8

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2))

    @staticmethod
    def load(path: str | Path) -> "ReliabilityCalibration":
        return ReliabilityCalibration(**json.loads(Path(path).read_text()))


def combined_reliability(r_intr, r_ext, w_intr: float, w_ext: float):
    """Per-prediction score r(s, ā) (Eq. 3). Higher = less reliable."""
    return w_intr * r_intr + w_ext * r_ext


def gate_weight(r, lam: float = 1.0):
    """Imagined-update gate w = exp(-λ r) (Eq. 4), in (0, 1]."""
    if isinstance(r, torch.Tensor):
        return torch.exp(-lam * r)
    return math.exp(-lam * r)


def is_triggered(r, tau: float):
    """Trigger (Eq. 5): True iff r exceeds τ."""
    return r > tau


def calibrate(
    r_intr_in_dist: torch.Tensor,
    r_ext_in_dist: torch.Tensor,
    target_mean: float = 0.5,
    quantile: float = 0.95,
    lam: float = 1.0,
) -> ReliabilityCalibration:
    """Calibrate on held-out in-distribution predictions pooled over all tasks."""
    w_intr = target_mean / float(r_intr_in_dist.float().mean())
    w_ext = target_mean / float(r_ext_in_dist.float().mean())
    r = combined_reliability(r_intr_in_dist.float(), r_ext_in_dist.float(), w_intr, w_ext)
    tau = float(torch.quantile(r, quantile))
    return ReliabilityCalibration(w_intr=w_intr, w_ext=w_ext, tau=tau, lam=lam)


class ReliabilityScorer:
    """Score a batch of predicted chunks with both signals."""

    def __init__(
        self,
        calibration: ReliabilityCalibration,
        probe: DepthPoseProbe,
        id_model: InverseDynamicsMLP,
    ):
        self.cal = calibration
        self.probe = probe
        self.id_model = id_model.eval()

    @torch.no_grad()
    def __call__(
        self,
        wm,
        state: torch.Tensor,                    # [B, state_dim]
        action_chunk: torch.Tensor,             # [B, H, action_dim]
        t5_embeddings: torch.Tensor,            # [B, L, D]
        cond_latent: torch.Tensor,              # [B, C, 1, H', 3W']
        target_latents: torch.Tensor,           # [B, C, 2, H', 3W']
        decoded: Dict[str, torch.Tensor],       # {'left'|'right'|'high': [B, 3, H, h, w] in [0, 1]}
        start_pose: Optional[torch.Tensor] = None,  # [B, 7] end-effector pose at the chunk start
        seed: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        r_intr = velocity_field_residual(
            wm, state, action_chunk, t5_embeddings, target_latents,
            cond_latent=cond_latent,
            k_probes=self.cal.k_probes,
            sigma_range=(self.cal.sigma_min, self.cal.sigma_max),
            seed=seed,
        )["score"]
        r_ext = torch.tensor([
            implied_action_residual(
                self.probe, self.id_model,
                {cam: frames[b] for cam, frames in decoded.items()},
                action_chunk[b],
                None if start_pose is None else start_pose[b],
            )["score"]
            for b in range(state.shape[0])
        ])
        r = combined_reliability(r_intr, r_ext, self.cal.w_intr, self.cal.w_ext)
        return {
            "r_intr": r_intr,
            "r_ext": r_ext,
            "r": r,
            "weight": gate_weight(r, self.cal.lam),
            "triggered": is_triggered(r, self.cal.tau),
        }
