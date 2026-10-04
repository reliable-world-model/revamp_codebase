"""Geometric faithfulness — intrinsic velocity-field self-consistency (Eq. 1).

After the world model predicts a clean target latent x̂ for a chunk, we re-noise
it with K fresh (noise, σ) pairs under the same conditioning and compare the
velocity field with the path velocity:

    for k in 1..K:
        z_k ~ N(0, I)
        σ_k ~ U[0.2, 0.8]   (nearest grid point of the FM scheduler)
        x_k = (1 - σ_k) * x̂ + σ_k * z_k          (target slots only; cond untouched)
        r_k = || v_θ(x_k, σ_k, cond, ā, s) - (z_k - x̂) ||²
    r_intr = mean_k r_k

This is the model's own training loss evaluated at x̂. It stays at its
training-data level while x̂ lies on the training manifold and grows off it.
It needs no ground-truth future.

Cost: K DiT forwards per chunk.
"""
from __future__ import annotations

from typing import Optional

import torch

from revamp.common.constants import (
    COND_LATENT_IDX,
    TARGET_LATENT_INDICES,
    NUM_LATENT_FRAMES,
)


@torch.no_grad()
def velocity_field_residual(
    wm,
    state: torch.Tensor,                 # [B, state_dim]
    action_chunk: torch.Tensor,          # [B, horizon, action_dim]
    t5_embeddings: torch.Tensor,         # [B, L, D]
    clean_target_latent: torch.Tensor,   # [B, C, len(TARGET_LATENT_INDICES), H', 3W']
    cond_latent: Optional[torch.Tensor] = None,        # [B, C, 1, H', 3W']
    cam_left_segment: Optional[torch.Tensor] = None,   # [B, 3, T, H, W] (used if cond_latent is None)
    cam_right_segment: Optional[torch.Tensor] = None,
    cam_high_segment: Optional[torch.Tensor] = None,
    k_probes: int = 4,
    sigma_range: tuple = (0.2, 0.8),
    seed: Optional[int] = None,
) -> dict:
    """Probe the velocity field at K random (z, σ) pairs; return the per-sample mean residual.

    Args:
        wm: a ``revamp.common.world_model.WorldModel`` instance.
        clean_target_latent: x̂, the target latents of the predicted chunk.
        cond_latent: condition latent of the chunk; built from the camera
            segments when not given.

    Returns:
        dict with ``score`` ([B] r_intr in FM-loss units), ``per_probe``
        ([K, B] residuals), ``sigma_used`` ([K, B]) and ``k``.
    """
    device = state.device
    B = state.shape[0]
    gen = None
    if seed is not None:
        gen = torch.Generator(device="cpu")
        gen.manual_seed(seed)

    if cond_latent is None:
        cond_latent = wm.build_full_latent_sequence(
            cam_left_segment, cam_right_segment, cam_high_segment,
        )[:, :, COND_LATENT_IDX:COND_LATENT_IDX + 1]
    clean_full = torch.cat(
        [cond_latent.to(device=device, dtype=wm.dtype), clean_target_latent.to(device=device, dtype=wm.dtype)],
        dim=2,
    )                                                                   # [B, C, 3, H', 3W']

    # σ ~ U[sigma_min, sigma_max]; the DiT is queried at the nearest grid point
    # of the FM scheduler so that (σ, t) stay consistent with training.
    fm_sched = wm.fm_scheduler
    sigmas_full = fm_sched.sigmas.to(device=device, dtype=torch.float32)
    timesteps_full = fm_sched.timesteps.to(device=device, dtype=wm.dtype)
    sigma_min, sigma_max = sigma_range

    per_probe = []
    sigmas_used = []
    for _ in range(k_probes):
        u = torch.rand(B, generator=gen) if gen is not None else torch.rand(B)
        sigma_draw = (sigma_min + (sigma_max - sigma_min) * u).to(device)              # [B]
        idx = torch.argmin((sigmas_full[None, :] - sigma_draw[:, None]).abs(), dim=1)  # [B]
        sigma = sigmas_full[idx].to(wm.dtype)                                          # [B]
        t = timesteps_full[idx]                                                        # [B]
        sigmas_used.append(sigma.float().cpu())

        if gen is not None:
            z = torch.randn(clean_full.shape, generator=gen).to(device=device, dtype=wm.dtype)
        else:
            z = torch.randn_like(clean_full)

        # Per-frame σ: zero on the cond slot, σ on the target slots only.
        sigma_per_frame = torch.zeros(B, 1, NUM_LATENT_FRAMES, 1, 1, device=device, dtype=wm.dtype)
        per_frame_t = torch.zeros(B, NUM_LATENT_FRAMES, device=device, dtype=wm.dtype)
        for slot in TARGET_LATENT_INDICES:
            sigma_per_frame[:, 0, slot, 0, 0] = sigma
            per_frame_t[:, slot] = t
        noisy = clean_full * (1 - sigma_per_frame) + z * sigma_per_frame

        video_tokens, time_emb, grid_sizes = wm._run_dit(
            noisy, per_frame_t, t5_embeddings, action_chunk, state, n_blocks=None,
        )
        pred = wm._dit_to_pred(video_tokens, time_emb, grid_sizes)

        # FM target velocity (world model's convention): v = noise - clean.
        target_v = z - clean_full
        diff = (pred[:, :, TARGET_LATENT_INDICES].float() - target_v[:, :, TARGET_LATENT_INDICES].float())
        per_probe.append(diff.pow(2).flatten(1).sum(dim=1).cpu())                       # [B] squared L2

    per_probe_t = torch.stack(per_probe, dim=0)                                         # [K, B]
    return {
        "score": per_probe_t.mean(dim=0),
        "per_probe": per_probe_t,
        "sigma_used": torch.stack(sigmas_used, dim=0),
        "k": k_probes,
    }
