"""Action faithfulness — external behavioral verification (Eq. 2).

Two auxiliary networks, both fit on real trajectories only, check that a
prediction follows the commanded action chunk:

- ``DepthPoseProbe``: decoded RGB -> end-effector pose. Depth Anything 3
  gives metric depth, a gripper segmentation model gives the gripper mask, the
  masked depth is back-projected with each camera's calibration and fused in
  the robot base frame, and a gripper template point set is registered to it
  with Arun/Horn rigid alignment (closed-form SVD, iterated with
  nearest-neighbour correspondences).
- ``InverseDynamicsMLP``: a three-layer MLP mapping consecutive states
  (s_t, s_{t+1}) to the action a_t, trained on the demonstration trajectories.

At inference:

    world model                      ->  H predicted frames per camera
    probe(frame_k)                   ->  ŝ_k  (end-effector pose per frame)
    inverse dynamics(ŝ_k, ŝ_{k+1})   ->  â_k  (implied action chunk ā_impl)
    r_ext = || ā_impl - ā ||_1
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional

import torch
import torch.nn as nn

# --------------------------------------------------------------------------- #
# Geometry
# --------------------------------------------------------------------------- #
def backproject(depth: torch.Tensor, mask: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
    """Masked metric depth [H, W] -> camera-frame points [N, 3] with intrinsics K [3, 3]."""
    v, u = torch.nonzero(mask, as_tuple=True)
    z = depth[v, u]
    keep = z > 0
    u, v, z = u[keep].float(), v[keep].float(), z[keep]
    x = (u - K[0, 2]) * z / K[0, 0]
    y = (v - K[1, 2]) * z / K[1, 1]
    return torch.stack([x, y, z], dim=-1)


def transform_points(T: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
    """Apply a homogeneous transform T [4, 4] to points [N, 3]."""
    return points @ T[:3, :3].T + T[:3, 3]


def arun_horn(src: torch.Tensor, dst: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Least-squares rigid transform (R, t) with R @ src_i + t ≈ dst_i (Arun et al. 1987; Horn 1987)."""
    mu_s, mu_d = src.mean(dim=0), dst.mean(dim=0)
    H = (src - mu_s).T @ (dst - mu_d)
    U, _, Vt = torch.linalg.svd(H)
    D = torch.eye(3, dtype=src.dtype, device=src.device)
    D[2, 2] = torch.sign(torch.det(Vt.T @ U.T))
    R = Vt.T @ D @ U.T
    t = mu_d - R @ mu_s
    return R, t


def register_template(
    template: torch.Tensor,
    observed: torch.Tensor,
    R0: Optional[torch.Tensor] = None,
    t0: Optional[torch.Tensor] = None,
    iters: int = 20,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Register the gripper template [M, 3] to observed points [N, 3]; Arun/Horn step per iteration."""
    R = torch.eye(3, dtype=template.dtype, device=template.device) if R0 is None else R0
    t = observed.mean(dim=0) - template.mean(dim=0) if t0 is None else t0
    for _ in range(iters):
        moved = template @ R.T + t
        nn_idx = torch.cdist(moved, observed).argmin(dim=1)
        R, t = arun_horn(template, observed[nn_idx])
    return R, t


def quaternion_to_matrix(q: torch.Tensor) -> torch.Tensor:
    """Unit quaternion [x, y, z, w] -> rotation matrix [3, 3]."""
    x, y, z, w = (q / q.norm()).unbind(-1)
    return torch.stack([
        torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)]),
        torch.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)]),
        torch.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]),
    ])


def matrix_to_quaternion(R: torch.Tensor) -> torch.Tensor:
    """Rotation matrix [3, 3] -> unit quaternion [x, y, z, w]."""
    m = R
    tr = m[0, 0] + m[1, 1] + m[2, 2]
    if tr > 0:
        s = torch.sqrt(tr + 1.0) * 2
        q = torch.stack([(m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s, 0.25 * s])
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = torch.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
        q = torch.stack([0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s, (m[2, 1] - m[1, 2]) / s])
    elif m[1, 1] > m[2, 2]:
        s = torch.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
        q = torch.stack([(m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s, (m[0, 2] - m[2, 0]) / s])
    else:
        s = torch.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
        q = torch.stack([(m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s, (m[1, 0] - m[0, 1]) / s])
    return q / q.norm()


# --------------------------------------------------------------------------- #
# Visual probe: RGB -> end-effector pose
# --------------------------------------------------------------------------- #
@dataclass
class CameraCalibration:
    K: torch.Tensor            # [3, 3] intrinsics
    T_base_cam: torch.Tensor   # [4, 4] camera-to-base extrinsics


class DepthPoseProbe:
    """Depth Anything 3 + gripper segmentation + Arun/Horn template alignment.

    Args:
        depth_model: callable RGB [3, H, W] in [0, 1] -> metric depth [H, W]
            (Depth Anything 3).
        segmenter: callable RGB [3, H, W] -> boolean gripper mask [H, W].
        gripper_template: [M, 3] points of the gripper model in the
            end-effector frame.
        cameras: per-camera calibration, keyed like the world-model outputs
            ("high", "left", "right").
    """

    def __init__(
        self,
        depth_model: Callable[[torch.Tensor], torch.Tensor],
        segmenter: Callable[[torch.Tensor], torch.Tensor],
        gripper_template: torch.Tensor,
        cameras: Dict[str, CameraCalibration],
        icp_iters: int = 20,
        max_points: int = 4096,
    ):
        self.depth_model = depth_model
        self.segmenter = segmenter
        self.template = gripper_template.float()
        self.cameras = cameras
        self.icp_iters = icp_iters
        self.max_points = max_points

    @torch.no_grad()
    def __call__(
        self,
        frames: Dict[str, torch.Tensor],
        init: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """frames: {cam: RGB [3, H, W] in [0, 1]} -> (pose [7] = pos(3) | quat(4), (R, t))."""
        points = []
        for cam, rgb in frames.items():
            calib = self.cameras[cam]
            depth = self.depth_model(rgb)
            mask = self.segmenter(rgb)
            pts = backproject(depth, mask, calib.K.to(depth.device))
            points.append(transform_points(calib.T_base_cam.to(depth.device), pts))
        observed = torch.cat(points, dim=0)
        if observed.shape[0] > self.max_points:
            observed = observed[torch.randperm(observed.shape[0], device=observed.device)[: self.max_points]]
        R0, t0 = init if init is not None else (None, None)
        R, t = register_template(self.template.to(observed.device), observed, R0, t0, iters=self.icp_iters)
        return torch.cat([t, matrix_to_quaternion(R)]), (R, t)


def load_depth_anything3(checkpoint: str, device: str = "cuda") -> Callable[[torch.Tensor], torch.Tensor]:
    """Return a metric-depth callable backed by Depth Anything 3."""
    import numpy as np
    from depth_anything_3.api import DepthAnything3  # external dependency, see ENVIRONMENT.md

    model = DepthAnything3.from_pretrained(checkpoint).to(device=device)

    @torch.no_grad()
    def depth_fn(rgb: torch.Tensor) -> torch.Tensor:
        image = (rgb.clamp(0.0, 1.0).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        prediction = model.inference([image])
        return torch.as_tensor(np.asarray(prediction.depth[0]), dtype=torch.float32, device=rgb.device)

    return depth_fn


# --------------------------------------------------------------------------- #
# Inverse dynamics: (s_t, s_{t+1}) -> a_t
# --------------------------------------------------------------------------- #
class InverseDynamicsMLP(nn.Module):
    """Three-layer MLP f(s_t, s_{t+1}) -> a_t, trained on the demonstration trajectories."""

    def __init__(self, state_ee_dim: int = 7, action_dim: int = 12, hidden: int = 256):
        super().__init__()
        in_dim = state_ee_dim * 3  # s_t, s_{t+1}, Δ
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, action_dim),
        )

    def forward(self, s_k: torch.Tensor, s_kp1: torch.Tensor) -> torch.Tensor:
        delta = s_kp1 - s_k
        return self.net(torch.cat([s_k, s_kp1, delta], dim=-1))


# --------------------------------------------------------------------------- #
# Inference chain
# --------------------------------------------------------------------------- #
@torch.no_grad()
def probe_decoded(
    probe: DepthPoseProbe,
    decoded_per_cam: Dict[str, torch.Tensor],
    start_pose: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Run the probe over the predicted frames of one sample -> per-frame pose [n_frames, 7].

    Args:
        decoded_per_cam: {'left': [3, n_frames, h, w], 'right': ..., 'high': ...} in [0, 1].
        start_pose: end-effector pose [7] at the start of the chunk, used to
            initialise the registration of the first frame.
    """
    n_frames = decoded_per_cam["left"].shape[1]
    poses = []
    init = None if start_pose is None else (quaternion_to_matrix(start_pose[3:7]), start_pose[:3])
    for f in range(n_frames):
        frames = {cam: decoded_per_cam[cam][:, f].clamp(0.0, 1.0) for cam in decoded_per_cam}
        pose, init = probe(frames, init)  # warm-start the registration from the previous frame
        poses.append(pose)
    return torch.stack(poses, dim=0)


@torch.no_grad()
def implied_action_residual(
    probe: DepthPoseProbe,
    id_model: InverseDynamicsMLP,
    decoded_per_cam: Dict[str, torch.Tensor],
    action_chunk_commanded: torch.Tensor,  # (H, action_dim)
    start_pose: Optional[torch.Tensor] = None,
) -> dict:
    """External score r_ext (Eq. 2): L1 distance between the implied and the commanded chunk."""
    s_chunk = probe_decoded(probe, decoded_per_cam, start_pose)   # (H, 7)
    implied = id_model(s_chunk[:-1], s_chunk[1:])                 # (H-1, action_dim)
    cmd = action_chunk_commanded[: implied.shape[0]].to(implied.device)
    diff = (implied - cmd).abs()
    return {
        "score": diff.sum().item(),
        "per_step_l1": diff.sum(dim=-1).detach().cpu(),
        "implied": implied.detach().cpu(),
        "commanded": cmd.detach().cpu(),
    }
