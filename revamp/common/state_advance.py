"""Advance the proprioceptive state by a commanded action chunk (imagination).

Layouts (LeRobot export order, PandaOmron):
    state  (17) = [base_pos(3), base_quat(4), eef_pos(3), eef_quat(4), gripper_qpos(2), contact(1)]
    action (12) = [base_motion(4), control_mode(1), eef_dpos(3), eef_drot(3), gripper(1)]

Arm actions are OSC deltas in [-1, 1] scaled by the controller output limits;
the base command is a planar velocity (x, y, yaw) plus torso. Quaternions are
[x, y, z, w].
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

BASE_POS, BASE_QUAT = slice(0, 3), slice(3, 7)
EEF_POS, EEF_QUAT = slice(7, 10), slice(10, 14)
GRIPPER_QPOS, CONTACT = slice(14, 16), 16
A_BASE, A_EEF_DPOS, A_EEF_DROT, A_GRIPPER = slice(0, 4), slice(5, 8), slice(8, 11), 11


@dataclass(frozen=True)
class AdvanceScales:
    pos: float = 0.05            # OSC output limit, metres per unit action
    rot: float = 0.5             # OSC output limit, radians per unit action
    base_lin: float = 0.0        # base controller output limit, metres per unit action
    base_yaw: float = 0.0        # base controller output limit, radians per unit action
    gripper_open: float = 0.04   # finger joint position when open
    gripper_closed: float = 0.0  # finger joint position when closed


def quat_mul(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    x1, y1, z1, w1 = q1.unbind(-1)
    x2, y2, z2, w2 = q2.unbind(-1)
    return torch.stack([
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
    ], dim=-1)


def axis_angle_to_quat(v: torch.Tensor) -> torch.Tensor:
    angle = v.norm(dim=-1, keepdim=True)
    axis = v / angle.clamp(min=1e-8)
    half = 0.5 * angle
    return torch.cat([axis * torch.sin(half), torch.cos(half)], dim=-1)


def yaw_of(q: torch.Tensor) -> torch.Tensor:
    x, y, z, w = q.unbind(-1)
    return torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def advance_proprio(
    state: torch.Tensor,          # [B, 17] raw (unnormalized)
    chunk: torch.Tensor,          # [B, H, 12] raw commanded chunk
    contact: torch.Tensor,        # [B] contact bit at the chunk's last frame
    scales: AdvanceScales = AdvanceScales(),
) -> torch.Tensor:
    """State at the end of the chunk, obtained by integrating the commanded actions."""
    s = state.clone()
    a = chunk.clamp(-1.0, 1.0)
    eef_quat = s[:, EEF_QUAT]
    base_quat = s[:, BASE_QUAT]
    for h in range(a.shape[1]):
        s[:, EEF_POS] = s[:, EEF_POS] + scales.pos * a[:, h, A_EEF_DPOS]
        eef_quat = quat_mul(axis_angle_to_quat(scales.rot * a[:, h, A_EEF_DROT]), eef_quat)
        yaw = yaw_of(base_quat)
        vx, vy, wz = a[:, h, 0], a[:, h, 1], a[:, h, 2]
        s[:, 0] = s[:, 0] + scales.base_lin * (torch.cos(yaw) * vx - torch.sin(yaw) * vy)
        s[:, 1] = s[:, 1] + scales.base_lin * (torch.sin(yaw) * vx + torch.cos(yaw) * vy)
        dyaw = torch.zeros_like(a[:, h, A_EEF_DROT]); dyaw[:, 2] = scales.base_yaw * wz
        base_quat = quat_mul(axis_angle_to_quat(dyaw), base_quat)
    s[:, EEF_QUAT] = eef_quat / eef_quat.norm(dim=-1, keepdim=True)
    s[:, BASE_QUAT] = base_quat / base_quat.norm(dim=-1, keepdim=True)
    closed = (a[:, -1, A_GRIPPER] > 0).unsqueeze(-1)
    s[:, GRIPPER_QPOS] = torch.where(
        closed,
        torch.full_like(s[:, GRIPPER_QPOS], scales.gripper_closed),
        torch.tensor([scales.gripper_open, -scales.gripper_open], device=s.device, dtype=s.dtype).expand_as(s[:, GRIPPER_QPOS]),
    )
    s[:, CONTACT] = contact.to(s.dtype)
    return s


def unnormalize_state(state_norm: torch.Tensor, state_min: torch.Tensor, state_max: torch.Tensor) -> torch.Tensor:
    """Inverse of the dataset's min-max normalization to [-1, 1]."""
    return (state_norm + 1) * 0.5 * (state_max - state_min) + state_min


def normalize_state(state: torch.Tensor, state_min: torch.Tensor, state_max: torch.Tensor) -> torch.Tensor:
    state = torch.maximum(torch.minimum(state, state_max), state_min)
    return (state - state_min) / (state_max - state_min + 1e-8) * 2 - 1
