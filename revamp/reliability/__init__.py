"""Reliability-aware mechanism (paper Sec. 3.3).

Scores every world-model prediction with two complementary signals and combines
them into one per-prediction reliability score used by the closed loop
(Sec. 3.4):

- Geometric faithfulness (intrinsic) — ``geometric.velocity_field_residual``:
  velocity-field self-consistency residual ``r_intr`` (Eq. 1). Catches
  predictions that have drifted off the training manifold (objects distort,
  change color, disappear).
- Action faithfulness (external) — ``action_faithfulness.implied_action_residual``:
  a visual probe (Depth Anything 3, gripper segmentation, Arun/Horn alignment)
  maps decoded RGB to end-effector pose and an inverse-dynamics MLP recovers
  the implied action chunk; ``r_ext`` is its L1 distance to the commanded chunk
  (Eq. 2). Catches plausible-looking predictions that do not follow the
  commanded action.

``score`` combines the two signals (Eq. 3), turns the score into the
imagined-update gate weight (Eq. 4) and into the real-rollout trigger (Eq. 5),
and calibrates the weights and the threshold.
"""
from __future__ import annotations

from revamp.reliability.geometric import velocity_field_residual
from revamp.reliability.action_faithfulness import (
    CameraCalibration,
    DepthPoseProbe,
    InverseDynamicsMLP,
    arun_horn,
    implied_action_residual,
    load_depth_anything3,
    probe_decoded,
    register_template,
)
from revamp.reliability.score import (
    ReliabilityCalibration,
    ReliabilityScorer,
    calibrate,
    combined_reliability,
    gate_weight,
    is_triggered,
)

__all__ = [
    "velocity_field_residual",
    "CameraCalibration",
    "DepthPoseProbe",
    "InverseDynamicsMLP",
    "arun_horn",
    "implied_action_residual",
    "load_depth_anything3",
    "probe_decoded",
    "register_template",
    "ReliabilityCalibration",
    "ReliabilityScorer",
    "calibrate",
    "combined_reliability",
    "gate_weight",
    "is_triggered",
]
