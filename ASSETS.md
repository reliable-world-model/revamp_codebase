# External Assets And Experiment Files

This source package stores launch configs and source code only. Datasets
and checkpoints are kept outside the repository and placed at the repository
root when experiments are run.

## Included In This Code Package

- `configs/turn_on_sink_faucet.json`: main launch config.
- `configs/world_model/*.source.yaml`: policy-improvement, dynamics-refit, and Q-refit configs.

## Expected Dataset Artifact

Place the dataset artifact at the repository root so these paths exist:

- `datasets/assets/t5_embedding/prompts/robocasa/TurnOnSinkFaucet.pt`: prompt embedding.
- `datasets/origin`: the 150 logged trajectories (`success/` 100, `failure/` 50) used for the policy, the world model, and the progress model.
- `datasets/targeted_rollouts`: targeted rollouts of the feedback rounds (`success/`, `failure/`), written by `run_targeted_collection.py`.
- `datasets/assets/reliability/`:
  - `calibration.json`: w_intr, w_ext, τ, λ (from `revamp.reliability.calibrate`).
  - `depth_anything_3/`: Depth Anything 3 weights.
  - `gripper_segmenter.pt`: TorchScript gripper segmentation model.
  - `gripper_template.npy`: gripper points in the end-effector frame.
  - `cameras.json`: per-camera intrinsics `K` and camera-to-base extrinsics `T_base_cam`.
  - `inverse_dynamics.pt`: inverse-dynamics MLP weights.

## Expected Checkpoint Artifact

Place the checkpoint artifact at the repository root so these paths exist:

- `checkpoints/stage2_q_initial`: world model with the fitted Q-head and contact head, used by the imagination and Q-gradient servers.
- `checkpoints/openpi_pi0_turn`: OpenPI π0 checkpoint after SFT on the logged trajectories.
- `checkpoints/world_model_initial`: dynamics checkpoint used as the resume point of the dynamics refit.
- `checkpoints/world_model_after_online_update`: dynamics checkpoint after the refit; resume point of the Q/contact refit and input of the rollout check.

## Checkpoint Packaging

Do not commit full checkpoint weights to the main source repository. This
source package's `.gitignore` ignores the whole `checkpoints/` and
`datasets/` trees by default.

Minimum checkpoint sets by entrypoint:

- `run_imagination_policy.py`: `stage2_q_initial` and `openpi_pi0_turn`.
- `run_targeted_collection.py`: `openpi_pi0_turn` (current policy).
- `run_world_model_update.py`: `world_model_initial`.
- `run_world_model_update.py --with-rollout-check`: `world_model_initial` and
  `world_model_after_online_update`.

## External Large Assets

Wan2.2-TI2V-5B weights and RoboCasa model assets are not copied by default.
Point to them with:

```bash
export REVAMP_WAN_CHECKPOINT_DIR=/path/to/Wan2.2-TI2V-5B
export REVAMP_ROBOCASA_ASSETS_ROOT=/path/to/robocasa_models/assets
```
