# Reliability-Aware World Model for Robot Policy Improvement

Code for the ICLR 2027 submission (TurnOnSinkFaucet, RoboCasa). REVAMP is the code name of the method (Python package `revamp/`).

<a href="https://reliable-world-model.github.io/reliable-world-model/"><img alt="Project Page" src="https://img.shields.io/badge/Project_Page-1565C0?style=for-the-badge"></a>

## Overview

A robot policy (OpenPI π0) is improved inside a learned video world model, and
the world model's own prediction says how far it can be trusted.

- **World model.** A Wan2.2-TI2V-5B DiT conditioned on the action chunk, the
  proprioceptive state, and a contact signal, with a dynamics head that
  predicts the next eight multi-view frames, a Q-head that pools the
  conditioning-frame tokens of the first 15 of 30 DiT blocks together with
  state and action-chunk embeddings (two three-layer MLPs, minimum used), and a
  small contact head that predicts contact at the chunk's last frame
  (`revamp/common/world_model.py`, `revamp/stage1_2_world_model/`).
- **Reliability score.** Every predicted chunk is scored by
  r = w_intr r_intr + w_ext r_ext: the velocity-field self-consistency residual
  (K = 4 noise pairs, σ ~ U[0.2, 0.8]) and the L1 distance between the
  commanded chunk and the action implied by the predicted frames (Depth
  Anything 3 + gripper segmentation + Arun/Horn alignment, then a three-layer
  inverse-dynamics MLP). Weights scale each term to an in-distribution mean of
  0.5; τ is the 95th percentile of r on held-out in-distribution predictions
  (`revamp/reliability/`).
- **Imagined cycle.** Rollouts start at logged real states and continue chunk
  by chunk until r > τ or the episode ends; the next chunk's state is the
  current state advanced by the commanded chunk and its contact input is the
  contact head's prediction. Each chunk's QAM adjoint-matching loss is weighted
  by exp(-λ r), only LoRA adapters (rank 32) on the action expert are trained,
  and the chunk that crosses τ is recorded with its logged start and action
  prefix (`revamp/stage3_qam/`).
- **Real cycle.** B = 20 triggers per round are drawn uniformly across bins of
  end-effector position (5 cm cells) and gripper state, at most two per bin.
  For each, the simulator resets to the logged start, replays the prefix,
  executes the flagged chunk, and the policy continues; the outcome is
  recorded either way (`revamp/online_update/collect_targeted_rollouts.py`).
  The world model, then the Q-head and contact head, are refit on the logged
  and targeted data.

## Installation

```bash
conda env create -f environment.yml
conda activate revamp
python -m pip install -r requirements.txt
```

Install CUDA-matched PyTorch/JAX first, then the remaining requirements. See
`ENVIRONMENT.md` for the setup order, GPU notes, `flash-attn`, and Depth
Anything 3.

## Usage

One feedback round:

```bash
python run_imagination_policy.py      # 1. policy improvement on imagined rollouts; writes the trigger buffer
python run_targeted_collection.py     # 2. select B triggers and collect their real outcomes
python run_world_model_update.py      # 3. refit the dynamics, then the Q-head and contact head
```

`run_imagination_policy.py` starts the Q-gradient server, the world-model
imagination server, and the OpenPI trainer. `run_world_model_update.py` trains
on `datasets/origin` together with `datasets/targeted_rollouts`.

## Repository Layout

```text
repository/
├── run_imagination_policy.py          # imagined cycle (policy improvement)
├── run_targeted_collection.py         # real cycle: targeted collection
├── run_world_model_update.py          # real cycle: dynamics, Q-head, contact-head refit
├── configs/
│   ├── turn_on_sink_faucet.json       # experiment, asset, and GPU config
│   └── world_model/*.source.yaml      # policy improvement, dynamics refit, Q refit
├── revamp/
│   ├── common/                        # world model, datasets, rewards, state advance
│   ├── reliability/                   # r_intr, r_ext (probe + inverse dynamics), calibration
│   ├── stage1_2_world_model/          # dynamics and Q/contact training
│   ├── stage3_qam/                    # imagination server, Q server, QAM trainer
│   ├── online_update/                 # targeted collection
│   └── launch/
├── third_party/                       # trimmed openpi/, robocasa/, wan/ subsets
├── ENVIRONMENT.md
├── ASSETS.md
├── environment.yml
└── requirements.txt
```

## Configuration

All launchers default to `configs/turn_on_sink_faucet.json`, which controls
dataset/checkpoint/output paths, third-party code paths, ports, and GPU
assignment. Adjust the `cuda_visible_devices` fields before running on a
different machine. Outputs are written to `outputs/turn_on_sink_faucet/`; for
machines without online logging, run with `export WANDB_MODE=offline`.

The included `third_party/` subsets are used by default, so the path variables
below are optional — set them only to point at a different local checkout:

```bash
export REVAMP_OPENPI_ROOT=$PWD/third_party/openpi
export REVAMP_OPENPI_CLIENT_SRC=$PWD/third_party/openpi/packages/openpi-client/src
export REVAMP_ROBOCASA_ROOT=$PWD/third_party/robocasa
export REVAMP_WAN_CODE_ROOT=$PWD/third_party/wan
```
