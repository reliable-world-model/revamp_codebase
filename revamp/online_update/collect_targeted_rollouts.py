"""Targeted real-data collection at flagged interactions (Sec. 3.4).

Reads the candidate buffer written by imagined rollouts (one row per trigger:
logged start s_0, action prefix ā_{0:k-1}, flagged chunk ā_π(ŝ_k)), selects B
triggers per round, and for each one

  1. restores the logged start s_0 (the simulator resets exactly to s_0),
  2. replays the prefix ā_{0:k-1},
  3. executes the flagged chunk first, so the real outcome answers the action
     the model was flagged on,
  4. continues under the current policy until success, failure, or the step cap,

and records the trajectory with its outcome (successful and failed rollouts
alike). Trigger selection bins the buffer by end-effector position (5 cm
cells) and gripper state and draws B triggers uniformly across the non-empty
bins, at most two per bin; if fewer exist, all are used and the count is
reported.
"""

from __future__ import annotations

import argparse
import collections
import gzip
import importlib.util
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import time
from typing import Any

import numpy as np
import pandas as pd

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
os.environ.setdefault("ROBOCASA_MJCF_TMPDIR", "/tmp/robocasa_mjcf_tmp")


try:
    from omegaconf import OmegaConf
except ImportError:  # pragma: no cover - fallback for lean envs
    import yaml

    class _AttrDict(dict):
        def __getattr__(self, name):
            try:
                return self[name]
            except KeyError as exc:
                raise AttributeError(name) from exc

    def _attrify(x):
        if isinstance(x, dict):
            return _AttrDict({k: _attrify(v) for k, v in x.items()})
        if isinstance(x, list):
            return [_attrify(v) for v in x]
        return x

    class OmegaConf:  # type: ignore[no-redef]
        @staticmethod
        def load(path):
            with open(path, "r", encoding="utf-8") as f:
                return _attrify(yaml.safe_load(f))


_RELEASE_ROOT = Path(os.environ.get("REVAMP_RELEASE_ROOT", Path.cwd())).resolve()
ROBOCASA_ROOT = Path(os.environ.get("REVAMP_ROBOCASA_ROOT", _RELEASE_ROOT / "third_party" / "robocasa")).resolve()
ROBOSUITE_ROOT_RAW = os.environ.get("REVAMP_ROBOSUITE_ROOT", "")
ROBOSUITE_ROOT = Path(ROBOSUITE_ROOT_RAW).resolve() if ROBOSUITE_ROOT_RAW else None
ROBOCASA_SCRIPTS = ROBOCASA_ROOT / "robocasa" / "scripts"
DEFAULT_OPENPI_ROOT = Path(os.environ.get("REVAMP_OPENPI_ROOT", _RELEASE_ROOT / "third_party" / "openpi")).resolve()
if ROBOSUITE_ROOT is not None and str(ROBOSUITE_ROOT) not in sys.path:
    sys.path.insert(0, str(ROBOSUITE_ROOT))
if str(ROBOCASA_ROOT) not in sys.path:
    sys.path.insert(0, str(ROBOCASA_ROOT))
if str(ROBOCASA_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(ROBOCASA_SCRIPTS))
for _openpi_path in (
    DEFAULT_OPENPI_ROOT / "src",
    DEFAULT_OPENPI_ROOT / "packages" / "openpi-client" / "src",
):
    if _openpi_path.exists() and str(_openpi_path) not in sys.path:
        sys.path.insert(0, str(_openpi_path))

import robocasa.utils.lerobot_utils as LU  # noqa: E402
from generate_contact_features import (  # noqa: E402
    extract_contact_features_for_current_state,
    get_finger_geom_ids,
    get_robot_geom_ids,
    write_contact_features,
)
from generate_fail_data import (  # noqa: E402
    GenConfig,
    SourceEpisode,
    build_env,
    build_output_features,
    check_success,
    copy_source_metadata_templates,
    ensure_dir,
    extract_state_from_obs,
    finalize_output_dataset,
    get_observations,
    json_dump,
    restore_env_state,
)
from openpi_client import image_tools  # noqa: E402
from openpi_client import websocket_client_policy as _websocket_client_policy  # noqa: E402


DEFAULT_SRC_ROOT = str(_RELEASE_ROOT / "datasets/origin")
DEFAULT_OUT_ROOT = str(_RELEASE_ROOT / "datasets/targeted_rollouts")
DEFAULT_TRIGGERS = str(_RELEASE_ROOT / "outputs/turn_on_sink_faucet/triggers/triggers.jsonl")
DEFAULT_CONFIG = str(
    _RELEASE_ROOT
    / "configs/world_model/phase3_qam_turn_openpi_imagination.source.yaml"
)


def _log(msg: str) -> None:
    print(f"[branch_collect] {time.strftime('%Y-%m-%d %H:%M:%S')} {msg}", flush=True)


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _wait_for_port(host: str, port: int, timeout_s: float) -> None:
    deadline = time.time() + timeout_s
    last_error: Exception | None = None
    while time.time() < deadline:
        try:
            with socket.create_connection((host, port), timeout=2.0):
                return
        except OSError as exc:
            last_error = exc
            time.sleep(1.0)
    raise TimeoutError(f"Policy server did not open {host}:{port}") from last_error


def _start_policy_server(
    *,
    openpi_root: Path,
    config_name: str,
    checkpoint: Path,
    prompt: str,
    port: int,
    server_gpu: str | None,
) -> subprocess.Popen:
    env = os.environ.copy()
    env.setdefault("MUJOCO_GL", "egl")
    env.setdefault("PYOPENGL_PLATFORM", "egl")
    if server_gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = server_gpu

    cmd = [
        sys.executable,
        str(openpi_root / "scripts" / "serve_policy.py"),
        f"--port={port}",
        f"--default-prompt={prompt}",
        "policy:checkpoint",
        f"--policy.config={config_name}",
        f"--policy.dir={checkpoint}",
    ]
    _log("starting policy server: " + " ".join(cmd))
    return subprocess.Popen(cmd, cwd=str(openpi_root), env=env)


def _load_openpi_robocasa_main(openpi_root: Path):
    src = openpi_root / "src"
    client_src = openpi_root / "packages" / "openpi-client" / "src"
    for p in (src, client_src):
        if p.exists() and str(p) not in sys.path:
            sys.path.insert(0, str(p))

    module_path = openpi_root / "examples" / "robocasa" / "main.py"
    spec = importlib.util.spec_from_file_location("openpi_robocasa_eval_main", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load OpenPI RoboCasa helper from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _existing_episode_count(dataset_root: Path) -> int:
    episodes_path = dataset_root / "meta" / "episodes.jsonl"
    if not episodes_path.exists():
        return 0
    return sum(1 for line in episodes_path.read_text().splitlines() if line.strip())


def _has_existing_dataset(dataset_root: Path) -> bool:
    episodes_path = dataset_root / "meta" / "episodes.jsonl"
    return episodes_path.exists() and _existing_episode_count(dataset_root) > 0


class _DatasetContext:
    def __init__(self, dataset_dir: Path, env_meta: dict[str, Any]):
        self.dataset_dir = dataset_dir
        self.env_meta = env_meta
        self.info = _read_json(dataset_dir / "meta" / "info.json")


def _load_env_meta(dataset_dir: Path) -> dict[str, Any]:
    try:
        return LU.get_env_metadata(dataset_dir)
    except Exception as exc:
        raise RuntimeError(
            f"{dataset_dir}/extras/dataset_meta.json does not expose env_args; "
            "the source dataset should carry RoboCasa env_args directly."
        ) from exc


def _create_or_open_output_dataset(
    *,
    src_dataset_dir: Path,
    out_dataset_dir: Path,
    cfg: GenConfig,
    overwrite: bool,
    env_meta: dict[str, Any],
):
    if out_dataset_dir.exists() and overwrite:
        shutil.rmtree(out_dataset_dir)

    reader = _DatasetContext(src_dataset_dir, env_meta)
    if _has_existing_dataset(out_dataset_dir):
        dataset = LU.LerobotDatasetWrapper(
            repo_id=out_dataset_dir.name,
            root=out_dataset_dir,
            download_videos=cfg.save_video,
        )
        if cfg.save_video and (cfg.image_writer_threads or cfg.image_writer_processes):
            dataset.start_image_writer(
                num_processes=cfg.image_writer_processes,
                num_threads=cfg.image_writer_threads,
            )
        return dataset, reader

    if out_dataset_dir.exists():
        shutil.rmtree(out_dataset_dir)
    features = build_output_features(reader.info, include_videos=cfg.save_video)
    dataset = LU.LerobotDatasetWrapper.create(
        repo_id=out_dataset_dir.name,
        root=out_dataset_dir,
        fps=reader.info["fps"],
        robot_type=reader.info.get("robot_type"),
        features=features,
        use_videos=cfg.save_video,
        image_writer_threads=cfg.image_writer_threads if cfg.save_video else 0,
        image_writer_processes=cfg.image_writer_processes if cfg.save_video else 0,
    )
    copy_source_metadata_templates(src_dataset_dir, out_dataset_dir)
    extras_dir = out_dataset_dir / "extras"
    ensure_dir(extras_dir)
    source_meta = _read_json(src_dataset_dir / "extras" / "dataset_meta.json")
    source_meta["env_args"] = env_meta
    source_meta["targeted_collection"] = {
        "source_dataset_dir": str(src_dataset_dir.resolve()),
        "generator": "revamp/online_update/collect_targeted_rollouts.py",
        "save_video": cfg.save_video,
    }
    json_dump(source_meta, extras_dir / "dataset_meta.json")
    return dataset, reader


def _load_source_episode(src_failure_dir: Path, episode_index: int, env_meta: dict[str, Any]) -> SourceEpisode:
    ep_meta = LU.get_episode_meta(src_failure_dir, episode_index)
    states = LU.get_episode_states(src_failure_dir, episode_index)
    actions = LU.get_episode_actions(src_failure_dir, episode_index)
    model_xml = None
    try:
        model_xml = LU.get_episode_model_xml(src_failure_dir, episode_index)
    except FileNotFoundError:
        pass
    return SourceEpisode(
        episode_index=episode_index,
        episode_id=f"episode_{episode_index:06d}",
        actions=actions.astype(np.float32),
        model_xml=model_xml,
        initial_state=states[0].astype(np.float64),
        sim_states=states.astype(np.float64),
        ep_meta=ep_meta,
        env_meta=env_meta,
        task_name=env_meta["env_name"],
        layout_id=ep_meta.get("layout_id"),
        style_id=ep_meta.get("style_id"),
        instruction=ep_meta.get("lang"),
    )


def _current_model_xml(env) -> str:
    if hasattr(env, "model") and hasattr(env.model, "get_xml"):
        return env.model.get_xml()
    if hasattr(env, "sim") and hasattr(env.sim, "model") and hasattr(env.sim.model, "get_xml"):
        return env.sim.model.get_xml()
    raise RuntimeError("Could not get current MuJoCo model XML from env")


def _episode_parquet_path(dataset_dir: Path, episode_index: int) -> Path:
    info = _read_json(dataset_dir / "meta" / "info.json")
    chunks_size = int(info.get("chunks_size", 1000))
    episode_chunk = episode_index // chunks_size
    template = info.get("data_path", "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet")
    return dataset_dir / template.format(episode_chunk=episode_chunk, episode_index=episode_index)


def _render_frames(env, camera_names: tuple[str, ...], height: int, width: int) -> dict[str, np.ndarray]:
    frames: dict[str, np.ndarray] = {}
    for camera_name in camera_names:
        frame = env.sim.render(height=height, width=width, camera_name=camera_name)[::-1]
        frames[camera_name] = np.ascontiguousarray(frame).astype(np.uint8)
    return frames


def _add_frame_checked(dataset, frame: dict[str, Any], task_lang: str) -> None:
    for key, feature in dataset.features.items():
        if key.startswith("annotation.") and key not in frame:
            dtype = np.int64 if str(feature.get("dtype", "int64")).startswith("int") else np.float32
            shape = tuple(feature.get("shape") or (1,))
            frame[key] = np.zeros(shape, dtype=dtype)
    try:
        dataset.add_frame(frame, task=task_lang)
    except ValueError as exc:
        shapes = {
            key: {
                "shape": tuple(np.asarray(value).shape),
                "dtype": str(np.asarray(value).dtype),
            }
            for key, value in frame.items()
        }
        raise ValueError(f"{exc}\nframe shapes: {shapes}") from exc


def _policy_image(frame: np.ndarray, resize_size: int) -> np.ndarray:
    return image_tools.convert_to_uint8(image_tools.resize_with_pad(frame, resize_size, resize_size))


def _policy_state_from_obs(obs: Any, fallback_state: np.ndarray) -> np.ndarray:
    """Match the state order used by openpi/examples/robocasa/main.py.

    The LeRobot rows used by the world model keep RoboCasa's dataset order:
    base pose, eef relative pose, gripper. The deployed OpenPI evaluator,
    however, feeds pi0 eef relative pose first. Keep those two contracts
    separate so branch data remains WM-compatible while pi0 sees familiar
    observations.
    """
    if isinstance(obs, dict):
        raw_keys = (
            "robot0_base_to_eef_pos",
            "robot0_base_to_eef_quat",
            "robot0_base_pos",
            "robot0_base_quat",
            "robot0_gripper_qpos",
        )
        if all(k in obs for k in raw_keys):
            return np.concatenate([np.asarray(obs[k]).reshape(-1) for k in raw_keys], axis=0).astype(np.float32)

        gym_keys = (
            "state.end_effector_position_relative",
            "state.end_effector_rotation_relative",
            "state.base_position",
            "state.base_rotation",
            "state.gripper_qpos",
        )
        if all(k in obs for k in gym_keys):
            return np.concatenate([np.asarray(obs[k]).reshape(-1) for k in gym_keys], axis=0).astype(np.float32)

    return np.asarray(fallback_state, dtype=np.float32)


def _datanew_action_to_env_action(action: np.ndarray) -> np.ndarray:
    """Convert stored ViVa/datanew action order to RoboCasa/OpenPI action order."""
    a = np.asarray(action, dtype=np.float32).reshape(-1)
    if a.shape[0] < 12:
        return a
    return np.concatenate(
        [
            a[5:8],    # end_effector_position
            a[8:11],   # end_effector_rotation
            a[11:12],  # gripper_close
            a[0:4],    # base_motion
            a[4:5],    # control_mode
        ],
        axis=0,
    ).astype(np.float32)


def _env_action_to_datanew_action(action: np.ndarray) -> np.ndarray:
    """Convert RoboCasa/OpenPI action order to the WM dataset action order."""
    a = np.asarray(action, dtype=np.float32).reshape(-1)
    if a.shape[0] < 12:
        return a
    return np.concatenate(
        [
            a[7:11],   # base_motion
            a[11:12],  # control_mode
            a[0:3],    # end_effector_position
            a[3:6],    # end_effector_rotation
            a[6:7],    # gripper_close
        ],
        axis=0,
    ).astype(np.float64)


def _reset_episode_start(env, ep: SourceEpisode) -> None:
    if ep.model_xml:
        restore_env_state(env, ep.model_xml, ep.ep_meta, ep.initial_state)
        return

    if hasattr(env, "set_ep_meta"):
        env.set_ep_meta(ep.ep_meta)
    if hasattr(env, "set_attrs_from_ep_meta"):
        env.set_attrs_from_ep_meta(ep.ep_meta)
    if hasattr(env, "hard_reset"):
        env.hard_reset = True
    if hasattr(env, "deterministic_reset"):
        env.deterministic_reset = False
    env.reset()
    if hasattr(env, "update_sites"):
        env.update_sites()
    if hasattr(env, "update_state"):
        env.update_state()


def _query_policy(
    client,
    *,
    frames: dict[str, np.ndarray],
    state: np.ndarray,
    task_lang: str,
    resize_size: int,
    replan_steps: int,
) -> collections.deque:
    element = {
        "observation/image": _policy_image(frames["robot0_agentview_left"], resize_size),
        "observation/wrist_image": _policy_image(frames["robot0_eye_in_hand"], resize_size),
        "observation/state": np.asarray(state, dtype=np.float32),
        "prompt": task_lang,
    }
    action_chunk = client.infer(element)["actions"]
    if len(action_chunk) < replan_steps:
        raise RuntimeError(
            f"Policy returned {len(action_chunk)} actions, shorter than replan_steps={replan_steps}"
        )
    return collections.deque(np.asarray(a, dtype=np.float32) for a in action_chunk[:replan_steps])


def _safe_step(env, action: np.ndarray):
    out = env.step(action)
    if len(out) == 4:
        obs, reward, done, info = out
    elif len(out) == 5:
        obs, reward, terminated, truncated, info = out
        done = bool(terminated or truncated)
    else:
        raise RuntimeError(f"Unexpected env.step output length: {len(out)}")
    return obs, float(reward), bool(done), info


def _resolve_contact_sets(env):
    left_ids, right_ids = get_finger_geom_ids(env, arm="right")
    robot_ids = get_robot_geom_ids(env)
    return left_ids, right_ids, robot_ids


def _current_contact(env, contact_sets) -> np.ndarray:
    left_ids, right_ids, robot_ids = contact_sets
    return extract_contact_features_for_current_state(
        env,
        left_finger_geom_ids=left_ids,
        right_finger_geom_ids=right_ids,
        robot_geom_ids=robot_ids,
    )


# --------------------------------------------------------------------------- #
# Trigger selection
# --------------------------------------------------------------------------- #
def load_triggers(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def select_triggers(
    triggers: list[dict[str, Any]],
    budget: int,
    rng: np.random.Generator,
    cell: float = 0.05,
    per_bin: int = 2,
) -> list[dict[str, Any]]:
    """Bin by end-effector position (5 cm cells) and gripper state; draw uniformly across bins, ≤2 per bin."""
    bins: dict[tuple, list[dict[str, Any]]] = collections.defaultdict(list)
    for t in triggers:
        key = tuple(int(np.floor(v / cell)) for v in t["ee_pos"]) + (bool(t["gripper_closed"]),)
        bins[key].append(t)
    pools = {k: list(rng.permutation(len(v))) for k, v in bins.items()}
    taken = collections.Counter()
    selected: list[dict[str, Any]] = []
    while len(selected) < budget:
        open_bins = [k for k in bins if taken[k] < per_bin and pools[k]]
        if not open_bins:
            break
        k = open_bins[int(rng.integers(len(open_bins)))]
        selected.append(bins[k][pools[k].pop()])
        taken[k] += 1
    return selected


# --------------------------------------------------------------------------- #
# Logged start s_0
# --------------------------------------------------------------------------- #
def _episode_lengths(dataset_dir: Path) -> list[tuple[int, int]]:
    rows = []
    with (dataset_dir / "meta" / "episodes.jsonl").open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                e = json.loads(line)
                rows.append((int(e["episode_index"]), int(e["length"])))
    return sorted(rows)


def locate_logged_start(data_dirs: list[Path], global_index: int) -> tuple[Path, int, int]:
    """Dataset row (concatenated over data_dirs, as in WorldModelDataset) -> (dataset dir, episode, frame)."""
    offset = 0
    for d in data_dirs:
        for episode_index, length in _episode_lengths(d):
            if global_index < offset + length:
                return d, episode_index, global_index - offset
            offset += length
    raise IndexError(f"global_index {global_index} is outside the logged data")


def restore_logged_start(env, ep: SourceEpisode, frame: int) -> None:
    """The simulator resets exactly to s_0: episode reset, then the logged MuJoCo state at the start frame."""
    _reset_episode_start(env, ep)
    env.sim.set_state_from_flattened(np.asarray(ep.sim_states[frame], dtype=np.float64))
    env.sim.forward()
    if hasattr(env, "update_state"):
        env.update_state()


# --------------------------------------------------------------------------- #
# Targeted rollout
# --------------------------------------------------------------------------- #
def targeted_rollout(
    *,
    env,
    ep: SourceEpisode,
    start_frame: int,
    prefix: list[np.ndarray],
    flagged_chunk: np.ndarray,
    client,
    task_lang: str,
    cfg: GenConfig,
    resize_size: int,
    chunk_length: int,
    max_steps: int,
    post_success_steps: int,
) -> dict[str, Any]:
    """Restore s_0, replay the prefix, execute the flagged chunk, continue under the policy; record everything."""
    restore_logged_start(env, ep, start_frame)
    current_obs = get_observations(env)
    contact_sets = _resolve_contact_sets(env)
    scripted = collections.deque(
        _datanew_action_to_env_action(a)
        for chunk in list(prefix) + [flagged_chunk]
        for a in np.asarray(chunk, dtype=np.float32)
    )
    n_scripted = len(scripted)
    action_plan: collections.deque = collections.deque()
    traj = {
        "states": [], "actions": [], "sim_states": [], "contacts": [],
        "frames": {cam: [] for cam in cfg.camera_names}, "handle_states": [], "success_trace": [],
    }
    first_success_step = None
    for step in range(max_steps):
        obs_state = extract_state_from_obs(current_obs)
        frames = _render_frames(env, cfg.camera_names, cfg.camera_height, cfg.camera_width)
        if scripted:
            action = np.asarray(scripted.popleft(), dtype=np.float32)
        else:
            if not action_plan:
                action_plan = _query_policy(
                    client,
                    frames=frames,
                    state=_policy_state_from_obs(current_obs, obs_state),
                    task_lang=task_lang,
                    resize_size=resize_size,
                    replan_steps=chunk_length,
                )
            action = np.asarray(action_plan.popleft(), dtype=np.float32)

        traj["states"].append(np.asarray(obs_state, dtype=np.float64))
        traj["actions"].append(_env_action_to_datanew_action(action))
        traj["sim_states"].append(np.array(env.sim.get_state().flatten(), dtype=np.float64))
        traj["contacts"].append(_current_contact(env, contact_sets))
        for cam, frame in frames.items():
            traj["frames"][cam].append(frame)

        next_obs, _reward, done, _info = _safe_step(env, action)
        success_now = bool(check_success(env))
        if hasattr(env, "update_state"):
            env.update_state()
        traj["success_trace"].append(success_now)
        traj["handle_states"].append(env.sink.get_handle_state(env) if hasattr(env, "sink") else {})
        if success_now and first_success_step is None:
            first_success_step = step
        current_obs = get_observations(env) or next_obs
        if first_success_step is not None and step >= first_success_step + post_success_steps:
            break
        if done:
            break
    traj["success"] = first_success_step is not None
    traj["first_success_step"] = first_success_step
    traj["num_scripted_steps"] = n_scripted
    return traj


def write_rollout_episode(
    *,
    dataset,
    out_dataset_dir: Path,
    env,
    ep: SourceEpisode,
    traj: dict[str, Any],
    trigger: dict[str, Any],
    task_lang: str,
    cfg: GenConfig,
) -> int:
    episode_index = _existing_episode_count(out_dataset_dir)
    for i, action in enumerate(traj["actions"]):
        frame = {"observation.state": traj["states"][i], "action": np.asarray(action, dtype=np.float64)}
        if cfg.save_video:
            for cam_name in cfg.camera_names:
                frame[f"observation.images.{cam_name}"] = traj["frames"][cam_name][i]
        _add_frame_checked(dataset, frame, task_lang)
    dataset.save_episode()

    ep_dir = out_dataset_dir / "extras" / f"episode_{episode_index:06d}"
    ensure_dir(ep_dir)
    contacts = np.asarray(traj["contacts"], dtype=np.float32)
    np.savez_compressed(ep_dir / "states.npz", states=np.asarray(traj["sim_states"], dtype=np.float64))
    np.save(ep_dir / "contact_features.npy", contacts)
    write_contact_features(episode_dir=ep_dir, features=contacts, overwrite=True, write_json_flag=True)
    json_dump(ep.ep_meta, ep_dir / "ep_meta.json")
    json_dump(traj["handle_states"], ep_dir / "handle_states.json")
    with gzip.open(ep_dir / "model.xml.gz", "wb") as f:
        f.write((ep.model_xml or _current_model_xml(env)).encode("utf-8"))
    json_dump(
        {
            "env_name": ep.task_name,
            "episode_idx": int(episode_index),
            "success": bool(traj["success"]),
            "first_success_step": traj["first_success_step"],
            "num_steps": int(len(traj["actions"])),
            "task_lang": task_lang,
            "logged_start": {"source_episode_index": int(ep.episode_index), "frame": int(trigger["start_frame"])},
            "trigger": {k: trigger[k] for k in ("global_index", "depth", "r", "ee_pos", "gripper_closed")},
            "num_scripted_steps": int(traj["num_scripted_steps"]),  # replayed prefix + flagged chunk
            "success_trace": [bool(x) for x in traj["success_trace"]],
            "camera_mapping_for_viva": {
                "cam_high": "robot0_agentview_left",
                "cam_left_wrist": "robot0_eye_in_hand",
                "cam_right_wrist": "robot0_agentview_right",
            },
        },
        ep_dir / "rollout_meta.json",
    )
    return episode_index


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Collect targeted rollouts at flagged interactions.")
    parser.add_argument("--triggers", default=DEFAULT_TRIGGERS, help="Candidate buffer written by imagined rollouts.")
    parser.add_argument("--src-root", default=DEFAULT_SRC_ROOT, help="Logged data (success/ and failure/).")
    parser.add_argument("--out-root", default=DEFAULT_OUT_ROOT)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--checkpoint", default=None, help="Current policy checkpoint.")
    parser.add_argument("--budget", type=int, default=20, help="B: targeted rollouts per round.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8020)
    parser.add_argument("--no-start-policy-server", action="store_true")
    parser.add_argument("--server-gpu", default=None)
    parser.add_argument("--startup-timeout-s", type=float, default=300.0)
    parser.add_argument("--resize-size", type=int, default=224)
    parser.add_argument("--chunk-length", type=int, default=8)
    parser.add_argument("--post-success-steps", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=400)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument("--overwrite-output", action="store_true")
    parser.add_argument("--image-writer-threads", type=int, default=8)
    parser.add_argument("--image-writer-processes", type=int, default=0)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    config = OmegaConf.load(args.config)
    pi0_cfg = config.model.pi0
    openpi_root = Path(pi0_cfg.openpi_root).expanduser().resolve()
    checkpoint = Path(args.checkpoint or pi0_cfg.checkpoint_path).expanduser().resolve()
    prompt = str(pi0_cfg.get("prompt", "Turn on the sink faucet."))
    src_root = Path(args.src_root).expanduser().resolve()
    data_dirs = [src_root / "success", src_root / "failure"]
    env_meta = _load_env_meta(data_dirs[0])
    rng = np.random.default_rng(args.seed)

    triggers = load_triggers(Path(args.triggers))
    selected = select_triggers(triggers, args.budget, rng)
    _log(f"{len(triggers)} triggers in the buffer; selected {len(selected)} (budget {args.budget})")

    writers = {}
    for outcome in ("success", "failure"):
        cfg_o = GenConfig(
            src_dataset_dir=str(data_dirs[0]),
            out_dataset_dir=str(Path(args.out_root) / outcome),
            env_name=str(env_meta["env_name"]),
            save_video=not args.no_video,
            image_writer_threads=args.image_writer_threads,
            image_writer_processes=args.image_writer_processes,
            overwrite_output=args.overwrite_output,
            seed=args.seed,
        )
        cfg_o.camera_names = ("robot0_eye_in_hand", "robot0_agentview_left", "robot0_agentview_right")
        out_dir = Path(args.out_root).expanduser().resolve() / outcome
        dataset, reader = _create_or_open_output_dataset(
            src_dataset_dir=data_dirs[0], out_dataset_dir=out_dir, cfg=cfg_o,
            overwrite=args.overwrite_output, env_meta=env_meta,
        )
        writers[outcome] = (dataset, reader, cfg_o, out_dir)
    cfg = writers["success"][2]

    server = None
    if not args.no_start_policy_server:
        server = _start_policy_server(
            openpi_root=openpi_root,
            config_name=str(pi0_cfg.config_name),
            checkpoint=checkpoint,
            prompt=prompt,
            port=args.port,
            server_gpu=args.server_gpu,
        )
        _wait_for_port(args.host, args.port, args.startup_timeout_s)
    _load_openpi_robocasa_main(openpi_root)
    client = _websocket_client_policy.WebsocketClientPolicy(args.host, args.port)

    summary: dict[str, Any] = {
        "triggers_in_buffer": len(triggers),
        "budget": int(args.budget),
        "selected": len(selected),
        "rollouts": [],
    }
    env = None
    try:
        for trigger in selected:
            data_dir, episode_index, start_frame = locate_logged_start(data_dirs, int(trigger["global_index"]))
            trigger["start_frame"] = int(start_frame)
            ep = _load_source_episode(data_dir, episode_index, env_meta)
            if env is None:
                env = build_env(cfg, ep)
            traj = targeted_rollout(
                env=env,
                ep=ep,
                start_frame=start_frame,
                prefix=[np.asarray(a, dtype=np.float32) for a in trigger["prefix"]],
                flagged_chunk=np.asarray(trigger["flagged_chunk"], dtype=np.float32),
                client=client,
                task_lang=prompt,
                cfg=cfg,
                resize_size=args.resize_size,
                chunk_length=args.chunk_length,
                max_steps=args.max_steps,
                post_success_steps=args.post_success_steps,
            )
            outcome = "success" if traj["success"] else "failure"
            dataset, _reader, cfg_o, out_dir = writers[outcome]
            new_ep = write_rollout_episode(
                dataset=dataset, out_dataset_dir=out_dir, env=env, ep=ep,
                traj=traj, trigger=trigger, task_lang=prompt, cfg=cfg_o,
            )
            summary["rollouts"].append({
                "global_index": int(trigger["global_index"]),
                "depth": int(trigger["depth"]),
                "r": float(trigger["r"]),
                "outcome": outcome,
                "output_episode": f"{outcome}/episode_{new_ep:06d}",
            })
            _log(f"trigger at row {trigger['global_index']} (depth {trigger['depth']}): {outcome}")
        for outcome, (dataset, reader, cfg_o, _out_dir) in writers.items():
            finalize_output_dataset(dataset, reader, cfg_o, summary)
        _log(f"done: {len(summary['rollouts'])} targeted rollouts written to {args.out_root}")
    finally:
        if env is not None:
            try:
                env.close()
            except Exception:
                pass
        if server is not None:
            server.terminate()
            try:
                server.wait(timeout=20)
            except subprocess.TimeoutExpired:
                server.kill()
                server.wait(timeout=20)


if __name__ == "__main__":
    main()
