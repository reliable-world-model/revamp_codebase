"""Local world-model imagination server for OpenPI QAM.

The JAX/OpenPI trainer drives an imagined rollout chunk by chunk. For each
chunk it sends the current (logged or imagined) context and the policy's action
chunk; this server runs the frozen world model and returns the predicted
frames, the contact head's end-of-chunk prediction, and the reliability score
of the prediction (Eq. 1-3).
"""

from __future__ import annotations

import argparse
from http.server import BaseHTTPRequestHandler
from http.server import ThreadingHTTPServer
from io import BytesIO
import json
from pathlib import Path
import threading
import time
import traceback

import numpy as np
import torch
from omegaconf import OmegaConf

from revamp.common.world_model import WorldModel
from revamp.reliability import (
    CameraCalibration,
    DepthPoseProbe,
    InverseDynamicsMLP,
    ReliabilityCalibration,
    ReliabilityScorer,
    load_depth_anything3,
)


def _checkpoint_file(path: str | Path) -> Path:
    path = Path(path)
    if path.is_dir():
        model_safe = path / "model.safetensors"
        model_bin = path / "pytorch_model.bin"
        if model_safe.exists():
            return model_safe
        if model_bin.exists():
            return model_bin
        raise FileNotFoundError(f"No model.safetensors or pytorch_model.bin in {path}")
    return path


def _load_state_dict(path: Path) -> dict[str, torch.Tensor]:
    if path.suffix == ".safetensors":
        from safetensors.torch import load_file

        return load_file(str(path), device="cpu")
    ckpt = torch.load(path, map_location="cpu")
    return ckpt.get("world_model", ckpt)


def _npz_bytes(payload: dict[str, np.ndarray], *, compressed: bool = True) -> bytes:
    bio = BytesIO()
    if compressed:
        np.savez_compressed(bio, **payload)
    else:
        np.savez(bio, **payload)
    return bio.getvalue()


class WorldModelImagOracle:
    """World model + reliability scorer behind the imagination endpoint.

    One request = one imagined chunk for a batch of contexts: the chunk ā_t is
    predicted from the context (s_t, recent frames); the response carries the
    predicted frames, the contact head's end-of-chunk prediction, and the
    reliability score r = w_intr r_intr + w_ext r_ext of the prediction.
    """

    def __init__(self, config_path: str, checkpoint: str, device: str):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
        self.config = OmegaConf.load(config_path)
        self.device = torch.device(device)
        self.model = WorldModel(self.config).to(self.device)
        ckpt = _checkpoint_file(checkpoint)
        missing, unexpected = self.model.load_state_dict(_load_state_dict(ckpt), strict=False)
        if missing or unexpected:
            print(
                "[world_model_imag_server] checkpoint load warning: "
                f"missing={len(missing)}, unexpected={len(unexpected)}",
                flush=True,
            )
        for p in self.model.parameters():
            p.requires_grad = False
        self.model.eval()

        t5 = torch.load(self.config.dataset.tasks[0].t5_embedding_path, map_location=self.device).to(self.model.dtype)
        self.t5 = t5.unsqueeze(0) if t5.dim() == 2 else t5
        self.scorer = build_reliability_scorer(self.config.reliability, self.device)

    def predict(self, arrays: dict[str, np.ndarray], num_inference_steps: int) -> dict[str, np.ndarray]:
        state = torch.as_tensor(arrays["state"], device=self.device).float()
        cam_left = torch.as_tensor(arrays["cam_left_segment"], device=self.device).float()
        cam_right = torch.as_tensor(arrays["cam_right_segment"], device=self.device).float()
        cam_high = torch.as_tensor(arrays["cam_high_segment"], device=self.device).float()
        action = torch.as_tensor(arrays["action"], device=self.device).float()          # [B, H, D]
        start_pose = torch.as_tensor(arrays["start_pose"], device=self.device).float()  # [B, 7]
        t5_batch = self.t5.expand(state.shape[0], -1, -1)

        with torch.no_grad():
            cond_latent = self.model.build_full_latent_sequence(cam_left, cam_right, cam_high)[:, :, :1]
            pred = self.model.predict_chunk_from_cond_latent(
                cond_latent=cond_latent,
                state=state,
                action_chunk=action,
                t5_embeddings=t5_batch,
                num_inference_steps=int(num_inference_steps),
                decode_rgb=True,
            )
            frames = {}
            for key in ("next_cam_left", "next_cam_right", "next_cam_high"):
                f = pred[key].detach().float()
                frames[key] = ((f + 1.0) * 0.5).clamp(0.0, 1.0)                        # [-1, 1] -> [0, 1]
            score = self.scorer(
                self.model, state, action, t5_batch,
                cond_latent=cond_latent,
                target_latents=pred["next_cam_concat_latent"],
                decoded={"left": frames["next_cam_left"], "right": frames["next_cam_right"], "high": frames["next_cam_high"]},
                start_pose=start_pose,
            )

        out = {key: frames[key].cpu().numpy().astype(np.float32) for key in frames}
        out["contact_prob"] = pred["contact_prob"].cpu().numpy().astype(np.float32)
        for key in ("r", "r_intr", "r_ext", "weight"):
            out[key] = score[key].float().cpu().numpy().astype(np.float32)
        out["triggered"] = score["triggered"].cpu().numpy().astype(bool)
        out["num_pred_frames"] = np.asarray(out["next_cam_left"].shape[2], dtype=np.int32)
        return out


def build_reliability_scorer(cfg, device: torch.device) -> ReliabilityScorer:
    """Probe (Depth Anything 3 + gripper segmentation + Arun/Horn), inverse dynamics, calibration."""
    calib = json.loads(Path(cfg.camera_calibration).read_text())
    cameras = {
        cam: CameraCalibration(K=torch.tensor(v["K"], dtype=torch.float32), T_base_cam=torch.tensor(v["T_base_cam"], dtype=torch.float32))
        for cam, v in calib.items()
    }
    segmenter_model = torch.jit.load(cfg.gripper_segmenter, map_location=device).eval()

    def segmenter(rgb: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return segmenter_model(rgb.unsqueeze(0).to(device))[0, 0] > 0

    probe = DepthPoseProbe(
        depth_model=load_depth_anything3(cfg.depth_anything3, str(device)),
        segmenter=segmenter,
        gripper_template=torch.as_tensor(np.load(cfg.gripper_template), dtype=torch.float32),
        cameras=cameras,
    )
    id_model = InverseDynamicsMLP(action_dim=int(cfg.get("action_dim", 12)))
    id_model.load_state_dict(torch.load(cfg.inverse_dynamics, map_location="cpu"))
    return ReliabilityScorer(
        calibration=ReliabilityCalibration.load(cfg.calibration),
        probe=probe,
        id_model=id_model.to(device),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="World-model config (with the reliability section).")
    parser.add_argument("--checkpoint", required=True, help="World-model checkpoint dir or file (dynamics, Q-head, contact head).")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-inference-steps", type=int, default=20)
    parser.add_argument(
        "--response-compression",
        choices=("compressed", "uncompressed"),
        default="uncompressed",
    )
    args = parser.parse_args()

    oracle = WorldModelImagOracle(args.config, args.checkpoint, args.device)
    oracle_lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def do_GET(self):  # noqa: N802
            if self.path != "/health":
                self.send_error(404)
                return
            body = json.dumps({"ok": True}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self):  # noqa: N802
            if self.path != "/predict_chunk":
                self.send_error(404)
                return
            try:
                length = int(self.headers["Content-Length"])
                raw = self.rfile.read(length)
                t0 = time.time()
                data = np.load(BytesIO(raw))
                arrays = {k: data[k] for k in data.files}
                decode_s = time.time() - t0
                steps = int(arrays.get("num_inference_steps", np.asarray(args.num_inference_steps)).reshape(-1)[0])

                wait_t0 = time.time()
                with oracle_lock:
                    wait_s = time.time() - wait_t0
                    compute_t0 = time.time()
                    pred = oracle.predict(arrays, steps)
                    compute_s = time.time() - compute_t0

                all_frames = np.concatenate(
                    [pred["next_cam_left"], pred["next_cam_right"], pred["next_cam_high"]],
                    axis=-1,
                )
                payload = {
                    **pred,
                    "pred_pixel_mean": np.asarray(float(all_frames.mean()), dtype=np.float32),
                    "pred_pixel_std": np.asarray(float(all_frames.std()), dtype=np.float32),
                    "pred_pixel_min": np.asarray(float(all_frames.min()), dtype=np.float32),
                    "pred_pixel_max": np.asarray(float(all_frames.max()), dtype=np.float32),
                    "decode_s": np.asarray(decode_s, dtype=np.float32),
                    "wait_s": np.asarray(wait_s, dtype=np.float32),
                    "compute_s": np.asarray(compute_s, dtype=np.float32),
                    "encode_s": np.asarray(0.0, dtype=np.float32),
                }
                t0 = time.time()
                body = _npz_bytes(payload, compressed=(args.response_compression == "compressed"))
                encode_s = time.time() - t0
                payload["encode_s"] = np.asarray(encode_s, dtype=np.float32)
                body = _npz_bytes(payload, compressed=(args.response_compression == "compressed"))

                self.send_response(200)
                self.send_header("Content-Type", "application/octet-stream")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Connection", "keep-alive")
                self.end_headers()
                self.wfile.write(body)
            except Exception as exc:  # pragma: no cover - server diagnostics
                traceback.print_exc()
                self.send_error(500, f"{type(exc).__name__}: {exc}")

        def log_message(self, fmt, *args):
            return

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(
        f"[world_model_imag_server] listening on {args.host}:{args.port}, "
        f"device={args.device}, num_inference_steps={args.num_inference_steps}, "
        f"response_compression={args.response_compression}",
        flush=True,
    )
    server.serve_forever()


if __name__ == "__main__":
    main()
