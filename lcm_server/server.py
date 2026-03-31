"""
DEIO LCM Online Inference Server.

Subscribes to deio.TimestampPair_t on LCM_INPUT_CHANNEL.
For each received message: extracts events, runs DEVO, publishes deio.DeltaPose_t.

Usage:
    conda run -n DEIO python -m lcm_server.server \\
        --weights checkpoints/model.pth \\
        --h5_file /data/events_left.h5 \\
        --devo_config config/infer_base.yaml \\
        --config config/lcm_server.yaml

Optional overrides:
    --lcm_url       udpm://239.255.76.67:7667?ttl=1
    --input_channel  DEIO_TIMESTAMPS
    --output_channel DEIO_DELTA_POSE
    --rectify_map   /data/rectify_map_left.h5
"""

import argparse
import sys
import os
import yaml

# Force non-interactive matplotlib backend before any evo/viz imports.
# evo reads its own settings and overrides MPLBACKEND, so patch it directly.
os.environ.setdefault("MPLBACKEND", "Agg")
try:
    import evo.tools.settings as _evo_settings
    _evo_settings.SETTINGS.plot_backend = "Agg"
except Exception:
    pass

import torch

# Ensure repo root is on path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import lcm
from lcm_types.deio import TimestampPair_t, DeltaPose_t

from devo.config import cfg as devo_cfg
from lcm_server.devo_runner import DEVORunner, RunnerConfig
from lcm_server.devo_runner import STATUS_VALID, STATUS_NOT_INIT, STATUS_NO_EVENTS, STATUS_POSE_ERROR

_STATUS_NAMES = {
    STATUS_VALID:      "valid",
    STATUS_NOT_INIT:   "not_initialized",
    STATUS_NO_EVENTS:  "no_events",
    STATUS_POSE_ERROR: "pose_error",
}


def _build_response(req_utime, t0_us, t1_us, status, delta_np):
    msg = DeltaPose_t()
    msg.utime  = int(req_utime)
    msg.t0_us  = int(t0_us)
    msg.t1_us  = int(t1_us)
    msg.status = int(status)

    if delta_np is not None and status == STATUS_VALID:
        msg.tx = float(delta_np[0])
        msg.ty = float(delta_np[1])
        msg.tz = float(delta_np[2])
        msg.qx = float(delta_np[3])
        msg.qy = float(delta_np[4])
        msg.qz = float(delta_np[5])
        msg.qw = float(delta_np[6])
    else:
        # Identity pose as fallback
        msg.tx = msg.ty = msg.tz = 0.0
        msg.qx = msg.qy = msg.qz = 0.0
        msg.qw = 1.0

    return msg


class DEIOServer:
    def __init__(self, runner: DEVORunner, lcm_url: str,
                 input_channel: str, output_channel: str):
        self.lc = lcm.LCM(lcm_url) if lcm_url else lcm.LCM()
        self.input_channel  = input_channel
        self.output_channel = output_channel
        self.runner = runner

        self._sub = self.lc.subscribe(self.input_channel, self._on_timestamps)
        print(f"[DEIOServer] Listening on '{self.input_channel}' -> "
              f"publishing to '{self.output_channel}'")

    def _on_timestamps(self, channel: str, data: bytes):
        req = TimestampPair_t.decode(data)
        t0_us = int(req.t0_us)
        t1_us = int(req.t1_us)

        if t0_us >= t1_us:
            print(f"[DEIOServer] Invalid window: t0={t0_us} >= t1={t1_us}, skipping")
            return

        status, delta_np = self.runner.process_frame(t0_us, t1_us)

        resp = _build_response(req.utime, t0_us, t1_us, status, delta_np)
        self.lc.publish(self.output_channel, resp.encode())

        if status == STATUS_VALID:
            print(f"[DEIOServer] t={t1_us/1e6:.3f}s  "
                  f"delta=({resp.tx:.4f}, {resp.ty:.4f}, {resp.tz:.4f})")
        else:
            print(f"[DEIOServer] t={t1_us/1e6:.3f}s  "
                  f"status={_STATUS_NAMES.get(status, status)}")

    def spin(self):
        try:
            while True:
                self.lc.handle()
        except KeyboardInterrupt:
            print("\n[DEIOServer] Shutting down.")
        finally:
            self.runner.close()


def _load_server_yaml(path: str) -> dict:
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return yaml.safe_load(f) or {}


def main():
    parser = argparse.ArgumentParser(description="DEIO LCM Online Inference Server")
    parser.add_argument("--weights",      required=True,
                        help="Path to DEVO .pth checkpoint")
    parser.add_argument("--h5_file",      required=True,
                        help="Path to events H5 file")
    parser.add_argument("--devo_config",  default="config/infer_base.yaml",
                        help="Path to DEVO YAML config")
    parser.add_argument("--config",       default="config/lcm_server.yaml",
                        help="Path to server YAML config")
    parser.add_argument("--lcm_url",      default=None,
                        help="LCM URL (e.g. udpm://239.255.76.67:7667?ttl=1)")
    parser.add_argument("--input_channel",  default=None)
    parser.add_argument("--output_channel", default=None)
    parser.add_argument("--rectify_map",  default=None,
                        help="Optional path to rectify_map_{side}.h5 for lens undistortion")
    args = parser.parse_args()

    # Load DEVO config
    devo_cfg.merge_from_file(args.devo_config)

    # Load server config (YAML)
    scfg = _load_server_yaml(args.config)

    runner_cfg = RunnerConfig(
        h5_file       = args.h5_file,
        intrinsics    = scfg.get("INTRINSICS", [320.0, 320.0, 320.0, 240.0]),
        img_height    = scfg.get("IMG_HEIGHT", 480),
        img_width     = scfg.get("IMG_WIDTH",  640),
        voxel_bins    = scfg.get("VOXEL_BINS", 5),
        hot_pixel_filter = scfg.get("HOT_PIXEL_FILTER", True),
        hot_pixel_stds   = scfg.get("HOT_PIXEL_STDS",   10),
        rectify_map_file = args.rectify_map or scfg.get("RECTIFY_MAP_FILE", ""),
    )

    lcm_url        = args.lcm_url      or scfg.get("LCM_URL", "")
    input_channel  = args.input_channel  or scfg.get("LCM_INPUT_CHANNEL",  "DEIO_TIMESTAMPS")
    output_channel = args.output_channel or scfg.get("LCM_OUTPUT_CHANNEL", "DEIO_DELTA_POSE")

    torch.set_grad_enabled(False)

    runner = DEVORunner(runner_cfg, devo_cfg, args.weights)
    server = DEIOServer(runner, lcm_url, input_channel, output_channel)
    server.spin()


if __name__ == "__main__":
    main()
