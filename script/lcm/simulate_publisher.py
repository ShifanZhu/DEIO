"""
Simulated LCM publisher for DEIO.

Reads consecutive frame timestamps from a Vector-format H5 file and sends
them as deio.TimestampPair_t messages to the DEVORunner, then prints the
received DeltaPose_t responses.

Runs in a single thread: publisher and server share the same process.
LCM messages are published and subscribed to show the full message flow
while keeping h5py access on a single thread.

Usage:
    conda run -n DEIO python script/lcm/simulate_publisher.py \\
        --h5      /media/s/rell/tro/vector-processed-old/vector/mountain-fast/mountain-fast.h5 \\
        --weights weight/DEVO.pth \\
        --devo_config config/infer_base.yaml \\
        --n_frames 30

Options:
    --h5          Path to Vector H5 event file
    --weights     Path to DEVO checkpoint (.pth)
    --devo_config Path to DEVO config YAML  (default: config/infer_base.yaml)
    --n_frames    Number of frames to process (default: 30)
"""

import argparse
import os
import sys
import time

# Set non-interactive backend before any evo/pyplot imports
os.environ.setdefault("MPLBACKEND", "Agg")
try:
    import evo.tools.settings as _s
    _s.SETTINGS.plot_backend = "Agg"
except Exception:
    pass

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import h5py
import numpy as np
import lcm

from lcm_types.deio import TimestampPair_t, DeltaPose_t
from devo.config import cfg as devo_cfg
from lcm_server.devo_runner import DEVORunner, RunnerConfig
from lcm_server.devo_runner import STATUS_VALID, STATUS_NOT_INIT, STATUS_NO_EVENTS, STATUS_POSE_ERROR
from lcm_server.vector_slicer import VectorEventSlicer

_STATUS_NAMES = {
    STATUS_VALID:      "VALID",
    STATUS_NOT_INIT:   "not_initialized (warmup)",
    STATUS_NO_EVENTS:  "no_events",
    STATUS_POSE_ERROR: "pose_error",
}

INPUT_CHANNEL  = "DEIO_TIMESTAMPS"
OUTPUT_CHANNEL = "DEIO_DELTA_POSE"


def build_response(t0_us, t1_us, status, delta_np) -> DeltaPose_t:
    msg = DeltaPose_t()
    msg.utime  = t0_us
    msg.t0_us  = t0_us
    msg.t1_us  = t1_us
    msg.status = int(status)
    if status == STATUS_VALID and delta_np is not None:
        msg.tx, msg.ty, msg.tz = float(delta_np[0]), float(delta_np[1]), float(delta_np[2])
        msg.qx, msg.qy, msg.qz, msg.qw = (
            float(delta_np[3]), float(delta_np[4]),
            float(delta_np[5]), float(delta_np[6]),
        )
    else:
        msg.tx = msg.ty = msg.tz = 0.0
        msg.qx = msg.qy = msg.qz = 0.0
        msg.qw = 1.0
    return msg


def print_response(resp: DeltaPose_t):
    status_name = _STATUS_NAMES.get(resp.status, f"status={resp.status}")
    t_s = resp.t1_us / 1e6
    if resp.status == STATUS_VALID:
        mag = np.sqrt(resp.tx**2 + resp.ty**2 + resp.tz**2)
        print(
            f"    -> VALID  |t|={mag:.4f}m  "
            f"delta=({resp.tx:+.4f}, {resp.ty:+.4f}, {resp.tz:+.4f})  "
            f"q=({resp.qx:.4f},{resp.qy:.4f},{resp.qz:.4f},{resp.qw:.4f})"
        )
    else:
        print(f"    -> {status_name}")


def main():
    parser = argparse.ArgumentParser(description="DEIO LCM publisher simulation")
    parser.add_argument("--h5",      required=True, help="Path to Vector H5 file")
    parser.add_argument("--weights", required=True, help="Path to DEVO .pth checkpoint")
    parser.add_argument("--devo_config", default="config/infer_base.yaml")
    parser.add_argument("--n_frames",   type=int, default=30,
                        help="Number of frames to publish (default 30)")
    args = parser.parse_args()

    # ── Load DEVO config ──
    devo_cfg.merge_from_file(args.devo_config)

    # ── Open H5 once (single thread — no h5py threading issues) ──
    h5 = h5py.File(args.h5, "r")
    frame_t_us = np.array(h5["frames/depth_t_us"], dtype=np.int64)
    K = np.array(h5["meta/K_event_undist"])
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])

    print(f"[sim] H5          : {args.h5}")
    print(f"[sim] Weights     : {args.weights}")
    print(f"[sim] Frames      : {len(frame_t_us)}  (publishing first {args.n_frames})")
    print(f"[sim] Intrinsics  : fx={fx:.2f}  fy={fy:.2f}  cx={cx:.2f}  cy={cy:.2f}")
    print(f"[sim] Frame gap   : {np.mean(np.diff(frame_t_us[:10]))/1e3:.1f} ms avg")

    # ── Build runner (uses the same h5 handle via VectorEventSlicer) ──
    runner_cfg = RunnerConfig(
        h5_file      = args.h5,
        intrinsics   = [fx, fy, cx, cy],
        img_height   = 480,
        img_width    = 640,
        voxel_bins   = 5,
        hot_pixel_filter = True,
        hot_pixel_stds   = 10,
    )
    slicer = VectorEventSlicer(h5)
    runner = DEVORunner(runner_cfg, devo_cfg, args.weights, slicer=slicer)
    print("[sim] DEVO initialized.\n")

    # ── LCM: one instance for both publish and subscribe ──
    lc = lcm.LCM()
    received_responses = []

    def on_delta_pose(channel, data):
        resp = DeltaPose_t.decode(data)
        received_responses.append(resp)
        print_response(resp)

    lc.subscribe(OUTPUT_CHANNEL, on_delta_pose)

    # ── Main loop: publish → process → publish response → handle ──
    n = min(args.n_frames, len(frame_t_us) - 1)
    print(f"{'Frame':>5}  {'t0_us':>20}  {'t1_us':>20}  {'gap_ms':>8}")
    print("─" * 62)

    t_start_wall = time.time()

    for i in range(n):
        t0_us = int(frame_t_us[i])
        t1_us = int(frame_t_us[i + 1])
        gap_ms = (t1_us - t0_us) / 1e3

        print(f"{i:>5}  {t0_us:>20}  {t1_us:>20}  {gap_ms:>7.1f}ms")

        # 1. Simulate publisher: send request over LCM
        req = TimestampPair_t()
        req.utime = t0_us
        req.t0_us = t0_us
        req.t1_us = t1_us
        lc.publish(INPUT_CHANNEL, req.encode())

        # 2. Simulate server: process the request directly (same thread, safe h5py access)
        status, delta_np = runner.process_frame(t0_us, t1_us)

        # 3. Simulate server: publish response over LCM
        resp_msg = build_response(t0_us, t1_us, status, delta_np)
        lc.publish(OUTPUT_CHANNEL, resp_msg.encode())

        # 4. Simulate subscriber: drain pending LCM messages (receive our own response)
        lc.handle_timeout(1)

    # Drain any remaining messages
    for _ in range(10):
        lc.handle_timeout(20)

    h5.close()

    # ── Summary ──
    elapsed = time.time() - t_start_wall
    valid = [r for r in received_responses if r.status == STATUS_VALID]

    print("\n" + "═" * 62)
    print(f"[sim] Processed  : {n} frames in {elapsed:.1f}s  ({n/elapsed:.1f} fps)")
    print(f"[sim] Responses  : {len(received_responses)} received  ({len(valid)} valid poses)")

    if valid:
        txs = np.array([r.tx for r in valid])
        tys = np.array([r.ty for r in valid])
        tzs = np.array([r.tz for r in valid])
        mags = np.sqrt(txs**2 + tys**2 + tzs**2)
        print(f"[sim] Displacement per frame: mean={mags.mean():.4f}m  max={mags.max():.4f}m")
        print(f"[sim] Cumulative path length : {mags.sum():.4f}m")


if __name__ == "__main__":
    main()
