#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys

os.environ.setdefault("MPLBACKEND", "Agg")

def _repo_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))


sys.path.insert(0, _repo_root())

import h5py
import numpy as np
import torch
from itertools import islice
from scipy.spatial.transform import Rotation as R

from devo.config import cfg as devo_cfg_base
from lcm_server.devo_runner import DEVORunner, RunnerConfig, STATUS_VALID
from lcm_server.vector_slicer import VectorEventSlicer
from utils.eval_utils import run_DEIO2
from utils.load_utils import mvsec_h5_iterator, vector_preprocessed_h5_iterator


def _pose7_to_matrix(pose7):
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R.from_quat(np.asarray(pose7[3:7], dtype=np.float64)).as_matrix()
    T[:3, 3] = np.asarray(pose7[:3], dtype=np.float64)
    return T


def _accumulate_devo_deltas(runner, depth_t, i0, n_windows, recorder_like: bool = True):
    """Mirror LCM: pairs (depth_t[i], depth_t[i+1]) for i in [i0, i0+n_windows)."""
    T_wc = np.eye(4, dtype=np.float64)
    valid = 0
    status_hist = {}
    for k in range(n_windows):
        i = i0 + k
        if i + 1 >= len(depth_t):
            break
        t0, t1 = int(depth_t[i]), int(depth_t[i + 1])
        st, delta = runner.process_frame(t0, t1)
        status_hist[st] = status_hist.get(st, 0) + 1
        if st == STATUS_VALID and delta is not None:
            valid += 1
            T_wc = _pose7_to_matrix(delta) @ T_wc
    # Match current subscriber TrajectoryRecorder.export_tum: store c2w in file
    T_cw = np.linalg.inv(T_wc)
    pos = T_cw[:3, 3].copy()
    return pos, valid, status_hist, T_wc, T_cw


def _compare_event_windows(h5_path: str, n_check: int = 15):
    h5 = h5py.File(h5_path, "r")
    depth_t = np.array(h5["frames/depth_t_us"], dtype=np.int64)
    slices = np.array(h5["events/slices_index"], dtype=np.int64)
    slicer = VectorEventSlicer(h5)
    mism = []
    for i in range(min(n_check, len(depth_t) - 1)):
        t0, t1 = int(depth_t[i]), int(depth_t[i + 1])
        a0, a1 = int(slices[i]), int(slices[i + 1])
        n_slice = a1 - a0
        ev = slicer.get_events(t0, t1)
        n_win = 0 if ev is None else len(ev["t"])
        if n_slice != n_win:
            mism.append((i, t1 - t0, n_slice, n_win))
    h5.close()
    return mism


def _run_deio2_subset(h5_path, cfg, weights, iterator_builder, n, H, W, imu):
    # run_DEIO2 mutates IMU timestamps in-place (seconds); always pass a copy.
    imu = imu.copy()
    it = iterator_builder(h5_path, H=H, W=W)
    it = islice(it, n)
    traj, tss, _, _ = run_DEIO2(
        str(os.path.dirname(h5_path)),
        cfg,
        weights,
        viz=False,
        iterator=it,
        _all_imu=imu,
        _all_gt=None,
        _all_gt_keys=None,
        timing=False,
        H=H,
        W=W,
        viz_flow=False,
    )
    if len(traj) == 0:
        return None, None
    # DEIO2 terminate: camera-to-world positions in traj[:, :3]
    return traj[-1, :3].copy(), len(traj)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--devo_cfg_infer", default="config/infer_base.yaml")
    ap.add_argument("--devo_cfg_vector", default="config/vector.yaml")
    ap.add_argument("--n", type=int, default=80, help="number of frame windows")
    ap.add_argument("--skip", type=int, default=0, help="first window index i (pair t[i],t[i+1])")
    args = ap.parse_args()

    torch.set_grad_enabled(False)
    h5_path = os.path.abspath(args.h5)

    with h5py.File(h5_path, "r") as f:
        has_ti = "events" in f and "time_images" in f["events"]
        K = np.array(f["meta/K_event_undist"])
        depth_t = np.array(f["frames/depth_t_us"], dtype=np.int64)

    print("=== H5 ===")
    print("path:", h5_path)
    print("frames:", len(depth_t), "has events/time_images:", has_ti)

    mism = _compare_event_windows(h5_path, n_check=30)
    if mism:
        print("\n=== Event count mismatch (slice vs [t0,t1)) ===")
        for row in mism[:10]:
            print("  frame_idx", row[0], "dt_us", row[1], "n_slice", row[2], "n_window", row[3])
        if len(mism) > 10:
            print("  ...", len(mism) - 10, "more")
    else:
        print("\n=== Event windows: slices[i:i+1] match VectorEventSlicer [t_i,t_{i+1}) for first 30 pairs ===")

    # IMU for DEIO2 (same as subscriber)
    with h5py.File(h5_path, "r") as f:
        imu_t = f["imu/t_us"][:].astype(np.float64)
        imu_d = f["imu/data_raw"][:].astype(np.float64)
    nimu = min(len(imu_t), len(imu_d))
    all_imu = np.zeros((nimu, 7), dtype=np.float64)
    all_imu[:, 0] = imu_t[:nimu]
    all_imu[:, 1:4] = imu_d[:nimu, :3] * (180.0 / np.pi)
    all_imu[:, 4:7] = imu_d[:nimu, 3:6]
    all_imu = all_imu[all_imu[:, 0].argsort()]

    intr = [float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2])]
    rc = RunnerConfig(
        h5_file=h5_path,
        intrinsics=intr,
        img_height=480,
        img_width=640,
    )

    n_w = min(args.n, max(0, len(depth_t) - 1 - args.skip))
    i0 = args.skip
    print("\n=== DEVORunner (online / LCM stack) ===")
    for cfg_name, cfg_file in [("infer_base", args.devo_cfg_infer), ("vector", args.devo_cfg_vector)]:
        devo_cfg = devo_cfg_base.clone()
        devo_cfg.merge_from_file(cfg_file)
        h5 = h5py.File(h5_path, "r")
        runner = DEVORunner(rc, devo_cfg, args.weights, slicer=VectorEventSlicer(h5))
        pos, valid, hist, _, _ = _accumulate_devo_deltas(runner, depth_t, i0, n_w)
        runner.close()
        h5.close()
        print(f"  cfg={cfg_name} ({cfg_file})  valid={valid}/{n_w}  status_hist={hist}")
        print(f"  final |p| (c2w from composed deltas): {np.linalg.norm(pos):.4f} m  p={pos}")

    print("\n=== run_DEIO2 (same weights; IMU on) ===")
    for cfg_name, cfg_file in [("infer_base", args.devo_cfg_infer), ("vector", args.devo_cfg_vector)]:
        devo_cfg = devo_cfg_base.clone()
        devo_cfg.merge_from_file(cfg_file)

        pos_m, len_m = _run_deio2_subset(
            h5_path, devo_cfg, args.weights,
            lambda hp, H, W: mvsec_h5_iterator(hp, H=H, W=W),
            n_w, 480, 640, all_imu,
        )
        if pos_m is None:
            print(f"  cfg={cfg_name}  mvsec iterator: no poses")
        else:
            print(f"  cfg={cfg_name}  mvsec:  n_poses={len_m}  final |p|={np.linalg.norm(pos_m):.4f} m")

        if has_ti:
            pos_t, len_t = _run_deio2_subset(
                h5_path, devo_cfg, args.weights,
                lambda hp, H, W: vector_preprocessed_h5_iterator(hp, H=H, W=W),
                n_w, 480, 640, all_imu,
            )
            if pos_t is None:
                print(f"  cfg={cfg_name}  time_images: no poses")
            else:
                print(f"  cfg={cfg_name}  time_images: n_poses={len_t}  final |p|={np.linalg.norm(pos_t):.4f} m")

    print("\n=== Interpretation ===")
    print("- DEVORunner (LCM) is DEVO-only; run_DEIO2 is VI (DBA). Different trajectories are expected.")
    print("- For VECTOR H5s, infer_h5.py uses mvsec_h5_iterator (raw events), not time_images.")
    print("  subscriber._build_infer_h5_iterator must match that; time_images+vector.yaml can yield NaN.")
    print("- Default eval skip: desk_normal imstart=65 in load_utils.get_imstart_imstop_vector (infer_h5);")
    print("  LCM from frame 0 includes extra warmup frames → compare with same --skip when benchmarking.")


if __name__ == "__main__":
    main()
