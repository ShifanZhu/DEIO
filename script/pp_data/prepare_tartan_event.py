"""
Preprocess TartanAir-Event raw data into voxel grid format for DEIO training.

Raw format (per sequence):
  events.h5        -> events/{x, y, t, p}  (ns timestamps)
  indices.txt      -> N comma-separated event boundary indices (one per frame)
  timestamps.txt   -> N newline-separated ns timestamps (one per frame)
  depth_left/      -> 000000_left_depth.npy ... (N files)
  pose_left.txt    -> N lines of "tx ty tz qx qy qz qw"

Output format (mirrors original folder hierarchy):
  <out_root>/<scene>/<difficulty>/<seqnum>/
    evs_left/h5/0000000001.h5  ...  (key='voxel', shape=(5,H,W), float16)
    depth_left/000000_left_depth.npy  ...  (symlink or copy)
    pose_left.txt                          (symlink or copy)

Usage:
  python script/prepare_tartan_event.py \
      --src /media/s/rell/tartan_raw \
      --dst /media/s/rell/tartan \
      --n_bins 5 --H 480 --W 640 \
      [--sequences ocean/Easy/P000 ocean/Hard/P002 ...]
      [--overwrite]
"""

import os
import sys
import glob
import shutil
import argparse
import numpy as np
import h5py
import hdf5plugin  # must be imported before reading compressed h5 files
import torch
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed


# ──────────────────────────────────────────────────────────────────────────────
# Voxel grid (same algorithm as utils/event_utils.py::to_voxel_grid)
# ──────────────────────────────────────────────────────────────────────────────

def to_voxel_grid(xs, ys, ts, ps, H=480, W=640, nb_of_time_bins=5):
    """Accumulate events into a (nb_of_time_bins, H, W) voxel grid (float32)."""
    voxel_grid = torch.zeros(nb_of_time_bins, H, W, dtype=torch.float32)
    voxel_grid_flat = voxel_grid.flatten()

    ps = ps.astype(np.int8)
    ps[ps == 0] = -1  # remap 0→-1

    xs = xs.astype(np.float32)
    ys = ys.astype(np.float32)
    ts = ts.astype(np.float64)

    duration = ts[-1] - ts[0]
    if duration == 0:
        return voxel_grid  # empty window

    t = torch.from_numpy((ts - ts[0]) * (nb_of_time_bins - 1) / duration)
    x = torch.from_numpy(xs)
    y = torch.from_numpy(ys)
    polarity = torch.from_numpy(ps.astype(np.float32))

    left_t, right_t = t.floor(), t.floor() + 1

    for lim_x in [x.floor(), x.floor() + 1]:
        for lim_y in [y.floor(), y.floor() + 1]:
            for lim_t in [left_t, right_t]:
                mask = (
                    (0 <= lim_x) & (lim_x <= W - 1) &
                    (0 <= lim_y) & (lim_y <= H - 1) &
                    (0 <= lim_t) & (lim_t <= nb_of_time_bins - 1)
                )
                lin_idx = (
                    lim_x.long() +
                    lim_y.long() * W +
                    lim_t.long() * W * H
                )
                weight = (
                    polarity *
                    (1 - (lim_x - x).abs()) *
                    (1 - (lim_y - y).abs()) *
                    (1 - (lim_t - t).abs())
                )
                voxel_grid_flat.index_add_(0, lin_idx[mask], weight[mask].float())

    return voxel_grid


# ──────────────────────────────────────────────────────────────────────────────
# Per-sequence processing
# ──────────────────────────────────────────────────────────────────────────────

def process_sequence(src_seq: Path, dst_seq: Path, n_bins: int, H: int, W: int,
                     overwrite: bool, show_progress: bool = True):
    """Convert one raw sequence directory → flat output directory."""

    # ── load raw event data ────────────────────────────────────────────────
    ev_path = src_seq / "events.h5"
    idx_path = src_seq / "indices.txt"
    ts_path  = src_seq / "timestamps.txt"
    pose_path = src_seq / "pose_left.txt"
    depth_dir = src_seq / "depth_left"

    if not ev_path.exists():
        print(f"  [SKIP] {src_seq}: no events.h5")
        return

    with h5py.File(ev_path, "r") as f:
        ev_x = f["events/x"][:]
        ev_y = f["events/y"][:]
        ev_t = f["events/t"][:]  # nanoseconds
        ev_p = f["events/p"][:]

    # indices.txt: 2 lines (left cam, right cam), each with N comma-separated float indices
    lines = open(idx_path).readlines()
    import re
    indices = np.array([int(float(v)) for v in re.split(r'[\s,]+', lines[0].strip()) if v])

    N = len(indices)  # number of frames

    # ── set up output directories ──────────────────────────────────────────
    h5_dir = dst_seq / "evs_left" / "h5"
    h5_dir.mkdir(parents=True, exist_ok=True)

    # ── symlink / copy depth_left and pose_left.txt ────────────────────────
    dst_depth = dst_seq / "depth_left"
    if not dst_depth.exists():
        if depth_dir.exists():
            try:
                dst_depth.symlink_to(depth_dir.resolve())
            except (OSError, NotImplementedError):
                shutil.copytree(depth_dir, dst_depth)
        else:
            print(f"  [WARN] depth_left not found at {depth_dir}")

    dst_pose = dst_seq / "pose_left.txt"
    if not dst_pose.exists():
        if pose_path.exists():
            try:
                dst_pose.symlink_to(pose_path.resolve())
            except (OSError, NotImplementedError):
                shutil.copy2(pose_path, dst_pose)
        else:
            print(f"  [WARN] pose_left.txt not found at {pose_path}")

    # ── build voxels ───────────────────────────────────────────────────────
    # Frame i (1-indexed) accumulates events from indices[i-1] to indices[i].
    # Frame 0 has no events (no preceding frame), so we skip it.
    # This matches the training code's `[1:]` slicing on poses and depths.

    existing = set(p.name for p in h5_dir.glob("*.h5"))
    start_idx = 0  # events before first boundary index (not used)

    for i in tqdm(range(1, N), desc=str(dst_seq.name), unit="frame", leave=False,
                  disable=not show_progress):
        fname = f"{i:010d}.h5"
        if fname in existing and not overwrite:
            continue

        i0 = indices[i - 1]
        i1 = indices[i]

        if i1 <= i0:
            # empty window → zero voxel
            voxel = torch.zeros(n_bins, H, W, dtype=torch.float32)
        else:
            xs = ev_x[i0:i1]
            ys = ev_y[i0:i1]
            ts = ev_t[i0:i1].astype(np.float64)
            ps = ev_p[i0:i1]
            voxel = to_voxel_grid(xs, ys, ts, ps, H=H, W=W, nb_of_time_bins=n_bins)

        out_path = h5_dir / fname
        with h5py.File(out_path, "w") as f:
            f.create_dataset("voxel", data=voxel.numpy().astype(np.float16),
                             compression="lzf")


# ──────────────────────────────────────────────────────────────────────────────
# Parallel worker (must be top-level for multiprocessing pickle)
# ──────────────────────────────────────────────────────────────────────────────

def _worker(args):
    rel, src_root, dst_root, n_bins, H, W, overwrite = args
    src_seq = Path(src_root) / rel
    dst_seq = Path(dst_root) / rel
    try:
        process_sequence(src_seq, dst_seq, n_bins=n_bins, H=H, W=W,
                         overwrite=overwrite, show_progress=False)
        return rel, None
    except Exception as e:
        return rel, str(e)


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--src", required=True,
                        help="Root of raw TartanAir-Event data")
    parser.add_argument("--dst", required=True,
                        help="Root of output data (flat structure for training)")
    parser.add_argument("--sequences", nargs="*", default=None,
                        help="Relative paths like 'ocean/Hard/P002'. "
                             "If omitted, all sequences under --src are processed.")
    parser.add_argument("--n_bins", type=int, default=5)
    parser.add_argument("--H", type=int, default=480)
    parser.add_argument("--W", type=int, default=640)
    parser.add_argument("--overwrite", action="store_true",
                        help="Re-generate existing voxel files")
    parser.add_argument("--workers", type=int, default=4,
                        help="Number of parallel worker processes (default: 4)")
    args = parser.parse_args()

    src_root = Path(args.src)
    dst_root = Path(args.dst)
    dst_root.mkdir(parents=True, exist_ok=True)

    if args.sequences:
        rel_paths = args.sequences
    else:
        # auto-discover: any folder containing events.h5
        rel_paths = []
        for ev in sorted(src_root.rglob("events.h5")):
            rel_paths.append(str(ev.parent.relative_to(src_root)))

    n_workers = min(args.workers, len(rel_paths))
    print(f"Found {len(rel_paths)} sequences to process (workers={n_workers}).")

    if n_workers <= 1:
        for rel in rel_paths:
            src_seq = src_root / rel
            dst_seq = dst_root / rel
            print(f"\n→ {rel}  →  {dst_seq}")
            process_sequence(src_seq, dst_seq,
                             n_bins=args.n_bins, H=args.H, W=args.W,
                             overwrite=args.overwrite)
    else:
        worker_args = [
            (rel, str(src_root), str(dst_root), args.n_bins, args.H, args.W, args.overwrite)
            for rel in rel_paths
        ]
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = {executor.submit(_worker, a): a[0] for a in worker_args}
            with tqdm(total=len(futures), desc="sequences", unit="seq") as pbar:
                for fut in as_completed(futures):
                    rel, err = fut.result()
                    if err:
                        tqdm.write(f"  [ERROR] {rel}: {err}")
                    else:
                        tqdm.write(f"  [DONE]  {rel}")
                    pbar.update(1)

    print("\nDone.")
    print(f"Processed sequences written to: {dst_root}")
    print("Add their relative paths (e.g. ocean/Hard/P002) to your split .txt files.")


if __name__ == "__main__":
    main()
