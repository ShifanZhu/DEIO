"""
Script to inspect and save visualizations of DEIO input data.

Input data pipeline:
  Raw events (x, y, t, p) from .h5 file
      → EventSlicer slices a time window [t0, t1]
      → to_voxel_grid() converts to (5, H, W) voxel grid
      → Yielded per-frame as (voxel, intrinsics, timestamp_us)

Usage:
    conda run -n DEIO python script/inspect_input_data.py \
        --datadir /path/to/davis240c/sequence \
        --outdir /tmp/deio_inspect \
        --dataset davis240c \
        --num_frames 10
"""

import argparse
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import h5py
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from utils.event_utils import EventSlicer, to_voxel_grid
from utils.viz_utils import render


def inspect_raw_h5(h5path):
    """Print structure and stats of a raw events .h5 file."""
    print(f"\n{'='*60}")
    print(f"HDF5 file: {h5path}")
    print(f"{'='*60}")
    with h5py.File(h5path, "r") as f:
        def _print_tree(name, obj):
            if isinstance(obj, h5py.Dataset):
                print(f"  [{name}]  shape={obj.shape}  dtype={obj.dtype}")
        f.visititems(_print_tree)

        # Show first few raw events
        if "events/x" in f:
            prefix = "events/"
        else:
            prefix = ""
        x = np.asarray(f[f"{prefix}x"][:10])
        y = np.asarray(f[f"{prefix}y"][:10])
        t = np.asarray(f[f"{prefix}t"][:10])
        p = np.asarray(f[f"{prefix}p"][:10])
        print(f"\n  First 10 events:")
        print(f"  {'x':>6} {'y':>6} {'t (us)':>12} {'p':>4}")
        for i in range(len(x)):
            print(f"  {x[i]:>6} {y[i]:>6} {t[i]:>12} {p[i]:>4}")

        total = f[f"{prefix}t"].shape[0]
        t0 = int(f[f"{prefix}t"][0])
        t1 = int(f[f"{prefix}t"][-1])
        print(f"\n  Total events : {total:,}")
        print(f"  Duration     : {(t1-t0)/1e6:.3f} s  ({t0} us → {t1} us)")
        print(f"  Event rate   : {total/(t1-t0)*1e6/1e3:.1f} kHz")


def save_event_frame(events, H, W, outpath):
    """Render raw events as RGB image (blue=ON, red=OFF, white=empty)."""
    img = render(events["x"].astype(np.float32),
                 events["y"].astype(np.float32),
                 events["p"], H, W)
    plt.figure(figsize=(6, 4))
    plt.imshow(img)
    plt.title(f"Raw events  ({len(events['t']):,} events)\n"
              f"t=[{events['t'][0]}, {events['t'][-1]}] us")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(outpath, dpi=120)
    plt.close()
    print(f"  Saved raw event frame → {outpath}")


def save_voxel_grid(voxel, ts_us, outpath, frame_idx=0):
    """Save all 5 temporal bins of a voxel grid as a figure."""
    # voxel: (5, H, W) torch tensor
    v = voxel.cpu().numpy()  # (5, H, W)
    nbins = v.shape[0]

    fig, axes = plt.subplots(1, nbins, figsize=(4 * nbins, 4))
    fig.suptitle(f"Voxel grid  frame={frame_idx}  ts={ts_us/1e6:.4f}s\n"
                 f"shape={tuple(v.shape)}  min={v.min():.3f}  max={v.max():.3f}",
                 fontsize=11)

    for b in range(nbins):
        ax = axes[b]
        img = v[b]
        vmax = max(abs(img.max()), abs(img.min()), 1e-3)
        ax.imshow(img, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        ax.set_title(f"bin {b}")
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(outpath, dpi=120)
    plt.close()
    print(f"  Saved voxel grid      → {outpath}")


def save_voxel_summary(voxels_list, outpath):
    """Save a grid of collapsed voxels (sum over bins) for multiple frames."""
    n = len(voxels_list)
    ncols = min(n, 8)
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(3 * ncols, 3 * nrows))
    axes = np.array(axes).reshape(-1)

    for i, (voxel, ts_us) in enumerate(voxels_list):
        v = voxel.cpu().numpy()
        collapsed = v.sum(axis=0)  # (H, W)
        vmax = max(abs(collapsed.max()), abs(collapsed.min()), 1e-3)
        axes[i].imshow(collapsed, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        axes[i].set_title(f"t={ts_us/1e6:.3f}s", fontsize=8)
        axes[i].axis("off")

    for i in range(n, len(axes)):
        axes[i].axis("off")

    fig.suptitle("Voxel grids (sum over 5 bins) — one per frame", fontsize=12)
    plt.tight_layout()
    plt.savefig(outpath, dpi=120)
    plt.close()
    print(f"  Saved voxel summary   → {outpath}")


def save_event_stats(h5path, outpath, H, W):
    """Plot polarity histogram and pixel activity map from raw events."""
    with h5py.File(h5path, "r") as f:
        prefix = "events/" if "events/x" in f else ""
        n = f[f"{prefix}x"].shape[0]
        sample = min(n, 500_000)
        idx = np.random.choice(n, sample, replace=False)
        idx.sort()
        x = np.asarray(f[f"{prefix}x"][idx])
        y = np.asarray(f[f"{prefix}y"][idx])
        p = np.asarray(f[f"{prefix}p"][idx])

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    # Pixel activity map
    activity = np.zeros((H, W), dtype=np.float32)
    np.add.at(activity, (y, x), 1)
    axes[0].imshow(np.log1p(activity), cmap="inferno")
    axes[0].set_title(f"Pixel activity map (log scale)\n{sample:,} sampled events")
    axes[0].axis("off")

    # ON / OFF event maps
    on_map = np.zeros((H, W), dtype=np.float32)
    off_map = np.zeros((H, W), dtype=np.float32)
    np.add.at(on_map,  (y[p == 1], x[p == 1]), 1)
    np.add.at(off_map, (y[p == 0], x[p == 0]), 1)
    overlay = np.stack([np.log1p(off_map) / (np.log1p(off_map).max() + 1e-6),
                        np.zeros((H, W)),
                        np.log1p(on_map)  / (np.log1p(on_map).max()  + 1e-6)], axis=-1)
    axes[1].imshow(overlay)
    axes[1].set_title("ON (blue) vs OFF (red) events")
    axes[1].axis("off")

    # Polarity histogram
    labels = ["OFF (0)", "ON (1)"]
    counts = [int((p == 0).sum()), int((p == 1).sum())]
    axes[2].bar(labels, counts, color=["red", "blue"])
    axes[2].set_title("Polarity histogram")
    axes[2].set_ylabel("count")
    for bar, c in zip(axes[2].patches, counts):
        axes[2].text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 0.5,
                     f"{c:,}", ha="center", color="white", fontsize=11)

    plt.tight_layout()
    plt.savefig(outpath, dpi=120)
    plt.close()
    print(f"  Saved event stats     → {outpath}")


def print_voxel_info(voxel, intrinsics, ts_us, frame_idx):
    v = voxel.cpu()
    print(f"\n  Frame {frame_idx:>4d}  ts={ts_us/1e6:.4f}s")
    print(f"    voxel shape  : {tuple(v.shape)}  (bins, H, W)")
    print(f"    voxel dtype  : {v.dtype}")
    print(f"    voxel range  : [{v.min():.4f}, {v.max():.4f}]")
    print(f"    voxel mean   : {v.mean():.6f}   std: {v.std():.4f}")
    nonzero_frac = (v.abs() > 1e-4).float().mean().item()
    print(f"    nonzero frac : {nonzero_frac:.4f}  ({nonzero_frac*100:.2f}%)")
    print(f"    intrinsics   : fx={intrinsics[0]:.2f}  fy={intrinsics[1]:.2f}  "
          f"cx={intrinsics[2]:.2f}  cy={intrinsics[3]:.2f}")


def main():
    parser = argparse.ArgumentParser(description="Inspect DEIO input data")
    parser.add_argument("--datadir", required=True,
                        help="Path to a single sequence directory")
    parser.add_argument("--outdir", default="/tmp/deio_inspect",
                        help="Directory to save output images")
    parser.add_argument("--dataset", default="davis240c",
                        choices=["davis240c", "monohku", "eds", "tumvie", "mvsec",
                                 "vector", "dsec", "ecmd", "cear", "fpv"],
                        help="Dataset type (determines iterator and resolution)")
    parser.add_argument("--side", default="left",
                        help="Camera side (left/right) for stereo datasets")
    parser.add_argument("--num_frames", type=int, default=10,
                        help="Number of frames to inspect")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # --- Select iterator and raw h5 path ---
    dataset = args.dataset
    side = args.side

    if dataset == "davis240c":
        from utils.load_utils import davis240c_evs_iterator
        H, W = 180, 240
        iterator = davis240c_evs_iterator(args.datadir, side=side, H=H, W=W)
        h5path = os.path.join(args.datadir, f"evs_{side}.h5")

    elif dataset == "monohku":
        from utils.load_utils import monohku_evs_iterator
        H, W = 260, 346
        iterator = monohku_evs_iterator(args.datadir, H=H, W=W)
        h5path = os.path.join(args.datadir, "evs_davis346.h5")

    elif dataset == "eds":
        from utils.load_utils import eds_evs_iterator
        H, W = 480, 640
        iterator = eds_evs_iterator(args.datadir, H=H, W=W)
        h5path = os.path.join(args.datadir, "events.h5")

    elif dataset == "tumvie":
        from utils.load_utils import tumvie_evs_iterator
        H, W = 720, 1280
        camID = 2 if side == "left" else 3
        iterator = tumvie_evs_iterator(args.datadir, camID=camID, H=H, W=W)
        import glob
        h5path = glob.glob(os.path.join(args.datadir, f"*events_{side}.h5"))[0]

    elif dataset == "mvsec":
        from utils.load_utils import mvsec_h5_iterator
        H, W = 260, 346
        iterator = mvsec_h5_iterator(args.datadir, side=side)
        h5path = os.path.join(args.datadir, f"evs_{side}.h5")

    elif dataset == "vector":
        from utils.load_utils import vector_evs_iterator
        H, W = 480, 640
        iterator = vector_evs_iterator(args.datadir, side=side)
        h5path = os.path.join(args.datadir, f"evs_{side}.h5")

    elif dataset == "dsec":
        from utils.load_utils import dsec_evs_iterator
        H, W = 480, 640
        iterator = dsec_evs_iterator(args.datadir, side=side)
        h5path = os.path.join(args.datadir, f"evs_{side}.h5")

    elif dataset == "cear":
        from utils.load_utils import cear_evs_iterator
        H, W = 260, 346
        iterator = cear_evs_iterator(args.datadir, side=side)
        h5path = os.path.join(args.datadir, f"evs_{side}.h5")

    elif dataset == "fpv":
        from utils.load_utils import fpv_evs_iterator
        H, W = 260, 346
        iterator = fpv_evs_iterator(args.datadir, side=side)
        h5path = os.path.join(args.datadir, f"evs_{side}.h5")

    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    # --- 1. Inspect raw HDF5 file ---
    if os.path.exists(h5path):
        inspect_raw_h5(h5path)
        save_event_stats(h5path, os.path.join(args.outdir, "event_stats.png"), H, W)

        # Save one raw event frame from the first time window
        with h5py.File(h5path, "r") as f:
            prefix = "events/" if "events/x" in f else ""
            n = f[f"{prefix}x"].shape[0]
            chunk = min(n, 50_000)
            events_sample = {
                "x": np.asarray(f[f"{prefix}x"][:chunk]),
                "y": np.asarray(f[f"{prefix}y"][:chunk]),
                "t": np.asarray(f[f"{prefix}t"][:chunk]),
                "p": np.asarray(f[f"{prefix}p"][:chunk]),
            }
        save_event_frame(events_sample, H, W,
                         os.path.join(args.outdir, "raw_events_frame.png"))
    else:
        print(f"  [warn] h5 not found at {h5path}, skipping raw event inspection")

    # --- 2. Iterate voxel frames ---
    print(f"\n{'='*60}")
    print(f"Iterating {args.num_frames} voxel frames from iterator...")
    print(f"{'='*60}")

    voxels_for_summary = []
    for frame_idx, (voxel, intrinsics, ts_us) in enumerate(iterator):
        if frame_idx >= args.num_frames:
            break

        # Move to CPU for inspection (iterator yields .cuda() tensors)
        voxel_cpu = voxel.cpu() if voxel.is_cuda else voxel
        intr_cpu  = intrinsics.cpu() if intrinsics.is_cuda else intrinsics

        print_voxel_info(voxel_cpu, intr_cpu, ts_us, frame_idx)

        # Save individual voxel grid plot
        save_voxel_grid(voxel_cpu, ts_us,
                        os.path.join(args.outdir, f"voxel_frame_{frame_idx:04d}.png"),
                        frame_idx=frame_idx)

        # Also save as numpy
        np.save(os.path.join(args.outdir, f"voxel_frame_{frame_idx:04d}.npy"),
                voxel_cpu.numpy())

        voxels_for_summary.append((voxel_cpu, ts_us))

    # --- 3. Summary grid ---
    if voxels_for_summary:
        save_voxel_summary(voxels_for_summary,
                           os.path.join(args.outdir, "voxel_summary.png"))

    print(f"\nDone. All outputs saved to: {args.outdir}")
    print("\nData type summary:")
    print("  Raw events (from .h5) : dict with keys x,y,t,p — all np.ndarray")
    print("  Voxel grid (per frame): torch.Tensor  shape=(5, H, W)  float32")
    print("  Intrinsics            : torch.Tensor  shape=(4,)  [fx, fy, cx, cy]")
    print("  Timestamp             : int  microseconds (us)")


if __name__ == "__main__":
    main()
