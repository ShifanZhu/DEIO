#!/usr/bin/env python3
"""
Run patch tracking on event H5 files.

This script now supports a tracking-only mode that does not estimate poses:

1. Initialize patches once from the first usable frame/bin.
2. Pick the top-K scorer locations globally (default: 80).
3. Track those patches through the following frames/bins with the repo's
   learned feature extractor, correlation volume, and update operator.
4. Do not run bundle adjustment or visual-inertial optimization.

Legacy graph/SLAM-backed paths are still available through `--backend devo`
and `--backend deio`.

LD_LIBRARY_PATH=/home/s/repos/tool/miniconda3/envs/DEIO/lib:$LD_LIBRARY_PATH 
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/home/s/repos/DEIO 
/home/s/repos/tool/miniconda3/envs/DEIO/bin/python 
/home/s/repos/DEIO/script/eval_deio/track_h5_patches.py 
  --network ../SDEVO/DEVO/DEVO.pth 
  --config /home/s/repos/DEIO/config/vector.yaml 
  --h5 /media/s/rell/tro/vector-processed-old/vector/board-slow/board-slow.h5 
  --backend tracker 
  --save-raw 
  --no-render 
  --max-frames 8 
  --output-dir /home/s/repos/DEIO/results/track_h5_tracker_smoke

LD_LIBRARY_PATH=/home/s/repos/tool/miniconda3/envs/DEIO/lib:$LD_LIBRARY_PATH 
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/home/s/repos/DEIO 
/home/s/repos/tool/miniconda3/envs/DEIO/bin/python 
/home/s/repos/DEIO/script/eval_deio/track_h5_patches.py 
  --network ../SDEVO/DEVO/DEVO.pth 
  --config /home/s/repos/DEIO/config/vector.yaml 
  --h5 /media/s/rell/tro/vector-processed-old/vector/board-slow/board-slow.h5
  --backend tracker 
  --output-dir /home/s/repos/DEIO/results/track_h5

  LD_LIBRARY_PATH=/home/s/repos/tool/miniconda3/envs/DEIO/lib:$LD_LIBRARY_PATH CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/home/s/repos/DEIO /home/s/repos/tool/miniconda3/envs/DEIO/bin/python /home/s/repos/DEIO/script/eval_deio/track_h5_patches.py --network ../SDEVO/DEVO/DEVO.pth --config /home/s/repos/DEIO/config/vector.yaml --h5 /media/s/rell/tro/vector-processed-old/vector/board-slow/board-slow.h5 --backend tracker --output-dir /home/s/repos/DEIO/results/track_h5
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import tempfile
import time
from collections import OrderedDict
from pathlib import Path

import cv2
import h5py
import numpy as np
import torch

os.environ.setdefault("MPLCONFIGDIR", tempfile.mkdtemp(prefix="matplotlib-"))

from evo.tools.settings import SETTINGS

SETTINGS["plot_backend"] = "Agg"

from dpvo import altcorr
from devo.config import cfg as BASE_CFG
from devo.enet import CorrBlock, eVONet
from devo.selector import PatchSelector
from utils.load_utils import (
    cear_h5_iterator,
    mvsec_h5_iterator,
    vector_preprocessed_h5_iterator,
    vector_raw_iterator,
)
from utils.voxel_utils import rescale, std


def discover_h5_files(paths: list[str]) -> list[Path]:
    """Resolve input files / directories to a sorted list of H5/HDF5 files."""
    seen: dict[Path, bool] = {}
    for raw in paths:
        p = Path(raw).resolve()
        if p.is_file() and p.suffix in {".h5", ".hdf5"}:
            seen[p] = True
            continue
        if p.is_dir():
            for f in sorted(list(p.glob("**/*.h5")) + list(p.glob("**/*.hdf5"))):
                seen[f.resolve()] = True
            continue
        print(f"Warning: skipping non-existent or non-H5 path: {p}")
    return sorted(seen.keys())


def seq_to_subdir(seq_name: str) -> str:
    """Map sequence name to VECtor GT subdirectory: 'desk_fast1' -> 'desk-fast'."""
    base = re.sub(r"\d+$", "", seq_name)
    return base.rstrip("_").replace("_", "-")


def load_imu_h5(h5_path: str) -> np.ndarray:
    """Load H5 IMU as [t_us, gx_deg/s, gy_deg/s, gz_deg/s, ax, ay, az]."""
    with h5py.File(h5_path, "r") as f:
        imu_t = f["imu/t_us"][:].astype(np.float64)
        imu_d = f["imu/data_raw"][:].astype(np.float64)

    n = min(len(imu_t), len(imu_d))
    imu_t, imu_d = imu_t[:n], imu_d[:n]

    all_imu = np.zeros((n, 7), dtype=np.float64)
    all_imu[:, 0] = imu_t
    all_imu[:, 1:4] = imu_d[:, :3] * (180.0 / math.pi)
    all_imu[:, 4:7] = imu_d[:, 3:6]
    return all_imu[all_imu[:, 0].argsort()]


def load_imu_vector_raw(imu_txt: str) -> np.ndarray:
    """Load VECtor raw IMU txt as [t_us, gx_deg/s, gy_deg/s, gz_deg/s, ax, ay, az]."""
    data = np.loadtxt(imu_txt, comments="#")
    n = len(data)
    all_imu = np.zeros((n, 7), dtype=np.float64)
    all_imu[:, 0] = data[:, 0] * 1e6
    all_imu[:, 1:4] = data[:, 1:4] * (180.0 / math.pi)
    all_imu[:, 4:7] = data[:, 4:7]
    return all_imu[all_imu[:, 0].argsort()]


def load_vector_raw_frame_timestamps(gt_txt: str) -> np.ndarray:
    """Load absolute frame timestamps in us from VECtor GT txt."""
    data = np.loadtxt(gt_txt, comments="#")
    return (data[:, 0] * 1e6).astype(np.float64)


def detect_preprocessed_h5_type(h5_path: Path) -> str:
    """Return one of: cear, mvsec, vector_preprocessed."""
    with h5py.File(h5_path, "r") as f:
        if "events" in f and "time_images" in f["events"]:
            return "vector_preprocessed"
        if "events" in f and "x" in f["events"] and "slices_index" in f["events"]:
            parts = {p.lower() for p in h5_path.parts}
            if "cear" in parts:
                return "cear"
            return "mvsec"
    raise ValueError(f"Unsupported H5 layout for tracking: {h5_path}")


def infer_hw(dataset_type: str, height: int | None, width: int | None) -> tuple[int, int]:
    """Infer dataset-native resolution unless explicitly overridden."""
    if height is not None and width is not None:
        return height, width

    defaults = {
        "cear": (240, 320),
        "mvsec": (260, 346),
        "vector_preprocessed": (480, 640),
        "vector_raw": (480, 640),
    }
    if dataset_type not in defaults:
        raise ValueError(f"No default resolution known for dataset_type={dataset_type}")
    return defaults[dataset_type]


def build_iterator(
    h5_path: Path,
    stride: int,
    height: int | None,
    width: int | None,
    skip_start_s: float,
    gtdir: str | None,
):
    """Build the appropriate event iterator for the requested H5."""
    is_raw_vector = h5_path.suffix == ".hdf5"
    seq_name = h5_path.name.split(".")[0] if is_raw_vector else h5_path.stem

    if is_raw_vector:
        height, width = infer_hw("vector_raw", height, width)
        if gtdir is None:
            raise ValueError(f"Raw VECtor HDF5 requires --gtdir for {h5_path}")

        subdir = seq_to_subdir(seq_name)
        gt_txt = Path(gtdir) / subdir / f"{seq_name}.synced.gt.txt"
        imu_txt = Path(gtdir) / subdir / f"{seq_name}.synced.imu.txt"
        if not gt_txt.exists():
            raise FileNotFoundError(f"Missing VECtor GT txt: {gt_txt}")
        if not imu_txt.exists():
            raise FileNotFoundError(f"Missing VECtor IMU txt: {imu_txt}")

        tss_frames_us = load_vector_raw_frame_timestamps(str(gt_txt))
        iterator = vector_raw_iterator(
            str(h5_path), tss_frames_us, stride=stride, H=height, W=width
        )
        return {
            "seq_name": seq_name,
            "dataset_type": "vector_raw",
            "iterator": iterator,
            "imu": load_imu_vector_raw(str(imu_txt)),
            "height": height,
            "width": width,
        }

    dataset_type = detect_preprocessed_h5_type(h5_path)
    height, width = infer_hw(dataset_type, height, width)
    if dataset_type == "cear":
        iterator = cear_h5_iterator(
            str(h5_path), stride=stride, H=height, W=width, skip_start_s=skip_start_s
        )
    elif dataset_type == "vector_preprocessed":
        iterator = vector_preprocessed_h5_iterator(
            str(h5_path), stride=stride, H=height, W=width, skip_start_s=skip_start_s
        )
    else:
        iterator = mvsec_h5_iterator(
            str(h5_path), stride=stride, H=height, W=width, skip_start_s=skip_start_s
        )

    all_imu = None
    with h5py.File(h5_path, "r") as f:
        if "imu" in f and "t_us" in f["imu"] and "data_raw" in f["imu"]:
            all_imu = load_imu_h5(str(h5_path))

    return {
        "seq_name": seq_name,
        "dataset_type": dataset_type,
        "iterator": iterator,
        "imu": all_imu,
        "height": height,
        "width": width,
    }


def limit_iterator(iterator, max_frames: int | None):
    """Optionally stop an iterator after a fixed number of frames."""
    if max_frames is None or max_frames <= 0:
        yield from iterator
        return

    for i, item in enumerate(iterator):
        if i >= max_frames:
            break
        yield item


def load_tracking_network(args, runtime_cfg):
    """Load eVONet once for tracker-only mode."""
    checkpoint = torch.load(args.network, map_location="cpu", weights_only=False)
    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = OrderedDict(
            (k.replace("module.", ""), v)
            for k, v in checkpoint.items()
            if "update.lmbda" not in k
        )

    dim_inet = args.dim_inet
    dim_fnet = args.dim_fnet
    dim = args.dim
    if "patchify.inet.conv2.weight" in state_dict:
        dim_inet = state_dict["patchify.inet.conv2.weight"].shape[0]
    if "patchify.fnet.conv2.weight" in state_dict:
        dim_fnet = state_dict["patchify.fnet.conv2.weight"].shape[0]
    if "patchify.fnet.conv1.weight" in state_dict:
        dim = state_dict["patchify.fnet.conv1.weight"].shape[0]

    network = eVONet(
        runtime_cfg,
        dim_inet=dim_inet,
        dim_fnet=dim_fnet,
        dim=dim,
        patch_selector=runtime_cfg.PATCH_SELECTOR,
    )
    missing, unexpected = network.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"Warning: missing {len(missing)} checkpoint keys")
    if unexpected:
        print(f"Warning: unexpected {len(unexpected)} checkpoint keys")

    network.cuda().eval()
    print(
        f"Loaded tracking network from {args.network} "
        f"(dim_inet={network.dim_inet}, dim_fnet={network.dim_fnet}, dim={network.dim})"
    )
    return network


def normalize_event_voxel(image: torch.Tensor, norm: str) -> torch.Tensor:
    """Match the repo's voxel normalization used before patch extraction."""
    image = image[None, None].float()
    if image.shape[-1] == 346:
        image = image[..., 1:-1]

    norm = norm.lower()
    if norm == "none":
        return image
    if norm in {"rescale", "norm"}:
        return rescale(image)
    if norm in {"standard", "std"}:
        return std(image, sequence=False)
    if norm in {"standard2", "std2"}:
        return std(image)
    raise NotImplementedError(f"Unsupported normalization: {norm}")


def choose_tracker_norm(dataset_type: str, requested_norm: str, voxel: torch.Tensor) -> tuple[str, str | None]:
    """Avoid known-degenerate normalization for sparse one-hot preprocessed voxels."""
    norm = requested_norm.lower()
    if dataset_type == "vector_preprocessed" and norm in {"standard", "std", "standard2", "std2"}:
        nonzero = voxel[voxel != 0]
        if nonzero.numel() > 0:
            span = float((nonzero.max() - nonzero.min()).item())
            if span <= 1e-8:
                return "none", (
                    "vector_preprocessed voxels have constant nonzero values; "
                    f"{requested_norm} normalization becomes degenerate, so tracker mode is falling back to none"
                )
    return requested_norm, None


def value_map_to_bgr(value_map: np.ndarray) -> np.ndarray:
    """Render a signed 2D map into a light-weight BGR image."""
    pos = np.clip(value_map, 0.0, None)
    neg = np.clip(-value_map, 0.0, None)

    img = np.full((value_map.shape[0], value_map.shape[1], 3), 255, dtype=np.uint8)
    pos_max = float(pos.max()) if pos.size else 0.0
    neg_max = float(neg.max()) if neg.size else 0.0

    if pos_max > 0.0:
        pos_u8 = np.clip(255.0 * pos / pos_max, 0.0, 255.0).astype(np.uint8)
        mask = pos_u8 > 0
        img[mask, 0] = 255 - pos_u8[mask]
        img[mask, 1] = 255 - pos_u8[mask]
        img[mask, 2] = 255

    if neg_max > 0.0:
        neg_u8 = np.clip(255.0 * neg / neg_max, 0.0, 255.0).astype(np.uint8)
        mask = neg_u8 > 0
        img[mask, 0] = 255
        img[mask, 1] = 255 - neg_u8[mask]
        img[mask, 2] = 255 - neg_u8[mask]

    return img


def voxel_to_bgr(voxel: torch.Tensor) -> np.ndarray:
    """Render a voxel grid by summing all bins."""
    acc = voxel.detach().float().sum(dim=0).cpu().numpy()
    return value_map_to_bgr(acc)


def voxel_bins_to_bgr_grid(voxel: torch.Tensor, max_cols: int = 3, pad: int = 6) -> tuple[np.ndarray, list[tuple[int, int]]]:
    """Render each voxel bin as its own panel in a compact grid."""
    voxel_np = voxel.detach().float().cpu().numpy()
    num_bins, h, w = voxel_np.shape
    cols = min(max_cols, num_bins)
    rows = math.ceil(num_bins / cols)

    canvas_h = rows * h + (rows - 1) * pad
    canvas_w = cols * w + (cols - 1) * pad
    canvas = np.full((canvas_h, canvas_w, 3), 255, dtype=np.uint8)
    offsets: list[tuple[int, int]] = []

    for b in range(num_bins):
        row = b // cols
        col = b % cols
        y0 = row * (h + pad)
        x0 = col * (w + pad)
        offsets.append((x0, y0))

        tile = value_map_to_bgr(voxel_np[b])
        canvas[y0 : y0 + h, x0 : x0 + w] = tile
        cv2.putText(
            canvas,
            f"bin {b}",
            (x0 + 8, y0 + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (40, 40, 40),
            1,
            cv2.LINE_AA,
        )

    return canvas, offsets


def extract_frame_features(network, images: torch.Tensor, mixed_precision: bool):
    """Run scorer, fnet, and inet on a single frame/bin."""
    with torch.amp.autocast("cuda", enabled=mixed_precision):
        fmap = network.patchify.fnet(images) / 4.0
        imap = network.patchify.inet(images) / 4.0
    if not hasattr(network.patchify, "scorer"):
        raise RuntimeError("Tracker mode requires scorer-based patch selection")

    # Keep scorer selection in fp32 to avoid collapsing small score differences
    # into an all-zero map on sparse event frames.
    with torch.amp.autocast("cuda", enabled=False):
        score_logits = network.patchify.scorer(images.float())
        scores = torch.sigmoid(score_logits.float())

    return fmap.float(), imap.float(), scores.float()


def select_topk_patch_centers(
    score_map: torch.Tensor,
    num_patches: int,
    patch_radius: int,
    coord_offset: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pick the top-K highest-score patch centers from the first frame."""
    score_map = torch.nan_to_num(score_map.float(), nan=0.0, posinf=0.0, neginf=0.0)
    if not torch.isfinite(score_map).all():
        raise RuntimeError("score map contains non-finite values")
    if score_map.numel() == 0:
        raise RuntimeError("empty score map")
    if float(score_map.max().item() - score_map.min().item()) <= 1e-8:
        raise RuntimeError("degenerate flat score map")

    h, w = score_map.shape
    valid = torch.zeros_like(score_map, dtype=torch.bool)
    valid[patch_radius : h - patch_radius, patch_radius : w - patch_radius] = True
    masked = score_map.masked_fill(~valid, float("-inf")).reshape(-1)

    k = min(num_patches, int(valid.sum().item()))
    if k <= 0:
        raise RuntimeError("No valid patch locations remain after boundary masking")

    values, flat_idx = torch.topk(masked, k=k, dim=0)
    x = (flat_idx % w).view(1, k)
    y = torch.div(flat_idx, w, rounding_mode="floor").view(1, k)
    centers = torch.stack([x + coord_offset, y + coord_offset], dim=-1).float()
    return centers, values.view(k)


def select_patch_centers_native(
    score_map: torch.Tensor,
    num_patches: int,
    selector_mode: str,
    selector_use_grid: bool,
    coord_offset: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Select patch centers with the same PatchSelector used by network.patchify()."""
    scores = torch.nan_to_num(score_map.float(), nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)
    selector = PatchSelector(selector_mode, grid=selector_use_grid)
    x, y = selector(scores[None, None], num_patches)
    coords = torch.stack([x, y], dim=-1).float()
    values = altcorr.patchify(scores[None, None], coords, 0).view(-1).float()
    centers = coords.view(1, -1, 2) + coord_offset
    return centers, values


def build_fallback_score_map(
    voxel: torch.Tensor,
    fmap: torch.Tensor,
    feat_h: int,
    feat_w: int,
) -> torch.Tensor:
    """Fallback saliency map when the learned scorer is flat.

    For VECTOR preprocessed H5 the input is often a one-hot temporal bin per pixel,
    so plain event activity can be spatially constant. Use a temporally weighted
    surface first; if that is still flat, fall back to feature magnitude.
    """
    bins = voxel.shape[0]
    bin_weights = torch.linspace(0.0, 1.0, bins, device=voxel.device, dtype=torch.float32)
    weighted = (voxel.float() * bin_weights[:, None, None]).sum(dim=0, keepdim=True).unsqueeze(0)
    weighted = torch.nn.functional.interpolate(
        weighted,
        size=(feat_h, feat_w),
        mode="bilinear",
        align_corners=False,
    )[0, 0]
    weighted = torch.nan_to_num(weighted, nan=0.0, posinf=0.0, neginf=0.0)
    if float(weighted.max().item() - weighted.min().item()) > 1e-8:
        return weighted

    fmap_score = fmap[0, 0].float().pow(2).sum(dim=0).sqrt()
    fmap_score = torch.nan_to_num(fmap_score, nan=0.0, posinf=0.0, neginf=0.0)
    return fmap_score


def extract_patch_features(
    fmap: torch.Tensor,
    imap: torch.Tensor,
    centers: torch.Tensor,
    patch_size: int,
    dim_fnet: int,
    dim_inet: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Extract patch descriptors and context features at given centers."""
    b = fmap.shape[0]
    if b != 1:
        raise ValueError(f"Tracker expects batch size 1, got {b}")

    gmap = altcorr.patchify(fmap[0], centers, patch_size // 2)
    gmap = gmap.view(b, -1, dim_fnet, patch_size, patch_size).float()

    ctx = altcorr.patchify(imap[0], centers, 0)
    ctx = ctx.view(b, -1, dim_inet, 1, 1).squeeze(-1).squeeze(-1).float()
    return gmap, ctx


def centers_to_patch_coords(centers: torch.Tensor, patch_size: int) -> torch.Tensor:
    """Expand patch centers into a dense PxP coordinate grid."""
    radius = patch_size // 2
    offs = torch.arange(-radius, radius + 1, device=centers.device, dtype=centers.dtype)
    oy, ox = torch.meshgrid(offs, offs, indexing="ij")
    offsets = torch.stack([ox, oy], dim=-1).view(1, 1, patch_size, patch_size, 2)
    return centers[:, :, None, None, :] + offsets


def clamp_patch_centers(
    centers: torch.Tensor, feat_h: int, feat_w: int, patch_size: int
) -> torch.Tensor:
    """Keep patch centers inside the feature map so patch extraction stays valid."""
    radius = patch_size // 2
    centers = centers.clone()
    centers[..., 0] = centers[..., 0].clamp(radius, feat_w - radius - 1)
    centers[..., 1] = centers[..., 1].clamp(radius, feat_h - radius - 1)
    return centers


def valid_track_mask(
    centers: torch.Tensor,
    feat_h: int,
    feat_w: int,
    patch_size: int,
) -> torch.Tensor:
    """Return a mask for tracks whose patch centers remain safely in-bounds."""
    radius = patch_size // 2
    finite = torch.isfinite(centers).all(dim=-1)
    valid_x = (centers[..., 0] >= radius) & (centers[..., 0] <= (feat_w - radius - 1))
    valid_y = (centers[..., 1] >= radius) & (centers[..., 1] <= (feat_h - radius - 1))
    return finite & valid_x & valid_y


def build_event_support_maps(
    voxel: torch.Tensor,
    patch_size: int,
    feature_stride: int = 4,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build dense image-space support maps for local event mass and active pixels."""
    activity = torch.nan_to_num(voxel.float(), nan=0.0, posinf=0.0, neginf=0.0).abs().sum(dim=0)
    if activity.shape[-1] == 346:
        activity = activity[..., 1:-1]

    active = (activity > eps).float()
    radius_px = max(1, (patch_size * feature_stride) // 2)
    kernel_size = 2 * radius_px + 1
    kernel = torch.ones(
        (1, 1, kernel_size, kernel_size),
        device=activity.device,
        dtype=activity.dtype,
    )

    activity = activity[None, None]
    active = active[None, None]
    support_mass = torch.nn.functional.conv2d(activity, kernel, padding=radius_px)
    support_count = torch.nn.functional.conv2d(active, kernel, padding=radius_px)
    return support_mass[0, 0], support_count[0, 0]


def supported_track_mask(
    centers: torch.Tensor,
    support_mass_map: torch.Tensor,
    support_count_map: torch.Tensor,
    min_event_support: float,
    min_event_pixels: float,
    feature_stride: int = 4,
) -> torch.Tensor:
    """Return a mask for tracks whose current footprint has enough event support."""
    if centers.shape[1] == 0:
        return torch.zeros((0,), dtype=torch.bool, device=centers.device)

    x = (feature_stride * (centers[0, :, 0] + 0.5)).round().long().clamp(0, support_mass_map.shape[1] - 1)
    y = (feature_stride * (centers[0, :, 1] + 0.5)).round().long().clamp(0, support_mass_map.shape[0] - 1)
    mass_ok = support_mass_map[y, x] >= float(min_event_support)
    count_ok = support_count_map[y, x] >= float(min_event_pixels)
    return mass_ok & count_ok


def centers_to_pixel_points(
    centers: torch.Tensor,
    feature_stride: float = 4.0,
    center_offset: float = 0.5,
) -> np.ndarray:
    """Convert feature-map centers to image-space patch centers."""
    if centers.shape[1] == 0:
        return np.zeros((0, 2), dtype=np.float32)
    points = feature_stride * (centers[0].detach().float().cpu().numpy() + center_offset)
    return np.asarray(points, dtype=np.float32)


def normalize_pixel_points(points_px: np.ndarray, intrinsics: torch.Tensor) -> np.ndarray:
    """Normalize image points with pinhole intrinsics [fx, fy, cx, cy]."""
    if points_px.size == 0:
        return np.zeros((0, 2), dtype=np.float32)

    fx, fy, cx, cy = intrinsics.detach().float().cpu().tolist()
    fx = max(float(fx), 1e-6)
    fy = max(float(fy), 1e-6)

    points_n = points_px.astype(np.float32, copy=True)
    points_n[:, 0] = (points_n[:, 0] - float(cx)) / fx
    points_n[:, 1] = (points_n[:, 1] - float(cy)) / fy
    return points_n


def ransac_epipolar_inlier_mask(
    prev_centers: torch.Tensor,
    curr_centers: torch.Tensor,
    intrinsics: torch.Tensor,
    reproj_thresh: float,
    confidence: float,
    min_points: int,
    min_inliers: int,
) -> torch.Tensor:
    """Return a conservative inlier mask from normalized-point FM_RANSAC."""
    num_tracks = curr_centers.shape[1]
    keep_all = torch.ones(num_tracks, dtype=torch.bool, device=curr_centers.device)

    required_points = max(8, int(min_points))
    required_inliers = max(8, int(min_inliers))
    if num_tracks < required_points:
        return keep_all

    pts0_px = centers_to_pixel_points(prev_centers)
    pts1_px = centers_to_pixel_points(curr_centers)
    pts0_n = np.ascontiguousarray(normalize_pixel_points(pts0_px, intrinsics))
    pts1_n = np.ascontiguousarray(normalize_pixel_points(pts1_px, intrinsics))

    try:
        _F, mask = cv2.findFundamentalMat(
            pts0_n,
            pts1_n,
            cv2.FM_RANSAC,
            float(reproj_thresh),
            float(confidence),
        )
    except cv2.error:
        return keep_all

    if mask is None:
        return keep_all

    inliers = np.asarray(mask).reshape(-1).astype(bool)
    if inliers.shape[0] != num_tracks or int(inliers.sum()) < required_inliers:
        return keep_all

    return torch.from_numpy(inliers).to(device=curr_centers.device, dtype=torch.bool)


def filter_track_state(
    centers: torch.Tensor,
    template_gmap: torch.Tensor,
    template_ctx: torch.Tensor,
    net_state: torch.Tensor | None,
    weights: torch.Tensor | None,
    survivor_idx: torch.Tensor,
    keep_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None, torch.Tensor]:
    """Apply a 1D survivor mask to all per-track tensors."""
    centers = centers[:, keep_mask]
    template_gmap = template_gmap[:, keep_mask]
    template_ctx = template_ctx[:, keep_mask]
    if net_state is not None:
        net_state = net_state[:, keep_mask]
    if weights is not None:
        weights = weights[:, keep_mask]
    survivor_idx = survivor_idx[keep_mask]
    return centers, template_gmap, template_ctx, net_state, weights, survivor_idx


def filter_track_history(track_history_px: list[np.ndarray], survivor_idx: np.ndarray) -> list[np.ndarray]:
    """Keep only surviving tracks across the current interval history."""
    return [hist[survivor_idx] for hist in track_history_px]


@torch.no_grad()
def track_patches_step(
    network,
    template_gmap: torch.Tensor,
    template_ctx: torch.Tensor,
    prev_centers: torch.Tensor,
    fmap_tgt: torch.Tensor,
    tracker_steps: int,
    corr_radius: int,
    event_support_mass_map: torch.Tensor | None = None,
    event_support_count_map: torch.Tensor | None = None,
    min_event_support: float = 1.0,
    min_event_pixels: float = 1.0,
    net_state: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Track one set of patches into the next frame/bin without BA."""
    num_tracks = prev_centers.shape[1]
    feat_h, feat_w = fmap_tgt.shape[-2:]
    fmap_tgt = torch.nan_to_num(fmap_tgt.float(), nan=0.0, posinf=0.0, neginf=0.0)
    template_gmap = torch.nan_to_num(template_gmap.float(), nan=0.0, posinf=0.0, neginf=0.0)
    template_ctx = torch.nan_to_num(template_ctx.float(), nan=0.0, posinf=0.0, neginf=0.0)
    centers = prev_centers.clone()
    survivor_idx = torch.arange(num_tracks, dtype=torch.long, device=prev_centers.device)

    if num_tracks == 0:
        empty_coords = prev_centers.new_zeros((1, 0, network.P, network.P, 2))
        empty_weights = prev_centers.new_zeros((1, 0, 2))
        empty_net = prev_centers.new_zeros((1, 0, network.dim_inet))
        return centers, empty_coords, empty_weights, template_gmap, template_ctx, survivor_idx, empty_net

    if net_state is None or net_state.shape[1] != num_tracks:
        net = torch.zeros(
            1,
            num_tracks,
            network.dim_inet,
            device=prev_centers.device,
            dtype=torch.float,
        )
    else:
        net = net_state.clone().float()
    weights = torch.ones(1, num_tracks, 2, device=prev_centers.device, dtype=torch.float)

    for _ in range(tracker_steps):
        if centers.shape[1] == 0:
            break
        num_tracks = centers.shape[1]
        ii = torch.zeros(num_tracks, dtype=torch.long, device=prev_centers.device)
        jj = torch.zeros(num_tracks, dtype=torch.long, device=prev_centers.device)
        kk = torch.arange(num_tracks, dtype=torch.long, device=prev_centers.device)
        corr_fn = CorrBlock(fmap_tgt, template_gmap, radius=corr_radius, dropout=0.0, levels=[1, 4])
        coords = centers_to_patch_coords(centers, network.P)
        corr = corr_fn(kk, jj, coords.permute(0, 1, 4, 2, 3).contiguous())
        corr = torch.nan_to_num(corr.float(), nan=0.0, posinf=0.0, neginf=0.0)
        net = torch.nan_to_num(net.float(), nan=0.0, posinf=0.0, neginf=0.0)
        net, (delta, weights, _) = network.update(net, template_ctx, corr, None, ii, jj, kk)
        net = torch.nan_to_num(net.float(), nan=0.0, posinf=0.0, neginf=0.0)
        delta = torch.nan_to_num(delta.float(), nan=0.0, posinf=0.0, neginf=0.0)
        weights = torch.nan_to_num(weights.float(), nan=0.0, posinf=0.0, neginf=0.0)
        centers = centers + delta

        valid = valid_track_mask(centers, feat_h, feat_w, network.P)[0]
        if not valid.any():
            centers, template_gmap, template_ctx, net, weights, survivor_idx = filter_track_state(
                centers,
                template_gmap,
                template_ctx,
                net,
                weights,
                survivor_idx,
                valid,
            )
            break

        if (~valid).any():
            centers, template_gmap, template_ctx, net, weights, survivor_idx = filter_track_state(
                centers,
                template_gmap,
                template_ctx,
                net,
                weights,
                survivor_idx,
                valid,
            )

    if event_support_mass_map is not None and event_support_count_map is not None and centers.shape[1] > 0:
        supported = supported_track_mask(
            centers,
            event_support_mass_map,
            event_support_count_map,
            min_event_support,
            min_event_pixels,
        )
        if (~supported).any():
            centers, template_gmap, template_ctx, net, weights, survivor_idx = filter_track_state(
                centers,
                template_gmap,
                template_ctx,
                net,
                weights,
                survivor_idx,
                supported,
            )

    if centers.shape[1] == 0:
        coords = prev_centers.new_zeros((1, 0, network.P, network.P, 2))
    else:
        coords = centers_to_patch_coords(centers, network.P)
    return centers, coords, weights.float(), template_gmap, template_ctx, survivor_idx, net.float()


def save_tracker_frame(
    tracks_dir: Path,
    frame_idx: int,
    timestamp_us: float,
    centers_feat: torch.Tensor,
    coords_feat: torch.Tensor,
    scores_init: np.ndarray,
    weights: torch.Tensor | None,
    init_frame_idx: int,
):
    """Dump one tracker frame to a compressed NPZ."""
    centers_feature = centers_feat[0].detach().cpu().numpy().astype(np.float32)
    coords_feature = coords_feat[0].detach().cpu().numpy().astype(np.float32)
    centers_pixel = centers_feature * 4.0
    coords_pixel = coords_feature * 4.0

    if weights is None:
        weights_np = np.full((centers_feature.shape[0], 2), np.nan, dtype=np.float32)
    else:
        weights_np = weights[0].detach().cpu().numpy().astype(np.float32)

    np.savez_compressed(
        tracks_dir / f"t{frame_idx:05d}.npz",
        frame_idx=np.int32(frame_idx),
        timestamp_us=np.float64(timestamp_us),
        init_frame_idx=np.int32(init_frame_idx),
        patch_idx=np.arange(len(centers_feature), dtype=np.int32),
        score_init=np.asarray(scores_init, dtype=np.float32),
        weights=weights_np,
        centers_feature=centers_feature,
        coords_feature=coords_feature,
        centers_pixel=centers_pixel,
        coords_pixel=coords_pixel,
    )


def save_summary_csv(csv_path: Path, rows: list[tuple], headers: list[str]) -> None:
    """Write a simple CSV summary without NumPy binary output."""
    with csv_path.open("w", encoding="utf-8") as f:
        f.write(",".join(headers) + "\n")
        for row in rows:
            f.write(",".join(str(v) for v in row) + "\n")


def save_tracker_meta_json(meta_path: Path, args, init_frame_idx: int | None, scores_init_np) -> None:
    """Write tracker metadata as JSON instead of NumPy binary."""
    meta = {
        "init_frame_idx": -1 if init_frame_idx is None else int(init_frame_idx),
        "tracker_patches": int(args.tracker_patches),
        "min_event_support": float(args.min_event_support),
        "min_event_pixels": float(args.min_event_pixels),
        "ransac_reproj_thresh": float(args.ransac_reproj_thresh),
        "ransac_confidence": float(args.ransac_confidence),
        "ransac_min_points": int(args.ransac_min_points),
        "ransac_min_inliers": int(args.ransac_min_inliers),
        "reinit_every": int(args.reinit_every),
        "tracker_steps": int(args.tracker_steps),
        "tracker_radius": int(args.tracker_radius),
        "refresh_template": bool(args.refresh_template),
        "score_init": [] if scores_init_np is None else np.asarray(scores_init_np, dtype=np.float32).tolist(),
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")


def render_tracker_frame(
    viz_dir: Path,
    frame_idx: int,
    base_img: np.ndarray,
    panel_offsets: list[tuple[int, int]],
    init_points_px: np.ndarray,
    track_history_px: list[np.ndarray],
    trail_length: int,
):
    """Render tracking on the current frame visualization, repeated over bin panels."""
    canvas = base_img.copy()
    start = max(0, len(track_history_px) - trail_length)
    recent = track_history_px[start:]
    num_tracks = recent[-1].shape[0]
    red = (0, 0, 255)
    green = (0, 255, 0)

    for x_off, y_off in panel_offsets:
        for k in range(num_tracks):
            p_init = init_points_px[k]
            if np.isfinite(p_init).all():
                cv2.circle(
                    canvas,
                    (int(round(p_init[0] + x_off)), int(round(p_init[1] + y_off))),
                    3,
                    red,
                    thickness=-1,
                    lineType=cv2.LINE_AA,
                )

            for i in range(1, len(recent)):
                p0 = recent[i - 1][k]
                p1 = recent[i][k]
                if not (np.isfinite(p0).all() and np.isfinite(p1).all()):
                    continue
                cv2.line(
                    canvas,
                    (int(round(p0[0] + x_off)), int(round(p0[1] + y_off))),
                    (int(round(p1[0] + x_off)), int(round(p1[1] + y_off))),
                    green,
                    1,
                    cv2.LINE_AA,
                )

            p = recent[-1][k]
            if np.isfinite(p).all():
                cv2.circle(
                    canvas,
                    (int(round(p[0] + x_off)), int(round(p[1] + y_off))),
                    3,
                    green,
                    thickness=-1,
                    lineType=cv2.LINE_AA,
                )

    cv2.imwrite(str(viz_dir / f"t{frame_idx:05d}.png"), canvas)


@torch.no_grad()
def run_tracker_only(h5_path: Path, runtime_cfg, args, out_root: Path, network) -> dict:
    """Run fixed-patch sequential tracking without pose optimization."""
    info = build_iterator(
        h5_path=h5_path,
        stride=args.stride,
        height=args.height,
        width=args.width,
        skip_start_s=args.skip_start_s,
        gtdir=args.gtdir,
    )
    iterator = limit_iterator(info["iterator"], args.max_frames)

    seq_name = info["seq_name"]
    dataset_type = info["dataset_type"]
    out_dir = out_root / Path(args.network).stem / dataset_type / seq_name
    out_dir.mkdir(parents=True, exist_ok=True)

    tracks_dir = out_dir / "tracks"
    viz_dir = out_dir / "flow_viz"
    if args.save_raw:
        tracks_dir.mkdir(parents=True, exist_ok=True)
    if not args.no_render:
        viz_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"\n=== Tracking {seq_name} ({dataset_type}) with backend=tracker "
        f"(patches={args.tracker_patches}, reinit_every={args.reinit_every}, "
        f"steps={args.tracker_steps}) ==="
    )

    scores_init_np = None
    init_points_px = None
    prev_centers = None
    template_gmap = None
    template_ctx = None
    tracker_net = None
    init_frame_idx = None
    effective_norm = runtime_cfg.NORM
    warned_norm_fallback = False

    track_history_px: list[np.ndarray] = []
    frame_rows = []

    t_start = time.perf_counter()
    num_tracked_frames = 0

    for frame_idx, (voxel, intrinsics, timestamp_us) in enumerate(iterator):
        effective_norm, reason = choose_tracker_norm(dataset_type, effective_norm, voxel)
        if reason is not None and not warned_norm_fallback:
            print(f"Frame {frame_idx}: {reason}")
            warned_norm_fallback = True

        voxel_norm = normalize_event_voxel(voxel, effective_norm)
        if not torch.isfinite(voxel_norm).all():
            if effective_norm.lower() != "none":
                print(
                    f"Frame {frame_idx}: normalization '{effective_norm}' produced non-finite values; "
                    "falling back to none for tracker mode"
                )
                effective_norm = "none"
                voxel_norm = normalize_event_voxel(voxel, effective_norm)
            if not torch.isfinite(voxel_norm).all():
                print(f"Skipping frame {frame_idx}: normalized voxel is still non-finite")
                continue

        voxel_for_viz = voxel
        if voxel_for_viz.shape[-1] == 346:
            voxel_for_viz = voxel_for_viz[..., 1:-1]
        fmap, imap, scores = extract_frame_features(network, voxel_norm, runtime_cfg.MIXED_PRECISION)
        feat_h, feat_w = fmap.shape[-2:]
        event_support_mass_map, event_support_count_map = build_event_support_maps(voxel, network.P)

        should_reinitialize = (
            prev_centers is None
            or prev_centers.shape[1] == 0
            or init_frame_idx is None
            or (frame_idx - init_frame_idx) >= args.reinit_every
        )

        if should_reinitialize:
            try:
                if args.tracker_selector == "native":
                    centers, score_vals = select_patch_centers_native(
                        scores[0, 0],
                        num_patches=args.tracker_patches,
                        selector_mode=runtime_cfg.SCORER_EVAL_MODE,
                        selector_use_grid=runtime_cfg.SCORER_EVAL_USE_GRID,
                        coord_offset=1,
                    )
                else:
                    centers, score_vals = select_topk_patch_centers(
                        scores[0, 0],
                        num_patches=args.tracker_patches,
                        patch_radius=network.P // 2,
                        coord_offset=1,
                    )
            except RuntimeError as exc:
                activity_scores = build_fallback_score_map(voxel, fmap, feat_h, feat_w)
                try:
                    centers, score_vals = select_topk_patch_centers(
                        activity_scores,
                        num_patches=args.tracker_patches,
                        patch_radius=network.P // 2,
                        coord_offset=0,
                    )
                    print(f"Frame {frame_idx}: using activity fallback because scorer selection failed ({exc})")
                except RuntimeError:
                    print(f"Skipping frame {frame_idx}: {exc}")
                    continue

            support_valid = supported_track_mask(
                centers,
                event_support_mass_map,
                event_support_count_map,
                args.min_event_support,
                args.min_event_pixels,
            )
            if not support_valid.any():
                print(
                    f"Skipping frame {frame_idx}: selected patches have "
                    f"< {args.min_event_support:g} event support or "
                    f"< {args.min_event_pixels:g} active event pixels"
                )
                continue
            if (~support_valid).any():
                removed = int((~support_valid).sum().item())
                centers = centers[:, support_valid]
                score_vals = score_vals[support_valid]
                print(
                    f"Frame {frame_idx}: removed {removed} initialization patches with "
                    f"< {args.min_event_support:g} event support or "
                    f"< {args.min_event_pixels:g} active event pixels"
                )

            prev_centers = centers.cuda()
            template_gmap, template_ctx = extract_patch_features(
                fmap, imap, prev_centers, network.P, network.dim_fnet, network.dim_inet
            )
            tracker_net = None
            scores_init_np = score_vals.detach().cpu().numpy().astype(np.float32)
            init_frame_idx = frame_idx

            coords_feat = centers_to_patch_coords(prev_centers, network.P)
            centers_px = prev_centers[0].detach().cpu().numpy().astype(np.float32) * 4.0
            init_points_px = centers_px.copy()
            track_history_px = [centers_px]

            if args.save_raw:
                save_tracker_frame(
                    tracks_dir,
                    frame_idx,
                    float(timestamp_us),
                    prev_centers,
                    coords_feat,
                    scores_init_np,
                    weights=None,
                    init_frame_idx=init_frame_idx,
                )
            if not args.no_render:
                viz_img, viz_offsets = voxel_bins_to_bgr_grid(voxel_for_viz)
                render_tracker_frame(
                    viz_dir,
                    frame_idx,
                    viz_img,
                    viz_offsets,
                    init_points_px,
                    track_history_px,
                    args.trail_length,
                )

            frame_rows.append((frame_idx, float(timestamp_us), prev_centers.shape[1]))
            num_tracked_frames += 1
            print(f"Initialized {prev_centers.shape[1]} patches at frame {frame_idx}")
            continue

        source_centers = prev_centers.clone()
        old_num_tracks = prev_centers.shape[1]
        prev_centers, coords_feat, weights, template_gmap, template_ctx, survivor_idx, tracker_net = track_patches_step(
            network,
            template_gmap,
            template_ctx,
            prev_centers,
            fmap,
            tracker_steps=args.tracker_steps,
            corr_radius=args.tracker_radius,
            event_support_mass_map=event_support_mass_map,
            event_support_count_map=event_support_count_map,
            min_event_support=args.min_event_support,
            min_event_pixels=args.min_event_pixels,
            net_state=tracker_net,
        )

        removed_support = old_num_tracks - survivor_idx.numel()
        removed_ransac = 0
        if prev_centers.shape[1] > 0:
            source_centers = source_centers[:, survivor_idx]
            ransac_inliers = ransac_epipolar_inlier_mask(
                source_centers,
                prev_centers,
                intrinsics,
                reproj_thresh=args.ransac_reproj_thresh,
                confidence=args.ransac_confidence,
                min_points=args.ransac_min_points,
                min_inliers=args.ransac_min_inliers,
            )
            if (~ransac_inliers).any():
                removed_ransac = int((~ransac_inliers).sum().item())
                prev_centers, template_gmap, template_ctx, tracker_net, weights, survivor_idx = filter_track_state(
                    prev_centers,
                    template_gmap,
                    template_ctx,
                    tracker_net,
                    weights,
                    survivor_idx,
                    ransac_inliers,
                )
                coords_feat = centers_to_patch_coords(prev_centers, network.P)

        if survivor_idx.numel() != old_num_tracks:
            survivor_idx_np = survivor_idx.detach().cpu().numpy()
            track_history_px = filter_track_history(track_history_px, survivor_idx_np)
            init_points_px = init_points_px[survivor_idx_np]
            scores_init_np = scores_init_np[survivor_idx_np]

        if removed_support > 0:
            print(
                f"Frame {frame_idx}: removed {removed_support} "
                f"tracks that moved out of bounds or below {args.min_event_support:g} event support "
                f"or {args.min_event_pixels:g} active event pixels"
            )
        if removed_ransac > 0:
            print(
                f"Frame {frame_idx}: RANSAC removed {removed_ransac} tracks inconsistent with "
                "the normalized epipolar motion"
            )

        if prev_centers.shape[1] == 0:
            centers_px = np.empty((0, 2), dtype=np.float32)
            track_history_px.append(centers_px)
            if args.save_raw:
                save_tracker_frame(
                    tracks_dir,
                    frame_idx,
                    float(timestamp_us),
                    prev_centers,
                    coords_feat,
                    scores_init_np,
                    weights=weights,
                    init_frame_idx=init_frame_idx,
                )
            if not args.no_render:
                viz_img, viz_offsets = voxel_bins_to_bgr_grid(voxel_for_viz)
                render_tracker_frame(
                    viz_dir,
                    frame_idx,
                    viz_img,
                    viz_offsets,
                    init_points_px,
                    track_history_px,
                    args.trail_length,
                )
            frame_rows.append((frame_idx, float(timestamp_us), 0))
            num_tracked_frames += 1
            continue

        if args.refresh_template:
            template_gmap, template_ctx = extract_patch_features(
                fmap, imap, prev_centers, network.P, network.dim_fnet, network.dim_inet
            )

        centers_px = prev_centers[0].detach().cpu().numpy().astype(np.float32) * 4.0
        track_history_px.append(centers_px)

        if args.save_raw:
            save_tracker_frame(
                tracks_dir,
                frame_idx,
                float(timestamp_us),
                prev_centers,
                coords_feat,
                scores_init_np,
                weights=weights,
                init_frame_idx=init_frame_idx,
            )
        if not args.no_render:
            viz_img, viz_offsets = voxel_bins_to_bgr_grid(voxel_for_viz)
            render_tracker_frame(
                viz_dir,
                frame_idx,
                viz_img,
                viz_offsets,
                init_points_px,
                track_history_px,
                args.trail_length,
            )

        frame_rows.append((frame_idx, float(timestamp_us), prev_centers.shape[1]))
        num_tracked_frames += 1

    elapsed = time.perf_counter() - t_start
    avg_fps = num_tracked_frames / elapsed if elapsed > 0 and num_tracked_frames > 0 else None

    if args.save_raw and frame_rows:
        save_summary_csv(
            tracks_dir / "summary.csv",
            frame_rows,
            ["frame_idx", "timestamp_us", "num_tracks"],
        )
        save_tracker_meta_json(
            tracks_dir / "tracker_meta.json",
            args,
            init_frame_idx,
            scores_init_np,
        )

    if num_tracked_frames == 0:
        print(f"No valid tracker output produced for {seq_name}")
    else:
        print(f"Saved tracker output for {seq_name}: {num_tracked_frames} frames -> {out_dir}")

    if args.timing and avg_fps is not None:
        print(f"Tracker throughput: {avg_fps:.1f} FPS")

    return {
        "seq_name": seq_name,
        "dataset_type": dataset_type,
        "num_flow_frames": num_tracked_frames,
        "avg_fps": avg_fps,
        "out_dir": out_dir,
    }


def save_raw_tracks(out_dir: Path, flow_data: dict, timestamps_us: np.ndarray) -> None:
    """Save one compressed npz per tracked frame with patch/edge projections."""
    tracks_dir = out_dir / "tracks"
    tracks_dir.mkdir(parents=True, exist_ok=True)

    frame_rows = []
    for frame_idx, flow_i in flow_data.items():
        ii = flow_i["ii"].detach().cpu().numpy().astype(np.int32)
        jj = flow_i["jj"].detach().cpu().numpy().astype(np.int32)
        kk = flow_i["kk"].detach().cpu().numpy().astype(np.int32)
        coords_feature = flow_i["coords_est"][0].detach().cpu().numpy().astype(np.float32)

        p = coords_feature.shape[1]
        centers_feature = coords_feature[:, p // 2, p // 2, :]
        coords_pixel = coords_feature * 4.0
        centers_pixel = centers_feature * 4.0

        ts_us = float(timestamps_us[frame_idx]) if frame_idx < len(timestamps_us) else float("nan")
        np.savez_compressed(
            tracks_dir / f"t{frame_idx:05d}.npz",
            frame_idx=np.int32(frame_idx),
            timestamp_us=np.float64(ts_us),
            n_active_frames=np.int32(flow_i["n"]),
            host_frame_idx=ii,
            target_frame_idx=jj,
            patch_idx=kk,
            coords_feature=coords_feature,
            centers_feature=centers_feature,
            coords_pixel=coords_pixel,
            centers_pixel=centers_pixel,
        )

        frame_rows.append((frame_idx, ts_us, len(ii), int(flow_i["n"])))

    if frame_rows:
        save_summary_csv(
            tracks_dir / "summary.csv",
            frame_rows,
            ["frame_idx", "timestamp_us", "num_edges", "n_active_frames"],
        )


def run_legacy_backend(h5_path: Path, runtime_cfg, args, out_root: Path) -> dict:
    """Keep the previous DEVO / DEIO graph-backed flow export path available."""
    info = build_iterator(
        h5_path=h5_path,
        stride=args.stride,
        height=args.height,
        width=args.width,
        skip_start_s=args.skip_start_s,
        gtdir=args.gtdir,
    )

    iterator = limit_iterator(info["iterator"], args.max_frames)
    seq_name = info["seq_name"]
    dataset_type = info["dataset_type"]
    height = info["height"]
    width = info["width"]

    print(f"\n=== Tracking {seq_name} ({dataset_type}) with backend={args.backend} ===")

    from utils.eval_utils import EVO_run, run_DEIO2
    from utils.viz_utils import viz_flow_inference

    if args.backend == "deio":
        if info["imu"] is None:
            raise ValueError(f"Backend 'deio' requires IMU, but none was found for {h5_path}")
        _, timestamps_us, flow_data, avg_fps = run_DEIO2(
            str(h5_path.parent),
            runtime_cfg,
            args.network,
            iterator=iterator,
            _all_imu=info["imu"].copy(),
            viz=False,
            viz_flow=True,
            timing=args.timing,
            H=height,
            W=width,
            dim_inet=args.dim_inet,
            dim_fnet=args.dim_fnet,
            dim=args.dim,
        )
    else:
        _, timestamps_us, flow_data, avg_fps = EVO_run(
            str(h5_path.parent),
            runtime_cfg,
            args.network,
            iterator=iterator,
            viz=False,
            viz_flow=True,
            timing=args.timing,
            H=height,
            W=width,
            dim_inet=args.dim_inet,
            dim_fnet=args.dim_fnet,
            dim=args.dim,
        )

    out_dir = out_root / Path(args.network).stem / dataset_type / seq_name
    out_dir.mkdir(parents=True, exist_ok=True)

    if not flow_data:
        print(f"No flow_data was produced for {seq_name}")
        return {
            "seq_name": seq_name,
            "dataset_type": dataset_type,
            "num_flow_frames": 0,
            "avg_fps": avg_fps,
            "out_dir": out_dir,
        }

    if args.save_raw:
        save_raw_tracks(out_dir, flow_data, np.asarray(timestamps_us))

    if not args.no_render:
        viz_flow_inference(str(out_dir), flow_data, debug=args.debug_viz)

    print(f"Saved tracking for {seq_name}: {len(flow_data)} flow frames -> {out_dir}")

    return {
        "seq_name": seq_name,
        "dataset_type": dataset_type,
        "num_flow_frames": len(flow_data),
        "avg_fps": avg_fps,
        "out_dir": out_dir,
    }


def run_patch_tracking(h5_path: Path, runtime_cfg, args, out_root: Path, tracker_network=None) -> dict:
    if args.backend == "tracker":
        if tracker_network is None:
            raise RuntimeError("tracker backend requires a preloaded network")
        return run_tracker_only(h5_path, runtime_cfg, args, out_root, tracker_network)
    return run_legacy_backend(h5_path, runtime_cfg, args, out_root)


def main():
    parser = argparse.ArgumentParser(
        description="Run learned event patch tracking on H5 event data",
    )
    parser.add_argument("--network", required=True, help="Path to checkpoint (.pth/.pt)")
    parser.add_argument("--config", required=True, help="YAML config file")
    parser.add_argument("--h5", nargs="+", required=True, help="H5 file(s) or directories")
    parser.add_argument("--output-dir", default="results/track_h5", help="Output directory")
    parser.add_argument(
        "--backend",
        choices=["tracker", "devo", "deio"],
        default="tracker",
        help="Tracking backend: tracker-only, visual-only DEVO, or visual-inertial DEIO",
    )
    parser.add_argument(
        "--gtdir",
        default=None,
        help="Required for raw VECtor .hdf5 input to find GT/IMU txt files",
    )
    parser.add_argument("--height", type=int, default=None, help="Override input height")
    parser.add_argument("--width", type=int, default=None, help="Override input width")
    parser.add_argument("--stride", type=int, default=1, help="Use every N-th frame/window")
    parser.add_argument(
        "--skip-start-s",
        type=float,
        default=0.0,
        help="Skip the first N seconds for preprocessed H5",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Optional hard cap on processed frames per sequence",
    )
    parser.add_argument(
        "--save-raw",
        action="store_true",
        help="Save raw per-frame track arrays under tracks/ in addition to PNG visualization",
    )
    parser.add_argument(
        "--no-render",
        action="store_true",
        help="Skip PNG rendering and only run tracking / raw export",
    )
    parser.add_argument(
        "--debug-viz",
        action="store_true",
        help="Enable assertions in legacy viz_flow_inference",
    )
    parser.add_argument("--timing", action="store_true", help="Print runtime timing")
    parser.add_argument("--dim_inet", type=int, default=384)
    parser.add_argument("--dim_fnet", type=int, default=128)
    parser.add_argument("--dim", type=int, default=32)
    parser.add_argument(
        "--tracker-patches",
        type=int,
        default=None,
        help="Number of patches to track at each re-initialization (default: PATCHES_PER_FRAME from config)",
    )
    parser.add_argument(
        "--tracker-selector",
        choices=["native", "topk"],
        default="native",
        help="Patch initialization strategy: native PatchSelector from config, or global top-K scorer peaks",
    )
    parser.add_argument(
        "--reinit-every",
        type=int,
        default=20,
        help="Re-initialize from the current frame every N frames",
    )
    parser.add_argument(
        "--tracker-steps",
        type=int,
        default=6,
        help="Number of learned update iterations per frame pair",
    )
    parser.add_argument(
        "--tracker-radius",
        type=int,
        default=3,
        help="Local correlation radius used during tracker-only matching",
    )
    parser.add_argument(
        "--min-event-support",
        type=float,
        default=4.0,
        help="Minimum local event mass required to keep a patch during init and tracking",
    )
    parser.add_argument(
        "--min-event-pixels",
        type=float,
        default=4.0,
        help="Minimum number of active event pixels required to keep a patch during init and tracking",
    )
    parser.add_argument(
        "--ransac-reproj-thresh",
        type=float,
        default=3e-3,
        help="FM_RANSAC reprojection threshold in normalized-image coordinates",
    )
    parser.add_argument(
        "--ransac-confidence",
        type=float,
        default=0.9,
        help="FM_RANSAC confidence",
    )
    parser.add_argument(
        "--ransac-min-points",
        type=int,
        default=8,
        help="Minimum track count required before attempting FM_RANSAC",
    )
    parser.add_argument(
        "--ransac-min-inliers",
        type=int,
        default=4,
        help="Minimum inlier count required to accept the FM_RANSAC result",
    )
    parser.add_argument(
        "--trail-length",
        type=int,
        default=20,
        help="Number of recent frames to include in tracker visualization trails",
    )
    parser.add_argument(
        "--refresh-template",
        dest="refresh_template",
        action="store_true",
        help="Refresh patch descriptors after every frame (default)",
    )
    parser.add_argument(
        "--no-refresh-template",
        dest="refresh_template",
        action="store_false",
        help="Keep the first-frame descriptors fixed for the whole sequence",
    )
    parser.set_defaults(refresh_template=True)

    args = parser.parse_args()
    assert Path(args.network).is_file(), args.network

    runtime_cfg = BASE_CFG.clone()
    runtime_cfg.merge_from_file(args.config)
    if args.tracker_patches is None:
        args.tracker_patches = int(getattr(runtime_cfg, "PATCHES_PER_FRAME", 80))

    h5_files = discover_h5_files(args.h5)
    if not h5_files:
        raise RuntimeError("No .h5/.hdf5 files found from --h5 inputs")

    tracker_network = None
    if args.backend == "tracker":
        tracker_network = load_tracking_network(args, runtime_cfg)

    out_root = Path(args.output_dir).resolve()
    results = []
    for h5_path in h5_files:
        results.append(run_patch_tracking(h5_path, runtime_cfg, args, out_root, tracker_network))

    print("\n=== Patch Tracking Summary ===")
    for row in results:
        fps = f"{row['avg_fps']:.1f}" if row["avg_fps"] is not None else "N/A"
        print(
            f"{row['seq_name']}: frames={row['num_flow_frames']} "
            f"fps={fps} out={row['out_dir']}"
        )


if __name__ == "__main__":
    torch.manual_seed(1234)
    main()
