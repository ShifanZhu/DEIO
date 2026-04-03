#!/usr/bin/env python3
"""Debug tracker score maps on H5 event sequences.

This script reuses the tracker-only eval path to inspect exactly what the
scorer sees on each frame:
  - normalized voxel input
  - raw scorer logits
  - sigmoid score map
  - fallback score map
  - event-support maps
  - selected patch centers and which ones fail support filtering

Examples
--------
python script/debug/debug_score_map.py \
    --network checkpoints/tracker_base/010000.pth \
    --config config/vector.yaml \
    --h5 /media/s/rell/tro/vector-processed/vector/board-slow/board-slow.h5

python script/debug/debug_score_map.py \
    --network checkpoints/tracker_base/010000.pth \
    --config config/vector.yaml \
    --h5 /media/s/rell/tro/vector-processed/vector/board-slow/board-slow.h5 \
    --norms none std2 \
    --max-frames 10 \
    --frame-indices 0,5,9 \
    --tracker-selector topk

conda run -n DEIO python /home/s  DEIO /repos/DEIO/script/debug/debug_score_map.py --network /home/s/repos/DEIO/checkpoints/tracker_base/070000.pth --config /home/s/repos/DEIO/config/vector.yaml --h5 /media/s/rell/tro/vector-processed/vector/board-slow/board-slow.h5 --output-dir results/debug_score_map/base  --tracker-selector topk
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from devo.config import cfg as BASE_CFG
from script.eval_deio.track_h5_patches import (
    build_event_support_maps,
    build_fallback_score_map,
    build_iterator,
    choose_tracker_norm,
    discover_h5_files,
    extract_frame_features,
    limit_iterator,
    load_tracking_network,
    normalize_event_voxel,
    score_map_to_bgr,
    select_adjacent_top1_patch_centers,
    select_patch_centers_native,
    select_topk_patch_centers,
    supported_track_mask,
    value_map_to_bgr,
    voxel_bins_to_bgr_grid,
)


def sanitize_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(name)).strip("_")


def parse_frame_indices(spec: str | None) -> set[int] | None:
    if spec is None:
        return None

    result: set[int] = set()
    for chunk in spec.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "-" in chunk:
            start_s, end_s = chunk.split("-", 1)
            start = int(start_s)
            end = int(end_s)
            if end < start:
                start, end = end, start
            result.update(range(start, end + 1))
        else:
            result.add(int(chunk))
    return result


def label_tile(image: np.ndarray, text: str) -> np.ndarray:
    canvas = image.copy()
    cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 24), (0, 0, 0), -1)
    cv2.putText(
        canvas,
        text,
        (6, 17),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return canvas


def make_strip(images_with_labels: list[tuple[str, np.ndarray]]) -> np.ndarray:
    target_h = max(img.shape[0] for _, img in images_with_labels)
    tiles = []
    for label, img in images_with_labels:
        tile = np.asarray(img)
        if tile.ndim == 2:
            tile = cv2.cvtColor(tile.astype(np.uint8), cv2.COLOR_GRAY2BGR)
        elif tile.ndim == 3 and tile.shape[2] == 1:
            tile = cv2.cvtColor(tile[..., 0].astype(np.uint8), cv2.COLOR_GRAY2BGR)
        elif tile.ndim != 3 or tile.shape[2] != 3:
            raise ValueError(f"Unsupported image shape for make_strip(): {tile.shape}")

        if tile.dtype != np.uint8:
            tile = np.clip(tile, 0, 255).astype(np.uint8)

        if tile.shape[0] != target_h:
            target_w = int(round(tile.shape[1] * (target_h / tile.shape[0])))
            tile = cv2.resize(tile, (target_w, target_h), interpolation=cv2.INTER_NEAREST)

        tiles.append(label_tile(tile, label))

    return cv2.hconcat(tiles)


def signed_map_to_bgr(value_map: np.ndarray, upscale: int = 4) -> np.ndarray:
    image = value_map_to_bgr(np.asarray(value_map, dtype=np.float32))
    if upscale > 1:
        image = cv2.resize(
            image,
            (image.shape[1] * upscale, image.shape[0] * upscale),
            interpolation=cv2.INTER_NEAREST,
        )
    return image


def overlay_feature_points(
    base_bgr: np.ndarray,
    centers_feat: np.ndarray,
    supported_mask: np.ndarray | None = None,
    *,
    panel_offsets: list[tuple[int, int]] | None = None,
    scale: float = 4.0,
    center_offset: float = 0.5,
    subtitle: str | None = None,
) -> np.ndarray:
    canvas = base_bgr.copy()
    h, w = canvas.shape[:2]
    offsets = panel_offsets if panel_offsets is not None else [(0, 0)]
    if supported_mask is None:
        supported_mask = np.ones((len(centers_feat),), dtype=bool)

    for idx, xy in enumerate(centers_feat):
        px = int(round((float(xy[0]) + center_offset) * scale))
        py = int(round((float(xy[1]) + center_offset) * scale))
        color = (0, 220, 0) if bool(supported_mask[idx]) else (0, 0, 255)
        for x_off, y_off in offsets:
            x = int(np.clip(px + x_off, 0, w - 1))
            y = int(np.clip(py + y_off, 0, h - 1))
            cv2.circle(canvas, (x, y), 3, color, 1, cv2.LINE_AA)

    if subtitle:
        cv2.putText(
            canvas,
            subtitle,
            (6, h - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    return canvas


def summarize_array(values: np.ndarray) -> dict[str, float | int | bool | None]:
    arr = np.asarray(values, dtype=np.float32)
    finite = np.isfinite(arr)
    num_total = int(arr.size)
    num_finite = int(finite.sum())
    if num_finite == 0:
        return {
            "num_total": num_total,
            "num_finite": 0,
            "num_nonfinite": num_total,
            "min": None,
            "max": None,
            "mean": None,
            "std": None,
            "p01": None,
            "p50": None,
            "p95": None,
            "p99": None,
            "dynamic_range": None,
            "is_flat": False,
        }

    safe = arr[finite]
    min_v = float(safe.min())
    max_v = float(safe.max())
    dynamic_range = max_v - min_v
    return {
        "num_total": num_total,
        "num_finite": num_finite,
        "num_nonfinite": num_total - num_finite,
        "min": min_v,
        "max": max_v,
        "mean": float(safe.mean()),
        "std": float(safe.std()),
        "p01": float(np.percentile(safe, 1.0)),
        "p50": float(np.percentile(safe, 50.0)),
        "p95": float(np.percentile(safe, 95.0)),
        "p99": float(np.percentile(safe, 99.0)),
        "dynamic_range": float(dynamic_range),
        "is_flat": bool(dynamic_range <= 1e-8),
    }


def summarize_score_map(scores: np.ndarray) -> dict[str, float | int | bool | None]:
    summary = summarize_array(scores)
    finite = np.isfinite(scores)
    if int(finite.sum()) > 0:
        safe = scores[finite]
        summary.update(
            {
                "frac_ge_0.10": float(np.mean(safe >= 0.10)),
                "frac_ge_0.25": float(np.mean(safe >= 0.25)),
                "frac_ge_0.50": float(np.mean(safe >= 0.50)),
                "frac_ge_0.75": float(np.mean(safe >= 0.75)),
            }
        )
    else:
        summary.update(
            {
                "frac_ge_0.10": None,
                "frac_ge_0.25": None,
                "frac_ge_0.50": None,
                "frac_ge_0.75": None,
            }
        )
    return summary


def save_summary_csv(csv_path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    headers = list(rows[0].keys())
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)


def extract_frame_features_with_logits(
    network,
    images: torch.Tensor,
    mixed_precision: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    fmap, imap, scores = extract_frame_features(network, images, mixed_precision)
    with torch.amp.autocast("cuda", enabled=False):
        score_logits = network.patchify.scorer(images.float()).float()
    return fmap.float(), imap.float(), score_logits.float(), scores.float()


def resolve_norm_requests(args, runtime_cfg) -> list[str]:
    if args.norms:
        norms = [n.strip().lower() for n in args.norms if n.strip()]
    else:
        norms = [str(runtime_cfg.NORM).strip().lower()]

    resolved = []
    for norm in norms:
        if norm == "config":
            resolved.append(str(runtime_cfg.NORM).strip().lower())
        else:
            resolved.append(norm)
    return resolved


def run_selection(
    score_map: torch.Tensor,
    fallback_map: torch.Tensor,
    network,
    args,
    selector_mode: str,
    selector_use_grid: bool,
) -> tuple[torch.Tensor | None, torch.Tensor | None, bool, str | None]:
    try:
        if args.tracker_selector == "native":
            centers, values = select_patch_centers_native(
                score_map,
                num_patches=args.tracker_patches,
                selector_mode=selector_mode,
                selector_use_grid=selector_use_grid,
                coord_offset=1,
            )
        elif args.tracker_selector == "adjacent_top1":
            centers, values = select_adjacent_top1_patch_centers(
                score_map,
                num_patches=args.tracker_patches,
                patch_radius=network.P // 2,
                coord_offset=1,
            )
        else:
            centers, values = select_topk_patch_centers(
                score_map,
                num_patches=args.tracker_patches,
                patch_radius=network.P // 2,
                coord_offset=1,
            )
        return centers, values, False, None
    except RuntimeError as exc:
        try:
            if args.tracker_selector == "adjacent_top1":
                centers, values = select_adjacent_top1_patch_centers(
                    fallback_map,
                    num_patches=args.tracker_patches,
                    patch_radius=network.P // 2,
                    coord_offset=0,
                )
            else:
                centers, values = select_topk_patch_centers(
                    fallback_map,
                    num_patches=args.tracker_patches,
                    patch_radius=network.P // 2,
                    coord_offset=0,
                )
            return centers, values, True, str(exc)
        except RuntimeError as fallback_exc:
            return None, None, True, f"{exc}; fallback failed: {fallback_exc}"


def norm_tag(requested_norm: str, effective_norm: str) -> str:
    requested = sanitize_name(requested_norm)
    effective = sanitize_name(effective_norm)
    if requested == effective:
        return f"requested_{requested}"
    return f"requested_{requested}__effective_{effective}"


@torch.no_grad()
def process_sequence(h5_path: Path, runtime_cfg, args, out_root: Path, network) -> dict[str, object]:
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
    seq_out_dir = out_root / Path(args.network).stem / dataset_type / seq_name
    seq_out_dir.mkdir(parents=True, exist_ok=True)

    input_dir = seq_out_dir / "input"
    compare_dir = seq_out_dir / "compare"
    input_dir.mkdir(exist_ok=True)
    compare_dir.mkdir(exist_ok=True)

    requested_norms = resolve_norm_requests(args, runtime_cfg)
    selector_mode = args.selector_mode or str(runtime_cfg.SCORER_EVAL_MODE)
    selector_use_grid = (
        bool(args.selector_use_grid)
        if args.selector_use_grid is not None
        else bool(runtime_cfg.SCORER_EVAL_USE_GRID)
    )

    frame_filter = parse_frame_indices(args.frame_indices)
    per_frame_rows: list[dict[str, object]] = []
    per_norm_stats: dict[str, list[dict[str, object]]] = defaultdict(list)
    num_processed_frames = 0

    print(
        f"\n=== Debugging score maps for {seq_name} ({dataset_type}) "
        f"patches={args.tracker_patches} selector={args.tracker_selector} ==="
    )

    for frame_idx, (voxel, _intrinsics, timestamp_us) in enumerate(iterator):
        if (frame_idx % args.frame_step) != 0:
            continue
        if frame_filter is not None and frame_idx not in frame_filter:
            continue

        voxel_for_viz = voxel
        if voxel_for_viz.shape[-1] == 346:
            voxel_for_viz = voxel_for_viz[..., 1:-1]
        input_grid, panel_offsets = voxel_bins_to_bgr_grid(voxel_for_viz)
        input_path = input_dir / f"frame{frame_idx:05d}_voxel_bins.png"
        if not input_path.exists():
            cv2.imwrite(str(input_path), input_grid)

        support_mass_map, support_count_map = build_event_support_maps(voxel, network.P)
        support_mass_np = support_mass_map.detach().float().cpu().numpy().astype(np.float32)
        support_count_np = support_count_map.detach().float().cpu().numpy().astype(np.float32)
        support_mass_vis = score_map_to_bgr(support_mass_np, upscale=1)

        compare_tiles: list[tuple[str, np.ndarray]] = [("input", input_grid)]

        for requested_norm in requested_norms:
            effective_norm, reason = choose_tracker_norm(dataset_type, requested_norm, voxel)
            nonfinite_reason = None

            voxel_norm = normalize_event_voxel(voxel, effective_norm)
            if not torch.isfinite(voxel_norm).all():
                if effective_norm != "none":
                    fallback_norm = "none"
                    nonfinite_reason = (
                        f"normalization '{effective_norm}' produced non-finite values; "
                        "falling back to none"
                    )
                    effective_norm = fallback_norm
                    voxel_norm = normalize_event_voxel(voxel, effective_norm)
                if not torch.isfinite(voxel_norm).all():
                    summary_row = {
                        "frame_idx": frame_idx,
                        "timestamp_us": float(timestamp_us),
                        "requested_norm": requested_norm,
                        "effective_norm": effective_norm,
                        "status": "nonfinite_normalized_voxel",
                        "selection_source": "none",
                        "selected_count": 0,
                        "supported_count": 0,
                        "unsupported_count": 0,
                        "score_dynamic_range": None,
                        "score_mean": None,
                        "score_max": None,
                        "score_error": nonfinite_reason,
                    }
                    per_frame_rows.append(summary_row)
                    per_norm_stats[requested_norm].append(summary_row)
                    print(f"Frame {frame_idx:05d} [{requested_norm}]: normalized voxel remained non-finite")
                    continue

            fmap, imap, score_logits, scores = extract_frame_features_with_logits(
                network, voxel_norm, runtime_cfg.MIXED_PRECISION
            )
            feat_h, feat_w = fmap.shape[-2:]
            fallback_map = build_fallback_score_map(voxel, fmap, feat_h, feat_w)
            centers, selected_values, fallback_used, selection_error = run_selection(
                scores[0, 0],
                fallback_map,
                network,
                args,
                selector_mode,
                selector_use_grid,
            )

            if centers is None:
                centers_np = np.zeros((0, 2), dtype=np.float32)
                selected_values_np = np.zeros((0,), dtype=np.float32)
                supported_np = np.zeros((0,), dtype=bool)
                selection_source = "none"
            else:
                centers_np = centers[0].detach().cpu().numpy().astype(np.float32)
                selected_values_np = selected_values.detach().cpu().numpy().astype(np.float32)
                supported_np = supported_track_mask(
                    centers,
                    support_mass_map,
                    support_count_map,
                    args.min_event_support,
                    args.min_event_pixels,
                ).detach().cpu().numpy().astype(bool)
                selection_source = "fallback" if fallback_used else "score"

            score_np = scores[0, 0].detach().float().cpu().numpy().astype(np.float32)
            logits_np = score_logits[0, 0].detach().float().cpu().numpy().astype(np.float32)
            fallback_np = fallback_map.detach().float().cpu().numpy().astype(np.float32)

            score_stats = summarize_score_map(score_np)
            logits_stats = summarize_array(logits_np)
            fallback_stats = summarize_array(fallback_np)

            selection_mean = float(selected_values_np.mean()) if selected_values_np.size else None
            selection_min = float(selected_values_np.min()) if selected_values_np.size else None
            selection_max = float(selected_values_np.max()) if selected_values_np.size else None

            requested_dir = seq_out_dir / norm_tag(requested_norm, effective_norm)
            requested_dir.mkdir(exist_ok=True)

            label = requested_norm if requested_norm == effective_norm else f"{requested_norm}->{effective_norm}"
            subtitle = (
                f"{label}: sel={len(centers_np)} "
                f"ok={int(supported_np.sum())}/{len(supported_np)} "
                f"range={score_stats['dynamic_range']:.4g}"
                if score_stats["dynamic_range"] is not None
                else f"{label}: sel={len(centers_np)}"
            )

            score_heat = score_map_to_bgr(score_np)
            logits_heat = signed_map_to_bgr(logits_np)
            fallback_heat = score_map_to_bgr(fallback_np)
            score_selected = overlay_feature_points(
                score_heat,
                centers_np,
                supported_np,
                subtitle=subtitle,
            )
            input_selected = overlay_feature_points(
                input_grid,
                centers_np,
                supported_np,
                panel_offsets=panel_offsets,
                subtitle=subtitle,
            )

            base_stem = f"frame{frame_idx:05d}"
            np.savez_compressed(
                requested_dir / f"{base_stem}.npz",
                frame_idx=np.int32(frame_idx),
                timestamp_us=np.float64(timestamp_us),
                requested_norm=np.array(requested_norm),
                effective_norm=np.array(effective_norm),
                score_logits_feature=logits_np,
                score_feature=score_np,
                fallback_score_feature=fallback_np,
                support_mass_image=support_mass_np,
                support_count_image=support_count_np,
                selected_centers_feature=centers_np,
                selected_scores=selected_values_np,
                selected_supported=supported_np.astype(np.uint8),
            )
            cv2.imwrite(str(requested_dir / f"{base_stem}_score.png"), score_heat)
            cv2.imwrite(str(requested_dir / f"{base_stem}_score_selected.png"), score_selected)
            cv2.imwrite(str(requested_dir / f"{base_stem}_logits.png"), logits_heat)
            cv2.imwrite(str(requested_dir / f"{base_stem}_fallback.png"), fallback_heat)
            cv2.imwrite(str(requested_dir / f"{base_stem}_support_mass.png"), support_mass_vis)
            cv2.imwrite(str(requested_dir / f"{base_stem}_input_selected.png"), input_selected)

            compare_tiles.append((label, score_selected))

            summary_row = {
                "frame_idx": frame_idx,
                "timestamp_us": float(timestamp_us),
                "requested_norm": requested_norm,
                "effective_norm": effective_norm,
                "status": "ok",
                "selection_source": selection_source,
                "selected_count": int(len(centers_np)),
                "supported_count": int(supported_np.sum()),
                "unsupported_count": int((~supported_np).sum()) if supported_np.size else 0,
                "selected_score_mean": selection_mean,
                "selected_score_min": selection_min,
                "selected_score_max": selection_max,
                "score_dynamic_range": score_stats["dynamic_range"],
                "score_mean": score_stats["mean"],
                "score_max": score_stats["max"],
                "score_is_flat": score_stats["is_flat"],
                "logit_mean": logits_stats["mean"],
                "logit_max": logits_stats["max"],
                "fallback_dynamic_range": fallback_stats["dynamic_range"],
                "fallback_used": bool(fallback_used),
                "selection_error": selection_error,
                "norm_reason": reason if reason is not None else nonfinite_reason,
            }
            per_frame_rows.append(summary_row)
            per_norm_stats[requested_norm].append(summary_row)

            fallback_text = " fallback" if fallback_used else ""
            print(
                f"Frame {frame_idx:05d} [{label}]: "
                f"range={score_stats['dynamic_range']:.4g} "
                f"selected={len(centers_np)} "
                f"supported={int(supported_np.sum())}/{len(supported_np)}"
                f"{fallback_text}"
            )

        cv2.imwrite(str(compare_dir / f"frame{frame_idx:05d}.png"), make_strip(compare_tiles))
        num_processed_frames += 1

    save_summary_csv(seq_out_dir / "summary.csv", per_frame_rows)

    summary = {
        "h5_path": str(h5_path),
        "seq_name": seq_name,
        "dataset_type": dataset_type,
        "output_dir": str(seq_out_dir),
        "network": args.network,
        "config": args.config,
        "requested_norms": requested_norms,
        "selector_mode": selector_mode,
        "selector_use_grid": selector_use_grid,
        "tracker_selector": args.tracker_selector,
        "tracker_patches": int(args.tracker_patches),
        "min_event_support": float(args.min_event_support),
        "min_event_pixels": float(args.min_event_pixels),
        "num_processed_frames": num_processed_frames,
        "per_norm_mean": {},
        "rows": per_frame_rows,
    }

    for requested_norm, rows in per_norm_stats.items():
        ok_rows = [row for row in rows if row["status"] == "ok"]
        summary["per_norm_mean"][requested_norm] = {
            "num_rows": len(rows),
            "num_ok_rows": len(ok_rows),
            "mean_selected_count": float(np.mean([row["selected_count"] for row in ok_rows])) if ok_rows else None,
            "mean_supported_count": float(np.mean([row["supported_count"] for row in ok_rows])) if ok_rows else None,
            "mean_unsupported_count": float(np.mean([row["unsupported_count"] for row in ok_rows])) if ok_rows else None,
            "mean_score_dynamic_range": float(np.mean([row["score_dynamic_range"] for row in ok_rows])) if ok_rows else None,
            "num_fallback_used": int(sum(bool(row["fallback_used"]) for row in ok_rows)),
            "num_flat_scores": int(sum(bool(row["score_is_flat"]) for row in ok_rows)),
        }

    (seq_out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved score-map debug outputs to {seq_out_dir}")
    return {
        "seq_name": seq_name,
        "dataset_type": dataset_type,
        "out_dir": str(seq_out_dir),
        "num_processed_frames": num_processed_frames,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--network", required=True, help="Path to checkpoint (.pth/.pt)")
    parser.add_argument("--config", required=True, help="YAML config file")
    parser.add_argument("--h5", nargs="+", required=True, help="H5 file(s) or directories")
    parser.add_argument("--output-dir", default="results/debug_score_map", help="Output directory")
    parser.add_argument("--gtdir", default=None, help="Required for raw VECtor .hdf5 input")
    parser.add_argument("--height", type=int, default=None, help="Override input height")
    parser.add_argument("--width", type=int, default=None, help="Override input width")
    parser.add_argument("--stride", type=int, default=1, help="Use every N-th frame/window")
    parser.add_argument("--skip-start-s", type=float, default=0.0, help="Skip the first N seconds for preprocessed H5")
    parser.add_argument("--max-frames", type=int, default=None, help="Optional cap on processed frames per sequence")
    parser.add_argument(
        "--frame-indices",
        type=str,
        default=None,
        help="Optional subset of iterator frame indices, e.g. '0,5,10-15'",
    )
    parser.add_argument(
        "--frame-step",
        type=int,
        default=1,
        help="Process every Nth iterator frame; combine with --frame-indices to intersect both filters",
    )
    parser.add_argument(
        "--norms",
        nargs="*",
        default=None,
        help="Normalization variants to compare; default is the config norm. Use 'config' to reference the config value.",
    )
    parser.add_argument(
        "--tracker-selector",
        choices=["native", "topk", "adjacent_top1"],
        default="native",
        help="Patch selection strategy to debug",
    )
    parser.add_argument(
        "--tracker-patches",
        type=int,
        default=None,
        help="Number of selected patches; default is PATCHES_PER_FRAME from config",
    )
    parser.add_argument(
        "--selector-mode",
        type=str,
        default=None,
        help="Override PatchSelector mode for native selection; default comes from config",
    )
    parser.add_argument("--selector-use-grid", dest="selector_use_grid", action="store_true")
    parser.add_argument("--no-selector-grid", dest="selector_use_grid", action="store_false")
    parser.set_defaults(selector_use_grid=None)
    parser.add_argument("--min-event-support", type=float, default=4.0)
    parser.add_argument("--min-event-pixels", type=float, default=4.0)
    parser.add_argument("--dim_inet", type=int, default=384)
    parser.add_argument("--dim_fnet", type=int, default=128)
    parser.add_argument("--dim", type=int, default=32)
    return parser


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("This script requires CUDA")

    parser = build_arg_parser()
    args = parser.parse_args()
    if args.frame_step <= 0:
        raise ValueError(f"--frame-step must be >= 1, got {args.frame_step}")

    runtime_cfg = BASE_CFG.clone()
    runtime_cfg.merge_from_file(args.config)
    if args.tracker_patches is None:
        args.tracker_patches = int(getattr(runtime_cfg, "PATCHES_PER_FRAME", 80))

    h5_files = discover_h5_files(args.h5)
    if not h5_files:
        raise RuntimeError("No .h5/.hdf5 files found from --h5 inputs")

    network = load_tracking_network(args, runtime_cfg)
    if not hasattr(network.patchify, "scorer"):
        raise RuntimeError("This checkpoint/config does not expose a scorer head")

    out_root = Path(args.output_dir).resolve()
    results = []
    for h5_path in h5_files:
        results.append(process_sequence(h5_path, runtime_cfg, args, out_root, network))

    print("\n=== Score Map Debug Summary ===")
    for row in results:
        print(
            f"{row['seq_name']}: frames={row['num_processed_frames']} "
            f"out={row['out_dir']}"
        )


if __name__ == "__main__":
    torch.manual_seed(1234)
    main()
