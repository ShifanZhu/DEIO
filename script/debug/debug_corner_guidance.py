"""Compare `corner_guidance` modes on one tracker sample.

This script dumps side-by-side score maps and selected patch overlays for:
  - `none`
  - `harris`
  - `shitomasi`

By default it uses the same scorer sampling path as eval (`multi` + grid).
For deterministic comparisons, pass `--selector_mode topk`.

Examples
--------
python script/debug/debug_corner_guidance.py \
    -c config/train_tracker_base.conf \
    --resume checkpoints/tracker_base/010000.pth \
    --sample_idx 0

python script/debug/debug_corner_guidance.py \
    -c config/train_tracker_base.conf \
    --resume checkpoints/tracker_base/010000.pth \
    --scene_substr office/Easy/P001 \
    --selector_mode topk \
    --run_tracker

python script/debug/debug_corner_guidance.py \
    -c config/train_tracker_base.conf \
    --resume checkpoints/tracker_base/010000.pth \
    --folder /media/s/rell/tartan/ocean/Easy/P001 \
    --selector_mode topk
"""

import os
import re
import sys
import json
import random
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import cv2
import numpy as np
import torch
import configargparse

from dpvo import altcorr
from dpvo.lietorch import SE3

from devo.data_readers.factory import dataset_factory
from devo.patch_tracker import PatchTracker
from devo.selector import PatchSelector, SelectionMethod
from utils.voxel_utils import std, rescale


VALID_MODES = ("none", "harris", "shitomasi")
MODE_COLORS = {
    "none": (60, 200, 60),
    "harris": (0, 180, 255),
    "shitomasi": (220, 80, 220),
}


def score_map_to_bgr(score_map: np.ndarray, upscale: int = 4) -> np.ndarray:
    """Render a non-negative 2D map as a TURBO heatmap."""
    score_map = np.asarray(score_map, dtype=np.float32)
    finite = np.isfinite(score_map)
    if not finite.any():
        norm_u8 = np.zeros(score_map.shape, dtype=np.uint8)
    else:
        safe = np.where(finite, score_map, 0.0)
        lo, hi = float(safe[finite].min()), float(safe[finite].max())
        if hi - lo <= 1e-8:
            norm_u8 = np.zeros(score_map.shape, dtype=np.uint8)
        else:
            norm_u8 = np.clip(255.0 * (safe - lo) / (hi - lo), 0, 255).astype(np.uint8)
    heatmap = cv2.applyColorMap(norm_u8, cv2.COLORMAP_TURBO)
    if upscale > 1:
        heatmap = cv2.resize(
            heatmap,
            (score_map.shape[1] * upscale, score_map.shape[0] * upscale),
            interpolation=cv2.INTER_NEAREST,
        )
    return heatmap


def normalize_voxels(images: torch.Tensor, norm: str) -> torch.Tensor:
    """Match PatchTracker voxel normalization."""
    if norm == 'none':
        return images
    if norm in ('rescale', 'norm'):
        return rescale(images)
    if norm in ('standard', 'std'):
        return std(images, sequence=False)
    if norm in ('standard2', 'std2'):
        return std(images)
    raise NotImplementedError(f"norm '{norm}' not implemented")


def strip_quotes(value):
    if isinstance(value, str):
        return value.strip().strip('"').strip("'")
    return value


def parse_block_dims(value):
    value = strip_quotes(value)
    if isinstance(value, str):
        return [int(v.strip()) for v in value.split(',') if v.strip()]
    return list(value)


def sanitize_name(name: str) -> str:
    return re.sub(r'[^A-Za-z0-9._-]+', '_', name).strip('_')


def path_label(path_like: str, max_parts: int = 5) -> str:
    parts = [p for p in Path(path_like).parts if p not in ('/', '')]
    if not parts:
        return 'root'
    return sanitize_name('_'.join(parts[-max_parts:]))


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def label_tile(image: np.ndarray, text: str) -> np.ndarray:
    canvas = image.copy()
    cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 24), (0, 0, 0), -1)
    cv2.putText(canvas, text, (6, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (255, 255, 255), 1, cv2.LINE_AA)
    return canvas


def make_strip(images_with_labels):
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


def voxel_frame_to_bgr(voxel_frame: torch.Tensor) -> np.ndarray:
    """Convert one event voxel grid (C,H,W) to a grayscale BGR canvas."""
    frame = voxel_frame.detach().float().abs().sum(dim=0).cpu().numpy()
    finite = np.isfinite(frame)
    if finite.any():
        safe = np.where(finite, frame, 0.0)
        hi = float(np.percentile(safe[finite], 99.0))
        lo = float(safe[finite].min())
        if hi - lo <= 1e-8:
            norm = np.zeros_like(frame, dtype=np.uint8)
        else:
            norm = np.clip(255.0 * (safe - lo) / (hi - lo), 0, 255).astype(np.uint8)
    else:
        norm = np.zeros(frame.shape, dtype=np.uint8)
    return cv2.cvtColor(norm, cv2.COLOR_GRAY2BGR)


def overlay_patches(base_bgr: np.ndarray, x_feat: np.ndarray, y_feat: np.ndarray,
                    color, subtitle: str) -> np.ndarray:
    """Draw feature-resolution patch centers on the full-resolution voxel image."""
    canvas = base_bgr.copy()
    h, w = canvas.shape[:2]
    for x, y in zip(x_feat.tolist(), y_feat.tolist()):
        px = int(np.clip(round((float(x) + 0.5) * 4.0), 0, w - 1))
        py = int(np.clip(round((float(y) + 0.5) * 4.0), 0, h - 1))
        cv2.circle(canvas, (px, py), 3, color, 1, cv2.LINE_AA)
    cv2.putText(canvas, subtitle, (6, h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                color, 1, cv2.LINE_AA)
    return canvas


def select_coords_from_scores(scores: torch.Tensor, patches_per_image: int,
                             selector_mode: str, selector_use_grid: bool):
    """Replicate scorer eval-time selection from Patchifier.forward()."""
    selector = PatchSelector(selector_mode, grid=selector_use_grid)
    x, y = selector(scores, patches_per_image)
    coords = torch.stack([x, y], dim=-1).float()
    values = altcorr.patchify(scores[0, :, None], coords, 0).view(scores.shape[1], patches_per_image)
    return x + 1, y + 1, values


def pairwise_patch_overlap(mode_to_xy):
    """Count exact coordinate overlap between each pair of modes."""
    modes = list(mode_to_xy.keys())
    overlap = {}
    for mode_a in modes:
        overlap[mode_a] = {}
        for mode_b in modes:
            frame_counts = []
            for (xa, ya), (xb, yb) in zip(mode_to_xy[mode_a], mode_to_xy[mode_b]):
                set_a = {(int(x), int(y)) for x, y in zip(xa.tolist(), ya.tolist())}
                set_b = {(int(x), int(y)) for x, y in zip(xb.tolist(), yb.tolist())}
                frame_counts.append(len(set_a & set_b))
            overlap[mode_a][mode_b] = {
                "per_frame": frame_counts,
                "mean": float(np.mean(frame_counts)) if frame_counts else 0.0,
            }
    return overlap


def load_checkpoint(model: torch.nn.Module, checkpoint_path: str) -> None:
    checkpoint = torch.load(checkpoint_path, map_location='cuda')
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        return

    new_sd = {}
    for k, v in checkpoint.items():
        new_sd[k.replace('module.', '')] = v
    state = model.state_dict()
    update = {k: v for k, v in new_sd.items() if k in state and state[k].shape == v.shape}
    state.update(update)
    model.load_state_dict(state, strict=False)


def normalize_path_string(path_like: str) -> str:
    return Path(str(path_like).strip()).as_posix().rstrip('/').lower()


def scene_matches_folder(scene_id: str, folder: str) -> bool:
    scene_norm = normalize_path_string(scene_id)
    folder_norm = normalize_path_string(folder)
    scene_no_evs = re.sub(r'/evs_left$', '', scene_norm)

    candidates = (scene_norm, scene_no_evs)
    for candidate in candidates:
        if candidate == folder_norm:
            return True
        if candidate.startswith(folder_norm + '/'):
            return True
        if folder_norm.startswith(candidate + '/'):
            return True

    scene_parts = [p for p in scene_no_evs.split('/') if p]
    folder_parts = [p for p in folder_norm.split('/') if p]
    if folder_parts and len(folder_parts) <= len(scene_parts):
        span = len(folder_parts)
        for start in range(len(scene_parts) - span + 1):
            if scene_parts[start:start + span] == folder_parts:
                return True
    if scene_parts and len(scene_parts) <= len(folder_parts):
        span = len(scene_parts)
        for start in range(len(folder_parts) - span + 1):
            if folder_parts[start:start + span] == scene_parts:
                return True

    return folder_norm in scene_norm or folder_norm in scene_no_evs


def collect_dataset_entries(db):
    entries = []
    sub_datasets = db.datasets if hasattr(db, 'datasets') else [db]
    offset = 0
    for sub_db in sub_datasets:
        for local_idx, (scene_id, _frame_idx) in enumerate(sub_db.dataset_index):
            entries.append({
                'global_idx': offset + local_idx,
                'scene_id': scene_id,
                'frame_idx': _frame_idx,
            })
        offset += len(sub_db)
    return entries


def closest_scene_ids(entries, folder: str, limit: int = 8):
    folder_norm = normalize_path_string(folder)
    folder_parts = [p for p in folder_norm.split('/') if p]
    unique_scenes = sorted({entry['scene_id'] for entry in entries})
    scored = []
    for scene in unique_scenes:
        scene_norm = normalize_path_string(scene)
        scene_no_evs = re.sub(r'/evs_left$', '', scene_norm)
        scene_parts = [p for p in scene_no_evs.split('/') if p]
        overlap = len(set(folder_parts) & set(scene_parts))
        if folder_parts and folder_parts[-1] in scene_parts:
            overlap += 2
        if folder_parts and any(part in scene_no_evs for part in folder_parts[-2:]):
            overlap += 1
        scored.append((overlap, scene))
    scored.sort(key=lambda x: (-x[0], x[1]))
    return [scene for score, scene in scored[:limit] if score > 0]


def resolve_entries(db, sample_idx: int, scene_substr: str | None, folder: str | None,
                    max_samples: int | None):
    entries = collect_dataset_entries(db)

    if folder is not None:
        matches = [entry for entry in entries if scene_matches_folder(entry['scene_id'], folder)]
        if not matches:
            suggestions = closest_scene_ids(entries, folder)
            message = [f"No dataset entries matched folder='{folder}'"]
            if suggestions:
                message.append("Closest scene ids:")
                message.extend(f"  - {scene}" for scene in suggestions)
            raise RuntimeError('\n'.join(message))
        if max_samples is not None:
            matches = matches[:max_samples]
        return matches

    if scene_substr is None:
        if sample_idx < 0 or sample_idx >= len(entries):
            raise IndexError(f"sample_idx {sample_idx} out of range for dataset of size {len(entries)}")
        return [entries[sample_idx]]

    matches = [entry for entry in entries if scene_substr in entry['scene_id']]
    if not matches:
        raise RuntimeError(f"No dataset entry matched scene_substr='{scene_substr}'")
    if sample_idx < 0 or sample_idx >= len(matches):
        raise IndexError(f"sample_idx {sample_idx} out of range for {len(matches)} matching entries")
    return [matches[sample_idx]]


def build_dataset(args, split_mode: str):
    return dataset_factory(
        ['tartan_evs'],
        datapath=args.datapath,
        n_frames=args.n_frames,
        fgraph_pickle=args.fgraph_pickle,
        train_split=args.train_split,
        val_split=args.val_split,
        split_mode=split_mode,
        strict_split=False,
        sample=args.sample,
        return_fname=True,
        scale=args.scale,
    )


def resolve_folder_entries_across_splits(args):
    all_entries = []
    seen = set()
    suggestions = []

    for split_mode in ('train', 'val'):
        db = build_dataset(args, split_mode)
        split_entries = [
            entry for entry in collect_dataset_entries(db)
            if scene_matches_folder(entry['scene_id'], args.folder)
        ]
        if not split_entries:
            suggestions.extend(closest_scene_ids(collect_dataset_entries(db), args.folder))
            continue
        for entry in split_entries:
            key = (entry['scene_id'], entry['frame_idx'])
            if key in seen:
                continue
            seen.add(key)
            entry = dict(entry)
            entry['dataset'] = db
            entry['source_split'] = split_mode
            all_entries.append(entry)

    all_entries.sort(key=lambda e: (e['scene_id'], e['frame_idx']))
    if not all_entries:
        unique_suggestions = []
        for scene in suggestions:
            if scene not in unique_suggestions:
                unique_suggestions.append(scene)
        message = [f"No dataset entries matched folder='{args.folder}' across train/val splits"]
        if unique_suggestions:
            message.append("Closest scene ids:")
            message.extend(f"  - {scene}" for scene in unique_suggestions[:8])
        raise RuntimeError('\n'.join(message))
    if args.max_samples is not None:
        all_entries = all_entries[:args.max_samples]
    return all_entries


@torch.no_grad()
def run_tracker_mode(model: PatchTracker, images: torch.Tensor, poses: SE3, disps: torch.Tensor,
                     intrinsics: torch.Tensor, mode: str, iters: int, patches_per_image: int,
                     selector_mode: str, selector_use_grid: bool):
    original_mode = model.patchify.corner_guidance
    model.patchify.corner_guidance = mode
    try:
        traj = model(images, poses, disps, intrinsics,
                     STEPS=iters,
                     patches_per_image=patches_per_image,
                     scorer_eval_mode=selector_mode,
                     scorer_eval_use_grid=selector_use_grid)
    finally:
        model.patchify.corner_guidance = original_mode

    coords_est, coords_gt, valid, _weight, _scores, _kk = traj[-1]
    valid_mask = (valid > 0.5).reshape(-1)
    errors = (coords_est - coords_gt).norm(dim=-1).reshape(-1)
    if int(valid_mask.sum().item()) == 0:
        return float('nan')
    return float(errors[valid_mask].mean().item())


def build_arg_parser():
    parser = configargparse.ArgumentParser(
        description=__doc__,
        ignore_unknown_config_file_keys=True,
    )
    parser.add_argument('-c', '--config', is_config_file=True,
                        default='config/train_tracker_base.conf',
                        help='config file path')
    parser.add_argument('--name', '--expname', default='tracker_run')
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--datapath', type=str, default='')
    parser.add_argument('--fgraph_pickle', type=str, default='fgraph/TartanAirEVS.pickle')
    parser.add_argument('--train_split', type=str, default='script/splits/tartan/tartan_default_train.txt')
    parser.add_argument('--val_split', type=str, default='script/splits/tartan/tartan_default_val.txt')

    parser.add_argument('--n_frames', type=int, default=15)
    parser.add_argument('--patches_per_image', type=int, default=80)
    parser.add_argument('--patch_selector', type=str, default='scorer')
    parser.add_argument('--corner_guidance', type=str, default='none')
    parser.add_argument('--norm', type=str, default='std2')
    parser.add_argument('--scale', type=float, default=1.0)

    parser.add_argument('--dim_inet', type=int, default=384)
    parser.add_argument('--dim_fnet', type=int, default=128)
    parser.add_argument('--dim', type=int, default=32)
    parser.add_argument('--resnet', action='store_true')
    parser.add_argument('--block_dims', type=str, default='64,128,256')
    parser.add_argument('--initial_dim', type=int, default=64)
    parser.add_argument('--pretrain', type=str, default='resnet18')

    parser.add_argument('--split_mode', choices=['train', 'val'], default='val')
    parser.add_argument('--sample_idx', type=int, default=0,
                        help='global sample index, or match index when --scene_substr is used')
    parser.add_argument('--scene_substr', type=str, default=None,
                        help='restrict sample selection to scenes containing this substring')
    parser.add_argument('--folder', type=str, default=None,
                        help='process all dataset entries under this folder, searching both train and val splits')
    parser.add_argument('--max_samples', type=int, default=None,
                        help='optional cap for folder mode')
    parser.add_argument('--sample', action='store_true',
                        help='use stochastic frame sampling in the dataset')
    parser.add_argument('--seed', type=int, default=0)

    parser.add_argument('--modes', nargs='+', default=list(VALID_MODES))
    parser.add_argument('--selector_mode', type=str, default='multi',
                        help='PatchSelector mode. Use topk for deterministic comparisons.')
    parser.add_argument('--selector_use_grid', dest='selector_use_grid', action='store_true')
    parser.add_argument('--no_selector_grid', dest='selector_use_grid', action='store_false')
    parser.set_defaults(selector_use_grid=True)

    parser.add_argument('--iters', type=int, default=18)
    parser.add_argument('--run_tracker', action='store_true',
                        help='also run PatchTracker once per mode and report last-iter EPE')
    parser.add_argument('--max_frames', type=int, default=None,
                        help='limit the number of frames rendered')
    parser.add_argument('--outdir', type=str, default='results/debug_corner_guidance')
    return parser


@torch.no_grad()
def process_sample(model: PatchTracker, sample, entry, args, run_root: Path):
    global_idx = entry['global_idx']
    scene_id = entry['scene_id']
    frame_idx = entry['frame_idx']
    source_split = entry.get('source_split')
    split_tag = source_split if source_split is not None else args.split_mode

    images, poses, disps, intrinsics = sample[:4]
    images = images.unsqueeze(0).cuda().float()
    disps = disps.unsqueeze(0).cuda().float()
    intrinsics = intrinsics.unsqueeze(0).cuda().float()
    poses = SE3(poses.unsqueeze(0).cuda().float()).inv()

    images_norm = normalize_voxels(images.clone(), model.norm)
    base_scores, _, _ = model.patchify.compute_guided_scores(images_norm, corner_guidance='none')
    base_scores = base_scores.float()

    n_frames_total = images.shape[1]
    frame_limit = n_frames_total if args.max_frames is None else min(n_frames_total, args.max_frames)

    scene_safe = path_label(scene_id)
    outdir = run_root / f"{split_tag}_sample{global_idx:05d}_f{frame_idx:05d}_{scene_safe}"
    outdir.mkdir(parents=True, exist_ok=True)

    input_dir = outdir / 'input'
    compare_dir = outdir / 'compare'
    input_dir.mkdir(exist_ok=True)
    compare_dir.mkdir(exist_ok=True)

    mode_results = {}

    for mode in args.modes:
        seed_everything(args.seed)
        _scores, corner_map, guided_scores = model.patchify.compute_guided_scores(images_norm, corner_guidance=mode)
        guided_scores = guided_scores.float()
        x, y, selected_values = select_coords_from_scores(
            guided_scores,
            patches_per_image=args.patches_per_image,
            selector_mode=args.selector_mode,
            selector_use_grid=args.selector_use_grid,
        )

        tracker_epe = None
        if args.run_tracker:
            seed_everything(args.seed)
            tracker_epe = run_tracker_mode(
                model, images, poses, disps, intrinsics,
                mode=mode,
                iters=args.iters,
                patches_per_image=args.patches_per_image,
                selector_mode=args.selector_mode,
                selector_use_grid=args.selector_use_grid,
            )

        mode_dir = outdir / mode
        mode_dir.mkdir(exist_ok=True)

        frame_stats = []
        xy_per_frame = []
        for frame_idx in range(frame_limit):
            voxel_bgr = voxel_frame_to_bgr(images[0, frame_idx])
            input_path = input_dir / f'frame{frame_idx:04d}.png'
            if not input_path.exists():
                cv2.imwrite(str(input_path), voxel_bgr)

            base_np = base_scores[0, frame_idx].detach().cpu().numpy().astype(np.float32)
            guided_np = guided_scores[0, frame_idx].detach().cpu().numpy().astype(np.float32)
            corner_np = None if corner_map is None else corner_map[0, frame_idx].detach().cpu().numpy().astype(np.float32)

            frame_x = x[frame_idx].detach().cpu().numpy().astype(np.int32)
            frame_y = y[frame_idx].detach().cpu().numpy().astype(np.int32)
            frame_vals = selected_values[frame_idx].detach().cpu().numpy().astype(np.float32)
            xy_per_frame.append((frame_x, frame_y))

            patch_overlay = overlay_patches(
                voxel_bgr,
                frame_x,
                frame_y,
                MODE_COLORS[mode],
                f"{mode}: {len(frame_x)} patches, mean={frame_vals.mean():.4f}",
            )

            payload = {
                'base_score': base_np,
                'guided_score': guided_np,
                'x_feat': frame_x,
                'y_feat': frame_y,
                'selected_score': frame_vals,
            }
            if corner_np is not None:
                payload['corner_map'] = corner_np

            np.savez_compressed(mode_dir / f'frame{frame_idx:04d}.npz', **payload)
            cv2.imwrite(str(mode_dir / f'frame{frame_idx:04d}_patches.png'), patch_overlay)
            cv2.imwrite(str(mode_dir / f'frame{frame_idx:04d}_guided.png'), score_map_to_bgr(guided_np))
            if corner_np is not None:
                cv2.imwrite(str(mode_dir / f'frame{frame_idx:04d}_corner.png'), score_map_to_bgr(corner_np))

            frame_stats.append({
                'frame_idx': frame_idx,
                'mean_guided_score': float(guided_np.mean()),
                'max_guided_score': float(guided_np.max()),
                'mean_selected_score': float(frame_vals.mean()),
                'max_selected_score': float(frame_vals.max()),
            })

        mode_results[mode] = {
            'corner_map_present': corner_map is not None,
            'tracker_last_iter_epe_feature_px': tracker_epe,
            'frame_stats': frame_stats,
            'xy_per_frame': xy_per_frame,
        }

    for frame_idx in range(frame_limit):
        voxel_bgr = voxel_frame_to_bgr(images[0, frame_idx])

        patch_tiles = [('input', voxel_bgr)]
        for mode in args.modes:
            frame_x, frame_y = mode_results[mode]['xy_per_frame'][frame_idx]
            vals = mode_results[mode]['frame_stats'][frame_idx]['mean_selected_score']
            patch_tiles.append((
                mode,
                overlay_patches(voxel_bgr, frame_x, frame_y, MODE_COLORS[mode], f'{mode}: mean={vals:.4f}')
            ))
        cv2.imwrite(str(compare_dir / f'frame{frame_idx:04d}_patches.png'), make_strip(patch_tiles))

        score_tiles = [
            ('input', voxel_bgr),
            ('base scorer', score_map_to_bgr(base_scores[0, frame_idx].cpu().numpy().astype(np.float32))),
        ]
        for mode in args.modes:
            mode_npz = np.load(outdir / mode / f'frame{frame_idx:04d}.npz')
            if 'corner_map' in mode_npz:
                score_tiles.append((f'{mode} corner', score_map_to_bgr(mode_npz['corner_map'])))
            score_tiles.append((f'{mode} guided', score_map_to_bgr(mode_npz['guided_score'])))
        cv2.imwrite(str(compare_dir / f'frame{frame_idx:04d}_scores.png'), make_strip(score_tiles))

    overlaps = pairwise_patch_overlap({mode: mode_results[mode]['xy_per_frame'] for mode in args.modes})

    summary = {
        'scene_id': scene_id,
        'global_sample_idx': global_idx,
        'frame_idx': frame_idx,
        'source_split': source_split,
        'split_mode': split_tag,
        'seed': args.seed,
        'checkpoint': args.resume,
        'selector_mode': args.selector_mode,
        'selector_use_grid': args.selector_use_grid,
        'run_tracker': args.run_tracker,
        'patches_per_image': args.patches_per_image,
        'modes': args.modes,
        'pairwise_patch_overlap': overlaps,
        'per_mode': {
            mode: {
                'corner_map_present': mode_results[mode]['corner_map_present'],
                'tracker_last_iter_epe_feature_px': mode_results[mode]['tracker_last_iter_epe_feature_px'],
                'mean_selected_score': float(np.mean([s['mean_selected_score']
                                                      for s in mode_results[mode]['frame_stats']])),
                'mean_guided_score': float(np.mean([s['mean_guided_score']
                                                    for s in mode_results[mode]['frame_stats']])),
            }
            for mode in args.modes
        },
    }
    (outdir / 'summary.json').write_text(json.dumps(summary, indent=2))

    print(f"Saved debug outputs to {outdir}")
    print(json.dumps(summary['per_mode'], indent=2))
    return summary


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("This script requires CUDA because DEIO patch selection uses CUDA ops")

    parser = build_arg_parser()
    args = parser.parse_args()

    args.resume = strip_quotes(args.resume) or None
    args.datapath = strip_quotes(args.datapath)
    args.fgraph_pickle = strip_quotes(args.fgraph_pickle)
    args.train_split = strip_quotes(args.train_split)
    args.val_split = strip_quotes(args.val_split)
    args.scene_substr = strip_quotes(args.scene_substr) or None
    args.folder = strip_quotes(args.folder) or None
    args.patch_selector = strip_quotes(args.patch_selector).lower()
    args.corner_guidance = strip_quotes(args.corner_guidance).lower()
    args.norm = strip_quotes(args.norm).lower()
    args.pretrain = strip_quotes(args.pretrain)
    args.block_dims = parse_block_dims(args.block_dims)
    args.modes = [strip_quotes(m).lower() for m in args.modes]

    invalid_modes = [m for m in args.modes if m not in VALID_MODES]
    if invalid_modes:
        raise ValueError(f"Unsupported modes: {invalid_modes}. Valid values: {VALID_MODES}")
    if args.patch_selector != SelectionMethod.SCORER:
        raise ValueError("corner_guidance only applies to --patch_selector scorer")

    seed_everything(args.seed)

    if args.folder is not None:
        entries = resolve_folder_entries_across_splits(args)
        if not entries:
            raise RuntimeError(
                f"No dataset entries matched folder='{args.folder}' across train/val splits")
        db = None
    else:
        db = build_dataset(args, args.split_mode)
        entries = resolve_entries(db, args.sample_idx, args.scene_substr, args.folder, args.max_samples)

    model = PatchTracker(
        args,
        dim_inet=args.dim_inet,
        dim_fnet=args.dim_fnet,
        dim=args.dim,
        patch_selector=args.patch_selector,
        norm=args.norm,
        randaug=False,
        corner_guidance='none',
    ).cuda().eval()

    if args.resume:
        load_checkpoint(model, args.resume)

    if args.folder is not None:
        run_root = Path(args.outdir) / f"folder_{path_label(args.folder)}"
    else:
        run_root = Path(args.outdir)
    run_root.mkdir(parents=True, exist_ok=True)

    batch_summaries = []
    for idx, entry in enumerate(entries, start=1):
        source_split = entry.get('source_split', args.split_mode)
        print(f"[{idx}/{len(entries)}] processing {source_split} sample {entry['global_idx']} frame {entry['frame_idx']} :: {entry['scene_id']}")
        sample_db = entry.get('dataset', db)
        sample = sample_db[entry['global_idx']]
        batch_summaries.append(process_sample(model, sample, entry, args, run_root))

    if len(batch_summaries) > 1:
        aggregate = {
            'folder': args.folder,
            'split_mode': args.split_mode,
            'n_samples': len(batch_summaries),
            'modes': args.modes,
            'samples': [
                {
                    'scene_id': summary['scene_id'],
                    'global_sample_idx': summary['global_sample_idx'],
                    'frame_idx': summary['frame_idx'],
                    'source_split': summary['source_split'],
                    'per_mode': summary['per_mode'],
                }
                for summary in batch_summaries
            ],
            'per_mode_mean': {},
        }

        for mode in args.modes:
            selected_vals = [s['per_mode'][mode]['mean_selected_score'] for s in batch_summaries]
            guided_vals = [s['per_mode'][mode]['mean_guided_score'] for s in batch_summaries]
            epe_vals = [s['per_mode'][mode]['tracker_last_iter_epe_feature_px']
                        for s in batch_summaries
                        if s['per_mode'][mode]['tracker_last_iter_epe_feature_px'] is not None]
            aggregate['per_mode_mean'][mode] = {
                'mean_selected_score': float(np.mean(selected_vals)) if selected_vals else float('nan'),
                'mean_guided_score': float(np.mean(guided_vals)) if guided_vals else float('nan'),
                'tracker_last_iter_epe_feature_px': float(np.mean(epe_vals)) if epe_vals else None,
            }

        (run_root / 'batch_summary.json').write_text(json.dumps(aggregate, indent=2))
        print(f"Saved batch summary to {run_root / 'batch_summary.json'}")


if __name__ == '__main__':
    main()
