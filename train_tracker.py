"""
train_tracker.py — Training script for PatchTracker

Simplified version of train.py: keeps the data pipeline, optimizer, and
checkpoint infrastructure but uses only the flow loss (+ optional scorer loss).
No pose loss, no BA, no CM refinement, no structure-only warmup.

Loss: exponentially-weighted flow loss across STEPS iterations (RAFT-style).

Scorer objective run commands:
    conda run -n DEIO python /home/s/repos/DEIO/train_tracker.py -c /home/s/repos/DEIO/config/train_tracker_base.conf --name tracker_baseline --score_objective baseline --eval
    conda run -n DEIO python /home/s/repos/DEIO/train_tracker.py -c /home/s/repos/DEIO/config/train_tracker_base.conf --name tracker_rank_only --score_objective rank_only --eval
    conda run -n DEIO python /home/s/repos/DEIO/train_tracker.py -c /home/s/repos/DEIO/config/train_tracker_base.conf --name tracker_corr_isotropy --score_objective corr_isotropy --eval
    conda run -n DEIO python /home/s/repos/DEIO/train_tracker.py -c /home/s/repos/DEIO/config/train_tracker_base.conf --name tracker_short_horizon_survivability --score_objective short_horizon_survivability --eval
    conda run -n DEIO python /home/s/repos/DEIO/train_tracker.py -c /home/s/repos/DEIO/config/train_tracker_base.conf --name tracker_repeatability --score_objective repeatability --eval
    conda run -n DEIO python /home/s/repos/DEIO/train_tracker.py -c /home/s/repos/DEIO/config/train_tracker_base.conf --name tracker_diversity --score_objective diversity --eval
    conda run -n DEIO python /home/s/repos/DEIO/train_tracker.py -c /home/s/repos/DEIO/config/train_tracker_base.conf --name tracker_multi_motion_consistency --score_objective multi_motion_consistency --eval
    conda run -n DEIO python /home/s/repos/DEIO/train_tracker.py -c /home/s/repos/DEIO/config/train_tracker_base.conf --name tracker_event_teacher --score_objective event_teacher --eval
    conda run -n DEIO python /home/s/repos/DEIO/train_tracker.py -c /home/s/repos/DEIO/config/train_tracker_base.conf --name tracker_info_gain_head --score_objective info_gain_head --eval
    conda run -n DEIO python /home/s/repos/DEIO/train_tracker.py -c /home/s/repos/DEIO/config/train_tracker_base.conf --name tracker_conditioning_head --score_objective conditioning_head --eval
    conda run -n DEIO python /home/s/repos/DEIO/train_tracker.py -c /home/s/repos/DEIO/config/train_tracker_base.conf --name tracker_forward_backward_cycle --score_objective forward_backward_cycle --eval
    conda run -n DEIO python /home/s/repos/DEIO/train_tracker.py -c /home/s/repos/DEIO/config/train_tracker_base.conf --name tracker_replay_stability --score_objective replay_stability --eval
    conda run -n DEIO python /home/s/repos/DEIO/train_tracker.py -c /home/s/repos/DEIO/config/train_tracker_base.conf --name tracker_cycle_stability --score_objective cycle_stability --eval
"""

import os
import json
import numpy as np
from collections import OrderedDict
from pathlib import Path
import contextlib

import cv2
import matplotlib
matplotlib.use('Agg')

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from dpvo.lietorch import SE3
from devo.data_readers.factory import dataset_factory
from devo.logger import Logger
from devo.patch_tracker import PatchTracker
from devo.scorer_objectives import (
    append_experiment_table_row,
    compute_patch_survival_metrics,
    compute_replay_metric_summary,
    compute_score_map_dynamic_range,
    compute_scorer_objective,
    compute_selected_patch_tensor_metrics,
    compute_selection_spread,
    extract_selected_centers_by_frame,
    implemented_scorer_objective_names,
    replay_scorer_objective_names,
    scorer_objective_names,
    summarize_score_behavior,
)
from devo.selector import SelectionMethod
from utils.voxel_utils import std, rescale

import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from tqdm import tqdm


def score_map_to_bgr(score_map: np.ndarray, upscale: int = 4) -> np.ndarray:
    """Render a non-negative scorer map as a TURBO heatmap PNG."""
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
        heatmap = cv2.resize(heatmap,
                             (score_map.shape[1] * upscale, score_map.shape[0] * upscale),
                             interpolation=cv2.INTER_NEAREST)
    return heatmap


def _normalize_voxel(images: torch.Tensor, norm: str) -> torch.Tensor:
    """Apply the same voxel normalization as PatchTracker.forward()."""
    if norm == 'none':
        return images
    elif norm in ('rescale', 'norm'):
        return rescale(images)
    elif norm in ('standard', 'std'):
        return std(images, sequence=False)
    elif norm in ('standard2', 'std2'):
        return std(images)
    raise NotImplementedError(f"norm '{norm}' not implemented")


def _activity_frame_to_bgr(images_frame: torch.Tensor) -> np.ndarray:
    activity = images_frame.detach().float().abs().sum(dim=0).cpu().numpy()
    finite = np.isfinite(activity)
    if finite.any():
        safe = np.where(finite, activity, 0.0)
        hi = float(np.percentile(safe[finite], 99.0))
        lo = float(safe[finite].min())
        if hi - lo <= 1e-8:
            norm = np.zeros_like(activity, dtype=np.uint8)
        else:
            norm = np.clip(255.0 * (safe - lo) / (hi - lo), 0, 255).astype(np.uint8)
    else:
        norm = np.zeros(activity.shape, dtype=np.uint8)
    return cv2.cvtColor(norm, cv2.COLOR_GRAY2BGR)


def _overlay_feature_centers(
    base_bgr: np.ndarray,
    centers_feat: torch.Tensor,
    *,
    scale: float,
    center_offset: float = 0.5,
    color: tuple[int, int, int] = (0, 255, 0),
    subtitle: str | None = None,
) -> np.ndarray:
    canvas = base_bgr.copy()
    h, w = canvas.shape[:2]
    for xy in centers_feat.detach().cpu().numpy():
        px = int(np.clip(round((float(xy[0]) + center_offset) * scale), 0, w - 1))
        py = int(np.clip(round((float(xy[1]) + center_offset) * scale), 0, h - 1))
        cv2.circle(canvas, (px, py), 3, color, 1, cv2.LINE_AA)

    if subtitle is not None:
        cv2.putText(canvas, subtitle, (6, h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (255, 255, 255), 1, cv2.LINE_AA)
    return canvas


def _overlay_metric_centers(
    base_bgr: np.ndarray,
    centers_feat: torch.Tensor,
    values: torch.Tensor,
    *,
    scale: float,
    center_offset: float = 0.5,
    subtitle: str | None = None,
) -> np.ndarray:
    canvas = base_bgr.copy()
    h, w = canvas.shape[:2]
    values_np = values.detach().cpu().numpy() if isinstance(values, torch.Tensor) else np.asarray(values)
    finite = np.isfinite(values_np)
    if finite.any():
        lo = float(values_np[finite].min())
        hi = float(values_np[finite].max())
    else:
        lo, hi = 0.0, 1.0

    for xy, value in zip(centers_feat.detach().cpu().numpy(), values_np):
        px = int(np.clip(round((float(xy[0]) + center_offset) * scale), 0, w - 1))
        py = int(np.clip(round((float(xy[1]) + center_offset) * scale), 0, h - 1))
        if np.isfinite(value) and hi - lo > 1e-8:
            norm = int(np.clip(round(255.0 * (float(value) - lo) / (hi - lo)), 0, 255))
            color = tuple(int(c) for c in cv2.applyColorMap(np.array([[norm]], dtype=np.uint8), cv2.COLORMAP_TURBO)[0, 0])
        elif np.isfinite(value):
            color = (0, 255, 255)
        else:
            color = (128, 128, 128)
        cv2.circle(canvas, (px, py), 3, color, -1, cv2.LINE_AA)

    if subtitle is not None:
        cv2.putText(canvas, subtitle, (6, h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (255, 255, 255), 1, cv2.LINE_AA)
    return canvas


def _label_tile(image: np.ndarray, text: str) -> np.ndarray:
    canvas = image.copy()
    cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 24), (0, 0, 0), -1)
    cv2.putText(canvas, text, (6, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (255, 255, 255), 1, cv2.LINE_AA)
    return canvas


def _make_strip(images_with_labels: list[tuple[str, np.ndarray]]) -> np.ndarray:
    target_h = max(img.shape[0] for _, img in images_with_labels)
    tiles = []
    for label, img in images_with_labels:
        tile = np.asarray(img)
        if tile.dtype != np.uint8:
            tile = np.clip(tile, 0, 255).astype(np.uint8)
        if tile.ndim == 2:
            tile = cv2.cvtColor(tile, cv2.COLOR_GRAY2BGR)
        if tile.shape[0] != target_h:
            target_w = int(round(tile.shape[1] * (target_h / tile.shape[0])))
            tile = cv2.resize(tile, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
        tiles.append(_label_tile(tile, label))
    return cv2.hconcat(tiles)


def _extract_seq_name(seq_field) -> str:
    if isinstance(seq_field, str):
        seq_name = seq_field
    elif isinstance(seq_field, (list, tuple)):
        seq_name = str(seq_field[0])
    else:
        seq_name = str(seq_field[0])
    return seq_name.replace('/', '_')


def _selected_values_by_frame(values: torch.Tensor | None, ix: torch.Tensor | None) -> list[torch.Tensor]:
    if values is None or ix is None or ix.numel() == 0:
        return []
    values = values.detach().reshape(-1)
    num_frames = int(ix.max().item()) + 1
    return [values[ix == frame_idx] for frame_idx in range(num_frames)]


@torch.no_grad()
def _collect_eval_scorer_artifacts(model, images, disps, args):
    if args.patch_selector != SelectionMethod.SCORER or not hasattr(model.patchify, 'scorer'):
        return None, None, None, None, None

    images_norm = _normalize_voxel(images, model.norm)
    with torch.amp.autocast('cuda', enabled=args.amp, dtype=torch.bfloat16):
        score_logits = model.patchify.scorer(images_norm)
    score_maps = torch.sigmoid(score_logits.float())

    disps_feature = disps[:, :, 1::4, 1::4].float() if disps is not None else None
    patchify_result = model.patchify(images_norm,
                                     patches_per_image=args.patches_per_image,
                                     disps=disps_feature)
    if len(patchify_result) == 6:
        fmap, gmap, imap, patches, ix, selected_scores = patchify_result
    else:
        fmap, gmap, imap, patches, ix = patchify_result
        selected_scores = None
        scorer_ctx = getattr(model.patchify, "last_scorer_context", None)
        if isinstance(scorer_ctx, dict):
            selected_logits = scorer_ctx.get("selected_score_logits")
            if selected_logits is not None:
                selected_scores = torch.sigmoid(selected_logits.float())
    return images_norm, score_maps, patches, ix, selected_scores


def _augment_scorer_context_with_replay(model, args, scorer_context, images, *, global_step=None):
    if scorer_context is None:
        return None

    if isinstance(scorer_context, dict):
        ctx = dict(scorer_context)
    else:
        ctx = dict(getattr(scorer_context, "__dict__", {}))

    replay_metrics = model.compute_replay_metrics(
        images,
        horizon=args.score_cycle_horizon,
        replay_steps=args.score_replay_steps,
        replay_runs=args.score_replay_runs,
    )
    ctx.update(replay_metrics)
    ctx["global_step"] = int(global_step) if global_step is not None else None
    ctx["total_steps"] = int(args.steps)
    return ctx


@torch.no_grad()
def evaluate_tracker(net, args, total_steps, val_loader):
    """
    Run PatchTracker on the validation set.

    Saves per-frame score maps (NPZ + PNG heatmap) and per-sequence track
    results (NPZ with coords_est, coords_gt, valid, weight at the last
    iteration).  Returns a metrics dict suitable for logger.push().

    Output layout:
        evals/{name}/{step:06d}/
            score_maps/seq{N:04d}_frame{F:04d}.{npz,png}
            tracks/seq{N:04d}.png
            meta.json
    """
    model = net.module if args.ddp else net
    model.eval()

    step_value = args.gpu_num * total_steps
    step_tag = f'{step_value:06d}'
    eval_root = Path('results/evals') / args.name
    eval_dir = eval_root / step_tag
    score_dir = eval_dir / 'score_maps'
    tracks_dir = eval_dir / 'tracks'
    score_dir.mkdir(parents=True, exist_ok=True)
    tracks_dir.mkdir(parents=True, exist_ok=True)

    all_epe = []
    all_survival = []
    all_rejection = []
    all_isotropy = []
    all_anisotropy = []
    all_diversity = []
    all_dynamic = []
    all_fb = []
    all_rep = []
    all_replay_valid = []
    per_seq = {}

    for seq_idx, data_blob in enumerate(val_loader):
        if args.eval_seqs > 0 and seq_idx >= args.eval_seqs:
            break

        seq_name = _extract_seq_name(data_blob[-1])
        data_blob = data_blob[:-1]
        images, poses, disps, intrinsics = [x.cuda().float() for x in data_blob]
        poses = SE3(poses).inv()

        seq_score_dir = score_dir / seq_name
        seq_tracks_dir = tracks_dir / seq_name
        seq_score_dir.mkdir(parents=True, exist_ok=True)
        seq_tracks_dir.mkdir(parents=True, exist_ok=True)

        images_norm, score_maps, patches_sel, ix_sel, selected_scores = _collect_eval_scorer_artifacts(
            model, images, disps, args
        )

        traj = model(images, poses, disps, intrinsics,
                     STEPS=args.iters,
                     patches_per_image=args.patches_per_image)

        scorer_ctx_eval = getattr(model.patchify, "last_scorer_context", None)
        if isinstance(scorer_ctx_eval, dict):
            images_norm = scorer_ctx_eval.get("images", images_norm)
            score_maps = scorer_ctx_eval.get("score_maps", score_maps)
            patches_sel = scorer_ctx_eval.get("patches", patches_sel)
            ix_sel = scorer_ctx_eval.get("ix", ix_sel)
            selected_logits = scorer_ctx_eval.get("selected_score_logits")
            if selected_logits is not None:
                selected_scores = torch.sigmoid(selected_logits.float())

        replay_ctx = None
        replay_summary = {}
        fb_values_by_frame = []
        stability_values_by_frame = []
        selected_centers_by_frame = []
        if patches_sel is not None and ix_sel is not None:
            selected_centers_by_frame = extract_selected_centers_by_frame(patches_sel, ix_sel)
            if args.patch_selector == SelectionMethod.SCORER:
                replay_ctx = _augment_scorer_context_with_replay(model, args, scorer_ctx_eval, images, global_step=step_value)
                replay_summary = compute_replay_metric_summary(
                    replay_ctx.get("fb_error_px"),
                    replay_ctx.get("stability_error_px"),
                    replay_ctx.get("replay_valid"),
                )
                fb_values_by_frame = _selected_values_by_frame(replay_ctx.get("fb_error_px"), ix_sel)
                stability_values_by_frame = _selected_values_by_frame(replay_ctx.get("stability_error_px"), ix_sel)

        if score_maps is not None:
            for frame_idx in range(score_maps.shape[1]):
                score_np = score_maps[0, frame_idx].detach().cpu().numpy().astype(np.float32)
                score_bgr = score_map_to_bgr(score_np)
                cv2.imwrite(str(seq_score_dir / f'frame{frame_idx:04d}_score.png'), score_bgr)

                if seq_idx < args.eval_viz_seqs and frame_idx < args.eval_viz_frames:
                    activity_bgr = _activity_frame_to_bgr(images[0, frame_idx])
                    centers_frame = selected_centers_by_frame[frame_idx] if frame_idx < len(selected_centers_by_frame) else torch.empty((0, 2))
                    mean_sel = float(selected_scores[frame_idx].mean().item()) if selected_scores is not None else float('nan')
                    subtitle = f"patches={centers_frame.shape[0]} mean_score={mean_sel:.4f}"
                    input_sel = _overlay_feature_centers(activity_bgr, centers_frame, scale=4.0, subtitle=subtitle)
                    score_sel = _overlay_feature_centers(score_bgr, centers_frame, scale=4.0, subtitle=subtitle)
                    compare_tiles = [
                        ('activity', activity_bgr),
                        ('input+selected', input_sel),
                        ('score', score_bgr),
                        ('score+selected', score_sel),
                    ]
                    if fb_values_by_frame and frame_idx < len(fb_values_by_frame):
                        fb_overlay = _overlay_metric_centers(
                            activity_bgr,
                            centers_frame,
                            fb_values_by_frame[frame_idx],
                            scale=4.0,
                            subtitle='fb_error_px',
                        )
                        cv2.imwrite(str(seq_score_dir / f'frame{frame_idx:04d}_fb_error.png'), fb_overlay)
                        compare_tiles.append(('fb_error', fb_overlay))
                    if stability_values_by_frame and frame_idx < len(stability_values_by_frame):
                        rep_overlay = _overlay_metric_centers(
                            activity_bgr,
                            centers_frame,
                            stability_values_by_frame[frame_idx],
                            scale=4.0,
                            subtitle='replay_stability_px',
                        )
                        cv2.imwrite(str(seq_score_dir / f'frame{frame_idx:04d}_stability.png'), rep_overlay)
                        compare_tiles.append(('stability', rep_overlay))
                    compare = _make_strip(compare_tiles)
                    cv2.imwrite(str(seq_score_dir / f'frame{frame_idx:04d}_compare.png'), compare)

        coords_est, coords_gt, valid, weight, _scores, kk_close = traj[-1]
        valid_mask = (valid > 0.5).reshape(-1)
        e = (coords_est - coords_gt).norm(dim=-1).reshape(-1)
        n_valid = int(valid_mask.sum().item())
        epe = e[valid_mask].mean().item() if n_valid > 0 else float('nan')
        all_epe.append(epe)

        est_np = coords_est[0].cpu().numpy()
        gt_np = coords_gt[0].cpu().numpy()
        vm_np = (valid[0] > 0.5).cpu().numpy()

        feat_h = score_maps.shape[-2] if score_maps is not None else int(coords_est[..., 1].max().item()) + 1
        feat_w = score_maps.shape[-1] if score_maps is not None else int(coords_est[..., 0].max().item()) + 1
        canvas = np.full((feat_h, feat_w, 3), 240, dtype=np.uint8)

        for idx in range(est_np.shape[0]):
            if not vm_np[idx]:
                continue
            ex = int(np.clip(round(float(est_np[idx, 0])), 0, feat_w - 1))
            ey = int(np.clip(round(float(est_np[idx, 1])), 0, feat_h - 1))
            gx = int(np.clip(round(float(gt_np[idx, 0])), 0, feat_w - 1))
            gy = int(np.clip(round(float(gt_np[idx, 1])), 0, feat_h - 1))
            cv2.line(canvas, (gx, gy), (ex, ey), (0, 200, 0), 1, cv2.LINE_AA)
            cv2.circle(canvas, (gx, gy), 2, (0, 0, 220), -1)
            cv2.circle(canvas, (ex, ey), 2, (0, 180, 0), -1)

        cv2.putText(canvas, f'EPE={epe:.3f}px  n={n_valid}', (4, 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.imwrite(str(seq_tracks_dir / 'tracks.png'), canvas)

        seq_metrics = {
            'epe_feature_px': epe,
            'num_valid_edges': n_valid,
        }
        if score_maps is not None and patches_sel is not None and ix_sel is not None and selected_scores is not None:
            survival_metrics = compute_patch_survival_metrics(
                valid, kk_close, num_patches=int(selected_scores.numel())
            )
            tensor_metrics = compute_selected_patch_tensor_metrics(images, patches_sel, ix_sel)
            diversity = compute_selection_spread(
                patches_sel, ix_sel, feat_h=score_maps.shape[-2], feat_w=score_maps.shape[-1]
            )
            dynamic_metrics = compute_score_map_dynamic_range(score_maps)
            seq_metrics.update({
                'patch_survival_rate': survival_metrics['patch_survival_rate'],
                'edge_rejection_rate': survival_metrics['edge_rejection_rate'],
                'mean_patch_valid_fraction': survival_metrics['mean_patch_valid_fraction'],
                'selected_patch_isotropy': tensor_metrics['selected_patch_isotropy'],
                'selected_patch_anisotropy': tensor_metrics['selected_patch_anisotropy'],
                'selection_diversity': diversity,
                'score_dynamic_range': dynamic_metrics['score_dynamic_range'],
                'qualitative_note': summarize_score_behavior(
                    dynamic_metrics['score_dynamic_range'],
                    tensor_metrics['selected_patch_isotropy'],
                    diversity,
                ),
            })
            seq_metrics.update(replay_summary)

            for series, value in (
                (all_survival, seq_metrics['patch_survival_rate']),
                (all_rejection, seq_metrics['edge_rejection_rate']),
                (all_isotropy, seq_metrics['selected_patch_isotropy']),
                (all_anisotropy, seq_metrics['selected_patch_anisotropy']),
                (all_diversity, seq_metrics['selection_diversity']),
                (all_dynamic, seq_metrics['score_dynamic_range']),
                (all_fb, seq_metrics.get('fb_cycle_error_px', float('nan'))),
                (all_rep, seq_metrics.get('replay_stability_px', float('nan'))),
                (all_replay_valid, seq_metrics.get('replay_valid_fraction', float('nan'))),
            ):
                if np.isfinite(value):
                    series.append(float(value))

        per_seq[seq_name] = seq_metrics

    valid_epes = [e for e in all_epe if not np.isnan(e)]
    mean_epe = float(np.mean(valid_epes)) if valid_epes else float('nan')
    mean_survival = float(np.mean(all_survival)) if all_survival else float('nan')
    mean_rejection = float(np.mean(all_rejection)) if all_rejection else float('nan')
    mean_isotropy = float(np.mean(all_isotropy)) if all_isotropy else float('nan')
    mean_anisotropy = float(np.mean(all_anisotropy)) if all_anisotropy else float('nan')
    mean_diversity = float(np.mean(all_diversity)) if all_diversity else float('nan')
    mean_dynamic = float(np.mean(all_dynamic)) if all_dynamic else float('nan')
    mean_fb = float(np.mean(all_fb)) if all_fb else float('nan')
    mean_rep = float(np.mean(all_rep)) if all_rep else float('nan')
    mean_replay_valid = float(np.mean(all_replay_valid)) if all_replay_valid else float('nan')

    meta = {
        'step': step_value,
        'score_objective': args.score_objective,
        'score_objective_implemented': args.score_objective in implemented_scorer_objective_names(),
        'scores_weight': args.scores_weight,
        'n_seqs': len(per_seq),
        'mean_epe_feature_px': mean_epe,
        'patch_survival_rate': mean_survival,
        'edge_rejection_rate': mean_rejection,
        'selected_patch_isotropy': mean_isotropy,
        'selected_patch_anisotropy': mean_anisotropy,
        'selection_diversity': mean_diversity,
        'score_dynamic_range': mean_dynamic,
        'fb_cycle_error_px': mean_fb,
        'replay_stability_px': mean_rep,
        'replay_valid_fraction': mean_replay_valid,
        'qualitative_note': summarize_score_behavior(mean_dynamic, mean_isotropy, mean_diversity),
        'per_seq': per_seq,
    }
    (eval_dir / 'meta.json').write_text(json.dumps(meta, indent=2))

    table_row = {
        'step': step_value,
        'score_objective': args.score_objective,
        'scores_weight': float(args.scores_weight),
        'val_epe_feature_px': mean_epe if np.isfinite(mean_epe) else "",
        'patch_survival_rate': mean_survival if np.isfinite(mean_survival) else "",
        'edge_rejection_rate': mean_rejection if np.isfinite(mean_rejection) else "",
        'selected_patch_isotropy': mean_isotropy if np.isfinite(mean_isotropy) else "",
        'selected_patch_anisotropy': mean_anisotropy if np.isfinite(mean_anisotropy) else "",
        'selection_diversity': mean_diversity if np.isfinite(mean_diversity) else "",
        'score_dynamic_range': mean_dynamic if np.isfinite(mean_dynamic) else "",
        'fb_cycle_error_px': mean_fb if np.isfinite(mean_fb) else "",
        'replay_stability_px': mean_rep if np.isfinite(mean_rep) else "",
        'replay_valid_fraction': mean_replay_valid if np.isfinite(mean_replay_valid) else "",
        'qualitative_note': meta['qualitative_note'],
    }
    append_experiment_table_row(eval_root / 'experiment_table.csv', table_row)

    print(f"[eval step {step_value}] mean EPE = {mean_epe:.4f} px (feature res) "
          f"over {len(valid_epes)}/{len(all_epe)} seqs — saved to {eval_dir}")

    model.train()
    metrics = {'val/epe_feature_px': mean_epe}
    if np.isfinite(mean_survival):
        metrics['val/patch_survival_rate'] = mean_survival
    if np.isfinite(mean_rejection):
        metrics['val/rejection_rate'] = mean_rejection
    if np.isfinite(mean_isotropy):
        metrics['val/selected_patch_isotropy'] = mean_isotropy
    if np.isfinite(mean_anisotropy):
        metrics['val/selected_patch_anisotropy'] = mean_anisotropy
    if np.isfinite(mean_diversity):
        metrics['val/selection_diversity'] = mean_diversity
    if np.isfinite(mean_dynamic):
        metrics['val/score_dynamic_range'] = mean_dynamic
    if np.isfinite(mean_fb):
        metrics['val/fb_cycle_error_px'] = mean_fb
    if np.isfinite(mean_rep):
        metrics['val/replay_stability_px'] = mean_rep
    if np.isfinite(mean_replay_valid):
        metrics['val/replay_valid_fraction'] = mean_replay_valid
    return metrics


def setup_ddp(rank, args):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = args.port
    dist.init_process_group(
        backend='nccl',
        init_method='env://',
        world_size=args.gpu_num,
        rank=rank)
    torch.manual_seed(0)
    torch.cuda.set_device(rank)


def train(rank, args):
    """Main training loop for PatchTracker."""

    if args.ddp:
        setup_ddp(rank, args)

    # ── Dataset ──────────────────────────────────────────────────────────────
    db = dataset_factory(
        ['tartan_evs'],
        datapath=args.datapath,
        n_frames=args.n_frames,
        fgraph_pickle=args.fgraph_pickle,
        train_split=args.train_split,
        val_split=args.val_split,
        split_mode='train',
        strict_split=False,
        sample=True,
        return_fname=True,
        scale=args.scale,
    )

    if args.ddp:
        sampler = torch.utils.data.distributed.DistributedSampler(
            db, shuffle=True, num_replicas=args.gpu_num, rank=rank)
        train_loader = DataLoader(db, batch_size=args.batch, sampler=sampler,
                                  num_workers=args.num_workers, pin_memory=True, prefetch_factor=4)
    else:
        train_loader = DataLoader(db, batch_size=args.batch, shuffle=True,
                                  num_workers=args.num_workers, pin_memory=True, prefetch_factor=4)

    # ── Network ───────────────────────────────────────────────────────────────
    net = PatchTracker(
        args,
        dim_inet=args.dim_inet,
        dim_fnet=args.dim_fnet,
        dim=args.dim,
        patch_selector=args.patch_selector.lower(),
        norm=args.norm,
        randaug=args.randaug,
        corner_guidance=args.corner_guidance,
    )
    net.train()
    net.cuda()

    if args.ddp:
        net = DDP(net, device_ids=[rank], find_unused_parameters=False)
    model_ref = net.module if args.ddp else net

    # ── Val loader (for periodic evaluation) ─────────────────────────────────
    val_loader = None
    if args.eval and rank == 0:
        val_db = dataset_factory(
            ['tartan_evs'],
            datapath=args.datapath,
            n_frames=args.n_frames,
            fgraph_pickle=args.fgraph_pickle,
            train_split=args.train_split,
            val_split=args.val_split,
            split_mode='val',
            strict_split=False,
            sample=False,
            return_fname=True,
            scale=args.scale,
        )
        # Pick the first window index for each unique scene — O(n) metadata scan,
        # avoids iterating through thousands of duplicate windows per sequence.
        # dataset_factory returns a ConcatDataset; iterate sub-datasets to build
        # a flat offset-adjusted index list.
        seen, one_per_seq = set(), []
        sub_datasets = val_db.datasets if hasattr(val_db, 'datasets') else [val_db]
        offset = 0
        for sub_db in sub_datasets:
            for local_idx, (scene_id, _) in enumerate(sub_db.dataset_index):
                if scene_id not in seen:
                    seen.add(scene_id)
                    one_per_seq.append(offset + local_idx)
            offset += len(sub_db)
        val_subset = torch.utils.data.Subset(val_db, one_per_seq)
        val_loader = DataLoader(val_subset, batch_size=1, shuffle=False,
                                num_workers=2, pin_memory=True)

    # ── Optimizer ─────────────────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-6)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, args.lr, args.steps,
        pct_start=0.01, cycle_momentum=False, anneal_strategy='linear')
    scaler = torch.amp.GradScaler('cuda', enabled=args.amp)

    total_steps = 0

    # ── Resume ────────────────────────────────────────────────────────────────
    if args.resume:
        print(f"Loading from {args.resume}")
        checkpoint = torch.load(args.resume)
        model = net.module if args.ddp else net
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        else:
            new_sd = OrderedDict()
            for k, v in checkpoint.items():
                new_sd[k.replace('module.', '')] = v
            update = {k: v for k, v in new_sd.items()
                      if k in model.state_dict() and model.state_dict()[k].shape == v.shape}
            state = model.state_dict()
            state.update(update)
            model.load_state_dict(state, strict=False)
        if 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        if 'steps' in checkpoint:
            total_steps = checkpoint['steps']

    if rank == 0:
        logger = Logger(args.name, scheduler, args.gpu_num * total_steps, args.gpu_num)
        print(
            f"Scorer objective: {args.score_objective} "
            f"(scores_weight={args.scores_weight}, tau_e={args.score_rank_tau_e}, "
            f"margin={args.score_rank_margin}, top_q={args.score_rank_top_quantile}, "
            f"bottom_q={args.score_rank_bottom_quantile})"
        )

    os.makedirs(f'checkpoints/{args.name}', exist_ok=True)

    # ── Training loop ─────────────────────────────────────────────────────────
    with contextlib.nullcontext():
        if rank == 0:
            pbar = tqdm(total=args.gpu_num * args.steps,
                        initial=args.gpu_num * total_steps,
                        desc=f"Training {args.name}")

        while True:
            for data_blob in train_loader:
                data_blob.pop()  # pop scene_id (return_fname=True appends it)
                images, poses, disps, intrinsics = [x.cuda().float() for x in data_blob]

                optimizer.zero_grad(set_to_none=True)

                # convert GT poses from c2w → w2c (same as train.py:188)
                poses = SE3(poses).inv()

                with torch.amp.autocast('cuda', enabled=args.amp, dtype=torch.bfloat16):
                    traj = net(images, poses, disps, intrinsics,
                               STEPS=args.iters,
                               patches_per_image=args.patches_per_image)

                # ── Loss ──────────────────────────────────────────────────────
                loss = torch.tensor(0.0, device='cuda')
                flow_loss_last = torch.tensor(0.0, device='cuda')
                scores_loss = torch.tensor(0.0, device='cuda')
                scorer_metrics = {}

                for i, (coords_est, coords_gt, valid, weight, scores, kk_close) in enumerate(traj):
                    # exponential iteration weighting (RAFT-style: earlier iters matter less)
                    w_i = args.gamma ** (len(traj) - i - 1)

                    valid_mask = (valid > 0.5).reshape(-1)
                    e = (coords_est - coords_gt).norm(dim=-1)  # (B, close_edges)
                    e_flat = e.reshape(-1)

                    if valid_mask.any():
                        fl = e_flat[valid_mask].mean()
                    else:
                        fl = torch.tensor(0.0, device='cuda')

                    loss = loss + w_i * args.flow_weight * fl

                    if i == len(traj) - 1:
                        flow_loss_last = fl

                    # scorer loss: at final iteration only
                    is_last = (i == len(traj) - 1)
                    if args.patch_selector == SelectionMethod.SCORER and is_last and scores is not None:
                        scorer_context = getattr(model_ref.patchify, "last_scorer_context", None)
                        if args.score_objective in replay_scorer_objective_names():
                            scorer_context = _augment_scorer_context_with_replay(
                                model_ref,
                                args,
                                scorer_context,
                                images,
                                global_step=total_steps,
                            )
                        objective_result = compute_scorer_objective(
                            args.score_objective,
                            coords_est=coords_est,
                            coords_gt=coords_gt,
                            valid=valid,
                            weight=weight.detach(),
                            scores=scores,
                            kk_close=kk_close,
                            args=args,
                            scorer_context=scorer_context,
                        )
                        scores_loss = objective_result.loss
                        scorer_metrics = objective_result.metrics
                        loss = loss + args.scores_weight * scores_loss

                if torch.isnan(loss):
                    print(f"NaN at step {total_steps}")
                    optimizer.zero_grad(set_to_none=True)
                    total_steps += 1
                    scheduler.step()
                    continue

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(net.parameters(), args.clip)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()

                total_steps += 1

                metrics = {
                    "loss/train": loss.item(),
                    "loss/flow_train": flow_loss_last.item(),
                    "loss/scores_train": scores_loss.item() if isinstance(scores_loss, torch.Tensor) else 0.0,
                }
                metrics.update(scorer_metrics)

                if rank == 0:
                    pbar.update(args.gpu_num)
                    logger.push(metrics)

                # ── Checkpoint ────────────────────────────────────────────────
                if total_steps % args.save_freq == 0 or total_steps >= args.steps:
                    torch.cuda.empty_cache()
                    if rank == 0:
                        path = f'checkpoints/{args.name}/{args.gpu_num * total_steps:06d}.pth'
                        torch.save({
                            'steps': total_steps,
                            'model_state_dict': net.module.state_dict() if args.ddp else net.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'scheduler_state_dict': scheduler.state_dict(),
                        }, path)

                        if args.eval and val_loader is not None:
                            val_metrics = evaluate_tracker(net, args, total_steps, val_loader)
                            logger.push(val_metrics)

                    torch.cuda.empty_cache()
                    net.train()

                if total_steps >= args.steps:
                    break
            else:
                continue
            break

    if rank == 0:
        logger.close()
    if args.ddp:
        dist.destroy_process_group()


if __name__ == '__main__':
    import configargparse
    parser = configargparse.ArgumentParser()
    parser.add_argument('-c', '--config', default='config/train_tracker_base.conf',
                        is_config_file=True, help='config file path')
    parser.add_argument('--name', '--expname', default='tracker_run')
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--fgraph_pickle', type=str, default='fgraph/TartanAirEVS.pickle')
    parser.add_argument('--datapath', default='')
    parser.add_argument('--train_split', type=str, default='script/splits/tartan/tartan_default_train.txt')
    parser.add_argument('--val_split', type=str, default='script/splits/tartan/tartan_default_val.txt')

    parser.add_argument('--batch', type=int, default=1)
    parser.add_argument('--steps', type=int, default=200000)
    parser.add_argument('--save_freq', type=int, default=10000)
    parser.add_argument('--iters', type=int, default=18)
    parser.add_argument('--lr', type=float, default=0.00008)
    parser.add_argument('--clip', type=float, default=10.0)
    parser.add_argument('--n_frames', type=int, default=15)
    parser.add_argument('--gamma', type=float, default=0.8,
                        help='exponential weight for earlier iterations (RAFT-style)')

    parser.add_argument('--flow_weight', type=float, default=1.0)
    parser.add_argument('--scores_weight', type=float, default=0.05)
    parser.add_argument('--score_objective', type=str, default='baseline',
                        choices=scorer_objective_names(),
                        help='training-time scorer objective; baseline matches the current loss')
    parser.add_argument('--score_rank_tau_e', type=float, default=1.0,
                        help='error temperature for rank_only utility u=exp(-e/tau_e)*conf')
    parser.add_argument('--score_rank_margin', type=float, default=0.2,
                        help='pairwise margin for rank_only scorer-logit ranking')
    parser.add_argument('--score_rank_top_quantile', type=float, default=0.25,
                        help='top utility quantile treated as positives for rank_only')
    parser.add_argument('--score_rank_bottom_quantile', type=float, default=0.25,
                        help='bottom utility quantile treated as negatives for rank_only')
    parser.add_argument('--score_repeatability_weight', type=float, default=1.0,
                        help='weight on repeatability consistency term')
    parser.add_argument('--score_repeatability_dropout', type=float, default=0.15,
                        help='event dropout applied to repeatability perturbations')
    parser.add_argument('--score_repeatability_bin_shift', type=int, default=1,
                        help='max temporal-bin roll applied to repeatability perturbations')
    parser.add_argument('--score_diversity_weight', type=float, default=0.25,
                        help='weight on selected-set diversity regularizer')
    parser.add_argument('--score_diversity_min_separation', type=float, default=0.06,
                        help='normalized minimum nearest-neighbor spacing target for diversity')
    parser.add_argument('--score_teacher_weight', type=float, default=0.5,
                        help='weight on event-teacher distillation term')
    parser.add_argument('--score_teacher_warmup_frac', type=float, default=0.2,
                        help='fraction of training over which replay-objective teacher warmup decays to zero')
    parser.add_argument('--score_aux_head_weight', type=float, default=1.0,
                        help='weight on auxiliary head regression terms')
    parser.add_argument('--score_cycle_horizon', type=int, default=2,
                        help='future frame offset used by replay-based scorer objectives')
    parser.add_argument('--score_replay_steps', type=int, default=4,
                        help='update iterations used by replay-based scorer objectives')
    parser.add_argument('--score_replay_runs', type=int, default=3,
                        help='number of replay runs for stability objectives (1 clean + perturbed runs)')
    parser.add_argument('--score_cycle_tau_fb', type=float, default=1.5,
                        help='temperature for forward-backward replay error utility')
    parser.add_argument('--score_cycle_tau_rep', type=float, default=1.0,
                        help='temperature for replay-stability utility')

    parser.add_argument('--patches_per_image', type=int, default=80)
    parser.add_argument('--patch_selector', type=str, default='scorer')
    parser.add_argument('--corner_guidance', type=str, default='none',
                        help='Multiply scorer map by corner response: none | harris | shitomasi')
    parser.add_argument('--norm', type=str, default='std2')
    parser.add_argument('--randaug', action='store_true')

    parser.add_argument('--eval', action='store_true', default=False)
    parser.add_argument('--eval_seqs', type=int, default=10,
                        help='Max number of val sequences to evaluate at each checkpoint')
    parser.add_argument('--eval_viz_seqs', type=int, default=3,
                        help='How many fixed val sequences get score-map/selection visual dumps')
    parser.add_argument('--eval_viz_frames', type=int, default=3,
                        help='How many frames per val sequence to dump for scorer diagnostics')
    parser.add_argument('--evs', action='store_true', default=True)
    parser.add_argument('--scale', type=float, default=1.0)
    parser.add_argument('--ddp', action='store_true')
    parser.add_argument('--gpu_num', type=int, default=1)
    parser.add_argument('--port', default='12349')
    parser.add_argument('--amp', action='store_true')
    parser.add_argument('--num_workers', type=int, default=4)

    parser.add_argument('--dim_inet', type=int, default=384)
    parser.add_argument('--dim_fnet', type=int, default=128)
    parser.add_argument('--dim', type=int, default=32)
    parser.add_argument('--resnet', action='store_true')
    parser.add_argument('--block_dims', type=str, default='64,128,256')

    args = parser.parse_args()
    if args.score_objective not in implemented_scorer_objective_names():
        raise NotImplementedError(
            f"score objective '{args.score_objective}' is reserved in the roadmap but not implemented yet. "
            f"Implemented objectives: {implemented_scorer_objective_names()}"
        )
    if not (0.0 < args.score_rank_top_quantile <= 0.5):
        raise ValueError(f'--score_rank_top_quantile must be in (0, 0.5], got {args.score_rank_top_quantile}')
    if not (0.0 < args.score_rank_bottom_quantile <= 0.5):
        raise ValueError(f'--score_rank_bottom_quantile must be in (0, 0.5], got {args.score_rank_bottom_quantile}')
    if not (0.0 <= args.score_repeatability_dropout < 1.0):
        raise ValueError(f'--score_repeatability_dropout must be in [0, 1), got {args.score_repeatability_dropout}')
    if args.score_repeatability_bin_shift < 0:
        raise ValueError('--score_repeatability_bin_shift must be >= 0')
    if args.score_diversity_min_separation < 0.0:
        raise ValueError('--score_diversity_min_separation must be >= 0')
    if args.score_teacher_warmup_frac < 0.0:
        raise ValueError('--score_teacher_warmup_frac must be >= 0')
    if args.score_cycle_horizon < 1:
        raise ValueError('--score_cycle_horizon must be >= 1')
    if args.score_replay_steps < 1:
        raise ValueError('--score_replay_steps must be >= 1')
    if args.score_replay_runs < 1:
        raise ValueError('--score_replay_runs must be >= 1')
    if args.score_cycle_tau_fb <= 0.0:
        raise ValueError('--score_cycle_tau_fb must be > 0')
    if args.score_cycle_tau_rep <= 0.0:
        raise ValueError('--score_cycle_tau_rep must be > 0')
    if args.eval_viz_seqs < 0 or args.eval_viz_frames < 0:
        raise ValueError('--eval_viz_seqs and --eval_viz_frames must be >= 0')

    if args.ddp:
        mp.spawn(train, nprocs=args.gpu_num, args=(args,))
    else:
        train(0, args)
