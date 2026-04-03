from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F

# baseline: the original loss. It rewards high scores on patches with low tracking error and adds a -log(score) barrier so scores do not collapse to zero. See devo/scorer_objectives.py (line 471).
# rank_only: instead of pushing raw score magnitude up, it ranks patches within each frame. Patch utility is exp(-error / tau_e) * confidence, and the loss makes high-utility patches outrank low-utility ones by a margin. See devo/scorer_objectives.py (line 501).
# corr_isotropy: same ranking setup, but the target utility is multiplied by an isotropy term from the local structure tensor. So corner-like patches get boosted and edge-like patches get suppressed. See devo/scorer_objectives.py (line 537).
# short_horizon_survivability: ranks patches by a survivability proxy: valid-fraction across supervised edges, mean tracking error, and confidence. This is “stay valid and accurate” rather than “look strong once.” See devo/scorer_objectives.py (line 578).
# repeatability: starts from rank_only, then adds a consistency loss. It perturbs the event voxel input with dropout, bin roll, and gain jitter, and penalizes changes in the selected patch scores under that perturbation. See devo/scorer_objectives.py (line 614).
# diversity: starts from rank_only, then adds a penalty when selected patches are too spatially clustered and have similar local orientation/anistropy. This discourages wasting many patches on the same edge fragment. See devo/scorer_objectives.py (line 655).
# multi_motion_consistency: ranks patches higher when their supervised motions span more than one direction. If a patch only supports one motion direction, it scores lower; if its associated motions are more directionally spread, it scores higher. See devo/scorer_objectives.py (line 676).
# event_teacher: builds an event-native soft teacher from a temporally weighted event surface plus a corner-like structure-tensor response, then combines ranking with a distillation term that aligns scorer outputs to that teacher. See devo/scorer_objectives.py (line 716).
# info_gain_head: uses the auxiliary info_gain head. That head is trained to predict a utility proxy based on validity, error, and confidence, and then the main score head is trained to rank patches according to the aux prediction. See devo/scorer_objectives.py (line 758).
# conditioning_head: uses the auxiliary conditioning head. That head is trained to predict patch isotropy, and the main score head is then trained to rank patches according to utility weighted by the predicted conditioning quality. See devo/scorer_objectives.py (line 804).
# forward_backward_cycle: ranks patches by how well they replay forward and then return to their source location under a backward replay pass.
# replay_stability: ranks patches by how little their replay endpoint moves across small event perturbations.
# cycle_stability: combines forward-backward and replay-stability signals into one replay-based reliability target.

# A practical grouping is:
# Pure scorer-loss variants: baseline, rank_only, corr_isotropy, short_horizon_survivability, repeatability, diversity, multi_motion_consistency, event_teacher
# Aux-head variants: info_gain_head, conditioning_head


EPS = 1e-6


@dataclass
class ScorerObjectiveResult:
    loss: torch.Tensor
    metrics: dict[str, float]


@dataclass
class ScorerContext:
    images: torch.Tensor | None = None
    patches: torch.Tensor | None = None
    ix: torch.Tensor | None = None
    selected_score_coords: torch.Tensor | None = None
    selected_score_logits: torch.Tensor | None = None
    score_logits_map: torch.Tensor | None = None
    score_maps: torch.Tensor | None = None
    aux_logits_maps: dict[str, torch.Tensor] | None = None
    aux_selected_logits: dict[str, torch.Tensor] | None = None
    scorer_module: object | None = None
    fb_error_px: torch.Tensor | None = None
    stability_error_px: torch.Tensor | None = None
    replay_valid: torch.Tensor | None = None
    forward_endpoint: torch.Tensor | None = None
    forward_conf: torch.Tensor | None = None
    global_step: int | None = None
    total_steps: int | None = None


@dataclass
class PatchUtilityStats:
    valid_mask: torch.Tensor
    edge_error: torch.Tensor
    edge_error_valid: torch.Tensor
    edge_conf_valid: torch.Tensor
    patch_indices_all: torch.Tensor
    patch_indices_valid: torch.Tensor
    patch_scores: torch.Tensor
    patch_logits: torch.Tensor
    patch_utility: torch.Tensor
    patch_valid_counts: torch.Tensor
    patch_total_counts: torch.Tensor
    patch_valid_fraction: torch.Tensor
    patch_error_mean: torch.Tensor
    patch_conf_mean: torch.Tensor


def _safe_zero(device: torch.device) -> torch.Tensor:
    return torch.zeros((), device=device, dtype=torch.float32)


def _safe_logit(x: torch.Tensor, eps: float = EPS) -> torch.Tensor:
    x = x.clamp(min=eps, max=1.0 - eps)
    return torch.log(x) - torch.log1p(-x)


def _to_context(ctx: ScorerContext | dict | None) -> ScorerContext | None:
    if ctx is None:
        return None
    if isinstance(ctx, ScorerContext):
        return ctx
    return ScorerContext(**ctx)


def _selected_patch_logits(stats: PatchUtilityStats, scorer_context: ScorerContext | None) -> torch.Tensor:
    if scorer_context is not None and scorer_context.selected_score_logits is not None:
        return scorer_context.selected_score_logits.reshape(-1).float()
    return stats.patch_logits


def _sample_selected_map_values(map_tensor: torch.Tensor, selected_score_coords: torch.Tensor) -> torch.Tensor:
    x = selected_score_coords[..., 0].long().clamp(0, map_tensor.shape[-1] - 1)
    y = selected_score_coords[..., 1].long().clamp(0, map_tensor.shape[-2] - 1)
    frame_idx = torch.arange(map_tensor.shape[1], device=map_tensor.device).view(-1, 1).expand_as(x)
    return map_tensor[0, frame_idx, y, x]


def _get_selected_aux_logits(
    scorer_context: ScorerContext | None,
    name: str,
) -> torch.Tensor | None:
    if scorer_context is None:
        return None
    if scorer_context.aux_selected_logits is not None and name in scorer_context.aux_selected_logits:
        return scorer_context.aux_selected_logits[name].reshape(-1).float()
    if (
        scorer_context.aux_logits_maps is not None
        and name in scorer_context.aux_logits_maps
        and scorer_context.selected_score_coords is not None
    ):
        return _sample_selected_map_values(
            scorer_context.aux_logits_maps[name],
            scorer_context.selected_score_coords,
        ).reshape(-1).float()
    return None


def _get_replay_signal(
    scorer_context: ScorerContext | None,
    name: str,
    num_patches: int,
    device: torch.device,
) -> torch.Tensor | None:
    if scorer_context is None:
        return None
    value = getattr(scorer_context, name, None)
    if value is None:
        return None
    value = value.reshape(-1).to(device=device)
    if value.numel() != num_patches:
        return None
    return value.float()


def _teacher_warmup_weight(args, scorer_context: ScorerContext | None) -> float:
    warmup_frac = float(getattr(args, "score_teacher_warmup_frac", 0.2))
    base_weight = float(getattr(args, "score_teacher_weight", 0.3))
    if warmup_frac <= 0.0 or base_weight <= 0.0:
        return 0.0

    total_steps = None
    global_step = None
    if scorer_context is not None:
        total_steps = scorer_context.total_steps
        global_step = scorer_context.global_step

    if total_steps is None:
        total_steps = int(getattr(args, "steps", 0))
    if global_step is None:
        global_step = 0
    if total_steps <= 0:
        return base_weight

    warmup_steps = max(int(round(total_steps * warmup_frac)), 1)
    progress = min(max(float(global_step) / float(warmup_steps), 0.0), 1.0)
    return base_weight * (1.0 - progress)


def _compute_patch_utility_stats(
    coords_est: torch.Tensor,
    coords_gt: torch.Tensor,
    valid: torch.Tensor,
    weight: torch.Tensor,
    scores: torch.Tensor,
    kk_close: torch.Tensor,
    tau_e: float,
    scorer_context: ScorerContext | None = None,
) -> PatchUtilityStats:
    device = coords_est.device
    b = coords_est.shape[0]
    num_patches = int(scores.numel())

    valid_mask = (valid > 0.5).reshape(-1)
    edge_error = (coords_est - coords_gt).norm(dim=-1).reshape(-1)
    edge_conf_valid = torch.empty(0, device=device, dtype=torch.float32)
    edge_error_valid = edge_error[valid_mask]
    patch_indices_all = kk_close[None].expand(b, -1).reshape(-1)
    patch_indices_valid = patch_indices_all[valid_mask]

    if valid_mask.any():
        weight_flat = weight.reshape(-1, 2)
        edge_conf_valid = (-0.5 * weight_flat[valid_mask].mean(dim=-1)).exp()

    patch_scores = scores.reshape(-1).float()
    patch_logits = _selected_patch_logits(
        PatchUtilityStats(
            valid_mask=valid_mask,
            edge_error=edge_error,
            edge_error_valid=edge_error_valid,
            edge_conf_valid=edge_conf_valid,
            patch_indices_all=patch_indices_all,
            patch_indices_valid=patch_indices_valid,
            patch_scores=patch_scores,
            patch_logits=_safe_logit(patch_scores),
            patch_utility=torch.empty(0, device=device),
            patch_valid_counts=torch.empty(0, device=device),
            patch_total_counts=torch.empty(0, device=device),
            patch_valid_fraction=torch.empty(0, device=device),
            patch_error_mean=torch.empty(0, device=device),
            patch_conf_mean=torch.empty(0, device=device),
        ),
        scorer_context,
    )

    patch_utility_sum = torch.zeros(num_patches, device=device, dtype=torch.float32)
    patch_error_sum = torch.zeros(num_patches, device=device, dtype=torch.float32)
    patch_conf_sum = torch.zeros(num_patches, device=device, dtype=torch.float32)
    patch_valid_counts = torch.zeros(num_patches, device=device, dtype=torch.float32)
    patch_total_counts = torch.zeros(num_patches, device=device, dtype=torch.float32)

    if patch_indices_all.numel() > 0:
        patch_total_counts.index_add_(
            0,
            patch_indices_all,
            torch.ones_like(patch_indices_all, dtype=torch.float32),
        )

    if patch_indices_valid.numel() > 0:
        edge_utility_valid = torch.exp(-edge_error_valid / max(float(tau_e), EPS)) * edge_conf_valid
        patch_utility_sum.index_add_(0, patch_indices_valid, edge_utility_valid)
        patch_error_sum.index_add_(0, patch_indices_valid, edge_error_valid)
        patch_conf_sum.index_add_(0, patch_indices_valid, edge_conf_valid)
        patch_valid_counts.index_add_(
            0,
            patch_indices_valid,
            torch.ones_like(patch_indices_valid, dtype=torch.float32),
        )

    patch_utility = patch_utility_sum / patch_valid_counts.clamp_min(1.0)
    patch_valid_fraction = patch_valid_counts / patch_total_counts.clamp_min(1.0)
    patch_error_mean = patch_error_sum / patch_valid_counts.clamp_min(1.0)
    patch_conf_mean = patch_conf_sum / patch_valid_counts.clamp_min(1.0)

    return PatchUtilityStats(
        valid_mask=valid_mask,
        edge_error=edge_error,
        edge_error_valid=edge_error_valid,
        edge_conf_valid=edge_conf_valid,
        patch_indices_all=patch_indices_all,
        patch_indices_valid=patch_indices_valid,
        patch_scores=patch_scores,
        patch_logits=patch_logits,
        patch_utility=patch_utility,
        patch_valid_counts=patch_valid_counts,
        patch_total_counts=patch_total_counts,
        patch_valid_fraction=patch_valid_fraction,
        patch_error_mean=patch_error_mean,
        patch_conf_mean=patch_conf_mean,
    )


def _framewise_rank_loss(
    patch_logits: torch.Tensor,
    target: torch.Tensor,
    num_frames: int,
    patches_per_image: int,
    margin: float,
    top_q: float,
    bottom_q: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    device = patch_logits.device
    loss_terms = []
    num_frames_used = 0
    num_pairs = 0
    pos_mean = []
    neg_mean = []

    for frame_idx in range(num_frames):
        start = frame_idx * patches_per_image
        end = start + patches_per_image

        logits_frame = patch_logits[start:end]
        target_frame = target[start:end]
        count = int(logits_frame.numel())
        if count < 2:
            continue

        max_each = max(1, count // 2)
        k_pos = max(1, min(int(round(count * top_q)), max_each))
        k_neg = max(1, min(int(round(count * bottom_q)), max_each))
        order = torch.argsort(target_frame)
        neg_idx = order[:k_neg]
        pos_idx = order[-k_pos:]

        pos_logits = logits_frame[pos_idx]
        neg_logits = logits_frame[neg_idx]
        frame_loss = F.relu(margin - (pos_logits[:, None] - neg_logits[None, :])).mean()
        loss_terms.append(frame_loss)
        num_frames_used += 1
        num_pairs += int(pos_logits.numel() * neg_logits.numel())
        pos_mean.append(float(target_frame[pos_idx].mean().detach().item()))
        neg_mean.append(float(target_frame[neg_idx].mean().detach().item()))

    if loss_terms:
        loss = torch.stack(loss_terms).mean()
    else:
        loss = _safe_zero(device)

    metrics = {
        "scorer/rank_frames_train": float(num_frames_used),
        "scorer/rank_pairs_train": float(num_pairs),
        "scorer/pos_utility_mean_train": float(sum(pos_mean) / len(pos_mean)) if pos_mean else 0.0,
        "scorer/neg_utility_mean_train": float(sum(neg_mean) / len(neg_mean)) if neg_mean else 0.0,
    }
    return loss, metrics


def _structure_tensor_fields(images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    device = images.device
    activity = images[0].detach().float().abs().sum(dim=1, keepdim=True)
    activity = F.avg_pool2d(activity, kernel_size=4, stride=4)

    sobel_x = torch.tensor(
        [[[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]],
        device=device,
        dtype=activity.dtype,
    ).view(1, 1, 3, 3)
    sobel_y = torch.tensor(
        [[[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]],
        device=device,
        dtype=activity.dtype,
    ).view(1, 1, 3, 3)

    gx = F.conv2d(activity, sobel_x, padding=1)
    gy = F.conv2d(activity, sobel_y, padding=1)
    jxx = F.avg_pool2d(gx * gx, kernel_size=3, stride=1, padding=1)
    jyy = F.avg_pool2d(gy * gy, kernel_size=3, stride=1, padding=1)
    jxy = F.avg_pool2d(gx * gy, kernel_size=3, stride=1, padding=1)
    return activity, jxx, jyy, jxy


def _selected_patch_structure(
    images: torch.Tensor,
    patches: torch.Tensor,
    ix: torch.Tensor,
) -> dict[str, torch.Tensor]:
    p = patches.shape[-1]
    centers = patches[0, :, :2, p // 2, p // 2].detach().float()
    if centers.numel() == 0:
        empty = torch.empty(0, device=images.device)
        return {
            "isotropy": empty,
            "anisotropy": empty,
            "orientation": empty,
            "density": empty,
        }

    activity, jxx, jyy, jxy = _structure_tensor_fields(images)
    centers_x = centers[:, 0].round().long().clamp(0, jxx.shape[-1] - 1)
    centers_y = centers[:, 1].round().long().clamp(0, jxx.shape[-2] - 1)
    frame_idx = ix.long()

    a = jxx[frame_idx, 0, centers_y, centers_x]
    c = jyy[frame_idx, 0, centers_y, centers_x]
    b = jxy[frame_idx, 0, centers_y, centers_x]
    delta = torch.sqrt((a - c).pow(2) + 4.0 * b.pow(2) + EPS)
    eig_max = 0.5 * (a + c + delta)
    eig_min = 0.5 * (a + c - delta)

    isotropy = eig_min / eig_max.clamp_min(EPS)
    anisotropy = (eig_max - eig_min) / eig_max.clamp_min(EPS)
    orientation = 0.5 * torch.atan2(2.0 * b, a - c + EPS)
    density = activity[frame_idx, 0, centers_y, centers_x]
    density = density / density.amax().clamp_min(EPS)

    return {
        "isotropy": isotropy,
        "anisotropy": anisotropy,
        "orientation": orientation,
        "density": density,
    }


def _event_teacher_map(images: torch.Tensor) -> torch.Tensor:
    b, n, bins, h, w = images.shape
    weights = torch.linspace(0.25, 1.0, bins, device=images.device, dtype=images.dtype)
    surface = (images * weights.view(1, 1, bins, 1, 1)).sum(dim=2).view(b * n, 1, h, w)
    density = images.abs().sum(dim=2).view(b * n, 1, h, w)

    sobel_x = torch.tensor(
        [[[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]],
        device=images.device,
        dtype=surface.dtype,
    ).view(1, 1, 3, 3) / 8.0
    sobel_y = torch.tensor(
        [[[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]],
        device=images.device,
        dtype=surface.dtype,
    ).view(1, 1, 3, 3) / 8.0

    sx = F.conv2d(surface, sobel_x, padding=1)
    sy = F.conv2d(surface, sobel_y, padding=1)
    avg = torch.ones(1, 1, 3, 3, dtype=surface.dtype, device=surface.device) / 9.0
    sxx = F.conv2d(sx * sx, avg, padding=1)
    syy = F.conv2d(sy * sy, avg, padding=1)
    sxy = F.conv2d(sx * sy, avg, padding=1)

    trace = sxx + syy
    det = sxx * syy - sxy * sxy
    lam_min = 0.5 * (trace - (trace * trace - 4.0 * det).clamp(min=0).sqrt())
    lam_min = F.avg_pool2d(lam_min.clamp(min=0), kernel_size=4, stride=4)
    density = F.avg_pool2d(density, kernel_size=4, stride=4)
    teacher = lam_min * density.sqrt()
    teacher = teacher.view(b, n, teacher.shape[-2], teacher.shape[-1])
    teacher = teacher / teacher.amax(dim=(-2, -1), keepdim=True).clamp_min(EPS)
    return teacher


def _selected_event_teacher(scorer_context: ScorerContext | None) -> torch.Tensor | None:
    if scorer_context is None or scorer_context.images is None or scorer_context.selected_score_coords is None:
        return None
    teacher_map = _event_teacher_map(scorer_context.images.detach())
    return _sample_selected_map_values(teacher_map, scorer_context.selected_score_coords).reshape(-1).float()


def _apply_event_perturbation(images: torch.Tensor, args) -> torch.Tensor:
    images = images.detach()
    keep_prob = 1.0 - float(getattr(args, "score_repeatability_dropout", 0.15))
    if keep_prob < 1.0:
        mask = (torch.rand_like(images) < keep_prob).to(images.dtype)
        images = images * mask / max(keep_prob, EPS)

    max_shift = int(getattr(args, "score_repeatability_bin_shift", 1))
    if max_shift > 0:
        shift = int(torch.randint(-max_shift, max_shift + 1, size=(1,), device=images.device).item())
        if shift != 0:
            images = torch.roll(images, shifts=shift, dims=2)

    gain = 0.9 + 0.2 * torch.rand(images.shape[0], images.shape[1], 1, 1, 1, device=images.device, dtype=images.dtype)
    return images * gain


def _selected_centers_and_structure(
    scorer_context: ScorerContext | None,
) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
    if scorer_context is None or scorer_context.patches is None or scorer_context.ix is None:
        return None, None, None
    struct = _selected_patch_structure(scorer_context.images, scorer_context.patches, scorer_context.ix)
    p = scorer_context.patches.shape[-1]
    centers = scorer_context.patches[0, :, :2, p // 2, p // 2].detach().float()
    return centers, struct["anisotropy"], struct["orientation"]


def _diversity_penalty(scorer_context: ScorerContext | None, args) -> tuple[torch.Tensor | None, float]:
    if scorer_context is None or scorer_context.patches is None or scorer_context.ix is None:
        return None, float("nan")

    centers, anisotropy, orientation = _selected_centers_and_structure(scorer_context)
    if centers is None or anisotropy is None or orientation is None:
        return None, float("nan")

    ix = scorer_context.ix.long()
    feat_h = int(scorer_context.score_logits_map.shape[-2]) if scorer_context.score_logits_map is not None else 1
    feat_w = int(scorer_context.score_logits_map.shape[-1]) if scorer_context.score_logits_map is not None else 1
    diag = math.sqrt(float(feat_h * feat_h + feat_w * feat_w))
    if diag <= 0.0:
        return None, float("nan")

    min_sep = float(getattr(args, "score_diversity_min_separation", 0.06))
    losses = []

    for frame_idx in range(int(ix.max().item()) + 1 if ix.numel() > 0 else 0):
        mask = ix == frame_idx
        frame_centers = centers[mask]
        if frame_centers.shape[0] < 2:
            continue
        dmat = torch.cdist(frame_centers, frame_centers) / diag
        dmat.fill_diagonal_(float("inf"))
        nn = dmat.min(dim=-1).values

        frame_anisotropy = anisotropy[mask].detach()
        frame_orientation = orientation[mask]
        angle_diffs = torch.abs(frame_orientation[:, None] - frame_orientation[None, :])
        angle_diffs = torch.minimum(angle_diffs, math.pi - angle_diffs.clamp(max=math.pi))
        angle_diffs.fill_diagonal_(math.pi / 2.0)
        orient_sim = torch.cos(angle_diffs).abs().amax(dim=-1)

        losses.append((F.relu(min_sep - nn) * (0.5 + 0.5 * frame_anisotropy) * orient_sim).mean())

    if not losses:
        return None, float("nan")

    penalty = torch.stack(losses).mean()
    return penalty, float(penalty.detach().item())


def _multi_motion_spread(
    coords_gt: torch.Tensor,
    stats: PatchUtilityStats,
    scorer_context: ScorerContext | None,
) -> torch.Tensor | None:
    if scorer_context is None or scorer_context.patches is None:
        return None
    p = scorer_context.patches.shape[-1]
    patch_centers = scorer_context.patches[0, :, :2, p // 2, p // 2].detach().float()
    if stats.patch_indices_valid.numel() == 0:
        return torch.zeros_like(stats.patch_utility)

    source_centers_valid = patch_centers[stats.patch_indices_valid]
    motion_valid = coords_gt.reshape(-1, 2)[stats.valid_mask] - source_centers_valid
    motion_norm = motion_valid.norm(dim=-1, keepdim=True).clamp_min(EPS)
    unit_motion = motion_valid / motion_norm

    num_patches = stats.patch_utility.numel()
    sum_unit = torch.zeros(num_patches, 2, device=coords_gt.device, dtype=torch.float32)
    counts = torch.zeros(num_patches, device=coords_gt.device, dtype=torch.float32)
    sum_unit.index_add_(0, stats.patch_indices_valid, unit_motion)
    counts.index_add_(
        0,
        stats.patch_indices_valid,
        torch.ones_like(stats.patch_indices_valid, dtype=torch.float32),
    )

    mean_unit = sum_unit / counts[:, None].clamp_min(1.0)
    spread = 1.0 - mean_unit.norm(dim=-1).clamp(max=1.0)
    spread = torch.where(counts > 1, spread, torch.zeros_like(spread))
    return spread


def scorer_objective_baseline(
    coords_est: torch.Tensor,
    coords_gt: torch.Tensor,
    valid: torch.Tensor,
    weight: torch.Tensor,
    scores: torch.Tensor,
    kk_close: torch.Tensor,
    args,
    scorer_context: ScorerContext | None = None,
) -> ScorerObjectiveResult:
    device = coords_est.device
    stats = _compute_patch_utility_stats(
        coords_est, coords_gt, valid, weight, scores, kk_close,
        tau_e=max(float(args.score_rank_tau_e), EPS),
        scorer_context=scorer_context,
    )
    scores_clamped = scores.clamp_min(EPS)

    if stats.patch_indices_valid.numel() > 0:
        patch_scores_valid = stats.patch_scores[stats.patch_indices_valid]
        regression = (stats.edge_conf_valid * patch_scores_valid * stats.edge_error_valid).mean()
    else:
        regression = _safe_zero(device)

    log_barrier = (-scores_clamped.log()).mean()
    loss = regression + log_barrier
    metrics = {
        "scorer/regression_train": float(regression.detach().item()),
        "scorer/logbarrier_train": float(log_barrier.detach().item()),
        "scorer/utility_mean_train": float(stats.patch_utility.mean().detach().item()),
        "scorer/valid_patch_fraction_train": float(
            (stats.patch_valid_counts > 0).float().mean().detach().item()
        ),
    }
    return ScorerObjectiveResult(loss=loss, metrics=metrics)


def scorer_objective_rank_only(
    coords_est: torch.Tensor,
    coords_gt: torch.Tensor,
    valid: torch.Tensor,
    weight: torch.Tensor,
    scores: torch.Tensor,
    kk_close: torch.Tensor,
    args,
    scorer_context: ScorerContext | None = None,
) -> ScorerObjectiveResult:
    stats = _compute_patch_utility_stats(
        coords_est, coords_gt, valid, weight, scores, kk_close,
        tau_e=max(float(args.score_rank_tau_e), EPS),
        scorer_context=scorer_context,
    )

    rank_loss, rank_metrics = _framewise_rank_loss(
        stats.patch_logits,
        stats.patch_utility.detach(),
        num_frames=scores.shape[0],
        patches_per_image=scores.shape[1],
        margin=float(args.score_rank_margin),
        top_q=float(args.score_rank_top_quantile),
        bottom_q=float(args.score_rank_bottom_quantile),
    )

    metrics = {
        **rank_metrics,
        "scorer/utility_mean_train": float(stats.patch_utility.mean().detach().item()),
        "scorer/valid_patch_fraction_train": float(
            (stats.patch_valid_counts > 0).float().mean().detach().item()
        ),
    }
    return ScorerObjectiveResult(loss=rank_loss, metrics=metrics)


def scorer_objective_corr_isotropy(
    coords_est: torch.Tensor,
    coords_gt: torch.Tensor,
    valid: torch.Tensor,
    weight: torch.Tensor,
    scores: torch.Tensor,
    kk_close: torch.Tensor,
    args,
    scorer_context: ScorerContext | None = None,
) -> ScorerObjectiveResult:
    scorer_context = _to_context(scorer_context)
    stats = _compute_patch_utility_stats(
        coords_est, coords_gt, valid, weight, scores, kk_close,
        tau_e=max(float(args.score_rank_tau_e), EPS),
        scorer_context=scorer_context,
    )
    if scorer_context is None or scorer_context.images is None or scorer_context.patches is None or scorer_context.ix is None:
        return scorer_objective_rank_only(coords_est, coords_gt, valid, weight, scores, kk_close, args, scorer_context)

    struct = _selected_patch_structure(scorer_context.images, scorer_context.patches, scorer_context.ix)
    isotropy = struct["isotropy"].detach()
    target = stats.patch_utility.detach() * (0.25 + 0.75 * isotropy)
    patch_logits = _selected_patch_logits(stats, scorer_context)
    loss, rank_metrics = _framewise_rank_loss(
        patch_logits,
        target,
        num_frames=scores.shape[0],
        patches_per_image=scores.shape[1],
        margin=float(args.score_rank_margin),
        top_q=float(args.score_rank_top_quantile),
        bottom_q=float(args.score_rank_bottom_quantile),
    )
    metrics = {
        **rank_metrics,
        "scorer/utility_mean_train": float(target.mean().item()),
        "scorer/isotropy_mean_train": float(isotropy.mean().item()),
        "scorer/valid_patch_fraction_train": float(stats.patch_valid_fraction.mean().detach().item()),
    }
    return ScorerObjectiveResult(loss=loss, metrics=metrics)


def scorer_objective_short_horizon_survivability(
    coords_est: torch.Tensor,
    coords_gt: torch.Tensor,
    valid: torch.Tensor,
    weight: torch.Tensor,
    scores: torch.Tensor,
    kk_close: torch.Tensor,
    args,
    scorer_context: ScorerContext | None = None,
) -> ScorerObjectiveResult:
    stats = _compute_patch_utility_stats(
        coords_est, coords_gt, valid, weight, scores, kk_close,
        tau_e=max(float(args.score_rank_tau_e), EPS),
        scorer_context=_to_context(scorer_context),
    )
    survival = stats.patch_valid_fraction.detach()
    accuracy = torch.exp(-stats.patch_error_mean.detach() / max(float(args.score_rank_tau_e), EPS))
    target = survival * accuracy * stats.patch_conf_mean.detach()
    loss, rank_metrics = _framewise_rank_loss(
        stats.patch_logits,
        target,
        num_frames=scores.shape[0],
        patches_per_image=scores.shape[1],
        margin=float(args.score_rank_margin),
        top_q=float(args.score_rank_top_quantile),
        bottom_q=float(args.score_rank_bottom_quantile),
    )
    metrics = {
        **rank_metrics,
        "scorer/utility_mean_train": float(target.mean().item()),
        "scorer/survival_mean_train": float(survival.mean().item()),
        "scorer/valid_patch_fraction_train": float(stats.patch_valid_fraction.mean().detach().item()),
    }
    return ScorerObjectiveResult(loss=loss, metrics=metrics)


def scorer_objective_repeatability(
    coords_est: torch.Tensor,
    coords_gt: torch.Tensor,
    valid: torch.Tensor,
    weight: torch.Tensor,
    scores: torch.Tensor,
    kk_close: torch.Tensor,
    args,
    scorer_context: ScorerContext | None = None,
) -> ScorerObjectiveResult:
    scorer_context = _to_context(scorer_context)
    base = scorer_objective_rank_only(coords_est, coords_gt, valid, weight, scores, kk_close, args, scorer_context)
    if (
        scorer_context is None
        or scorer_context.images is None
        or scorer_context.selected_score_coords is None
        or scorer_context.scorer_module is None
    ):
        return base

    perturbed_images = _apply_event_perturbation(scorer_context.images, args)
    with torch.amp.autocast('cuda', enabled=getattr(args, "amp", False), dtype=torch.bfloat16):
        perturbed_outputs = scorer_context.scorer_module.forward_all(perturbed_images)
    perturbed_logits = _sample_selected_map_values(
        perturbed_outputs["score"].float(),
        scorer_context.selected_score_coords,
    ).reshape(-1)

    stats = _compute_patch_utility_stats(
        coords_est, coords_gt, valid, weight, scores, kk_close,
        tau_e=max(float(args.score_rank_tau_e), EPS),
        scorer_context=scorer_context,
    )
    patch_logits = _selected_patch_logits(stats, scorer_context)
    repeat_loss = F.smooth_l1_loss(torch.sigmoid(perturbed_logits), torch.sigmoid(patch_logits))
    total = base.loss + float(getattr(args, "score_repeatability_weight", 1.0)) * repeat_loss
    metrics = dict(base.metrics)
    metrics["scorer/repeatability_train"] = float(repeat_loss.detach().item())
    return ScorerObjectiveResult(loss=total, metrics=metrics)


def scorer_objective_diversity(
    coords_est: torch.Tensor,
    coords_gt: torch.Tensor,
    valid: torch.Tensor,
    weight: torch.Tensor,
    scores: torch.Tensor,
    kk_close: torch.Tensor,
    args,
    scorer_context: ScorerContext | None = None,
) -> ScorerObjectiveResult:
    scorer_context = _to_context(scorer_context)
    base = scorer_objective_rank_only(coords_est, coords_gt, valid, weight, scores, kk_close, args, scorer_context)
    penalty, penalty_value = _diversity_penalty(scorer_context, args)
    if penalty is None:
        return base
    total = base.loss + float(getattr(args, "score_diversity_weight", 0.25)) * penalty
    metrics = dict(base.metrics)
    metrics["scorer/diversity_penalty_train"] = penalty_value
    return ScorerObjectiveResult(loss=total, metrics=metrics)


def scorer_objective_multi_motion_consistency(
    coords_est: torch.Tensor,
    coords_gt: torch.Tensor,
    valid: torch.Tensor,
    weight: torch.Tensor,
    scores: torch.Tensor,
    kk_close: torch.Tensor,
    args,
    scorer_context: ScorerContext | None = None,
) -> ScorerObjectiveResult:
    scorer_context = _to_context(scorer_context)
    stats = _compute_patch_utility_stats(
        coords_est, coords_gt, valid, weight, scores, kk_close,
        tau_e=max(float(args.score_rank_tau_e), EPS),
        scorer_context=scorer_context,
    )
    spread = _multi_motion_spread(coords_gt, stats, scorer_context)
    if spread is None:
        return scorer_objective_rank_only(coords_est, coords_gt, valid, weight, scores, kk_close, args, scorer_context)

    target = stats.patch_utility.detach() * (0.25 + 0.75 * spread.detach()) * (0.5 + 0.5 * stats.patch_valid_fraction.detach())
    patch_logits = _selected_patch_logits(stats, scorer_context)
    loss, rank_metrics = _framewise_rank_loss(
        patch_logits,
        target,
        num_frames=scores.shape[0],
        patches_per_image=scores.shape[1],
        margin=float(args.score_rank_margin),
        top_q=float(args.score_rank_top_quantile),
        bottom_q=float(args.score_rank_bottom_quantile),
    )
    metrics = {
        **rank_metrics,
        "scorer/utility_mean_train": float(target.mean().item()),
        "scorer/motion_spread_train": float(spread.mean().detach().item()),
        "scorer/valid_patch_fraction_train": float(stats.patch_valid_fraction.mean().detach().item()),
    }
    return ScorerObjectiveResult(loss=loss, metrics=metrics)


def scorer_objective_event_teacher(
    coords_est: torch.Tensor,
    coords_gt: torch.Tensor,
    valid: torch.Tensor,
    weight: torch.Tensor,
    scores: torch.Tensor,
    kk_close: torch.Tensor,
    args,
    scorer_context: ScorerContext | None = None,
) -> ScorerObjectiveResult:
    scorer_context = _to_context(scorer_context)
    stats = _compute_patch_utility_stats(
        coords_est, coords_gt, valid, weight, scores, kk_close,
        tau_e=max(float(args.score_rank_tau_e), EPS),
        scorer_context=scorer_context,
    )
    teacher = _selected_event_teacher(scorer_context)
    if teacher is None:
        return scorer_objective_rank_only(coords_est, coords_gt, valid, weight, scores, kk_close, args, scorer_context)

    patch_logits = _selected_patch_logits(stats, scorer_context)
    target = stats.patch_utility.detach() * (0.5 + 0.5 * teacher.detach())
    rank_loss, rank_metrics = _framewise_rank_loss(
        patch_logits,
        target,
        num_frames=scores.shape[0],
        patches_per_image=scores.shape[1],
        margin=float(args.score_rank_margin),
        top_q=float(args.score_rank_top_quantile),
        bottom_q=float(args.score_rank_bottom_quantile),
    )
    teacher_loss = F.smooth_l1_loss(torch.sigmoid(patch_logits), teacher.detach())
    total = rank_loss + float(getattr(args, "score_teacher_weight", 0.5)) * teacher_loss
    metrics = {
        **rank_metrics,
        "scorer/utility_mean_train": float(target.mean().item()),
        "scorer/teacher_alignment_train": float(teacher_loss.detach().item()),
        "scorer/valid_patch_fraction_train": float(stats.patch_valid_fraction.mean().detach().item()),
    }
    return ScorerObjectiveResult(loss=total, metrics=metrics)


def _replay_objective(
    coords_est: torch.Tensor,
    coords_gt: torch.Tensor,
    valid: torch.Tensor,
    weight: torch.Tensor,
    scores: torch.Tensor,
    kk_close: torch.Tensor,
    args,
    scorer_context: ScorerContext | None,
    *,
    use_fb: bool,
    use_stability: bool,
) -> ScorerObjectiveResult:
    scorer_context = _to_context(scorer_context)
    stats = _compute_patch_utility_stats(
        coords_est, coords_gt, valid, weight, scores, kk_close,
        tau_e=max(float(args.score_rank_tau_e), EPS),
        scorer_context=scorer_context,
    )
    num_patches = int(scores.numel())
    device = coords_est.device

    replay_valid = _get_replay_signal(scorer_context, "replay_valid", num_patches, device)
    fb_error = _get_replay_signal(scorer_context, "fb_error_px", num_patches, device)
    stability_error = _get_replay_signal(scorer_context, "stability_error_px", num_patches, device)
    if replay_valid is None or (use_fb and fb_error is None) or (use_stability and stability_error is None):
        return scorer_objective_rank_only(coords_est, coords_gt, valid, weight, scores, kk_close, args, scorer_context)

    patch_logits = _selected_patch_logits(stats, scorer_context)
    target = stats.patch_utility.detach() * replay_valid.detach()

    fb_score = torch.ones_like(target)
    rep_score = torch.ones_like(target)
    if use_fb:
        tau_fb = max(float(getattr(args, "score_cycle_tau_fb", 1.5)), EPS)
        fb_score = torch.zeros_like(target)
        finite_fb = torch.isfinite(fb_error)
        fb_score[finite_fb] = torch.exp(-fb_error[finite_fb].detach() / tau_fb)
        target = target * fb_score
    if use_stability:
        tau_rep = max(float(getattr(args, "score_cycle_tau_rep", 1.0)), EPS)
        rep_score = torch.zeros_like(target)
        finite_rep = torch.isfinite(stability_error)
        rep_score[finite_rep] = torch.exp(-stability_error[finite_rep].detach() / tau_rep)
        target = target * rep_score

    rank_loss, rank_metrics = _framewise_rank_loss(
        patch_logits,
        target,
        num_frames=scores.shape[0],
        patches_per_image=scores.shape[1],
        margin=float(args.score_rank_margin),
        top_q=float(args.score_rank_top_quantile),
        bottom_q=float(args.score_rank_bottom_quantile),
    )

    total = rank_loss
    metrics = {
        **rank_metrics,
        "scorer/utility_mean_train": float(target.mean().item()),
        "scorer/valid_patch_fraction_train": float(stats.patch_valid_fraction.mean().detach().item()),
        "scorer/replay_valid_fraction_train": float(replay_valid.mean().detach().item()),
    }
    if use_fb:
        finite_fb = torch.isfinite(fb_error)
        metrics["scorer/fb_cycle_error_px_train"] = float(
            fb_error[finite_fb].mean().item() if finite_fb.any() else float("nan")
        )
    if use_stability:
        finite_rep = torch.isfinite(stability_error)
        metrics["scorer/replay_stability_px_train"] = float(
            stability_error[finite_rep].mean().item() if finite_rep.any() else float("nan")
        )

    teacher = _selected_event_teacher(scorer_context)
    teacher_weight = _teacher_warmup_weight(args, scorer_context)
    if teacher is not None and teacher_weight > 0.0:
        teacher_loss = F.smooth_l1_loss(torch.sigmoid(patch_logits), teacher.detach())
        total = total + teacher_weight * teacher_loss
        metrics["scorer/teacher_alignment_train"] = float(teacher_loss.detach().item())
        metrics["scorer/teacher_weight_train"] = float(teacher_weight)

    return ScorerObjectiveResult(loss=total, metrics=metrics)


def scorer_objective_forward_backward_cycle(
    coords_est: torch.Tensor,
    coords_gt: torch.Tensor,
    valid: torch.Tensor,
    weight: torch.Tensor,
    scores: torch.Tensor,
    kk_close: torch.Tensor,
    args,
    scorer_context: ScorerContext | None = None,
) -> ScorerObjectiveResult:
    return _replay_objective(
        coords_est, coords_gt, valid, weight, scores, kk_close, args, scorer_context,
        use_fb=True,
        use_stability=False,
    )


def scorer_objective_replay_stability(
    coords_est: torch.Tensor,
    coords_gt: torch.Tensor,
    valid: torch.Tensor,
    weight: torch.Tensor,
    scores: torch.Tensor,
    kk_close: torch.Tensor,
    args,
    scorer_context: ScorerContext | None = None,
) -> ScorerObjectiveResult:
    return _replay_objective(
        coords_est, coords_gt, valid, weight, scores, kk_close, args, scorer_context,
        use_fb=False,
        use_stability=True,
    )


def scorer_objective_cycle_stability(
    coords_est: torch.Tensor,
    coords_gt: torch.Tensor,
    valid: torch.Tensor,
    weight: torch.Tensor,
    scores: torch.Tensor,
    kk_close: torch.Tensor,
    args,
    scorer_context: ScorerContext | None = None,
) -> ScorerObjectiveResult:
    return _replay_objective(
        coords_est, coords_gt, valid, weight, scores, kk_close, args, scorer_context,
        use_fb=True,
        use_stability=True,
    )


def scorer_objective_info_gain_head(
    coords_est: torch.Tensor,
    coords_gt: torch.Tensor,
    valid: torch.Tensor,
    weight: torch.Tensor,
    scores: torch.Tensor,
    kk_close: torch.Tensor,
    args,
    scorer_context: ScorerContext | None = None,
) -> ScorerObjectiveResult:
    scorer_context = _to_context(scorer_context)
    stats = _compute_patch_utility_stats(
        coords_est, coords_gt, valid, weight, scores, kk_close,
        tau_e=max(float(args.score_rank_tau_e), EPS),
        scorer_context=scorer_context,
    )
    aux_logits = _get_selected_aux_logits(scorer_context, "info_gain")
    if aux_logits is None:
        return scorer_objective_rank_only(coords_est, coords_gt, valid, weight, scores, kk_close, args, scorer_context)

    info_target = stats.patch_valid_fraction.detach() * torch.exp(
        -stats.patch_error_mean.detach() / max(float(args.score_rank_tau_e), EPS)
    ) * stats.patch_conf_mean.detach()
    info_pred = torch.sigmoid(aux_logits)
    head_loss = F.smooth_l1_loss(info_pred, info_target)

    patch_logits = _selected_patch_logits(stats, scorer_context)
    rank_loss, rank_metrics = _framewise_rank_loss(
        patch_logits,
        info_pred.detach(),
        num_frames=scores.shape[0],
        patches_per_image=scores.shape[1],
        margin=float(args.score_rank_margin),
        top_q=float(args.score_rank_top_quantile),
        bottom_q=float(args.score_rank_bottom_quantile),
    )
    total = rank_loss + float(getattr(args, "score_aux_head_weight", 1.0)) * head_loss
    metrics = {
        **rank_metrics,
        "scorer/utility_mean_train": float(info_target.mean().item()),
        "scorer/info_head_train": float(head_loss.detach().item()),
        "scorer/valid_patch_fraction_train": float(stats.patch_valid_fraction.mean().detach().item()),
    }
    return ScorerObjectiveResult(loss=total, metrics=metrics)


def scorer_objective_conditioning_head(
    coords_est: torch.Tensor,
    coords_gt: torch.Tensor,
    valid: torch.Tensor,
    weight: torch.Tensor,
    scores: torch.Tensor,
    kk_close: torch.Tensor,
    args,
    scorer_context: ScorerContext | None = None,
) -> ScorerObjectiveResult:
    scorer_context = _to_context(scorer_context)
    stats = _compute_patch_utility_stats(
        coords_est, coords_gt, valid, weight, scores, kk_close,
        tau_e=max(float(args.score_rank_tau_e), EPS),
        scorer_context=scorer_context,
    )
    aux_logits = _get_selected_aux_logits(scorer_context, "conditioning")
    if aux_logits is None or scorer_context is None or scorer_context.images is None or scorer_context.patches is None or scorer_context.ix is None:
        return scorer_objective_rank_only(coords_est, coords_gt, valid, weight, scores, kk_close, args, scorer_context)

    struct = _selected_patch_structure(scorer_context.images, scorer_context.patches, scorer_context.ix)
    conditioning_target = struct["isotropy"].detach()
    cond_pred = torch.sigmoid(aux_logits)
    head_loss = F.smooth_l1_loss(cond_pred, conditioning_target)

    patch_logits = _selected_patch_logits(stats, scorer_context)
    rank_target = stats.patch_utility.detach() * (0.25 + 0.75 * cond_pred.detach())
    rank_loss, rank_metrics = _framewise_rank_loss(
        patch_logits,
        rank_target,
        num_frames=scores.shape[0],
        patches_per_image=scores.shape[1],
        margin=float(args.score_rank_margin),
        top_q=float(args.score_rank_top_quantile),
        bottom_q=float(args.score_rank_bottom_quantile),
    )
    total = rank_loss + float(getattr(args, "score_aux_head_weight", 1.0)) * head_loss
    metrics = {
        **rank_metrics,
        "scorer/utility_mean_train": float(rank_target.mean().item()),
        "scorer/conditioning_head_train": float(head_loss.detach().item()),
        "scorer/isotropy_mean_train": float(conditioning_target.mean().item()),
        "scorer/valid_patch_fraction_train": float(stats.patch_valid_fraction.mean().detach().item()),
    }
    return ScorerObjectiveResult(loss=total, metrics=metrics)


_OBJECTIVE_REGISTRY = {
    "baseline": scorer_objective_baseline,
    "rank_only": scorer_objective_rank_only,
    "corr_isotropy": scorer_objective_corr_isotropy,
    "short_horizon_survivability": scorer_objective_short_horizon_survivability,
    "repeatability": scorer_objective_repeatability,
    "diversity": scorer_objective_diversity,
    "multi_motion_consistency": scorer_objective_multi_motion_consistency,
    "event_teacher": scorer_objective_event_teacher,
    "info_gain_head": scorer_objective_info_gain_head,
    "conditioning_head": scorer_objective_conditioning_head,
    "forward_backward_cycle": scorer_objective_forward_backward_cycle,
    "replay_stability": scorer_objective_replay_stability,
    "cycle_stability": scorer_objective_cycle_stability,
}


def scorer_objective_names() -> list[str]:
    return list(_OBJECTIVE_REGISTRY.keys())


def implemented_scorer_objective_names() -> list[str]:
    return list(_OBJECTIVE_REGISTRY.keys())


def replay_scorer_objective_names() -> set[str]:
    return {"forward_backward_cycle", "replay_stability", "cycle_stability"}


def compute_scorer_objective(
    objective_name: str,
    coords_est: torch.Tensor,
    coords_gt: torch.Tensor,
    valid: torch.Tensor,
    weight: torch.Tensor,
    scores: torch.Tensor | None,
    kk_close: torch.Tensor | None,
    args,
    scorer_context: ScorerContext | dict | None = None,
) -> ScorerObjectiveResult:
    device = coords_est.device
    if scores is None or kk_close is None:
        return ScorerObjectiveResult(loss=_safe_zero(device), metrics={})

    if objective_name not in _OBJECTIVE_REGISTRY:
        raise KeyError(f"Unknown scorer objective '{objective_name}'")

    return _OBJECTIVE_REGISTRY[objective_name](
        coords_est=coords_est,
        coords_gt=coords_gt,
        valid=valid,
        weight=weight,
        scores=scores,
        kk_close=kk_close,
        args=args,
        scorer_context=_to_context(scorer_context),
    )


def compute_patch_survival_metrics(
    valid: torch.Tensor,
    kk_close: torch.Tensor,
    num_patches: int,
) -> dict[str, float]:
    device = valid.device
    b = valid.shape[0]
    patch_indices_all = kk_close[None].expand(b, -1).reshape(-1)
    valid_flat = (valid > 0.5).reshape(-1).float()

    patch_total_counts = torch.zeros(num_patches, device=device, dtype=torch.float32)
    patch_valid_counts = torch.zeros(num_patches, device=device, dtype=torch.float32)
    if patch_indices_all.numel() > 0:
        patch_total_counts.index_add_(
            0,
            patch_indices_all,
            torch.ones_like(patch_indices_all, dtype=torch.float32),
        )
        patch_valid_counts.index_add_(0, patch_indices_all, valid_flat)

    patch_has_edge = patch_total_counts > 0
    if patch_has_edge.any():
        patch_survival_rate = float((patch_valid_counts[patch_has_edge] > 0).float().mean().item())
    else:
        patch_survival_rate = float("nan")

    rejection_rate = float((1.0 - valid_flat.mean()).item()) if valid_flat.numel() > 0 else float("nan")
    mean_patch_valid_fraction = float(
        (patch_valid_counts[patch_has_edge] / patch_total_counts[patch_has_edge].clamp_min(1.0)).mean().item()
    ) if patch_has_edge.any() else float("nan")

    return {
        "patch_survival_rate": patch_survival_rate,
        "edge_rejection_rate": rejection_rate,
        "mean_patch_valid_fraction": mean_patch_valid_fraction,
    }


def compute_score_map_dynamic_range(score_maps: torch.Tensor) -> dict[str, float]:
    score_maps = score_maps.detach().float()
    flat = score_maps.flatten(start_dim=-2)
    dyn = flat.max(dim=-1).values - flat.min(dim=-1).values
    return {
        "score_dynamic_range": float(dyn.mean().item()),
        "score_dynamic_range_min": float(dyn.min().item()),
        "score_dynamic_range_max": float(dyn.max().item()),
    }


def compute_replay_metric_summary(
    fb_error_px: torch.Tensor | None,
    stability_error_px: torch.Tensor | None,
    replay_valid: torch.Tensor | None,
) -> dict[str, float]:
    metrics = {
        "fb_cycle_error_px": float("nan"),
        "replay_stability_px": float("nan"),
        "replay_valid_fraction": float("nan"),
    }
    if replay_valid is not None and replay_valid.numel() > 0:
        metrics["replay_valid_fraction"] = float(replay_valid.detach().float().mean().item())
    if fb_error_px is not None:
        fb_error_px = fb_error_px.detach().float()
        finite = torch.isfinite(fb_error_px)
        if replay_valid is not None and replay_valid.numel() == fb_error_px.numel():
            finite = finite & replay_valid.detach().bool()
        if finite.any():
            metrics["fb_cycle_error_px"] = float(fb_error_px[finite].mean().item())
    if stability_error_px is not None:
        stability_error_px = stability_error_px.detach().float()
        finite = torch.isfinite(stability_error_px)
        if replay_valid is not None and replay_valid.numel() == stability_error_px.numel():
            finite = finite & replay_valid.detach().bool()
        if finite.any():
            metrics["replay_stability_px"] = float(stability_error_px[finite].mean().item())
    return metrics


def extract_selected_centers_by_frame(
    patches: torch.Tensor,
    ix: torch.Tensor,
) -> list[torch.Tensor]:
    p = patches.shape[-1]
    centers = patches[0, :, :2, p // 2, p // 2].detach().float()
    num_frames = int(ix.max().item()) + 1 if ix.numel() > 0 else 0
    return [centers[ix == frame_idx] for frame_idx in range(num_frames)]


def compute_selection_spread(
    patches: torch.Tensor,
    ix: torch.Tensor,
    feat_h: int,
    feat_w: int,
) -> float:
    centers_by_frame = extract_selected_centers_by_frame(patches, ix)
    diag = math.sqrt(float(feat_h * feat_h + feat_w * feat_w))
    if diag <= 0.0:
        return float("nan")

    spread_vals = []
    for centers in centers_by_frame:
        if centers.shape[0] < 2:
            continue
        dmat = torch.cdist(centers, centers)
        dmat.fill_diagonal_(float("inf"))
        nn = dmat.min(dim=-1).values
        spread_vals.append(float((nn.mean() / diag).item()))

    return float(sum(spread_vals) / len(spread_vals)) if spread_vals else float("nan")


def compute_selected_patch_tensor_metrics(
    images: torch.Tensor,
    patches: torch.Tensor,
    ix: torch.Tensor,
) -> dict[str, float]:
    struct = _selected_patch_structure(images, patches, ix)
    isotropy = struct["isotropy"]
    anisotropy = struct["anisotropy"]
    if isotropy.numel() == 0:
        return {
            "selected_patch_isotropy": float("nan"),
            "selected_patch_anisotropy": float("nan"),
        }

    return {
        "selected_patch_isotropy": float(isotropy.mean().item()),
        "selected_patch_anisotropy": float(anisotropy.mean().item()),
    }


def summarize_score_behavior(
    score_dynamic_range: float,
    selected_patch_isotropy: float,
    selection_diversity: float,
) -> str:
    notes = []
    if math.isnan(score_dynamic_range):
        notes.append("missing_score_maps")
    elif score_dynamic_range < 1e-3:
        notes.append("flat_scores")
    elif score_dynamic_range < 1e-2:
        notes.append("low_contrast_scores")
    else:
        notes.append("high_contrast_scores")

    if not math.isnan(selected_patch_isotropy):
        if selected_patch_isotropy < 0.15:
            notes.append("edge_like")
        elif selected_patch_isotropy > 0.35:
            notes.append("corner_like")

    if not math.isnan(selection_diversity):
        if selection_diversity < 0.03:
            notes.append("clustered")
        elif selection_diversity > 0.08:
            notes.append("spread")

    return ",".join(notes)


def append_experiment_table_row(table_path: Path, row: dict[str, object]) -> None:
    table_path.parent.mkdir(parents=True, exist_ok=True)
    existing_rows = []
    if table_path.exists():
        with table_path.open("r", encoding="utf-8", newline="") as f:
            existing_rows = list(csv.DictReader(f))

    baseline_row = None
    if row["score_objective"] == "baseline":
        row["delta_epe_feature_px"] = 0.0
        row["delta_patch_survival_rate"] = 0.0
        row["delta_selected_patch_isotropy"] = 0.0
        row["delta_selection_diversity"] = 0.0
        row["delta_fb_cycle_error_px"] = 0.0
        row["delta_replay_stability_px"] = 0.0
        row["delta_replay_valid_fraction"] = 0.0
    else:
        for prev in reversed(existing_rows):
            if prev.get("score_objective") == "baseline":
                baseline_row = prev
                break

        for delta_key, metric_key in (
            ("delta_epe_feature_px", "val_epe_feature_px"),
            ("delta_patch_survival_rate", "patch_survival_rate"),
            ("delta_selected_patch_isotropy", "selected_patch_isotropy"),
            ("delta_selection_diversity", "selection_diversity"),
            ("delta_fb_cycle_error_px", "fb_cycle_error_px"),
            ("delta_replay_stability_px", "replay_stability_px"),
            ("delta_replay_valid_fraction", "replay_valid_fraction"),
        ):
            if baseline_row is None:
                row[delta_key] = ""
                continue
            base_val = baseline_row.get(metric_key, "")
            curr_val = row.get(metric_key, "")
            if base_val in ("", None) or curr_val in ("", None):
                row[delta_key] = ""
            else:
                row[delta_key] = float(curr_val) - float(base_val)

    existing_rows.append({k: row.get(k, "") for k in row.keys()})
    headers = list(row.keys())
    with table_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(existing_rows)
