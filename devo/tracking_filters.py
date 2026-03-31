from __future__ import annotations

import cv2
import numpy as np
import torch
import torch.nn.functional as F


def _centers_2d(centers: torch.Tensor) -> torch.Tensor:
    """Normalize centers to shape (N, 2)."""
    if centers.ndim == 3:
        if centers.shape[0] != 1:
            raise ValueError(f"Expected batch size 1 centers, got {tuple(centers.shape)}")
        centers = centers[0]
    if centers.ndim != 2 or centers.shape[-1] != 2:
        raise ValueError(f"Expected centers with shape (N, 2) or (1, N, 2), got {tuple(centers.shape)}")
    return centers


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
    kernel = torch.ones((1, 1, kernel_size, kernel_size), device=activity.device, dtype=activity.dtype)

    support_mass = F.conv2d(activity[None, None], kernel, padding=radius_px)
    support_count = F.conv2d(active[None, None], kernel, padding=radius_px)
    return support_mass[0, 0], support_count[0, 0]


def supported_track_mask(
    centers: torch.Tensor,
    support_mass_map: torch.Tensor,
    support_count_map: torch.Tensor,
    min_event_support: float,
    min_event_pixels: float,
    feature_stride: int = 4,
) -> torch.Tensor:
    """Return a mask for centers whose image-space footprint has enough event support."""
    centers = _centers_2d(centers)
    if centers.shape[0] == 0:
        return torch.zeros((0,), dtype=torch.bool, device=centers.device)

    x = (feature_stride * (centers[:, 0] + 0.5)).round().long().clamp(0, support_mass_map.shape[1] - 1)
    y = (feature_stride * (centers[:, 1] + 0.5)).round().long().clamp(0, support_mass_map.shape[0] - 1)
    mass_ok = support_mass_map[y, x] >= float(min_event_support)
    count_ok = support_count_map[y, x] >= float(min_event_pixels)
    return mass_ok & count_ok


def centers_to_pixel_points(
    centers: torch.Tensor,
    feature_stride: float = 4.0,
    center_offset: float = 0.5,
) -> np.ndarray:
    """Convert feature-map centers to image-space patch centers."""
    centers = _centers_2d(centers)
    if centers.shape[0] == 0:
        return np.zeros((0, 2), dtype=np.float32)

    points = feature_stride * (centers.detach().float().cpu().numpy() + center_offset)
    return np.asarray(points, dtype=np.float32)


def normalize_pixel_points(points_px: np.ndarray, intrinsics: torch.Tensor) -> np.ndarray:
    """Normalize image points with pinhole intrinsics [fx, fy, cx, cy]."""
    if points_px.size == 0:
        return np.zeros((0, 2), dtype=np.float32)

    if isinstance(intrinsics, torch.Tensor):
        fx, fy, cx, cy = intrinsics.detach().float().cpu().tolist()
    else:
        fx, fy, cx, cy = np.asarray(intrinsics, dtype=np.float32).reshape(-1).tolist()

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
    feature_stride: float = 4.0,
) -> torch.Tensor:
    """Return a conservative inlier mask from normalized-point FM_RANSAC."""
    prev_centers = _centers_2d(prev_centers)
    curr_centers = _centers_2d(curr_centers)
    if prev_centers.shape[0] != curr_centers.shape[0]:
        raise ValueError("RANSAC center arrays must have the same number of tracks")

    num_tracks = curr_centers.shape[0]
    keep_all = torch.ones(num_tracks, dtype=torch.bool, device=curr_centers.device)

    required_points = max(8, int(min_points))
    required_inliers = max(8, int(min_inliers))
    if num_tracks < required_points:
        return keep_all

    pts0_px = centers_to_pixel_points(prev_centers, feature_stride=feature_stride)
    pts1_px = centers_to_pixel_points(curr_centers, feature_stride=feature_stride)
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
