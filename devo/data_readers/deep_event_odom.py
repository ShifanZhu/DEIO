"""Dataset adapter for deep_event_odometry HDF5 sequences.

Wraps MultiKeyframeDataset and converts its outputs to the format
expected by the feature-metric GN training pipeline:
    images:     (B, N, K_sub, H, W)  — event voxel grids per keyframe
    disps:      (B, N, H/4, W/4)     — inverse depth (downsampled to feature-map res)
    poses:      (B, N, 7)            — SE3 as [tx, ty, tz, qx, qy, qz, qw]
    intrinsics: (B, 4)               — [fx, fy, cx, cy] in pixel units
"""

import sys
import os
import numpy as np
import torch
from torch.utils.data import Dataset
from scipy.spatial.transform import Rotation

_DEO_ROOT = os.path.expanduser('~/repos/deep_event_odometry')


def _import_deo():
    """Import deep_event_odometry modules, handling the 'utils' package name collision.

    DEIO also has a top-level 'utils/' package, which Python may cache in
    sys.modules['utils'] before this file runs.  We temporarily remove it,
    import the DEO modules, then restore DEIO's utils so the rest of DEIO
    is unaffected.
    """
    if _DEO_ROOT not in sys.path:
        sys.path.insert(0, _DEO_ROOT)

    # Save all cached 'utils.*' entries that belong to DEIO
    saved = {k: v for k, v in sys.modules.items()
             if k == 'utils' or k.startswith('utils.')}
    for k in saved:
        del sys.modules[k]

    try:
        from preprocess.dataset import EventOdometryDataset
        from utils.keyframe_dataset import MultiKeyframeDataset
    finally:
        # Restore DEIO's utils (remove DEO's utils.* that were just loaded)
        deo_utils_keys = [k for k in sys.modules
                          if k == 'utils' or k.startswith('utils.')]
        for k in deo_utils_keys:
            del sys.modules[k]
        sys.modules.update(saved)

    return EventOdometryDataset, MultiKeyframeDataset


def _log_norm_to_metric(depth_log: np.ndarray, depth_k: float, depth_max_m: float) -> np.ndarray:
    """Invert the log-normalisation applied by EventOdometryDataset.

    Forward:  d_log = log1p(d_metric * depth_k) / log1p(depth_max_m * depth_k)
    Inverse:  d_metric = expm1(d_log * log1p(depth_max_m * depth_k)) / depth_k
    Invalid pixels (depth_log == 0) are returned as 0.
    """
    scale = np.log1p(depth_max_m * depth_k)
    d_metric = np.expm1(depth_log * scale) / depth_k
    # zero-depth pixels are invalid — keep as zero
    d_metric[depth_log <= 0] = 0.0
    return d_metric.astype(np.float32)


def _mat4_to_7vec(T: np.ndarray) -> np.ndarray:
    """Convert a (4,4) SE3 matrix to a 7-vector [tx, ty, tz, qx, qy, qz, qw]."""
    t = T[:3, 3]
    q = Rotation.from_matrix(T[:3, :3]).as_quat()  # (qx, qy, qz, qw)
    return np.concatenate([t, q]).astype(np.float32)


class DeepEventOdomDataset(Dataset):
    """Wraps MultiKeyframeDataset and outputs tensors for feature-metric training.

    Args:
        data_path: Root directory of the HDF5 dataset.
        n_keyframes: Number of keyframe boundaries per sample window (default: 4).
        interval_frames: Frames between consecutive keyframes (default: 5).
        sub_frame_stride: Stride for sub-frames within each interval (default: 1).
            K_sub = interval_frames // sub_frame_stride (number of voxel bins).
        d_max: Maximum metric depth used during log-normalisation (metres).
        depth_k: Depth scale factor used during log-normalisation.
        res_factor: Feature-map downsampling factor (default: 4, matching BasicEncoder4Evs).
        **base_kwargs: Extra kwargs forwarded to EventOdometryDataset.
    """

    def __init__(
        self,
        data_path: str,
        n_keyframes: int = 4,
        interval_frames: int = 5,
        sub_frame_stride: int = 1,
        d_max: float = 100.0,
        depth_k: float = 1.0,
        res_factor: int = 4,
        **base_kwargs,
    ):
        self.d_max = d_max
        self.depth_k = depth_k
        self.res_factor = res_factor

        EventOdometryDataset, MultiKeyframeDataset = _import_deo()

        base_ds = EventOdometryDataset(data_path, **base_kwargs)
        self.mkf_ds = MultiKeyframeDataset(
            base_ds,
            n_keyframes=n_keyframes,
            interval_frames=interval_frames,
            sub_frame_stride=sub_frame_stride,
        )

        # Cache per-sequence depth_k and depth_max_m from base dataset metadata
        self._seq_meta = base_ds._seq_meta

    def __len__(self) -> int:
        return len(self.mkf_ds)

    def __getitem__(self, idx: int) -> dict:
        sample = self.mkf_ds[idx]
        intervals = sample['intervals']        # list of N-1 dicts
        T_abs = sample['T_event0_keyframes']   # (N, 4, 4) float32 ndarray

        N = len(intervals) + 1  # number of keyframes

        # --- poses: (N, 7) ---
        poses = np.stack([_mat4_to_7vec(T_abs[i]) for i in range(N)], axis=0)  # (N, 7)

        # --- images and depths per keyframe ---
        # images[i] = time_images of the interval ENDING at keyframe i+1
        # images[0] = "before" keyframe 0: we use the first interval's images for
        #             keyframe 0 as a proxy (colocated in time).
        # More precisely: keyframe i corresponds to the START of interval i and
        # the END of interval i-1.  We associate the event voxels of interval i
        # with keyframe i+1 (the target frame).  Keyframe 0 gets a zero voxel.

        K_sub = intervals[0]['time_images'].shape[0]  # number of voxel bins
        H, W = intervals[0]['time_images'].shape[1:3]

        images_list = []   # (N, K_sub, H, W)
        disps_list  = []   # (N, H, W)  — full resolution inverse depth

        seq_id = intervals[0]['seq_id']
        meta = self._seq_meta.get(seq_id, {})
        depth_k   = float(meta.get('depth_k',   self.depth_k))
        depth_max = float(meta.get('depth_max_m', self.d_max))

        # Keyframe 0: use first interval's time_images (events leading up to KF1)
        # and depth_start of interval 0 (depth at KF0)
        images_list.append(intervals[0]['time_images'])       # (K_sub, H, W)
        depth_log_0 = intervals[0]['depth_start']             # (H, W) log-norm
        d_metric_0  = _log_norm_to_metric(depth_log_0, depth_k, depth_max)
        valid_0 = d_metric_0 > 0
        disp_0  = np.where(valid_0, 1.0 / np.maximum(d_metric_0, 1e-3), 0.0).astype(np.float32)
        disps_list.append(disp_0)

        for i, interval in enumerate(intervals):
            # Keyframe i+1: event voxel for this interval, depth at interval end
            images_list.append(interval['time_images'])       # (K_sub, H, W)
            depth_log = interval['depth_end']                 # (H, W) log-norm
            d_metric  = _log_norm_to_metric(depth_log, depth_k, depth_max)
            valid     = d_metric > 0
            disp      = np.where(valid, 1.0 / np.maximum(d_metric, 1e-3), 0.0).astype(np.float32)
            disps_list.append(disp)

        images_np = np.stack(images_list, axis=0)  # (N, K_sub, H, W)
        disps_np  = np.stack(disps_list,  axis=0)  # (N, H, W)

        # --- intrinsics: (4,) absolute pixel units ---
        intr_norm = intervals[0]['intrinsics'].astype(np.float32)  # [fx/W, fy/H, cx/W, cy/H]
        intrinsics = np.array([
            intr_norm[0] * W,
            intr_norm[1] * H,
            intr_norm[2] * W,
            intr_norm[3] * H,
        ], dtype=np.float32)

        # Downsample disparity to feature-map resolution (1/res_factor)
        rf = self.res_factor
        # Simple uniform subsampling (matches DEIO: disps[:, :, 1::4, 1::4])
        disps_ds = disps_np[:, 1::rf, 1::rf]  # (N, H/rf, W/rf)
        # Intrinsics also need to be scaled to feature-map space for the GN solver
        intr_feat = intrinsics / rf             # (4,) at feature-map resolution

        return {
            'images':         torch.from_numpy(images_np),     # (N, K_sub, H, W)
            'disps':          torch.from_numpy(disps_ds),      # (N, H/rf, W/rf)
            'poses':          torch.from_numpy(poses),         # (N, 7)
            'intrinsics':     torch.from_numpy(intr_feat),     # (4,)
            'intrinsics_full': torch.from_numpy(intrinsics),   # (4,) full-res for reference
        }
