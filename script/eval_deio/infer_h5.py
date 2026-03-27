#!/usr/bin/env python3
"""
DEIO inference on preprocessed H5 files from the deep_event_odometry pipeline.

Reports per-sequence and aggregate RPE / ATE using the same metrics as
deep_event_odometry/infer.py:
  - RPE: SO(3) log-map rotation error + Euclidean translation error
  - ATE: unaligned, predicted relative poses integrated from GT anchor

Usage:
    # CEAR (events need rectification from H5 raw coords)
    CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/home/s/repos/DEIO python script/eval_deio/infer_h5.py --network /home/s/repos/SDEVO/src/SDEVO/DEVO/DEVO.pth --config config/cear.yaml --h5 /media/s/2tb-2/deep_event_odometry/cear/indoor/ --output-dir results/deio_infer/

    # MVSEC (iterator auto-selected from directory name)
    CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/home/s/repos/DEIO python script/eval_deio/infer_h5.py --network /home/s/repos/SDEVO/src/SDEVO/DEVO/DEVO.pth --config config/mvsec.yaml --h5 /media/s/rell/deep_event_odometry/mvsec/ --height 260 --width 346 --output-dir results/deio_infer/mvsec/

    # VECtor preprocessed H5 (board-slow only; iterator auto-selected from directory name)
    CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/home/s/repos/DEIO python script/eval_deio/infer_h5.py --network /home/s/repos/SDEVO/src/SDEVO/DEVO/DEVO.pth --config config/vector.yaml --h5 /media/s/rell/deep_event_odometry/vector/ --height 480 --width 640 --output-dir results/deio_infer/vector/

# VECtor raw HDF5 (6 seqs: board_slow1 corner_slow1 desk_fast1 desk_normal1 robot_fast1 robot_normal1)
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/home/s/repos/DEIO python script/eval_deio/infer_h5.py --network /home/s/repos/SDEVO/src/SDEVO/DEVO/DEVO.pth --config config/vector.yaml --h5 /media/s/2tb-1/tro/vector_hdf5/ --gtdir /media/s/2tb-1/tro/VECtor --height 480 --width 640 --output-dir results/deio_infer/vector_raw/
"""

import argparse
import csv
import math
import re
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import torch
from scipy.spatial.transform import Rotation

from evo.tools.settings import SETTINGS
SETTINGS['plot_backend'] = 'Agg'

from devo.config import cfg as DEIO_CFG
from utils.load_utils import cear_h5_iterator, mvsec_h5_iterator, vector_raw_iterator, m3ed_raw_iterator
from utils.eval_utils import run_DEIO2


# ---------------------------------------------------------------------------
# Metric functions  (identical logic to deep_event_odometry/utils/metrics.py)
# ---------------------------------------------------------------------------

def _log_SO3(R: torch.Tensor) -> torch.Tensor:
    """SO(3) logarithm: (B,3,3) → (B,3) Rodrigues vector (float32)."""
    R = R.float()
    R = R.nan_to_num(nan=0.0, posinf=2.0, neginf=-2.0)
    trace = R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]
    cos_theta = ((trace - 1.0) / 2.0).clamp(-1.0 + 1e-6, 1.0 - 1e-6)
    theta = torch.acos(cos_theta)
    skew_vec = torch.stack([
        R[:, 2, 1] - R[:, 1, 2],
        R[:, 0, 2] - R[:, 2, 0],
        R[:, 1, 0] - R[:, 0, 1],
    ], dim=-1) * 0.5
    sin_theta = theta.sin()
    factor = torch.where(
        sin_theta.abs() > 1e-7,
        theta / sin_theta,
        torch.ones_like(theta),
    )
    return skew_vec * factor.unsqueeze(-1)


def rotation_error_deg(R_pred: torch.Tensor, R_gt: torch.Tensor) -> torch.Tensor:
    """(B,3,3),(B,3,3) → (B,) degrees, using SO(3) Riemannian distance."""
    R_err = R_gt.transpose(-1, -2) @ R_pred
    return _log_SO3(R_err).norm(dim=-1) * (180.0 / torch.pi)


def translation_error_m(t_pred: torch.Tensor, t_gt: torch.Tensor) -> torch.Tensor:
    """(B,3),(B,3) → (B,) Euclidean error in metres."""
    return (t_pred - t_gt).norm(dim=-1)


def compute_rpe_metrics(all_rot_errs: list, all_trans_errs: list) -> dict:
    rot_all   = torch.cat(all_rot_errs)
    trans_all = torch.cat(all_trans_errs)
    return {
        'rpe/rot_mean_deg':   rot_all.mean().item(),
        'rpe/rot_median_deg': rot_all.median().item(),
        'rpe/trans_mean_m':   trans_all.mean().item(),
        'rpe/trans_median_m': trans_all.median().item(),
    }


# ---------------------------------------------------------------------------
# File I/O helpers  (same interface as deep_event_odometry/infer.py)
# ---------------------------------------------------------------------------

def discover_h5_files(paths: list) -> list:
    """Resolve paths (files / dirs) to sorted (h5_path, dataset_name) tuples."""
    seen = {}
    for p in paths:
        p = Path(p).resolve()
        if p.is_file() and p.suffix in ('.h5', '.hdf5'):
            seen[p] = p.parent.name
        elif p.is_dir():
            found = sorted(list(p.glob('**/*.h5')) + list(p.glob('**/*.hdf5')))
            if not found:
                print(f'Warning: no .h5/.hdf5 files found under {p}')
            for f in found:
                seen[f] = p.name
        else:
            print(f'Warning: skipping non-existent or non-H5 path: {p}')
    return sorted(seen.items(), key=lambda x: x[0])


def _rotmat_to_quat(R):
    """(3,3) rotation matrix → (qx, qy, qz, qw)."""
    trace = R[0, 0] + R[1, 1] + R[2, 2]
    if trace > 0:
        s = 0.5 / math.sqrt(float(trace) + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * math.sqrt(1.0 + float(R[0, 0]) - float(R[1, 1]) - float(R[2, 2]))
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * math.sqrt(1.0 + float(R[1, 1]) - float(R[0, 0]) - float(R[2, 2]))
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * math.sqrt(1.0 + float(R[2, 2]) - float(R[0, 0]) - float(R[1, 1]))
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    return float(x), float(y), float(z), float(w)


def save_tum(traj_7, timestamps_us, path):
    """TUM format: ts_s tx ty tz qx qy qz qw.  traj_7 cols: [tx ty tz qx qy qz qw]."""
    with open(path, 'w') as f:
        f.write('# time(s) tx ty tz qx qy qz qw\n')
        for pose, t_us in zip(traj_7, timestamps_us):
            t_s = float(t_us) / 1e6
            f.write(f'{t_s:.6f} '
                    f'{float(pose[0]):.8f} {float(pose[1]):.8f} {float(pose[2]):.8f} '
                    f'{float(pose[3]):.8f} {float(pose[4]):.8f} '
                    f'{float(pose[5]):.8f} {float(pose[6]):.8f}\n')


def _save_trajectory_plot(pos_gt, pos_pred, seq_name, rpe, ate_mean, out_dir):
    """Save 2-D trajectory comparison (top-down + side view), same style as infer.py."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        ax = axes[0]
        ax.plot(pos_gt[:, 0],   pos_gt[:, 2],   'b-',  linewidth=1.5, label='GT')
        ax.plot(pos_pred[:, 0], pos_pred[:, 2],  'r--', linewidth=1.5, label='DEIO')
        ax.scatter([pos_gt[0, 0]], [pos_gt[0, 2]], c='g', s=60, zorder=5)
        ax.set_xlabel('X (m)'); ax.set_ylabel('Z (m)')
        ax.set_title('Top-down (X-Z)')
        ax.legend(); ax.set_aspect('equal'); ax.grid(True, alpha=0.3)

        ax = axes[1]
        ax.plot(pos_gt[:, 2],   pos_gt[:, 1],   'b-',  linewidth=1.5, label='GT')
        ax.plot(pos_pred[:, 2], pos_pred[:, 1],  'r--', linewidth=1.5, label='DEIO')
        ax.set_xlabel('Z (m)'); ax.set_ylabel('Y (m)')
        ax.set_title('Side view (Z-Y)')
        ax.legend(); ax.set_aspect('equal'); ax.grid(True, alpha=0.3)

        fig.suptitle(
            f"{seq_name} | "
            f"RPE t={rpe['rpe/trans_mean_m']*100:.2f} cm "
            f"r={rpe['rpe/rot_mean_deg']:.2f}° | "
            f"ATE={ate_mean*100:.2f} cm",
            fontsize=11,
        )
        plt.tight_layout()
        plot_path = out_dir / f'{seq_name}_trajectory.png'
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f'Trajectory plot  -> {plot_path}')
    except ImportError:
        print('matplotlib not available — skipping plot.')


# ---------------------------------------------------------------------------
# H5 data loading
# ---------------------------------------------------------------------------

def load_gt_h5(h5_path: str):
    """GT from H5: frames/T_event0_event + frames/depth_t_us.

    Returns:
        tss_us     : (N,) float64 µs
        traj       : (N, 7) float64 [tx ty tz qx qy qz qw]
        T_abs_list : list of (N,) (4,4) float32 tensors (for ATE anchoring)
    """
    with h5py.File(h5_path, 'r') as f:
        poses  = f['frames/T_event0_event'][:].astype(np.float64)
        tss_us = f['frames/depth_t_us'][:].astype(np.float64)
    traj     = np.zeros((len(poses), 7))
    T_abs_list = []
    for i, T in enumerate(poses):
        traj[i, :3] = T[:3, 3]
        traj[i, 3:] = Rotation.from_matrix(T[:3, :3]).as_quat()   # xyzw
        T_abs_list.append(torch.as_tensor(T, dtype=torch.float32))
    return tss_us, traj, T_abs_list


def load_imu_h5(h5_path: str):
    """IMU from H5: imu/t_us (µs) + imu/data_raw ([gx gy gz ax ay az] rad/s, m/s²)."""
    with h5py.File(h5_path, 'r') as f:
        imu_t = f['imu/t_us'][:].astype(np.float64)
        imu_d = f['imu/data_raw'][:].astype(np.float64)
    n = min(len(imu_t), len(imu_d))
    imu_t, imu_d = imu_t[:n], imu_d[:n]
    all_imu = np.zeros((n, 7))
    all_imu[:, 0]   = imu_t
    all_imu[:, 1:4] = imu_d[:, :3] * (180.0 / math.pi)   # rad/s → deg/s
    all_imu[:, 4:7] = imu_d[:, 3:6]
    return all_imu[all_imu[:, 0].argsort()]


# ---------------------------------------------------------------------------
# VECtor raw HDF5 helpers
# ---------------------------------------------------------------------------

# T_event_body: maps MoCap body frame → event camera frame.
# Derived from camera_mocap_extrinsic_results1.yaml.yaml (T_cam_body for left regular camera)
# composed with small_scale_joint_camera_extrinsic_results.yaml.yaml (T_cam0_cam2):
#   T_event_body = inv(T_leftcam_eventcam) @ T_leftcam_body
_T_VECTOR_EVENT_BODY = np.array([
    [-0.8541543792,  0.0423372690, -0.5182932106,  0.0906902530],
    [ 0.0007887351, -0.9965736857, -0.0827059063,  0.0189580389],
    [-0.5200189173, -0.0710524081,  0.8511943849, -0.0154160656],
    [ 0.0,           0.0,           0.0,           1.0         ],
], dtype=np.float64)
_T_VECTOR_BODY_EVENT = np.linalg.inv(_T_VECTOR_EVENT_BODY)


def seq_to_subdir(seq_name: str) -> str:
    """Map sequence name to VECtor GT subdirectory: 'desk_fast1' → 'desk-fast'."""
    base = re.sub(r'\d+$', '', seq_name)   # strip trailing digits
    return base.rstrip('_').replace('_', '-')


def load_gt_vector_raw(gt_txt: str):
    """Load VECtor GT from txt (MoCap body frame) and convert to event camera frame.

    GT txt format (2-line header):
        timestamp(s)  tx  ty  tz  qx  qy  qz  qw

    Returns:
        tss_us     : (N,) float64 absolute µs
        traj       : (N, 7) float64 [tx ty tz qx qy qz qw]  — relative to first frame
        T_abs_list : list of N (4,4) float32 tensors         — relative to first frame
    """
    data = np.loadtxt(gt_txt, comments='#')
    tss_us = (data[:, 0] * 1e6).astype(np.float64)

    # T_world_event = T_world_body @ T_body_event
    T_world_event_list = []
    for row in data:
        T_wb = np.eye(4)
        T_wb[:3, :3] = Rotation.from_quat(row[4:8]).as_matrix()  # xyzw
        T_wb[:3, 3] = row[1:4]
        T_world_event_list.append(T_wb @ _T_VECTOR_BODY_EVENT)

    # Normalize to first frame: T_e0_ei = inv(T_we0) @ T_wei
    T_we0_inv = np.linalg.inv(T_world_event_list[0])
    traj = np.zeros((len(data), 7))
    T_abs_list = []
    for i, T_we in enumerate(T_world_event_list):
        T_e0e = T_we0_inv @ T_we
        traj[i, :3] = T_e0e[:3, 3]
        traj[i, 3:] = Rotation.from_matrix(T_e0e[:3, :3]).as_quat()
        T_abs_list.append(torch.as_tensor(T_e0e, dtype=torch.float32))
    return tss_us, traj, T_abs_list


def load_imu_vector_raw(imu_txt: str):
    """Load VECtor IMU from txt.

    IMU txt format (3-line header):
        timestamp(s)  gx  gy  gz  ax  ay  az  qx  qy  qz  qw

    Returns all_imu: (N, 7) [t_us  gx_deg/s  gy_deg/s  gz_deg/s  ax  ay  az]
    """
    data = np.loadtxt(imu_txt, comments='#')
    n = len(data)
    all_imu = np.zeros((n, 7))
    all_imu[:, 0] = data[:, 0] * 1e6                       # s → µs
    all_imu[:, 1:4] = data[:, 1:4] * (180.0 / math.pi)    # rad/s → deg/s
    all_imu[:, 4:7] = data[:, 4:7]                         # m/s²
    return all_imu[all_imu[:, 0].argsort()]


# ---------------------------------------------------------------------------
# Build interval records from absolute DEIO poses
# (mirrors infer.py's frame_results structure)
# ---------------------------------------------------------------------------

def _match_gt_indices(tss_est_us: np.ndarray, tss_gt_us: np.ndarray) -> np.ndarray:
    """Match each estimated µs timestamp to the nearest GT µs timestamp."""
    idx      = np.searchsorted(tss_gt_us, tss_est_us).clip(0, len(tss_gt_us) - 1)
    idx_prev = (idx - 1).clip(0, len(tss_gt_us) - 1)
    closer   = np.abs(tss_gt_us[idx] - tss_est_us) > np.abs(tss_gt_us[idx_prev] - tss_est_us)
    idx[closer] = idx_prev[closer]
    return idx


def build_interval_records(traj_est: np.ndarray, tstamps: np.ndarray,
                           tss_gt_us: np.ndarray, T_abs_gt: list) -> list:
    """Convert consecutive DEIO keyframe pairs into interval records.

    Each record mirrors the structure used in infer.py:
        start_bdry, end_bdry  : GT frame indices bounding the interval
        R_pred, t_pred        : predicted relative pose (torch float32)
        R_gt,   t_gt          : GT relative pose       (torch float32)
        R_abs_start           : GT absolute rotation at interval start (for ATE)
        t_abs_start           : GT absolute translation at interval start (for ATE)

    Args:
        traj_est  : (N, 7) [tx ty tz qx qy qz qw] — DEIO absolute keyframe poses
        tstamps   : (N,)   µs — keyframe timestamps
        tss_gt_us : (M,)   µs — GT timestamps
        T_abs_gt  : list of M (4,4) float32 tensors — GT absolute poses
    """
    gt_idx = _match_gt_indices(tstamps, tss_gt_us)   # (N,) GT index per keyframe

    # Build (4,4) tensors for estimated poses
    def _to_mat(pose7):
        T = torch.eye(4, dtype=torch.float32)
        T[:3, 3]  = torch.as_tensor(pose7[:3], dtype=torch.float32)
        T[:3, :3] = torch.as_tensor(
            Rotation.from_quat(pose7[3:7]).as_matrix(), dtype=torch.float32)
        return T

    records = []
    for i in range(len(traj_est) - 1):
        T_est_i   = _to_mat(traj_est[i])
        T_est_ip1 = _to_mat(traj_est[i + 1])
        T_rel_pred = torch.linalg.inv(T_est_i) @ T_est_ip1

        T_gt_i   = T_abs_gt[gt_idx[i]]
        T_gt_ip1 = T_abs_gt[gt_idx[i + 1]]
        T_rel_gt = torch.linalg.inv(T_gt_i) @ T_gt_ip1

        records.append({
            'start_bdry': int(gt_idx[i]),
            'end_bdry':   int(gt_idx[i + 1]),
            'R_pred':     T_rel_pred[:3, :3],
            't_pred':     T_rel_pred[:3,  3],
            'R_gt':       T_rel_gt[:3, :3],
            't_gt':       T_rel_gt[:3,  3],
            # absolute GT at interval start (for ATE anchoring)
            'T_gt_start': T_gt_i,
        })
    return records


def build_absolute_trajectory(records: list, T_abs_gt: list):
    """Integrate predicted relative poses from GT anchor; compute ATE positions.

    Mirrors _build_absolute_trajectory_outputs from infer.py:
      - The first GT pose anchors the predicted trajectory.
      - Predicted relative poses are composed forward.
      - pos_gt_eval and pos_pred_eval are the END positions of each interval.

    Returns:
        pos_gt_eval   : (M, 3) GT end positions per interval (for ATE)
        pos_pred_eval : (M, 3) predicted end positions per interval (for ATE)
        pos_gt_plot   : (M+1, 3) full GT trajectory for plotting
        pos_pred_plot : (M+1, 3) full predicted trajectory for plotting
        pred_traj_7   : (M+1, 7) predicted absolute poses [tx ty tz qx qy qz qw]
        gt_traj_7     : (M+1, 7) GT absolute poses
        bdry_ts_us    : list of boundary timestamps (µs) from GT, length M+1
    """
    T_anchor   = records[0]['T_gt_start'].clone()   # first GT pose as anchor
    T_pred_abs = T_anchor.clone()

    pos_gt_eval,   pos_pred_eval   = [], []
    pos_gt_plot,   pos_pred_plot   = [T_anchor[:3, 3].clone()], [T_pred_abs[:3, 3].clone()]
    pred_traj_7,   gt_traj_7       = [], []
    bdry_ts_us                     = []

    def _mat_to_7(T):
        t = T[:3, 3].numpy()
        q = Rotation.from_matrix(T[:3, :3].numpy()).as_quat()
        return np.concatenate([t, q])

    # append start pose
    pred_traj_7.append(_mat_to_7(T_pred_abs))
    gt_traj_7.append(_mat_to_7(T_anchor))
    bdry_ts_us.append(0)   # placeholder; caller can replace

    for rec in records:
        T_rel = torch.eye(4, dtype=torch.float32)
        T_rel[:3, :3] = rec['R_pred']
        T_rel[:3,  3] = rec['t_pred']
        T_pred_abs = T_pred_abs @ T_rel

        T_gt_end = T_abs_gt[rec['end_bdry']]

        pos_pred_eval.append(T_pred_abs[:3, 3].clone())
        pos_gt_eval.append(T_gt_end[:3, 3].clone())
        pos_gt_plot.append(T_gt_end[:3, 3].clone())
        pos_pred_plot.append(T_pred_abs[:3, 3].clone())
        pred_traj_7.append(_mat_to_7(T_pred_abs))
        gt_traj_7.append(_mat_to_7(T_gt_end))
        bdry_ts_us.append(rec['end_bdry'])

    return {
        'pos_gt_eval':   torch.stack(pos_gt_eval),
        'pos_pred_eval': torch.stack(pos_pred_eval),
        'pos_gt_plot':   torch.stack(pos_gt_plot),
        'pos_pred_plot': torch.stack(pos_pred_plot),
        'pred_traj_7':   np.array(pred_traj_7),
        'gt_traj_7':     np.array(gt_traj_7),
        'bdry_gt_idx':   bdry_ts_us,
    }


# ---------------------------------------------------------------------------
# Per-sequence evaluation
# ---------------------------------------------------------------------------

def evaluate_sequence(
    h5_path:      Path,
    deio_cfg,
    network:      str,
    stride:       int       = 1,
    H:            int       = 240,
    W:            int       = 320,
    output_dir:   Path | None = None,
    ckpt_tag:     str       = '',
    no_plot:      bool      = False,
    dataset_name: str       = '',
    timing:       bool      = False,
    skip_start_s: float     = 0.0,
    gtdir:        str | None = None,
) -> dict:
    import quaternion

    # ── Detect raw VECtor HDF5 (*.hdf5, GT/IMU loaded from txt) ─────────
    is_raw_vector = h5_path.suffix == '.hdf5'

    if is_raw_vector:
        # Stem e.g. 'desk_fast1.synced.left_event' → seq_name 'desk_fast1'
        seq_name = h5_path.name.split('.')[0]
        if gtdir is None:
            raise ValueError(f'Raw VECtor HDF5 requires --gtdir (got None) for {h5_path}')
        subdir  = seq_to_subdir(seq_name)
        gt_txt  = Path(gtdir) / subdir / f'{seq_name}.synced.gt.txt'
        imu_txt = Path(gtdir) / subdir / f'{seq_name}.synced.imu.txt'
        if not gt_txt.exists() or not imu_txt.exists():
            print(f'  Skipping {seq_name}: GT/IMU txt not found '
                  f'({gt_txt}, {imu_txt})')
            return {
                'seq_name':     seq_name,
                'dataset_name': dataset_name or h5_path.parent.name,
                'n_frames':     0,
                'skipped':      True,
            }
        tss_gt_us, traj_gt, T_abs_gt = load_gt_vector_raw(str(gt_txt))
        all_imu = load_imu_vector_raw(str(imu_txt))
        tss_frames_us = tss_gt_us
    else:
        seq_name = h5_path.stem
        tss_gt_us, traj_gt, T_abs_gt = load_gt_h5(str(h5_path))
        all_imu = load_imu_h5(str(h5_path))

    print(f'\nSequence: {seq_name}')

    # ── all_gt dict for VI initialisation ────────────────────────────────
    all_gt = {}
    for ts_us, pose in zip(tss_gt_us, traj_gt):
        t_s = float(ts_us / 1e6)
        if t_s in all_gt:
            continue
        q       = quaternion.from_float_array([float(pose[6]), float(pose[3]),
                                               float(pose[4]), float(pose[5])])
        T       = np.eye(4)
        T[:3, :3] = quaternion.as_rotation_matrix(q)
        T[:3,  3] = pose[:3].astype(float)
        all_gt[t_s] = {'T': T}
    all_gt_keys = sorted(all_gt.keys())

    # ── Event iterator ────────────────────────────────────────────────────
    if is_raw_vector:
        iterator = vector_raw_iterator(str(h5_path), tss_frames_us,
                                       stride=stride, H=H, W=W)
    else:
        # Pick iterator based on dataset name inferred from path.
        # MVSEC and VECtor preprocessed events are already undistorted (float16).
        # CEAR events are raw distorted integer coords and need on-the-fly rectification.
        PRE_UNDISTORTED = {'mvsec', 'vector'}
        path_parts = [p.lower() for p in h5_path.parts]
        if any(p in PRE_UNDISTORTED for p in path_parts):
            iterator = mvsec_h5_iterator(str(h5_path), stride=stride, H=H, W=W,
                                         skip_start_s=skip_start_s)
        else:
            iterator = cear_h5_iterator(str(h5_path), stride=stride, H=H, W=W,
                                        skip_start_s=skip_start_s)

    # ── Run DEIO ──────────────────────────────────────────────────────────
    t0 = time.perf_counter()
    traj_est, tstamps, _, avg_fps = run_DEIO2(
        str(h5_path.parent), deio_cfg, network,
        viz=False,
        iterator=iterator,
        _all_imu=all_imu,
        _all_gt=all_gt,
        _all_gt_keys=all_gt_keys,
        timing=timing,
        H=H, W=W,
        viz_flow=False,
    )
    t1 = time.perf_counter()

    N            = len(traj_est)
    total_ms     = (t1 - t0) * 1000.0
    per_frame_ms = total_ms / N if N > 0 else float('nan')
    fps_str = f'{avg_fps:.1f} fps' if avg_fps is not None else 'fps N/A'
    print(f'Processed {N} keyframes  |  total {total_ms:.1f} ms  |  '
          f'{per_frame_ms:.2f} ms/frame  |  {fps_str}')

    if N < 2:
        print(f'  Skipping {seq_name}: too few output keyframes ({N}).')
        return {
            'seq_name':     seq_name,
            'dataset_name': dataset_name or h5_path.parent.name,
            'n_frames':     N,
            'skipped':      True,
        }

    # ── Build interval records ────────────────────────────────────────────
    records = build_interval_records(traj_est, tstamps, tss_gt_us, T_abs_gt)
    M = len(records)   # number of intervals = N - 1

    # ── RPE  (same as infer.py) ───────────────────────────────────────────
    R_preds = torch.stack([r['R_pred'] for r in records])   # (M, 3, 3)
    t_preds = torch.stack([r['t_pred'] for r in records])   # (M, 3)
    R_gts   = torch.stack([r['R_gt']   for r in records])
    t_gts   = torch.stack([r['t_gt']   for r in records])

    rot_errs   = rotation_error_deg(R_preds, R_gts)         # (M,)
    trans_errs = translation_error_m(t_preds, t_gts)        # (M,)
    rpe        = compute_rpe_metrics([rot_errs], [trans_errs])

    # ── ATE  (no alignment — anchored to GT, same as infer.py) ───────────
    traj_out      = build_absolute_trajectory(records, T_abs_gt)
    ate_per_frame = (traj_out['pos_pred_eval'] - traj_out['pos_gt_eval']).norm(dim=-1)
    ate_mean      = ate_per_frame.mean().item()
    ate_median    = ate_per_frame.median().item()

    # ── Print  (same format as infer.py) ──────────────────────────────────
    BW = 58
    print('═' * BW)
    print(f'  Sequence  : {seq_name}')
    print(f'  Keyframes : {N}  ({M} intervals)')
    print(f'  Infer     : {per_frame_ms:.2f} ms/frame  '
          f'(total {total_ms:.1f} ms)  ({fps_str})')
    print('─' * BW)
    print('  RPE (Relative Pose Error per keyframe interval)')
    print(f'    Trans  mean  : {rpe["rpe/trans_mean_m"]*100:7.2f} cm   '
          f'median: {rpe["rpe/trans_median_m"]*100:.2f} cm')
    print(f'    Rot    mean  : {rpe["rpe/rot_mean_deg"]:7.3f}°   '
          f'median: {rpe["rpe/rot_median_deg"]:.3f}°')
    print('─' * BW)
    print('  ATE (Absolute Trajectory Error, no alignment)')
    print(f'    Trans  mean  : {ate_mean*100:7.2f} cm   '
          f'median: {ate_median*100:.2f} cm')
    print('═' * BW)

    # ── Optional outputs ───────────────────────────────────────────────────
    if output_dir is not None:
        out_dir = output_dir / ckpt_tag / (dataset_name or h5_path.parent.name)
        out_dir.mkdir(parents=True, exist_ok=True)

        csv_path = out_dir / f'{seq_name}_errors.csv'
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['interval_idx', 'start_bdry', 'end_bdry',
                             'trans_err_m', 'rot_err_deg'])
            for i, rec in enumerate(records):
                writer.writerow([i, rec['start_bdry'], rec['end_bdry'],
                                 f'{trans_errs[i].item():.6f}',
                                 f'{rot_errs[i].item():.4f}'])
        print(f'Per-interval errors  -> {csv_path}')

        # TUM poses: pred (anchored to GT) and GT (at matched keyframe boundaries)
        bdry_idx    = traj_out['bdry_gt_idx']
        ts_us_list  = [int(tss_gt_us[i]) for i in bdry_idx]
        pred_tum    = out_dir / f'{seq_name}_poses_pred.txt'
        gt_tum      = out_dir / f'{seq_name}_poses_gt.txt'
        gt_full_tum = out_dir / f'{seq_name}_poses_gt_full.txt'
        save_tum(traj_out['pred_traj_7'], ts_us_list, pred_tum)
        save_tum(traj_out['gt_traj_7'],   ts_us_list, gt_tum)
        save_tum(traj_gt, tss_gt_us, gt_full_tum)
        print(f'TUM poses (pred)        -> {pred_tum}')
        print(f'TUM poses (gt)          -> {gt_tum}')
        print(f'TUM poses (gt, full)    -> {gt_full_tum}')

        if not no_plot:
            pos_gt   = traj_out['pos_gt_plot'].numpy()
            pos_pred = traj_out['pos_pred_plot'].numpy()
            _save_trajectory_plot(pos_gt, pos_pred, seq_name, rpe, ate_mean, out_dir)

    return {
        'seq_name':     seq_name,
        'dataset_name': dataset_name or h5_path.parent.name,
        'n_frames':     M,
        'rpe':          rpe,
        'ate_mean':     ate_mean,
        'ate_median':   ate_median,
        'rot_errs':     rot_errs,
        'trans_errs':   trans_errs,
    }


# ---------------------------------------------------------------------------
# Aggregate summary  (identical to deep_event_odometry/infer.py)
# ---------------------------------------------------------------------------

def _print_aggregate_summary(all_results: list) -> None:
    valid = [r for r in all_results if not r.get('skipped')]
    if not valid:
        return

    all_rot   = torch.cat([r['rot_errs']   for r in valid])
    all_trans = torch.cat([r['trans_errs'] for r in valid])
    agg_rpe   = compute_rpe_metrics([all_rot], [all_trans])

    ate_valid = [r for r in valid if math.isfinite(r['ate_mean'])]
    if ate_valid:
        total_n  = sum(r['n_frames'] for r in ate_valid)
        agg_ate  = sum(r['ate_mean'] * r['n_frames'] for r in ate_valid) / total_n
        agg_ate_str = f'{agg_ate*100:>7.2f}'
    else:
        agg_ate_str = f"{'N/A':>7}"

    name_w = max(max(len(r['seq_name']) for r in valid), len('AGGREGATE'))
    hdr    = (f"  {'Sequence':<{name_w}}  {'Frames':>6}  "
              f"{'RPE t(cm)':>9}  {'RPE r(°)':>8}  {'ATE(cm)':>7}")
    sep    = '-' * len(hdr)

    print('\n' + '=' * len(hdr))
    print('  AGGREGATE EVALUATION SUMMARY')
    print('=' * len(hdr))
    print(hdr)
    print(sep)
    for r in valid:
        ate_s = (f"{r['ate_mean']*100:>7.2f}" if math.isfinite(r['ate_mean'])
                 else f"{'N/A':>7}")
        print(f"  {r['seq_name']:<{name_w}}  {r['n_frames']:>6}  "
              f"{r['rpe']['rpe/trans_mean_m']*100:>9.2f}  "
              f"{r['rpe']['rpe/rot_mean_deg']:>8.3f}  {ate_s}")
    print(sep)
    total_frames = sum(r['n_frames'] for r in valid)
    print(f"  {'AGGREGATE':<{name_w}}  {total_frames:>6}  "
          f"{agg_rpe['rpe/trans_mean_m']*100:>9.2f}  "
          f"{agg_rpe['rpe/rot_mean_deg']:>8.3f}  {agg_ate_str}")
    print('=' * len(hdr))


def save_summary(all_results: list, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    valid        = [r for r in all_results if not r.get('skipped')]
    dataset_name = valid[0]['dataset_name'] if valid else 'summary'
    summary_path = out_dir / f'{dataset_name}_summary.csv'

    with open(summary_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'dataset', 'sequence', 'n_frames',
            'rpe_trans_mean_cm', 'rpe_trans_median_cm',
            'rpe_rot_mean_deg',  'rpe_rot_median_deg',
            'ate_mean_cm',       'ate_median_cm',
        ])
        for r in all_results:
            if r.get('skipped'):
                writer.writerow([r['dataset_name'], r['seq_name'], 0,
                                 *(['SKIPPED'] * 6)])
            else:
                rpe = r['rpe']
                writer.writerow([
                    r['dataset_name'], r['seq_name'], r['n_frames'],
                    f"{rpe['rpe/trans_mean_m']*100:.4f}",
                    f"{rpe['rpe/trans_median_m']*100:.4f}",
                    f"{rpe['rpe/rot_mean_deg']:.4f}",
                    f"{rpe['rpe/rot_median_deg']:.4f}",
                    f"{r['ate_mean']*100:.4f}"   if math.isfinite(r['ate_mean'])   else 'NA',
                    f"{r['ate_median']*100:.4f}" if math.isfinite(r['ate_median']) else 'NA',
                ])
        if valid:
            all_t  = torch.cat([r['trans_errs'] for r in valid])
            all_r  = torch.cat([r['rot_errs']   for r in valid])
            ate_v  = [r for r in valid if math.isfinite(r['ate_mean'])]
            total_n    = sum(r['n_frames'] for r in valid)
            agg_n      = sum(r['n_frames'] for r in ate_v) if ate_v else 1
            agg_ate_m  = (sum(r['ate_mean']   * r['n_frames'] for r in ate_v) / agg_n
                          if ate_v else float('nan'))
            agg_ate_md = (sum(r['ate_median'] * r['n_frames'] for r in ate_v) / agg_n
                          if ate_v else float('nan'))
            agg_rpe = compute_rpe_metrics([all_r], [all_t])
            writer.writerow([
                'AGGREGATE', 'ALL', total_n,
                f"{agg_rpe['rpe/trans_mean_m']*100:.4f}",
                f"{agg_rpe['rpe/trans_median_m']*100:.4f}",
                f"{agg_rpe['rpe/rot_mean_deg']:.4f}",
                f"{agg_rpe['rpe/rot_median_deg']:.4f}",
                f"{agg_ate_m*100:.4f}"  if math.isfinite(agg_ate_m)  else 'NA',
                f"{agg_ate_md*100:.4f}" if math.isfinite(agg_ate_md) else 'NA',
            ])
    print(f'\nSummary saved -> {summary_path}')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Run DEIO inference on preprocessed H5 files',
    )
    parser.add_argument('--network', required=True,
                        help='Path to DEVO.pth network checkpoint')
    parser.add_argument('--config',  required=True,
                        help='Path to DEIO YAML config (e.g. config/cear.yaml)')
    parser.add_argument('--h5', required=True, nargs='+',
                        help='H5 file(s) and/or directories to evaluate')
    parser.add_argument('--gtdir', default=None,
                        help='Root GT/IMU directory for raw VECtor HDF5 '
                             '(e.g. /media/s/2tb-1/tro/VECtor)')
    parser.add_argument('--stride',      type=int, default=1)
    parser.add_argument('--skip-start',  type=float, default=0.0,
                        help='Skip the first N seconds of each sequence')
    parser.add_argument('--height',      type=int, default=240, help='Sensor height px')
    parser.add_argument('--width',       type=int, default=320, help='Sensor width px')
    parser.add_argument('--output-dir',  default=None,
                        help='Save errors.csv, TUM poses, and trajectory.png here')
    parser.add_argument('--no-plot',     action='store_true')
    parser.add_argument('--skip-done',   action='store_true',
                        help='Skip sequences whose _errors.csv already exists in --output-dir')
    parser.add_argument('--timing',      action='store_true')
    parser.add_argument('--opts',        nargs='+', default=[],
                        help='Extra DEIO cfg overrides (key val ...)')
    parser.add_argument('--resnet',      action='store_true')
    parser.add_argument('--block-dims',  type=str, default='64,128,256')
    parser.add_argument('--initial-dim', type=int, default=64)
    parser.add_argument('--pretrain',    type=str, default='resnet18')
    args = parser.parse_args()

    DEIO_CFG.merge_from_file(args.config)
    DEIO_CFG.merge_from_list(args.opts)
    DEIO_CFG.resnet      = args.resnet
    DEIO_CFG.block_dims  = list(map(int, args.block_dims.split(',')))
    DEIO_CFG.initial_dim = args.initial_dim
    DEIO_CFG.pretrain    = args.pretrain
    assert not DEIO_CFG.CLASSIC_LOOP_CLOSURE
    print('\033[42m DEIO infer_h5 \033[0m  config loaded')
    print(DEIO_CFG, '\n')

    ckpt_tag   = Path(args.network).stem
    output_dir = Path(args.output_dir) if args.output_dir else None

    h5_files = discover_h5_files(args.h5)
    if not h5_files:
        print('ERROR: No H5 files found.  Check --h5 paths.')
        sys.exit(1)
    print(f'Found {len(h5_files)} H5 file(s) to evaluate.')

    all_results = []
    for h5_path, dataset_name in h5_files:
        if args.skip_done and output_dir is not None:
            csv_path = output_dir / ckpt_tag / (dataset_name or h5_path.parent.name) / f'{h5_path.stem}_errors.csv'
            if csv_path.exists():
                print(f'[SKIP] {h5_path.stem} already done ({csv_path})')
                continue

        result = evaluate_sequence(
            h5_path      = h5_path,
            deio_cfg     = DEIO_CFG,
            network      = args.network,
            stride       = args.stride,
            H            = args.height,
            W            = args.width,
            output_dir   = output_dir,
            ckpt_tag     = ckpt_tag,
            no_plot      = args.no_plot,
            dataset_name = dataset_name,
            timing       = args.timing,
            skip_start_s = args.skip_start,
            gtdir        = args.gtdir,
        )
        all_results.append(result)

    valid_results = [r for r in all_results if not r.get('skipped')]
    if len(all_results) > 1 and valid_results:
        _print_aggregate_summary(valid_results)
    if output_dir is not None and valid_results:
        save_summary(all_results, output_dir / ckpt_tag)

    print('\nDone!')


if __name__ == '__main__':
    main()
