"""
Evaluation script for DEIO on the CEAR dataset.

Data source priority (per sequence):
  Events : {h5dir}/{scene}.h5  >  {inputdir}/{scene}/events.txt
  GT     : MoCap.txt           >  gt_temp.txt           >  no GT (inference only)
  IMU    : from H5 (imu/data_raw + imu/t_us)  OR  vectornav.txt

Usage:
    conda activate DEIO
    CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/home/s/repos/DEIO python script/eval_deio/cear.py \
        --inputdir=/media/s/2tb-2/cear/indoor \
        --h5dir=/media/s/2tb-2/deep_event_odometry/cear/indoor \
        --config=config/cear.yaml \
        --val_split=script/splits/cear/cear_val.txt \
        --network=/home/s/repos/SDEVO/src/SDEVO/DEVO/DEVO.pth \
        --enable_event --trials=1
No h5:
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/home/s/repos/DEIO python script/eval_deio/cear.py --inputdir=/media/s/2tb-2/cear/indoor --h5dir=/media/s/2tb-2/deep_event_odometry/cear/indoor --config=config/cear.yaml --val_split=script/splits/cear/cear_val.txt --network=/home/s/repos/SDEVO/src/SDEVO/DEVO/DEVO.pth --enable_event --trials=1

Use h5:
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/home/s/repos/DEIO python script/eval_deio/cear.py --inputdir=/media/s/2tb-2/cear/indoor --h5dir=/media/s/2tb-2/deep_event_odometry/cear/indoor --config=config/cear.yaml --val_split=script/splits/cear/cear_val.txt --network=/home/s/repos/SDEVO/src/SDEVO/DEVO/DEVO.pth --enable_event --trials=1 --use_h5=True
"""

import math
import os

import numpy as np
import torch
import quaternion

from evo.tools.settings import SETTINGS
SETTINGS['plot_backend'] = 'Agg'

from devo.config import cfg
from utils.load_utils import cear_evs_iterator, cear_h5_iterator
from utils.eval_utils import log_results, compute_median_results, run_DEIO2


def load_gt_h5(h5_path):
    """Load GT from H5 (frames/T_event0_event + frames/depth_t_us).

    T_event0_event[i] is the 4x4 pose of the event camera at frame i
    expressed in the frame of the first event camera pose — i.e. already in
    the camera frame, which is what DEIO outputs.

    Returns:
        tss_us : (N,) timestamps in microseconds
        traj   : (N, 7) [tx ty tz qx qy qz qw]
    """
    import h5py
    from scipy.spatial.transform import Rotation
    f = h5py.File(h5_path, 'r')
    poses  = f['frames/T_event0_event'][:].astype(np.float64)  # (N, 4, 4)
    tss_us = f['frames/depth_t_us'][:].astype(np.float64)      # (N,) µs
    f.close()
    traj = np.zeros((len(poses), 7))
    for i, T in enumerate(poses):
        traj[i, :3] = T[:3, 3]
        q = Rotation.from_matrix(T[:3, :3]).as_quat()  # xyzw
        traj[i, 3:] = q
    return tss_us, traj


def load_cear_gt(gt_path):
    """Load a TUM-format GT file (ts_s tx ty tz qx qy qz qw).

    Returns:
        tss_us : (N,) timestamps in microseconds
        traj   : (N, 7) [tx ty tz qx qy qz qw]
    """
    data = np.loadtxt(gt_path)
    assert data.shape[1] == 8, f"Expected 8 columns in {gt_path}, got {data.shape[1]}"
    tss_us = data[:, 0] * 1e6
    traj   = data[:, 1:]
    assert np.all(np.diff(tss_us) >= 0), f"GT timestamps not sorted in {gt_path}"
    return tss_us, traj


def build_all_gt(tss_traj_us, traj_hf):
    """Build the all_gt dict (keyed by seconds) used for DEIO VI visualisation."""
    all_gt = {}
    for ts_us, pose in zip(tss_traj_us, traj_hf):
        t_s = float(ts_us / 1e6)
        if t_s in all_gt:
            continue
        qx, qy, qz, qw = pose[3], pose[4], pose[5], pose[6]
        q = quaternion.from_float_array([float(qw), float(qx), float(qy), float(qz)])
        R = quaternion.as_rotation_matrix(q)
        T = np.eye(4)
        T[:3, :3] = R
        T[:3,  3] = [float(pose[0]), float(pose[1]), float(pose[2])]
        all_gt[t_s] = {'T': T}
    return all_gt


def load_imu_h5(h5_path):
    """Load raw IMU from H5 (imu/t_us in µs, imu/data_raw [gx gy gz ax ay az])."""
    import h5py
    f = h5py.File(h5_path, 'r')
    imu_t  = f['imu/t_us'][:].astype(np.float64)       # µs
    imu_d  = f['imu/data_raw'][:].astype(np.float64)   # [gx gy gz ax ay az]
    f.close()
    n = min(len(imu_t), len(imu_d))
    imu_t, imu_d = imu_t[:n], imu_d[:n]
    all_imu = np.zeros((n, 7))
    all_imu[:, 0]   = imu_t
    all_imu[:, 1:4] = imu_d[:, :3] * 180.0 / math.pi  # gyro rad/s → deg/s
    all_imu[:, 4:7] = imu_d[:, 3:6]                    # accel m/s²
    return all_imu[all_imu[:, 0].argsort()]


def load_imu_vectornav(imu_path):
    """Load IMU from vectornav.txt (ts_us gx gy gz ax ay az ...)."""
    import pandas as pd
    data = pd.read_csv(imu_path, sep=r'\s+', header=None).values
    all_imu = data[:, :7].astype(np.float64)
    all_imu[:, 1:4] *= 180.0 / math.pi   # gyro rad/s → deg/s
    return all_imu[all_imu[:, 0].argsort()]


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--inputdir',  default='/media/s/2tb-2/cear/indoor',
                        help='Root directory containing raw CEAR sequences')
    parser.add_argument('--h5dir',     default='/media/s/2tb-2/deep_event_odometry/cear/indoor',
                        help='Directory with preprocessed {scene}.h5 files (preferred event source)')
    parser.add_argument('--network',   type=str, default='/home/s/repos/SDEVO/src/SDEVO/DEVO/DEVO.pth')
    parser.add_argument('--val_split', type=str, default='script/splits/cear/cear_val.txt')
    parser.add_argument('--config',    default='config/cear.yaml')
    parser.add_argument('--stride',    type=int, default=1)
    parser.add_argument('--viz',       action='store_true')
    parser.add_argument('--enable_event', action='store_true')
    parser.add_argument('--trials',    type=int, default=1)
    parser.add_argument('--plot',      action='store_true')
    parser.add_argument('--opts',      nargs='+', default=[])
    parser.add_argument('--save_trajectory', action='store_true')
    parser.add_argument('--timing',    action='store_true')
    parser.add_argument('--resnet',    action='store_true')
    parser.add_argument('--block_dims',  type=str, default='64,128,256')
    parser.add_argument('--initial_dim', type=int, default=64)
    parser.add_argument('--pretrain',    type=str, default='resnet18')
    args = parser.parse_args()

    cfg.merge_from_file(args.config)
    cfg.merge_from_list(args.opts)
    cfg.resnet      = args.resnet
    cfg.block_dims  = list(map(int, args.block_dims.split(',')))
    cfg.initial_dim = args.initial_dim
    cfg.pretrain    = args.pretrain

    print("\033[42m Running CEAR eval with config...\033[0m")
    print(cfg, "\n")

    assert not cfg.CLASSIC_LOOP_CLOSURE
    print("\033[41m with Global BA \033[0m" if cfg.LOOP_CLOSURE else "\033[41m no Global BA \033[0m")

    test_scenes = open(args.val_split).read().split()
    print(f"Scenes to evaluate ({len(test_scenes)}): {test_scenes}")

    dataset_name = "cear/EVO" if args.enable_event else "cear/VO"
    if cfg.LOOP_CLOSURE: dataset_name += "_GBA"
    if cfg.ENALBE_IMU:   dataset_name += "_IMU"

    results_dict_scene, figures = {}, {}
    all_results = []
    outfolder   = "results"

    for scene in test_scenes:
        print(f"\n=== Eval on {scene} ===")
        results_dict_scene[scene] = []

        scene_base   = scene[:-3] if scene.endswith('.h5') else scene
        datapath_val = os.path.join(args.inputdir, scene_base)
        h5_path      = os.path.join(args.h5dir, f"{scene_base}.h5")

        # ── Event source ──────────────────────────────────────────────────
        use_h5 = os.path.exists(h5_path)
        evs_txt = os.path.join(datapath_val, 'events.txt')
        if not use_h5 and not os.path.exists(evs_txt):
            print(f"[SKIP] No H5 at {h5_path} and no events.txt — cannot run {scene}")
            continue
        print(f"  Events source: {'H5' if use_h5 else 'events.txt'}")

        # ── IMU source ────────────────────────────────────────────────────
        imu_path = os.path.join(datapath_val, 'vectornav.txt')
        if not os.path.exists(imu_path):
            print(f"[SKIP] No vectornav.txt for {scene}")
            continue

        # ── GT source (optional) — H5 first for frame alignment ──────────
        if use_h5:
            tss_traj_us, traj_hf = load_gt_h5(h5_path)
            has_gt = True
            print(f"  GT source: H5 (T_event0_event, camera frame)")
        else:
            gt_path = None
            for candidate in ['MoCap.txt', 'gt_temp.txt']:
                p = os.path.join(datapath_val, candidate)
                if os.path.exists(p):
                    gt_path = p
                    break
            has_gt = gt_path is not None
            print(f"  GT source: {os.path.basename(gt_path) if has_gt else 'none (inference only)'}")
            tss_traj_us, traj_hf = load_cear_gt(gt_path) if has_gt else (None, None)

        for trial in range(args.trials):
            print(f"\nRunning trial {trial} of {scene}...")

            if not args.enable_event:
                raise NotImplementedError("CEAR eval requires --enable_event")

            # ── GT dict for VI visualisation ──────────────────────────────
            all_gt, all_gt_keys = None, None
            if has_gt:
                all_gt = build_all_gt(tss_traj_us, traj_hf)
                all_gt_keys = sorted(all_gt.keys())

            # ── IMU (from same source as events for consistency) ──────────
            all_imu = load_imu_h5(h5_path) if use_h5 else load_imu_vectornav(imu_path)

            # ── Event iterator ────────────────────────────────────────────
            if use_h5:
                iterator = cear_h5_iterator(h5_path, stride=args.stride, H=240, W=320)
            else:
                iterator = cear_evs_iterator(datapath_val, stride=args.stride,
                                             timing=args.timing, H=240, W=320)

            # ── Run DEIO ──────────────────────────────────────────────────
            traj_est, tstamps, flowdata, avg_fps = run_DEIO2(
                datapath_val, cfg, args.network,
                viz=args.viz,
                iterator=iterator,
                _all_imu=all_imu,
                _all_gt=all_gt,
                _all_gt_keys=all_gt_keys,
                timing=args.timing,
                H=240, W=320,
                viz_flow=False,
            )

            # ── Evaluate (only if GT available) ───────────────────────────
            if has_gt:
                data       = (traj_hf, tss_traj_us, traj_est, tstamps)
                hyperparam = (None, args.network, dataset_name, scene, trial, cfg, args)
                all_results, results_dict_scene, figures, outfolder = log_results(
                    data, hyperparam, all_results, results_dict_scene, figures,
                    plot=args.plot, save=True,
                    return_figure=False, stride=args.stride,
                    expname=scene, _n_to_align=1000, avg_fps=avg_fps,
                )
            else:
                # Save trajectory even without GT
                from utils.eval_utils import make_outfolder, save_trajectory_tum_format
                import datetime
                outfolder = os.path.join("results", dataset_name,
                                         datetime.date.today().isoformat(), scene)
                os.makedirs(outfolder, exist_ok=True)
                scene_name = scene.title().replace('-', '_')
                traj_path = os.path.join(outfolder, f"{scene_name}_Trial{trial:02d}.txt")
                np.savetxt(traj_path,
                           np.column_stack([tstamps / 1e6,
                                            traj_est[:, :3],
                                            traj_est[:, [6,3,4,5]]]),
                           fmt='%.9f')
                print(f"Trajectory saved (no GT for evaluation): {traj_path}")

        if has_gt:
            print(scene, sorted(results_dict_scene[scene]))

    if all_results:
        results_dict = compute_median_results(results_dict_scene, all_results,
                                              dataset_name, outfolder=outfolder)
        for k in results_dict:
            print(k, results_dict[k])

    print("Done!")
