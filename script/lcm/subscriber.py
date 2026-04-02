"""
Subscriber / DEIO server: listens for TimestampPair_t on DEIO_TIMESTAMPS,
runs DEVO inference, publishes DeltaPose_t on DEIO_DELTA_POSE, and accumulates
the returned relative poses into an absolute trajectory.

Start this BEFORE publisher.py.

Usage:
    conda run -n DEIO python script/lcm/subscriber.py \\
        --h5      /media/s/rell/tro/vector-processed-old/vector/mountain-fast/mountain-fast.h5 \\
        --weights weight/DEVO.pth \\
        [--devo_config config/infer_base.yaml] \\
        [--traj_out results/lcm/mountain-fast_trajectory.txt] \\
        [--plot_out results/lcm/mountain-fast_trajectory.png]

LD_LIBRARY_PATH=/home/s/repos/tool/miniconda3/envs/DEIO/lib:$LD_LIBRARY_PATH python script/lcm/subscriber.py --h5 /media/s/rell/tro/vector-processed-old/vector/mountain-fast/mountain-fast.h5 --weights ../SDEVO/DEVO/DEVO.pth
"""
import argparse, os, sys
os.environ.setdefault("MPLBACKEND", "Agg")
try:
    import evo.tools.settings as _s; _s.SETTINGS.plot_backend = "Agg"
except Exception: pass

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import h5py, numpy as np, torch, lcm
from itertools import islice
from pathlib import Path
from scipy.spatial.transform import Rotation as R

from lcm_types.deio import TimestampPair_t, DeltaPose_t
from devo.config import cfg as devo_cfg
from lcm_server.devo_runner import DEVORunner, RunnerConfig
from lcm_server.devo_runner import STATUS_VALID
from lcm_server.vector_slicer import VectorEventSlicer
from utils.eval_utils import run_DEIO2
from utils.load_utils import mvsec_h5_iterator, vector_preprocessed_h5_iterator

INPUT_CHANNEL  = "DEIO_TIMESTAMPS"
OUTPUT_CHANNEL = "DEIO_DELTA_POSE"

_STATUS = {0: "VALID", 1: "warmup", 2: "no_events", 3: "pose_error"}


def _pose7_to_matrix(pose7):
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R.from_quat(np.asarray(pose7[3:7], dtype=np.float64)).as_matrix()
    T[:3, 3] = np.asarray(pose7[:3], dtype=np.float64)
    return T


def _matrix_to_pose7(T):
    qx, qy, qz, qw = R.from_matrix(T[:3, :3]).as_quat()
    tx, ty, tz = T[:3, 3]
    return np.array([tx, ty, tz, qx, qy, qz, qw], dtype=np.float64)


def _save_tum(path, traj_7, timestamps_us):
    with Path(path).open("w") as f:
        f.write("# time(s) tx ty tz qx qy qz qw\n")
        for pose, t_us in zip(traj_7, timestamps_us):
            f.write(
                f"{float(t_us) / 1e6:.6f} "
                f"{float(pose[0]):.8f} {float(pose[1]):.8f} {float(pose[2]):.8f} "
                f"{float(pose[3]):.8f} {float(pose[4]):.8f} "
                f"{float(pose[5]):.8f} {float(pose[6]):.8f}\n"
            )


def _save_trajectory_plot(path, positions_xyz):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[srv] matplotlib unavailable; skipping trajectory plot.")
        return

    axis_names = np.array(["X", "Y", "Z"])
    vars_xyz = np.var(positions_xyz, axis=0)
    order = np.argsort(vars_xyz)
    ax0, ax1 = int(order[-1]), int(order[-2])

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.plot(positions_xyz[:, ax0], positions_xyz[:, ax1], "-", color="tab:blue", linewidth=1.5)
    ax.scatter(positions_xyz[0, ax0], positions_xyz[0, ax1], c="tab:green", s=60, label="start", zorder=3)
    ax.scatter(positions_xyz[-1, ax0], positions_xyz[-1, ax1], c="tab:red", s=60, label="end", zorder=3)
    ax.set_xlabel(f"{axis_names[ax0]} (m)")
    ax.set_ylabel(f"{axis_names[ax1]} (m)")
    ax.set_title(f"DEIO trajectory ({len(positions_xyz)} poses)")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _load_imu_h5(h5_path):
    with h5py.File(h5_path, "r") as f:
        imu_t = f["imu/t_us"][:].astype(np.float64)
        imu_d = f["imu/data_raw"][:].astype(np.float64)

    n = min(len(imu_t), len(imu_d))
    imu_t, imu_d = imu_t[:n], imu_d[:n]

    all_imu = np.zeros((n, 7), dtype=np.float64)
    all_imu[:, 0] = imu_t
    all_imu[:, 1:4] = imu_d[:, :3] * (180.0 / np.pi)
    all_imu[:, 4:7] = imu_d[:, 3:6]
    return all_imu[all_imu[:, 0].argsort()]


def _load_gt_h5(h5_path):
    with h5py.File(h5_path, "r") as f:
        if "frames" not in f or "T_event0_event" not in f["frames"] or "depth_t_us" not in f["frames"]:
            return None, None
        poses = f["frames/T_event0_event"][:].astype(np.float64)
        tss_us = f["frames/depth_t_us"][:].astype(np.float64)
    return tss_us, poses


def _match_gt_indices(tss_est_us, tss_gt_us):
    idx = np.searchsorted(tss_gt_us, tss_est_us).clip(0, len(tss_gt_us) - 1)
    idx_prev = (idx - 1).clip(0, len(tss_gt_us) - 1)
    closer_prev = np.abs(tss_gt_us[idx] - tss_est_us) > np.abs(tss_gt_us[idx_prev] - tss_est_us)
    idx[closer_prev] = idx_prev[closer_prev]
    return idx


def _anchor_like_infer_h5(traj_est, tstamps_us, gt_tss_us, gt_poses):
    if len(traj_est) == 0:
        return traj_est, tstamps_us

    gt_idx = _match_gt_indices(np.asarray(tstamps_us, dtype=np.float64), gt_tss_us)
    T_pred_abs = gt_poses[int(gt_idx[0])].copy()
    pred_traj_7 = [_matrix_to_pose7(T_pred_abs)]
    out_tss_us = [int(gt_tss_us[int(gt_idx[0])])]

    for i in range(len(traj_est) - 1):
        T_est_i = _pose7_to_matrix(traj_est[i])
        T_est_ip1 = _pose7_to_matrix(traj_est[i + 1])
        T_rel_pred = np.linalg.inv(T_est_i) @ T_est_ip1
        T_pred_abs = T_pred_abs @ T_rel_pred
        pred_traj_7.append(_matrix_to_pose7(T_pred_abs))
        out_tss_us.append(int(gt_tss_us[int(gt_idx[i + 1])]))

    return np.asarray(pred_traj_7, dtype=np.float64), np.asarray(out_tss_us, dtype=np.int64)


def _build_infer_h5_iterator(h5_path, img_height, img_width):
    h5_path = str(h5_path)
    path_parts = [p.lower() for p in Path(h5_path).parts]

    with h5py.File(h5_path, "r") as f:
        has_time_images = "events" in f and "time_images" in f["events"]

    if has_time_images:
        return vector_preprocessed_h5_iterator(h5_path, H=img_height, W=img_width)

    # Match infer_h5.py: vector, mvsec, and cear preprocessed H5s all go through
    # the same undistorted-event iterator.
    return mvsec_h5_iterator(h5_path, H=img_height, W=img_width)


def _save_infer_h5_matched_trajectory(h5_path, devo_cfg, weights_path, img_height,
                                      img_width, traj_out, plot_out, max_windows):
    all_imu = _load_imu_h5(h5_path)
    iterator = _build_infer_h5_iterator(h5_path, img_height, img_width)
    iterator = islice(iterator, max_windows)

    traj_est, tstamps_us, _, _ = run_DEIO2(
        str(Path(h5_path).parent),
        devo_cfg,
        weights_path,
        viz=False,
        iterator=iterator,
        _all_imu=all_imu.copy(),
        _all_gt=None,
        _all_gt_keys=None,
        timing=False,
        H=img_height,
        W=img_width,
        viz_flow=False,
    )

    if len(traj_est) == 0:
        raise RuntimeError("DEIO2 export produced no poses")

    gt_tss_us, gt_poses = _load_gt_h5(h5_path)
    if gt_tss_us is not None and gt_poses is not None:
        traj_est, tstamps_us = _anchor_like_infer_h5(traj_est, tstamps_us, gt_tss_us, gt_poses)

    traj_out = Path(traj_out)
    plot_out = Path(plot_out)
    traj_out.parent.mkdir(parents=True, exist_ok=True)
    plot_out.parent.mkdir(parents=True, exist_ok=True)

    _save_tum(traj_out, traj_est, tstamps_us)
    _save_trajectory_plot(plot_out, traj_est[:, :3])
    print(f"[srv] Saved infer_h5-matched TUM trajectory: {traj_out}")
    print(f"[srv] Saved infer_h5-matched PNG: {plot_out}")


class TrajectoryRecorder:
    """Accumulate DeltaPose_t messages into an absolute trajectory."""

    def __init__(self):
        self.timestamps_us = []
        self._poses_wc = []
        self._current_T_wc = np.eye(4, dtype=np.float64)

    def observe(self, t0_us, t1_us, status, delta_np):
        t0_us = int(t0_us)
        t1_us = int(t1_us)

        if not self.timestamps_us:
            self.timestamps_us.append(t0_us)
            self._poses_wc.append(self._current_T_wc.copy())
        elif self.timestamps_us[-1] != t0_us:
            print(
                f"[srv] Trajectory accumulator expected t0={self.timestamps_us[-1]} "
                f"but received t0={t0_us}; composing anyway."
            )

        # DeltaPose_t stores T_wc(t1) * T_wc(t0)^-1, so accumulate in T_wc.
        if status == STATUS_VALID and delta_np is not None:
            self._current_T_wc = _pose7_to_matrix(delta_np) @ self._current_T_wc

        if not self.timestamps_us or self.timestamps_us[-1] != t1_us:
            self.timestamps_us.append(t1_us)
            self._poses_wc.append(self._current_T_wc.copy())

    def export_tum(self):
        if not self.timestamps_us:
            return np.empty((0, 7), dtype=np.float64), np.empty((0,), dtype=np.int64)

        poses_cw = [np.linalg.inv(T_wc) for T_wc in self._poses_wc]
        traj_7 = np.stack([_matrix_to_pose7(T_cw) for T_cw in poses_cw], axis=0)
        tss_us = np.asarray(self.timestamps_us, dtype=np.int64)
        return traj_7, tss_us

    def save(self, traj_out, plot_out):
        traj_7, tss_us = self.export_tum()
        if traj_7.size == 0:
            print("[srv] No trajectory samples recorded; skipping trajectory export.")
            return

        traj_out = Path(traj_out)
        plot_out = Path(plot_out)
        traj_out.parent.mkdir(parents=True, exist_ok=True)
        plot_out.parent.mkdir(parents=True, exist_ok=True)

        _save_tum(traj_out, traj_7, tss_us)
        _save_trajectory_plot(plot_out, traj_7[:, :3])
        print(f"[srv] Saved TUM trajectory: {traj_out}")
        print(f"[srv] Saved trajectory PNG: {plot_out}")

    @property
    def num_windows(self):
        return max(0, len(self.timestamps_us) - 1)


def _default_output_paths(h5_path, traj_out_arg, plot_out_arg):
    stem = Path(h5_path).stem
    out_dir = Path("results/lcm")
    traj_out = Path(traj_out_arg) if traj_out_arg else out_dir / f"{stem}_trajectory.txt"
    plot_out = Path(plot_out_arg) if plot_out_arg else traj_out.with_suffix(".png")
    return traj_out, plot_out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5",          required=True)
    ap.add_argument("--weights",     required=True)
    ap.add_argument("--devo_config", default="config/infer_base.yaml")
    ap.add_argument("--traj_out",    default="",
                    help="Output path for the accumulated trajectory in TUM format")
    ap.add_argument("--plot_out",    default="",
                    help="Output path for the trajectory PNG plot")
    args = ap.parse_args()

    devo_cfg.merge_from_file(args.devo_config)
    torch.set_grad_enabled(False)

    traj_out, plot_out = _default_output_paths(args.h5, args.traj_out, args.plot_out)
    recorder = TrajectoryRecorder()

    h5 = h5py.File(args.h5, "r")
    K  = np.array(h5["meta/K_event_undist"])

    runner_cfg = RunnerConfig(
        h5_file      = args.h5,
        intrinsics   = [float(K[0,0]), float(K[1,1]), float(K[0,2]), float(K[1,2])],
        img_height   = 480,
        img_width    = 640,
    )
    slicer = VectorEventSlicer(h5)
    runner = DEVORunner(runner_cfg, devo_cfg, args.weights, slicer=slicer)
    print(f"[srv] DEVO ready.  Listening on {INPUT_CHANNEL}")
    print(f"[srv] Publishing responses on {OUTPUT_CHANNEL}\n")
    print(f"[srv] Trajectory output: {traj_out}")
    print(f"[srv] Trajectory plot : {plot_out}\n")

    lc = lcm.LCM()

    def on_request(channel, data):
        req = TimestampPair_t.decode(data)
        t0, t1 = int(req.t0_us), int(req.t1_us)

        status, delta_np = runner.process_frame(t0, t1)

        resp = DeltaPose_t()
        resp.utime  = req.utime
        resp.t0_us  = t0
        resp.t1_us  = t1
        resp.status = int(status)
        if status == STATUS_VALID and delta_np is not None:
            resp.tx, resp.ty, resp.tz = float(delta_np[0]), float(delta_np[1]), float(delta_np[2])
            resp.qx, resp.qy, resp.qz, resp.qw = (
                float(delta_np[3]), float(delta_np[4]),
                float(delta_np[5]), float(delta_np[6]))
        else:
            resp.tx = resp.ty = resp.tz = 0.0
            resp.qx = resp.qy = resp.qz = 0.0; resp.qw = 1.0

        recorder.observe(t0, t1, status, delta_np)
        lc.publish(OUTPUT_CHANNEL, resp.encode())
        print(f"[srv] t={t1/1e6:.3f}s  {_STATUS.get(status, status)}", flush=True)

    lc.subscribe(INPUT_CHANNEL, on_request)

    try:
        while True:
            lc.handle()
    except KeyboardInterrupt:
        print("\n[srv] Stopped.")
    finally:
        try:
            exported = False
            if recorder.num_windows > 0:
                try:
                    _save_infer_h5_matched_trajectory(
                        args.h5,
                        devo_cfg,
                        args.weights,
                        runner_cfg.img_height,
                        runner_cfg.img_width,
                        traj_out,
                        plot_out,
                        recorder.num_windows,
                    )
                    exported = True
                except Exception as exc:
                    print(f"[srv] infer_h5-matched export failed: {exc}")
                    print("[srv] Falling back to online accumulated DEVO deltas.")

            if not exported:
                recorder.save(traj_out, plot_out)
        finally:
            runner.close()
            h5.close()

if __name__ == "__main__":
    main()
