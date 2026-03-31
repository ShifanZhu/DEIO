"""
DEVORunner: wraps DEVO for online LCM-driven inference.

Handles:
- Opening and holding the event H5 file via EventSlicer
- Creating voxels on demand per (t0_us, t1_us) request
- Feeding voxels sequentially to DEVO
- Extracting and returning delta poses between consecutive frames
"""

import os
os.environ.setdefault("MPLBACKEND", "Agg")

# evo reads its own settings and forcibly calls mpl.use(plot_backend).
# Patch it to Agg before any downstream import triggers evo.tools.plot.
try:
    import evo.tools.settings as _evo_settings
    _evo_settings.SETTINGS.plot_backend = "Agg"
except Exception:
    pass

import numpy as np
import torch
import h5py
from dataclasses import dataclass, field
from typing import Optional, List

from devo.devo import DEVO
from utils.event_utils import EventSlicer, to_voxel_grid, RemoveHotPixelsVoxel
from lcm_server.pose_utils import (
    find_counter_for_timestamp,
    compute_delta_pose,
)

# Status codes (must match DeltaPose_t.status field)
STATUS_VALID       = 0
STATUS_NOT_INIT    = 1
STATUS_NO_EVENTS   = 2
STATUS_POSE_ERROR  = 3


@dataclass
class RunnerConfig:
    h5_file: str
    intrinsics: List[float]        # [fx, fy, cx, cy]
    img_height: int = 480
    img_width: int = 640
    voxel_bins: int = 5
    hot_pixel_filter: bool = True
    hot_pixel_stds: int = 10
    rectify_map_file: str = ""     # optional path to rectify_map_{side}.h5


class DEVORunner:
    """
    Stateful wrapper around DEVO for online per-request inference.

    Call process_frame(t0_us, t1_us) for each incoming LCM message.
    The runner must receive frames in strictly increasing timestamp order.
    """

    def __init__(self, cfg: RunnerConfig, devo_cfg, weights_path: str, slicer=None):
        """
        Args:
            slicer: optional pre-built event slicer (e.g. VectorEventSlicer).
                    If None, auto-detects format from cfg.h5_file:
                    - Vector format (has 'events/t_us'): uses VectorEventSlicer
                    - Standard format (has 'ms_to_idx'): uses EventSlicer
        """
        self.cfg = cfg

        # Open H5 event file (held open throughout)
        self._h5 = h5py.File(cfg.h5_file, "r")

        if slicer is not None:
            self.slicer = slicer
        elif "events/t_us" in self._h5:
            from lcm_server.vector_slicer import VectorEventSlicer
            self.slicer = VectorEventSlicer(self._h5)
        else:
            self.slicer = EventSlicer(self._h5)

        # Optional rectification map (H, W, 2) float array
        self.rectify_map = None
        if cfg.rectify_map_file:
            rmap_h5 = h5py.File(cfg.rectify_map_file, "r")
            self.rectify_map = np.array(rmap_h5["rectify_map"])
            rmap_h5.close()

        # Optional hot pixel filter
        self._hot_pixel_filter = (
            RemoveHotPixelsVoxel(num_stds=cfg.hot_pixel_stds)
            if cfg.hot_pixel_filter
            else None
        )

        self.intrinsics_tensor = torch.as_tensor(
            cfg.intrinsics, dtype=torch.float32
        ).cuda()

        # Initialize DEVO
        self.slam = DEVO(
            devo_cfg,
            weights_path,
            evs=True,
            ht=cfg.img_height,
            wd=cfg.img_width,
        )

        # Guard against out-of-order frames
        self._last_t1_us: int = -1

    def process_frame(self, t0_us: int, t1_us: int):
        """
        Process one (t0_us, t1_us) request.

        Steps:
          1. Validate ordering
          2. Extract events from H5 in [t0_us, t1_us]
          3. Build voxel grid (with optional rectification + hot-pixel removal)
          4. Feed voxel to DEVO (timestamp = t1_us)
          5. Compute and return delta pose between t0 and t1

        Returns:
            (status_code, delta_array_7d)
            delta_array_7d: [tx, ty, tz, qx, qy, qz, qw] as numpy float64,
                            or None if status != STATUS_VALID.
        """
        # 1. Reject out-of-order frames
        if t1_us <= self._last_t1_us:
            print(f"[DEVORunner] Out-of-order frame: t1={t1_us} <= last={self._last_t1_us}")
            return STATUS_POSE_ERROR, None

        # 2. Slice events from H5
        ev_batch = self.slicer.get_events(t0_us, t1_us)
        if ev_batch is None or len(ev_batch.get("t", [])) == 0:
            print(f"[DEVORunner] No events in [{t0_us}, {t1_us}]")
            return STATUS_NO_EVENTS, None

        # 3. Build voxel grid
        voxel = to_voxel_grid(
            ev_batch["x"].astype(np.float32),
            ev_batch["y"].astype(np.float32),
            ev_batch["t"].astype(np.float32),
            ev_batch["p"],
            H=self.cfg.img_height,
            W=self.cfg.img_width,
            nb_of_time_bins=self.cfg.voxel_bins,
            remapping_maps=self.rectify_map,
        )

        if self._hot_pixel_filter is not None:
            voxel = self._hot_pixel_filter(voxel)

        voxel = voxel.cuda()

        # 4. Feed to DEVO
        # Pass t1_us in microseconds directly — must be consistent with
        # find_counter_for_timestamp() lookups below.
        with torch.no_grad():
            self.slam(t1_us, voxel, self.intrinsics_tensor)

        self._last_t1_us = t1_us

        # 5. Not initialized yet (fewer than 8 keyframes)
        if not self.slam.is_initialized:
            return STATUS_NOT_INIT, None

        # 6. Find counter indices for t0 and t1
        c1 = find_counter_for_timestamp(self.slam, t1_us)
        c0 = find_counter_for_timestamp(self.slam, t0_us)

        if c0 is None or c1 is None:
            return STATUS_POSE_ERROR, None

        # 7. Compute delta pose
        delta_SE3, ok, errmsg = compute_delta_pose(self.slam, c0, c1)
        if not ok:
            print(f"[DEVORunner] Delta pose error: {errmsg}")
            return STATUS_POSE_ERROR, None

        delta_np = delta_SE3.data.cpu().numpy().flatten().astype(np.float64)
        return STATUS_VALID, delta_np

    def close(self):
        """Release the H5 file handle."""
        try:
            self._h5.close()
        except Exception:
            pass

    def __del__(self):
        self.close()
