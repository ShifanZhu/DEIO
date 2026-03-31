"""
VectorEventSlicer: adapts the Vector dataset H5 format to the same
get_events(t_start_us, t_end_us) interface used by EventSlicer.

Vector H5 structure:
  events/x, events/y  (float32)
  events/t_us         (int64, microseconds)
  events/p            (int8,  0/1)
  events/slices_index (int64, shape N_frames) — first event idx for each frame
  frames/depth_t_us   (int64, shape N_frames) — frame boundary timestamps
"""

import numpy as np
import h5py


class VectorEventSlicer:
    """
    Slices events from a Vector-format H5 file by timestamp window.

    Uses frames/depth_t_us + events/slices_index for O(log N_frames) lookup
    instead of scanning all event timestamps.
    """

    def __init__(self, h5f: h5py.File):
        self._ev_x  = h5f["events/x"]
        self._ev_y  = h5f["events/y"]
        self._ev_t  = h5f["events/t_us"]
        self._ev_p  = h5f["events/p"]

        # Frame boundary arrays (loaded once — small, O(N_frames))
        self._frame_t  = np.array(h5f["frames/depth_t_us"], dtype=np.int64)
        self._slice_idx = np.array(h5f["events/slices_index"], dtype=np.int64)
        self._n_events = len(self._ev_t)

    def get_events(self, t_start_us: int, t_end_us: int):
        """
        Return dict {x, y, t, p} for events with t_start_us <= t < t_end_us.
        Returns None if no events are found.

        Uses frame-level index (slices_index / depth_t_us) for fast lookup,
        then trims to exact microsecond boundaries.
        """
        t_start_us = int(t_start_us)
        t_end_us   = int(t_end_us)

        # Find frame index range that covers [t_start_us, t_end_us)
        # i_start: last frame whose timestamp <= t_start_us
        # i_end:   first frame whose timestamp >= t_end_us
        i_start = max(0, int(np.searchsorted(self._frame_t, t_start_us, side="right")) - 1)
        i_end   = int(np.searchsorted(self._frame_t, t_end_us,   side="left"))
        i_end   = min(i_end, len(self._frame_t) - 1)

        if i_start > i_end:
            return None

        # Map frame indices → event indices
        ev_start = int(self._slice_idx[i_start])
        ev_end   = (int(self._slice_idx[i_end + 1])
                    if i_end + 1 < len(self._slice_idx)
                    else self._n_events)

        if ev_start >= ev_end:
            return None

        # Load the candidate event block
        t_block = np.array(self._ev_t[ev_start:ev_end], dtype=np.int64)

        # Trim to exact microsecond window
        mask = (t_block >= t_start_us) & (t_block < t_end_us)
        if not mask.any():
            return None

        lo = int(np.argmax(mask))
        hi = int(len(mask) - np.argmax(mask[::-1]))

        sl = slice(ev_start + lo, ev_start + hi)

        # Return timestamps relative to window start (microseconds).
        # Absolute timestamps (~1.6e15 us) exceed float32 precision and would
        # collapse to a single value; relative values (~0..1e5 us) are fine.
        t_abs = np.array(self._ev_t[sl], dtype=np.int64)
        return {
            "x": np.array(self._ev_x[sl], dtype=np.float32),
            "y": np.array(self._ev_y[sl], dtype=np.float32),
            "t": (t_abs - t_start_us).astype(np.float32),
            "p": np.array(self._ev_p[sl]),
        }
