"""
Online pose retrieval utilities for DEVO during live inference.

DEVO.get_pose() requires self.traj to be built by terminate(), which isn't
called in the online server. These helpers replicate that logic without
modifying the DEVO state.
"""

from typing import Optional
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dpvo.lietorch import SE3


def get_pose_online(slam, counter_idx: int) -> SE3:
    """
    Retrieve SE3 (world-to-camera) pose for the given counter index
    without calling terminate().

    Mirrors the logic of DEVO.terminate() + DEVO.get_pose():
      - Builds a transient traj from current keyframe state
        (self.tstamps_[i] -> self.poses_[i] for i in range(self.n))
      - Recurses through self.delta for non-keyframe counter indices

    Args:
        slam: DEVO instance
        counter_idx: frame counter index (0 .. slam.counter - 1)

    Returns:
        SE3 object representing the world-to-camera pose at counter_idx

    Raises:
        KeyError if counter_idx is unreachable through keyframes + delta
    """
    traj = {}
    for i in range(slam.n):
        traj[slam.tstamps_[i].item()] = slam.poses_[i]

    def _recurse(t: int) -> SE3:
        if t in traj:
            return SE3(traj[t])
        if t not in slam.delta:
            raise KeyError(
                f"counter index {t} not reachable. "
                f"Keyframe counter indices: {sorted(traj.keys())}, "
                f"delta keys: {sorted(slam.delta.keys())}"
            )
        t0, dP = slam.delta[t]
        return dP * _recurse(t0)

    return _recurse(counter_idx)


def find_counter_for_timestamp(slam, target_us: int) -> Optional[int]:
    """
    Return the counter index whose tlist entry is closest to target_us.

    slam.tlist[i] holds the timestamp passed to slam.__call__() at call i
    (i == counter index at call time). Returns None if tlist is empty.
    """
    if not slam.tlist:
        return None
    diffs = [abs(float(t) - float(target_us)) for t in slam.tlist]
    return int(min(range(len(diffs)), key=lambda i: diffs[i]))


def compute_delta_pose(slam, c0: int, c1: int):
    """
    Compute the delta pose between counter indices c0 and c1.

    delta = P_t1 * P_t0.inv()
      where P = T_wc (world-to-camera), matching DEVO's internal convention.

    Args:
        slam: DEVO instance
        c0: counter index for the earlier frame (t0)
        c1: counter index for the later frame (t1)

    Returns:
        (SE3 delta, True, "") on success
        (None, False, error_message) on failure
    """
    try:
        P_t0 = get_pose_online(slam, c0)
        P_t1 = get_pose_online(slam, c1)
    except Exception as e:
        return None, False, str(e)

    return P_t1 * P_t0.inv(), True, ""
