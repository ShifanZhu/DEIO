"""
Publisher: reads frame timestamps from a Vector H5 file and sends
TimestampPair_t messages on DEIO_TIMESTAMPS at ~10 Hz.

Run AFTER starting subscriber.py (which hosts the DEIO server).

Usage:
    conda run -n DEIO python script/lcm/publisher.py 
        --h5 /media/s/rell/tro/vector-processed-old/vector/mountain-fast/mountain-fast.h5
        --n_frames 50 --hz 10
python script/lcm/publisher.py --h5 /media/s/rell/tro/vector-processed-old/vector/mountain-fast/mountain-fast.h5 --n_frames 50 --hz 10
"""
import argparse, os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import h5py, numpy as np, lcm
from lcm_types.deio import TimestampPair_t, DeltaPose_t

INPUT_CHANNEL  = "DEIO_TIMESTAMPS"
OUTPUT_CHANNEL = "DEIO_DELTA_POSE"

_STATUS = {0: "VALID", 1: "warmup", 2: "no_events", 3: "pose_error"}

def on_delta_pose(channel, data):
    r = DeltaPose_t.decode(data)
    t_s = r.t1_us / 1e6
    if r.status == 0:
        mag = (r.tx**2 + r.ty**2 + r.tz**2) ** 0.5
        print(f"  [t={t_s:.3f}s] VALID  |t|={mag:.4f}m  "
              f"({r.tx:+.4f}, {r.ty:+.4f}, {r.tz:+.4f})  "
              f"q=({r.qx:.3f},{r.qy:.3f},{r.qz:.3f},{r.qw:.3f})")
    else:
        print(f"  [t={t_s:.3f}s] {_STATUS.get(r.status, r.status)}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5",       required=True)
    ap.add_argument("--n_frames", type=int,   default=50)
    ap.add_argument("--hz",       type=float, default=10.0)
    args = ap.parse_args()

    h5 = h5py.File(args.h5, "r")
    frame_t = np.array(h5["frames/depth_t_us"], dtype=np.int64)
    h5.close()

    lc = lcm.LCM()
    lc.subscribe(OUTPUT_CHANNEL, on_delta_pose)

    n = min(args.n_frames, len(frame_t) - 1)
    period = 1.0 / args.hz
    print(f"[pub] Publishing {n} frames at {args.hz} Hz → {INPUT_CHANNEL}")
    print(f"[pub] Listening for responses on {OUTPUT_CHANNEL}\n")

    for i in range(n):
        msg = TimestampPair_t()
        msg.utime = int(frame_t[i])
        msg.t0_us = int(frame_t[i])
        msg.t1_us = int(frame_t[i + 1])
        lc.publish(INPUT_CHANNEL, msg.encode())
        print(f"[pub] frame {i:3d}  t0={msg.t0_us}  t1={msg.t1_us}  "
              f"gap={( msg.t1_us - msg.t0_us)/1e3:.1f}ms")

        deadline = time.time() + period
        while time.time() < deadline:
            lc.handle_timeout(max(1, int((deadline - time.time()) * 1000)))

    # drain remaining responses
    for _ in range(10):
        lc.handle_timeout(100)
    print("\n[pub] Done.")

if __name__ == "__main__":
    main()
