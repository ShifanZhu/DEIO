"""
Inference on all sequences within a folder.

Discovers sequences by looking for subdirectories containing evs_left/h5/*.h5.
Expects ground truth at pose_left.txt alongside evs_left/.

Usage:
    python script/infer_folder.py \
        --weights checkpoints/half_data_full_step_30000.pth \
        --datapath /media/s/rell/tartan/ocean/Easy \
        --config config/infer_base.yaml \
        --plot --save
"""

import os
import glob
import math
import numpy as np
import torch
import argparse

import evo
from evo.tools.settings import SETTINGS
SETTINGS['plot_backend'] = 'Agg'

from devo.config import cfg
from utils.eval_utils import EVO_run, log_results, write_raw_results, compute_median_results
from utils.load_utils import voxel_iterator


def discover_sequences(datapath):
    """Return list of sequence paths that contain evs_left/h5/*.h5 files."""
    sequences = []
    for h5dir in sorted(glob.glob(os.path.join(datapath, "**/evs_left/h5"), recursive=True)):
        h5files = glob.glob(os.path.join(h5dir, "*.h5"))
        if len(h5files) == 0:
            continue
        seq_root = os.path.dirname(os.path.dirname(h5dir))  # strip evs_left/h5
        sequences.append(seq_root)
    return sequences


@torch.no_grad()
def evaluate(config, args, net, train_step=None, datapath="", stride=1,
             plot=False, save=False, return_figure=False, viz=False,
             timing=False, scale=1.0, rpg_eval=False, expname="", **kwargs):

    dataset_name = "tartanair_evs"

    if config is None:
        config = cfg
        config.merge_from_file("config/infer_base.yaml")

    sequences = discover_sequences(datapath)
    if len(sequences) == 0:
        raise RuntimeError(f"No sequences found under {datapath} (expected evs_left/h5/*.h5)")

    print(f"Found {len(sequences)} sequence(s):")
    for s in sequences:
        print(f"  {s}")

    results_dict_scene, figures = {}, {}
    all_results = []
    outfolder = "results"

    for seq_root in sequences:
        scene_path = os.path.join(seq_root, "evs_left", "h5")
        traj_ref_path = os.path.join(seq_root, "pose_left.txt")

        # Use path relative to datapath as scene name
        scene = os.path.relpath(seq_root, os.path.dirname(datapath))
        results_dict_scene[scene] = []

        print(f"\n--- Running inference on: {scene} ---")

        if scale != 1.0:
            nH = math.floor(scale * 480)
            nW = math.floor(scale * 640)
            kwargs.update({"scale": scale, "H": nH, "W": nW})

        traj_est, tstamps, flowdata, avg_fps = EVO_run(
            scene_path, config, net, viz=viz,
            iterator=voxel_iterator(scene_path, timing=timing, stride=stride, scale=scale),
            timing=timing, scale=scale, **kwargs
        )

        PERM = [1, 2, 0, 4, 5, 3, 6]  # ned -> xyz
        traj_ref = np.loadtxt(traj_ref_path, delimiter=" ")[1::stride, PERM]
        if scale != 1.0:
            from utils.transform_utils import transform_rescale_poses
            traj_ref = transform_rescale_poses(scale, torch.from_numpy(traj_ref)).data.numpy()

        traj_ref_kf = traj_ref[tstamps.astype(int)]
        data = (traj_ref_kf, tstamps, traj_est, tstamps)
        hyperparam = (train_step, net, dataset_name, scene, 0, config, args)

        all_results, results_dict_scene, figures, outfolder = log_results(
            data, hyperparam, all_results, results_dict_scene, figures,
            plot=plot, save=save, return_figure=return_figure, rpg_eval=rpg_eval,
            stride=stride, expname=expname, avg_fps=avg_fps
        )

        print(f"{scene}: MPE = {sorted(results_dict_scene[scene])}")

    write_raw_results(all_results, outfolder, train_step)
    results_dict = compute_median_results(results_dict_scene, all_results, dataset_name)

    print("\n=== Results ===")
    for k, v in results_dict.items():
        print(f"  {k}: {v:.4f}")

    if return_figure:
        return results_dict, figures
    return results_dict, None


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run DEVO inference on all sequences in a folder")
    parser.add_argument('--weights',   required=True,              help='Path to checkpoint (.pth)')
    parser.add_argument('--datapath',  required=True,              help='Folder containing sequences (will recurse for evs_left/h5)')
    parser.add_argument('--config',    default="config/infer_base.yaml")
    parser.add_argument('--expname',   default="infer",            help='Name for output folder')
    parser.add_argument('--stride',    type=int,   default=1)
    parser.add_argument('--scale',     type=float, default=1.0,    help='Image scale factor')
    parser.add_argument('--plot',      action="store_true",        help='Save trajectory PDFs')
    parser.add_argument('--save',      action="store_true",        help='Save trajectory TUM files')
    parser.add_argument('--viz',       action="store_true",        help='Live visualization')
    parser.add_argument('--timing',    action="store_true")
    parser.add_argument('--rpg_eval', action="store_true",         help='Compute RPG rotation metrics')
    parser.add_argument('--dim_inet',  type=int,   default=384)
    parser.add_argument('--dim_fnet',  type=int,   default=128)
    parser.add_argument('--dim',       type=int,   default=32)

    args = parser.parse_args()
    assert os.path.isfile(args.weights) and (".pth" in args.weights or ".pt" in args.weights)

    cfg.merge_from_file(args.config)

    # Infer train_step from checkpoint filename (e.g. step_30000.pth -> 30000)
    ckpt_stem = os.path.splitext(os.path.basename(args.weights))[0]
    train_step = ckpt_stem

    torch.manual_seed(1234)

    kwargs = {"dim_inet": args.dim_inet, "dim_fnet": args.dim_fnet, "dim": args.dim}
    results, _ = evaluate(
        cfg, args, args.weights,
        train_step=train_step,
        datapath=args.datapath,
        stride=args.stride,
        plot=args.plot,
        save=args.save,
        viz=args.viz,
        timing=args.timing,
        scale=args.scale,
        rpg_eval=args.rpg_eval,
        expname=args.expname,
        **kwargs
    )
