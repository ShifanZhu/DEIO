"""
Remap a pre-computed TartanEvent frame-graph pickle to local file paths.

The expensive part of dataset initialisation is build_frame_graph(), which
computes an N×N optical-flow distance matrix for every sequence.  A pickle
built on another machine contains exactly that graph — plus poses and
intrinsics — all of which are path-independent.  Only the file-path lists
(images/depths/events) need to be replaced with local equivalents.

Subtlety: our EVSDDataset skips frame 0 (voxels start at frame 1), so the
graph, poses and intrinsics are trimmed and re-indexed accordingly:
  old frames  1 .. N-1   →   new frames  0 .. N-2

Usage:
    python script/remap_fgraph_pickle.py \
        --src  fgraph/TartanEvent.pickle \
        --data /media/s/HDD8/data/tartan \
        --dst  fgraph/TartanAirEVS.pickle
"""

import argparse
import glob
import os
import os.path as osp
import pickle

import numpy as np


# ──────────────────────────────────────────────────────────────────────────────

def relative_key(abs_key: str, known_scenes=('ocean', 'office2',
        'abandonedfactory', 'seasidetown', 'westerndesert', 'carwelding',
        'endofworld', 'gascola', 'hospital', 'japanesealley', 'neighborhood',
        'office', 'oldtown', 'soulcity', 'amusement', 'house', 'prison')):
    """Extract 'scene/difficulty/seqnum' from an absolute pickle key."""
    parts = abs_key.rstrip('/').split('/')
    for i, p in enumerate(parts):
        if p in known_scenes:
            return '/'.join(parts[i:i+3])
    return None


def reindex_graph(old_graph: dict, n_old: int):
    """
    Trim graph to frames 1..n_old-1 and re-index them to 0..n_old-2.

    old_graph[i] = (neighbor_indices_array, distances_array)
    """
    new_graph = {}
    for old_i in range(1, n_old):
        new_i = old_i - 1
        if old_i not in old_graph:
            continue
        nbrs, dists = old_graph[old_i]
        # keep only neighbours that are also in frames 1..n_old-1
        mask = nbrs >= 1
        nbrs_new  = nbrs[mask] - 1          # re-index: subtract 1
        dists_new = dists[mask]
        new_graph[new_i] = (nbrs_new, dists_new)
    return new_graph


# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--src',  required=True, help='Source pickle (foreign machine)')
    parser.add_argument('--data', required=True, help='Local datapath root')
    parser.add_argument('--dst',  required=True, help='Output pickle path')
    args = parser.parse_args()

    print(f'Loading {args.src} ...')
    raw = pickle.load(open(args.src, 'rb'))
    old_si = raw[0] if isinstance(raw, tuple) else raw
    print(f'  {len(old_si)} sequences in source pickle')

    # build lookup: rel_path → old entry
    old_by_rel = {}
    for k, v in old_si.items():
        rel = relative_key(k)
        if rel is not None:
            old_by_rel[rel] = v

    # discover local sequences
    local_evs_dirs = sorted(glob.glob(osp.join(args.data, '*/*/*/evs_left')))
    print(f'  {len(local_evs_dirs)} local sequences found under {args.data}')

    new_si = {}
    skipped = []

    for evs_dir in local_evs_dirs:
        seq_dir = osp.dirname(evs_dir)
        rel = osp.relpath(seq_dir, args.data)   # e.g. ocean/Easy/P000

        # local file lists
        voxels = sorted(glob.glob(osp.join(evs_dir, 'h5/*.h5')))
        depths = sorted(glob.glob(osp.join(seq_dir, 'depth_left/*.npy')))
        pose_path = osp.join(seq_dir, 'pose_left.txt')

        if not voxels:
            skipped.append((rel, 'no voxels'))
            continue
        if not depths:
            skipped.append((rel, 'no depths'))
            continue
        if not osp.exists(pose_path):
            skipped.append((rel, 'no pose_left.txt'))
            continue

        # EVSDDataset convention: skip frame 0
        depths = depths[1:]
        if len(voxels) != len(depths):
            skipped.append((rel, f'voxel/depth mismatch {len(voxels)} vs {len(depths)}'))
            continue

        n_local = len(voxels)   # number of usable frames (= N_orig - 1)

        # try to reuse graph from old pickle
        if rel in old_by_rel:
            old = old_by_rel[rel]
            n_old = len(old.get('poses', []))   # original frame count

            if n_old - 1 == n_local:
                # frame counts match — reuse graph, poses, intrinsics
                graph      = reindex_graph(old['graph'], n_old)
                poses      = np.array(old['poses'])[1:]           # (N-1, 7)
                intrinsics = list(np.array(old['intrinsics'])[1:]) # N-1 entries
                source = 'pickle'
            else:
                print(f'  [WARN] {rel}: frame count mismatch '
                      f'(pickle={n_old-1}, local={n_local}) — will recompute')
                graph = poses = intrinsics = None
                source = 'recompute'
        else:
            graph = poses = intrinsics = None
            source = 'recompute'

        if source == 'recompute':
            # fall back: load poses from file and mark graph as None
            # (caller must run build_frame_graph for these)
            raw_poses = np.loadtxt(pose_path)[1:]
            poses = raw_poses[:, [1, 2, 0, 4, 5, 3, 6]]
            poses[:, :3] /= 5.0  # DEPTH_SCALE
            intrinsics = [np.array([320.0, 320.0, 320.0, 240.0])] * n_local
            graph = None  # will be computed at runtime

        new_si[evs_dir] = {
            'voxels':     voxels,
            'depths':     depths,
            'poses':      poses,
            'intrinsics': intrinsics,
            'graph':      graph,
            '_source':    source,   # debug tag, ignored by training code
        }

    reused   = sum(1 for v in new_si.values() if v['_source'] == 'pickle')
    recompute= sum(1 for v in new_si.values() if v['_source'] == 'recompute')
    print(f'\nResults:')
    print(f'  {reused} sequences: graph reused from pickle')
    print(f'  {recompute} sequences: graph must be recomputed at runtime')
    print(f'  {len(skipped)} sequences skipped:')
    for rel, reason in skipped:
        print(f'    {rel}: {reason}')

    if recompute > 0:
        print(f'\n[WARN] {recompute} sequences have graph=None.')
        print('       Training will crash for those unless you remove them from your split')
        print('       or run build_frame_graph manually.')

    os.makedirs(osp.dirname(osp.abspath(args.dst)), exist_ok=True)
    with open(args.dst, 'wb') as f:
        pickle.dump((new_si,), f)
    print(f'\nSaved → {args.dst}  ({len(new_si)} sequences)')


if __name__ == '__main__':
    main()
