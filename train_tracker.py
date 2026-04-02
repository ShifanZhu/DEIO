"""
train_tracker.py — Training script for PatchTracker

Simplified version of train.py: keeps the data pipeline, optimizer, and
checkpoint infrastructure but uses only the flow loss (+ optional scorer loss).
No pose loss, no BA, no CM refinement, no structure-only warmup.

Loss: exponentially-weighted flow loss across STEPS iterations (RAFT-style).
"""

import os
import json
import numpy as np
from collections import OrderedDict
from pathlib import Path
import contextlib

import cv2
import matplotlib
matplotlib.use('Agg')

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from dpvo.lietorch import SE3
from devo.data_readers.factory import dataset_factory
from devo.logger import Logger
from devo.patch_tracker import PatchTracker
from devo.selector import SelectionMethod
from utils.voxel_utils import std, rescale

import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from tqdm import tqdm


def score_map_to_bgr(score_map: np.ndarray, upscale: int = 4) -> np.ndarray:
    """Render a non-negative scorer map as a TURBO heatmap PNG."""
    score_map = np.asarray(score_map, dtype=np.float32)
    finite = np.isfinite(score_map)
    if not finite.any():
        norm_u8 = np.zeros(score_map.shape, dtype=np.uint8)
    else:
        safe = np.where(finite, score_map, 0.0)
        lo, hi = float(safe[finite].min()), float(safe[finite].max())
        if hi - lo <= 1e-8:
            norm_u8 = np.zeros(score_map.shape, dtype=np.uint8)
        else:
            norm_u8 = np.clip(255.0 * (safe - lo) / (hi - lo), 0, 255).astype(np.uint8)
    heatmap = cv2.applyColorMap(norm_u8, cv2.COLORMAP_TURBO)
    if upscale > 1:
        heatmap = cv2.resize(heatmap,
                             (score_map.shape[1] * upscale, score_map.shape[0] * upscale),
                             interpolation=cv2.INTER_NEAREST)
    return heatmap


def _normalize_voxel(images: torch.Tensor, norm: str) -> torch.Tensor:
    """Apply the same voxel normalization as PatchTracker.forward()."""
    if norm == 'none':
        return images
    elif norm in ('rescale', 'norm'):
        return rescale(images)
    elif norm in ('standard', 'std'):
        return std(images, sequence=False)
    elif norm in ('standard2', 'std2'):
        return std(images)
    raise NotImplementedError(f"norm '{norm}' not implemented")


@torch.no_grad()
def evaluate_tracker(net, args, total_steps, val_loader):
    """
    Run PatchTracker on the validation set.

    Saves per-frame score maps (NPZ + PNG heatmap) and per-sequence track
    results (NPZ with coords_est, coords_gt, valid, weight at the last
    iteration).  Returns a metrics dict suitable for logger.push().

    Output layout:
        evals/{name}/{step:06d}/
            score_maps/seq{N:04d}_frame{F:04d}.{npz,png}
            tracks/seq{N:04d}.png
            meta.json
    """
    model = net.module if args.ddp else net
    model.eval()

    step_tag = f'{args.gpu_num * total_steps:06d}'
    eval_dir  = Path('results/evals') / args.name / step_tag
    score_dir = eval_dir / 'score_maps'
    tracks_dir = eval_dir / 'tracks'
    score_dir.mkdir(parents=True, exist_ok=True)
    tracks_dir.mkdir(parents=True, exist_ok=True)

    all_epe, all_seq_names = [], []

    for data_blob in val_loader:
        # return_fname=True appends scene_id as last element
        seq_name = data_blob[-1][0] if isinstance(data_blob[-1], (list, tuple)) else str(data_blob[-1][0])
        seq_name = seq_name.replace('/', '_')
        data_blob = data_blob[:-1]
        images, poses, disps, intrinsics = [x.cuda().float() for x in data_blob]
        poses = SE3(poses).inv()

        B, N, bins, H, W = images.shape

        # per-sequence output dirs
        seq_score_dir  = score_dir / seq_name
        seq_tracks_dir = tracks_dir / seq_name
        seq_score_dir.mkdir(parents=True, exist_ok=True)
        seq_tracks_dir.mkdir(parents=True, exist_ok=True)

        # ── Score maps ──────────────────────────────────────────────────────
        images_norm = _normalize_voxel(images, model.norm)
        with torch.amp.autocast('cuda', enabled=args.amp, dtype=torch.bfloat16):
            score_logits = model.patchify.scorer(images_norm)
        score_maps = torch.sigmoid(score_logits.float())  # (B, N, H/4, W/4)

        for frame_idx in range(N):
            score_np = score_maps[0, frame_idx].cpu().numpy().astype(np.float32)
            cv2.imwrite(str(seq_score_dir / f'frame{frame_idx:04d}.png'), score_map_to_bgr(score_np))

        # ── Tracker forward ──────────────────────────────────────────────────
        traj = model(images, poses, disps, intrinsics,
                     STEPS=args.iters,
                     patches_per_image=args.patches_per_image)

        # EPE at last iteration over close edges (dij ≤ 2)
        coords_est, coords_gt, valid, weight, _, _kk = traj[-1]
        valid_mask = (valid > 0.5).reshape(-1)
        e = (coords_est - coords_gt).norm(dim=-1).reshape(-1)
        n_valid = int(valid_mask.sum().item())
        epe = e[valid_mask].mean().item() if n_valid > 0 else float('nan')
        all_epe.append(epe)
        all_seq_names.append(seq_name)

        # ── Save tracks as PNG ───────────────────────────────────────────────
        est_np = coords_est[0].cpu().numpy()      # (close_edges, 2)
        gt_np  = coords_gt[0].cpu().numpy()       # (close_edges, 2)
        vm_np  = (valid[0] > 0.5).cpu().numpy()   # (close_edges,) bool

        feat_h = score_maps.shape[-2]
        feat_w = score_maps.shape[-1]
        canvas = np.full((feat_h, feat_w, 3), 240, dtype=np.uint8)

        for idx in range(est_np.shape[0]):
            if not vm_np[idx]:
                continue
            ex = int(np.clip(round(float(est_np[idx, 0])), 0, feat_w - 1))
            ey = int(np.clip(round(float(est_np[idx, 1])), 0, feat_h - 1))
            gx = int(np.clip(round(float(gt_np[idx, 0])),  0, feat_w - 1))
            gy = int(np.clip(round(float(gt_np[idx, 1])),  0, feat_h - 1))
            cv2.line(canvas, (gx, gy), (ex, ey), (0, 200, 0), 1, cv2.LINE_AA)
            cv2.circle(canvas, (gx, gy), 2, (0, 0, 220), -1)   # GT: red
            cv2.circle(canvas, (ex, ey), 2, (0, 180, 0), -1)   # est: green

        cv2.putText(canvas, f'EPE={epe:.3f}px  n={n_valid}', (4, 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.imwrite(str(seq_tracks_dir / 'tracks.png'), canvas)

    # ── Summary ─────────────────────────────────────────────────────────────
    valid_epes = [e for e in all_epe if not np.isnan(e)]
    mean_epe = float(np.mean(valid_epes)) if valid_epes else float('nan')

    meta = {
        'step': args.gpu_num * total_steps,
        'n_seqs': len(all_epe),
        'mean_epe_feature_px': mean_epe,
        'per_seq': {name: round(e, 4) if not np.isnan(e) else None
                    for name, e in zip(all_seq_names, all_epe)},
    }
    (eval_dir / 'meta.json').write_text(json.dumps(meta, indent=2))
    print(f"[eval step {args.gpu_num * total_steps}] mean EPE = {mean_epe:.4f} px (feature res) "
          f"over {len(valid_epes)}/{len(all_epe)} seqs — saved to {eval_dir}")

    model.train()
    return {'val/epe_feature_px': mean_epe}


def setup_ddp(rank, args):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = args.port
    dist.init_process_group(
        backend='nccl',
        init_method='env://',
        world_size=args.gpu_num,
        rank=rank)
    torch.manual_seed(0)
    torch.cuda.set_device(rank)


def train(rank, args):
    """Main training loop for PatchTracker."""

    if args.ddp:
        setup_ddp(rank, args)

    # ── Dataset ──────────────────────────────────────────────────────────────
    db = dataset_factory(
        ['tartan_evs'],
        datapath=args.datapath,
        n_frames=args.n_frames,
        fgraph_pickle=args.fgraph_pickle,
        train_split=args.train_split,
        val_split=args.val_split,
        split_mode='train',
        strict_split=False,
        sample=True,
        return_fname=True,
        scale=args.scale,
    )

    if args.ddp:
        sampler = torch.utils.data.distributed.DistributedSampler(
            db, shuffle=True, num_replicas=args.gpu_num, rank=rank)
        train_loader = DataLoader(db, batch_size=args.batch, sampler=sampler,
                                  num_workers=args.num_workers, pin_memory=True, prefetch_factor=4)
    else:
        train_loader = DataLoader(db, batch_size=args.batch, shuffle=True,
                                  num_workers=args.num_workers, pin_memory=True, prefetch_factor=4)

    # ── Network ───────────────────────────────────────────────────────────────
    net = PatchTracker(
        args,
        dim_inet=args.dim_inet,
        dim_fnet=args.dim_fnet,
        dim=args.dim,
        patch_selector=args.patch_selector.lower(),
        norm=args.norm,
        randaug=args.randaug,
        corner_guidance=args.corner_guidance,
    )
    net.train()
    net.cuda()

    if args.ddp:
        net = DDP(net, device_ids=[rank], find_unused_parameters=False)

    # ── Val loader (for periodic evaluation) ─────────────────────────────────
    val_loader = None
    if args.eval and rank == 0:
        val_db = dataset_factory(
            ['tartan_evs'],
            datapath=args.datapath,
            n_frames=args.n_frames,
            fgraph_pickle=args.fgraph_pickle,
            train_split=args.train_split,
            val_split=args.val_split,
            split_mode='val',
            strict_split=False,
            sample=False,
            return_fname=True,
            scale=args.scale,
        )
        # Pick the first window index for each unique scene — O(n) metadata scan,
        # avoids iterating through thousands of duplicate windows per sequence.
        # dataset_factory returns a ConcatDataset; iterate sub-datasets to build
        # a flat offset-adjusted index list.
        seen, one_per_seq = set(), []
        sub_datasets = val_db.datasets if hasattr(val_db, 'datasets') else [val_db]
        offset = 0
        for sub_db in sub_datasets:
            for local_idx, (scene_id, _) in enumerate(sub_db.dataset_index):
                if scene_id not in seen:
                    seen.add(scene_id)
                    one_per_seq.append(offset + local_idx)
            offset += len(sub_db)
        val_subset = torch.utils.data.Subset(val_db, one_per_seq)
        val_loader = DataLoader(val_subset, batch_size=1, shuffle=False,
                                num_workers=2, pin_memory=True)

    # ── Optimizer ─────────────────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-6)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, args.lr, args.steps,
        pct_start=0.01, cycle_momentum=False, anneal_strategy='linear')
    scaler = torch.amp.GradScaler('cuda', enabled=args.amp)

    total_steps = 0

    # ── Resume ────────────────────────────────────────────────────────────────
    if args.resume:
        print(f"Loading from {args.resume}")
        checkpoint = torch.load(args.resume)
        model = net.module if args.ddp else net
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        else:
            new_sd = OrderedDict()
            for k, v in checkpoint.items():
                new_sd[k.replace('module.', '')] = v
            update = {k: v for k, v in new_sd.items()
                      if k in model.state_dict() and model.state_dict()[k].shape == v.shape}
            state = model.state_dict()
            state.update(update)
            model.load_state_dict(state, strict=False)
        if 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        if 'steps' in checkpoint:
            total_steps = checkpoint['steps']

    if rank == 0:
        logger = Logger(args.name, scheduler, args.gpu_num * total_steps, args.gpu_num)

    os.makedirs(f'checkpoints/{args.name}', exist_ok=True)

    # ── Training loop ─────────────────────────────────────────────────────────
    with contextlib.nullcontext():
        if rank == 0:
            pbar = tqdm(total=args.gpu_num * args.steps,
                        initial=args.gpu_num * total_steps,
                        desc=f"Training {args.name}")

        while True:
            for data_blob in train_loader:
                data_blob.pop()  # pop scene_id (return_fname=True appends it)
                images, poses, disps, intrinsics = [x.cuda().float() for x in data_blob]

                optimizer.zero_grad(set_to_none=True)

                # convert GT poses from c2w → w2c (same as train.py:188)
                poses = SE3(poses).inv()

                with torch.amp.autocast('cuda', enabled=args.amp, dtype=torch.bfloat16):
                    traj = net(images, poses, disps, intrinsics,
                               STEPS=args.iters,
                               patches_per_image=args.patches_per_image)

                # ── Loss ──────────────────────────────────────────────────────
                loss = torch.tensor(0.0, device='cuda')
                flow_loss_last = torch.tensor(0.0, device='cuda')
                scores_loss = torch.tensor(0.0, device='cuda')

                for i, (coords_est, coords_gt, valid, weight, scores, kk_close) in enumerate(traj):
                    # exponential iteration weighting (RAFT-style: earlier iters matter less)
                    w_i = args.gamma ** (len(traj) - i - 1)

                    valid_mask = (valid > 0.5).reshape(-1)
                    e = (coords_est - coords_gt).norm(dim=-1)  # (B, close_edges)
                    e_flat = e.reshape(-1)

                    if valid_mask.any():
                        fl = e_flat[valid_mask].mean()
                    else:
                        fl = torch.tensor(0.0, device='cuda')

                    loss = loss + w_i * args.flow_weight * fl

                    if i == len(traj) - 1:
                        flow_loss_last = fl

                    # scorer loss: at final iteration only
                    is_last = (i == len(traj) - 1)
                    if args.patch_selector == SelectionMethod.SCORER and is_last and scores is not None:
                        # kk_close: patch indices for close edges → index into scores.view(-1)
                        ba_w = weight.detach()  # (B, close_edges, 2)
                        scores_clamped = torch.max(scores, torch.as_tensor(1e-6, device='cuda'))
                        conf = (-0.5 * ba_w.view(-1, 2)[valid_mask].mean(dim=-1)).exp()
                        # repeat kk_close for batch dim B, then select valid edges
                        patch_scores = scores.view(-1)[kk_close.repeat(coords_est.shape[0])][valid_mask]
                        scores_loss = (conf * patch_scores * e_flat[valid_mask]).mean()
                        scores_loss = scores_loss + (-scores_clamped.log()).mean()
                        loss = loss + args.scores_weight * scores_loss

                if torch.isnan(loss):
                    print(f"NaN at step {total_steps}")
                    optimizer.zero_grad(set_to_none=True)
                    total_steps += 1
                    scheduler.step()
                    continue

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(net.parameters(), args.clip)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()

                total_steps += 1

                metrics = {
                    "loss/train": loss.item(),
                    "loss/flow_train": flow_loss_last.item(),
                    "loss/scores_train": scores_loss.item() if isinstance(scores_loss, torch.Tensor) else 0.0,
                }

                if rank == 0:
                    pbar.update(args.gpu_num)
                    logger.push(metrics)

                # ── Checkpoint ────────────────────────────────────────────────
                if total_steps % args.save_freq == 0 or total_steps >= args.steps:
                    torch.cuda.empty_cache()
                    if rank == 0:
                        path = f'checkpoints/{args.name}/{args.gpu_num * total_steps:06d}.pth'
                        torch.save({
                            'steps': total_steps,
                            'model_state_dict': net.module.state_dict() if args.ddp else net.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'scheduler_state_dict': scheduler.state_dict(),
                        }, path)

                        if args.eval and val_loader is not None:
                            val_metrics = evaluate_tracker(net, args, total_steps, val_loader)
                            logger.push(val_metrics)

                    torch.cuda.empty_cache()
                    net.train()

                if total_steps >= args.steps:
                    break
            else:
                continue
            break

    if rank == 0:
        logger.close()
    if args.ddp:
        dist.destroy_process_group()


if __name__ == '__main__':
    import configargparse
    parser = configargparse.ArgumentParser()
    parser.add_argument('-c', '--config', default='config/train_tracker_base.conf',
                        is_config_file=True, help='config file path')
    parser.add_argument('--name', '--expname', default='tracker_run')
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--fgraph_pickle', type=str, default='fgraph/TartanAirEVS.pickle')
    parser.add_argument('--datapath', default='')
    parser.add_argument('--train_split', type=str, default='script/splits/tartan/tartan_default_train.txt')
    parser.add_argument('--val_split', type=str, default='script/splits/tartan/tartan_default_val.txt')

    parser.add_argument('--batch', type=int, default=1)
    parser.add_argument('--steps', type=int, default=200000)
    parser.add_argument('--save_freq', type=int, default=10000)
    parser.add_argument('--iters', type=int, default=18)
    parser.add_argument('--lr', type=float, default=0.00008)
    parser.add_argument('--clip', type=float, default=10.0)
    parser.add_argument('--n_frames', type=int, default=15)
    parser.add_argument('--gamma', type=float, default=0.8,
                        help='exponential weight for earlier iterations (RAFT-style)')

    parser.add_argument('--flow_weight', type=float, default=1.0)
    parser.add_argument('--scores_weight', type=float, default=0.05)

    parser.add_argument('--patches_per_image', type=int, default=80)
    parser.add_argument('--patch_selector', type=str, default='scorer')
    parser.add_argument('--corner_guidance', type=str, default='none',
                        help='Multiply scorer map by corner response: none | harris | shitomasi')
    parser.add_argument('--norm', type=str, default='std2')
    parser.add_argument('--randaug', action='store_true')

    parser.add_argument('--eval', action='store_true', default=False)
    parser.add_argument('--eval_seqs', type=int, default=10,
                        help='Max number of val sequences to evaluate at each checkpoint')
    parser.add_argument('--evs', action='store_true', default=True)
    parser.add_argument('--scale', type=float, default=1.0)
    parser.add_argument('--ddp', action='store_true')
    parser.add_argument('--gpu_num', type=int, default=1)
    parser.add_argument('--port', default='12349')
    parser.add_argument('--amp', action='store_true')
    parser.add_argument('--num_workers', type=int, default=4)

    parser.add_argument('--dim_inet', type=int, default=384)
    parser.add_argument('--dim_fnet', type=int, default=128)
    parser.add_argument('--dim', type=int, default=32)
    parser.add_argument('--resnet', action='store_true')
    parser.add_argument('--block_dims', type=str, default='64,128,256')

    args = parser.parse_args()

    if args.ddp:
        mp.spawn(train, nprocs=args.gpu_num, args=(args,))
    else:
        train(0, args)
