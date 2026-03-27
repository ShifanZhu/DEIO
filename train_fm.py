"""Training script for FMNet: Feature-Metric Event Odometry Network.

Usage:
    python train_fm.py --data_path /path/to/data --name my_experiment

Key differences from train.py:
  - Uses DeepEventOdomDataset instead of TartanAir
  - Uses FMNet instead of eVONet
  - Loss = pose_weight * L_pose + feat_weight * L_feat
  - L_feat encourages spatially-smooth, matchable features
"""

import os
import contextlib
from collections import OrderedDict

import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dpvo.lietorch import SE3

from devo.fmnet import FMNet
from devo.data_readers.factory import dataset_factory
from devo.logger import Logger

import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from tqdm import tqdm


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


def kabsch_umeyama(A, B):
    """Compute optimal scale (SIM3) minimising RMSD between two point sets."""
    n, m = A.shape
    EA   = torch.mean(A, axis=0)
    EB   = torch.mean(B, axis=0)
    VarA = torch.mean((A - EA).norm(dim=1) ** 2)
    H    = ((A - EA).T @ (B - EB)) / n
    U, D, VT = torch.svd(H)
    c = VarA / torch.trace(torch.diag(D))
    return c


def pose_loss(P1, P2, ii, jj):
    """Geodesic pose loss between estimated P1 and GT P2.

    Args:
        P1, P2: SE3 objects (B, N, 7)
        ii, jj: frame pair indices (K,)

    Returns:
        tr: (K,) translation error norms
        ro: (K,) rotation error norms
    """
    P1 = P1.inv()
    P2 = P2.inv()

    t1 = P1.matrix()[..., :3, 3]
    t2 = P2.matrix()[..., :3, 3]

    s = kabsch_umeyama(t2[0], t1[0]).detach().clamp(max=10.0)
    P1 = P1.scale(s.view(1, 1))

    dP  = P1[:, ii].inv() * P1[:, jj]
    dG  = P2[:, ii].inv() * P2[:, jj]
    e1  = (dP * dG.inv()).log()
    tr  = e1[..., 0:3].norm(dim=-1)
    ro  = e1[..., 3:6].norm(dim=-1)
    return tr, ro


def train(rank, args):
    """Main training loop."""

    if args.ddp:
        setup_ddp(rank, args)

    # ------------------------------------------------------------------
    # Dataset
    # ------------------------------------------------------------------
    db = dataset_factory(
        ['deep_event_odom'],
        data_path=args.data_path,
        n_keyframes=args.n_keyframes,
        interval_frames=args.interval_frames,
        sub_frame_stride=args.sub_frame_stride,
        d_max=args.d_max,
        depth_k=args.depth_k,
    )

    if args.ddp:
        sampler = torch.utils.data.distributed.DistributedSampler(
            db, shuffle=True, num_replicas=args.gpu_num, rank=rank)
        loader  = DataLoader(db, batch_size=args.batch, sampler=sampler, num_workers=4)
    else:
        loader  = DataLoader(db, batch_size=args.batch, shuffle=True, num_workers=4)

    # ------------------------------------------------------------------
    # Network
    # ------------------------------------------------------------------
    net = FMNet(
        bins=args.bins,
        dim_fnet=args.dim_fnet,
        dim_inet=args.dim_inet,
        dim=args.dim,
        patch_size=3,
        norm=args.norm,
        randaug=args.randaug,
    )
    net.train()
    net.cuda()

    if args.ddp:
        net = DDP(net, device_ids=[rank], find_unused_parameters=False)

    # ------------------------------------------------------------------
    # Optimiser
    # ------------------------------------------------------------------
    optimizer = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-6)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, args.lr, args.steps,
        pct_start=0.01, cycle_momentum=False, anneal_strategy='linear')

    total_steps = 0

    if args.checkpoint:
        print(f"Loading checkpoint: {args.checkpoint}")
        ckpt = torch.load(args.checkpoint)
        model = net.module if args.ddp else net
        if 'model_state_dict' in ckpt:
            model.load_state_dict(ckpt['model_state_dict'])
        else:
            new_sd = OrderedDict({k.replace('module.', ''): v for k, v in ckpt.items()})
            model.load_state_dict(new_sd, strict=False)
        if 'optimizer_state_dict' in ckpt:
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        if 'scheduler_state_dict' in ckpt:
            scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        if 'steps' in ckpt:
            total_steps = ckpt['steps']

    if rank == 0:
        logger = Logger(args.name, scheduler, args.gpu_num * total_steps, args.gpu_num)

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    if rank == 0:
        pbar = tqdm(total=args.gpu_num * args.steps, desc=f"Training {args.name}")

    loader_iter = iter(loader)

    while total_steps < args.steps:
            optimizer.zero_grad()

            loss_accum      = 0.0
            pose_loss_val   = 0.0
            feat_loss_val   = 0.0
            valid_samples   = 0

            for _ in range(args.accum_steps):
                try:
                    batch = next(loader_iter)
                except StopIteration:
                    loader_iter = iter(loader)
                    batch = next(loader_iter)

                # batch keys: images (1,N,K,H,W), disps (1,N,H',W'), poses (1,N,7), intrinsics (1,4)
                images     = batch['images'].cuda().float()      # (1, N, K, H, W)
                disps      = batch['disps'].cuda().float()       # (1, N, H', W')
                poses_np   = batch['poses'].cuda().float()       # (1, N, 7)
                intrinsics = batch['intrinsics'].cuda().float()  # (1, 4)

                # GT poses as SE3 (c2w → w2c like train.py)
                poses_gt = SE3(poses_np).inv()

                # Structure-only for first 1k steps (warm-up)
                no_ckpt = args.checkpoint is None or args.checkpoint == ''
                so = (total_steps < 1000 // args.gpu_num) and no_ckpt

                traj = net(
                    images, poses_gt, disps, intrinsics,
                    STEPS=args.iters,
                    patches_per_image=args.patches_per_image,
                    structure_only=so,
                    gn_iters=args.gn_iters,
                )

                # Build all-pairs frame index for pose loss
                N_used = traj[0][0].shape[1]
                ii_all, jj_all = torch.meshgrid(
                    torch.arange(N_used), torch.arange(N_used), indexing='ij')
                ii_all = ii_all.reshape(-1).cuda()
                jj_all = jj_all.reshape(-1).cuda()
                k      = ii_all != jj_all
                ii_all, jj_all = ii_all[k], jj_all[k]

                loss = 0.0
                for step_idx, (P_pred, P_gt, resids) in enumerate(traj):
                    # ------ Pose loss ------
                    if not so and step_idx >= 2:
                        tr, ro = pose_loss(P_pred, P_gt, ii_all, jj_all)
                        pl = tr.mean() + ro.mean()
                        loss += args.pose_weight * pl
                        pose_loss_val += pl.item()

                    # ------ Feature consistency loss ------
                    fl = (resids ** 2).mean()
                    loss += args.feat_weight * fl
                    feat_loss_val += fl.item()

                feat_loss_val /= max(len(traj), 1)

                if not torch.isnan(loss):
                    (loss / args.accum_steps).backward()
                    loss_accum += loss.item()
                    valid_samples += 1
                else:
                    print(f"NaN loss at step {total_steps}")

            if valid_samples > 0:
                torch.nn.utils.clip_grad_norm_(net.parameters(), args.clip)
                optimizer.step()
                scheduler.step()

            total_steps += 1

            metrics = {
                'loss/train':       loss_accum / max(valid_samples, 1),
                'loss/pose_train':  pose_loss_val / max(valid_samples, 1),
                'loss/feat_train':  feat_loss_val / max(valid_samples, 1),
            }
            if rank == 0:
                pbar.update(args.gpu_num)
                logger.push(metrics)

            # Checkpoint
            if total_steps % 10000 == 0 and rank == 0:
                torch.cuda.empty_cache()
                ckpt_dir = f'checkpoints/{args.name}'
                os.makedirs(ckpt_dir, exist_ok=True)
                PATH = f'{ckpt_dir}/{args.gpu_num * total_steps:06d}.pth'
                torch.save({
                    'steps':                total_steps,
                    'model_state_dict':     (net.module if args.ddp else net).state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                }, PATH)
                net.train()


    if rank == 0:
        logger.close()
    if args.ddp:
        dist.destroy_process_group()


if __name__ == '__main__':
    import configargparse
    parser = configargparse.ArgumentParser(
        default_config_files=['config/fmnet_base.conf'],
        description='Train FMNet (feature-metric event odometry)',
    )
    parser.add_argument('-c', '--config', is_config_file=True,
                        help='config file path (default: config/fmnet_base.conf)')

    # Experiment
    parser.add_argument('--name',       default='fmnet',  help='experiment name')
    parser.add_argument('--checkpoint', default=None,     help='path to checkpoint')

    # Data
    parser.add_argument('--data_path',        required=True,  help='HDF5 dataset root')
    parser.add_argument('--n_keyframes',      type=int,   default=4)
    parser.add_argument('--interval_frames',  type=int,   default=5)
    parser.add_argument('--sub_frame_stride', type=int,   default=1)
    parser.add_argument('--d_max',            type=float, default=100.0)
    parser.add_argument('--depth_k',          type=float, default=1.0)
    parser.add_argument('--bins',             type=int,   default=5,
                        help='event voxel bins (must match dataset K_sub)')

    # Network
    parser.add_argument('--dim_fnet',          type=int,   default=128)
    parser.add_argument('--dim_inet',          type=int,   default=384)
    parser.add_argument('--dim',               type=int,   default=32)
    parser.add_argument('--norm',              default='std2',
                        help='event normalisation: none|rescale|std|std2')
    parser.add_argument('--randaug',           action='store_true')
    parser.add_argument('--patches_per_image', type=int,   default=80)

    # Training
    parser.add_argument('--batch',        type=int,   default=1)
    parser.add_argument('--steps',        type=int,   default=100000)
    parser.add_argument('--iters',        type=int,   default=12,
                        help='GN refinement rounds per forward pass')
    parser.add_argument('--gn_iters',     type=int,   default=4,
                        help='GN iterations per round (split across two pyramid levels)')
    parser.add_argument('--lr',           type=float, default=8e-5)
    parser.add_argument('--clip',         type=float, default=10.0)
    parser.add_argument('--pose_weight',  type=float, default=10.0)
    parser.add_argument('--feat_weight',  type=float, default=0.1,
                        help='weight for feature-consistency loss (encourages gradient)')
    parser.add_argument('--accum_steps', type=int, default=1,
                        help='gradient accumulation (effective batch = accum_steps)')

    # Multi-GPU
    parser.add_argument('--ddp',      action='store_true')
    parser.add_argument('--gpu_num',  type=int, default=1)
    parser.add_argument('--port',     default='12349')

    args = parser.parse_args()
    args.steps = args.steps // args.gpu_num

    print("----------")
    print("Config:")
    print(args)
    print("----------")
    print(parser.format_values())
    print("----------")

    os.makedirs(f'checkpoints/{args.name}', exist_ok=True)

    if args.ddp:
        mp.spawn(train, nprocs=args.gpu_num, args=(args,))
    else:
        train(0, args)
