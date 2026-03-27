import glob
import os
# from multiprocessing import Process, Queue
from pathlib import Path

import cv2
import evo.main_ape as main_ape
import numpy as np
import torch
import quaternion
import math

# Handle visualization issues with evo on the server
import evo
from evo.tools.settings import SETTINGS
SETTINGS['plot_backend'] = 'Agg'

from evo.core import sync
from evo.core.metrics import PoseRelation
from evo.core.trajectory import PoseTrajectory3D
from evo.tools import file_interface

from devo.config import cfg # config file imported
# from dpvo.utils import Timer

from utils.load_utils import load_gt_us,davis240c_evs_iterator, davis240c_evs_iterator
from utils.eval_utils import log_results,compute_median_results,VO_run,EVO_run,EVO_run_GBA,run_DEIO2

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--inputdir', default="datasets") # Path to the dataset
    parser.add_argument('--network', type=str, default='dpvo.pth') # Path to the network model
    parser.add_argument('--val_split', type=str, default="splits") # Path to the validation split, which determines the validation sequences
    parser.add_argument('--config', default="config/***.yaml")
    parser.add_argument('--stride', type=int, default=1)
    parser.add_argument('--viz', action="store_true")
    parser.add_argument('--enable_event', action="store_true") # Whether to enable events; if enabled, images will no longer be used
    parser.add_argument('--show_img', action="store_true")
    parser.add_argument('--trials', type=int, default=1)
    parser.add_argument('--backend_thresh', type=float, default=64.0) # Threshold for determining whether to use backend optimization; originally 64.0
    parser.add_argument('--plot', action="store_true")
    parser.add_argument('--opts', nargs='+', default=[])
    parser.add_argument('--save_trajectory', action="store_true")
    parser.add_argument('--side', type=str, default="left")
    parser.add_argument('--timing', action="store_true")

    parser.add_argument('--resnet', action='store_true', help='use the ResNet backbone')
    parser.add_argument('--block_dims', type=str, default="64,128,256", help='channel dimensions of ResNet blocks')
    parser.add_argument('--initial_dim', type=int, default=64, help='initial channel dimension of ResNet')
    parser.add_argument('--pretrain', type=str, default="resnet18", help='pretrained ResNet model (resnet18, resnet34)')

    args = parser.parse_args()

    cfg.merge_from_file(args.config)
    # cfg.BACKEND_THRESH = args.backend_thresh # Threshold for determining whether to use backend optimization; passed directly via the parameter file
    cfg.merge_from_list(args.opts)

    # Construct override list, mapping four parameters to cfg keys
    # Directly assign args parameters to top-level properties of cfg
    cfg.resnet = args.resnet
    cfg.block_dims = list(map(int, args.block_dims.split(',')))  # Convert to list of integers
    cfg.initial_dim = args.initial_dim
    cfg.pretrain = args.pretrain

    print("\033[42m Running EVO with config...\033[0m ")
    print(cfg, "\n")

    # torch.manual_seed(1234) # Conversely, it is not conducive to testing multiple datasets simultaneously

    # Do not enable cfg.CLASSIC_LOOP_CLOSURE for now
    assert not cfg.CLASSIC_LOOP_CLOSURE # cfg.CLASSIC_LOOP_CLOSURE uses traditional methods for loop closure detection; it is not needed now, only camera proximity method is used
    if cfg.LOOP_CLOSURE:
        print("\033[41m with Global BA \033[0m ")
    else:
        print("\033[41m no Global BA \033[0m ")

    # Read the names of the scenes    
    test_scenes = open(args.val_split).read().split()
    print("the number of scenes is", len(test_scenes),"the input scenes are: ", test_scenes)

    dataset_name = "davis240c"
    if args.enable_event:
        dataset_name += "/EVO"
    else:
        dataset_name += "/VO"
    
    if cfg.LOOP_CLOSURE:
        dataset_name += "_GBA" # If loop closure detection is enabled, append _GBA
    
    if cfg.ENALBE_IMU:
        dataset_name += "_IMU" # If IMU is also enabled, append _IMU

    results_dict_scene, figures = {}, {}
    all_results = []
    for i, scene in enumerate(test_scenes):
        print(f"Eval on {scene}")
        results_dict_scene[scene] = []

        groundtruth = os.path.join(args.inputdir, scene, f"gt_stamped_left.txt") # Path to ground truth; note that this ground truth has a time offset
        imupath = os.path.join(args.inputdir, scene, f"imu_data.csv") # Path to IMU data

        for trial in range(args.trials):
            print(f"\nRunning trial {trial} of {scene}...")
            
            # Run the DPVO main program
            if not args.enable_event:
                print("This code is for event rather than image")
                raise NotImplementedError

            datapath_val = os.path.join(args.inputdir, scene)
            # load traj (this should be getting the GT trajectory values, read from the txt file)
            tss_traj_us, traj_hf = load_gt_us(groundtruth) # Pre-fetch ground truth timestamps and positions

            if cfg.LOOP_CLOSURE and not cfg.ENALBE_IMU: # If loop closure detection is enabled but IMU is not
                # traj_est, tstamps, flowdata, avg_fps = EVO_run_GBA(datapath_val, cfg, args.network, viz=args.viz, 
                #                         iterator=davis240c_evs_iterator(datapath_val, side=args.side, stride=args.stride, timing=False, H=180, W=240),
                #                         timing=args.timing, H=180, W=240, viz_flow=False)
                raise NotImplementedError("No IMU, please check the config file")
            # elif cfg.LOOP_CLOSURE and cfg.ENALBE_IMU: # If loop closure detection is enabled and IMU is also enabled
            elif cfg.ENALBE_IMU: # If IMU is enabled (can run with or without loop closure)
                """ Load GT trajectory (for visualization and VI intilization) """
                # all_gt_keys=tss_traj_us # Timestamps for all ground truth
                # #all_gt is timestamps (tss_traj_us) + poses (traj_hf) for all ground truth
                # all_gt = np.concatenate((all_gt_keys, traj_hf), axis=1)
                all_gt = {} # Stores timestamps + poses of ground truth
                # Iterate through each timestamp and corresponding trajectory data
                for sod, data in zip(tss_traj_us, traj_hf):
                    # sod is the timestamp in us; convert us to seconds
                    sod = float(sod / 1e6)
                    if sod not in all_gt: # If sod is not in all_gt, initialize an empty dictionary
                        all_gt[sod] = {}
                    
                    # Extract position (x, y, z)
                    x = data[0]
                    y = data[1]
                    z = data[2]
                    
                    # Extract quaternion components (qx, qy, qz, qw); note that GT is stored this way
                    # Note: Adjust indices according to the actual quaternion order in the data
                    qx = data[3]
                    qy = data[4]
                    qz = data[5]
                    qw = data[6]
                    
                    # Construct quaternion object and convert to rotation matrix
                    q = quaternion.from_float_array([float(qw), float(qx), float(qy), float(qz)])  # Note whether the quaternion order is (w, x, y, z)
                    R = quaternion.as_rotation_matrix(q)
                    
                    # Construct 4x4 transformation matrix
                    TTT = np.eye(4)
                    TTT[0:3, 0:3] = R
                    TTT[0:3, 3] = [float(x), float(y), float(z)]
                    
                    all_gt[sod]['T'] = TTT

                # Sort timestamps * (note that this ground truth has a time offset)
                all_gt_keys = sorted(all_gt.keys()) # Store ground truth timestamps; note this is in seconds
                assert np.all(all_gt_keys==tss_traj_us / 1e6)

                # t_offset_us = np.loadtxt(os.path.join(args.inputdir, scene, "t0_us.txt"))#读取时间偏移量
                raw_tss_imgs_ns = np.loadtxt(os.path.join(args.inputdir, scene, f"raw_tss_imgs_ns_left.txt"))#绝对时间戳
                raw_tss_imgs_us=raw_tss_imgs_ns/1e3#转换为微妙(us)
                tss_imgs_us = np.loadtxt(os.path.join(args.inputdir, scene, f"tss_imgs_us_left.txt"))#图像的时间戳(相对时间戳)
                t_offset_us=raw_tss_imgs_us[0]-tss_imgs_us[0]#第一帧的时间戳
                # assert t0_us == t_offset_us

                """ Load IMU data """
                all_imu = np.loadtxt(imupath,delimiter=',')#去掉第0列序号
                #将IMU的时间戳转换为微妙
                all_imu[:,0] /= 1e3 #读入的imu时间是纳秒，转换为微秒
                #将IMU的时间减去偏移量（这样可以跟图像的时间对齐，因为图像也有时间偏移量）
                all_imu[:,0] -= t_offset_us
                # 将时间小于0的数据去掉
                all_imu = all_imu[all_imu[:,0]>0]
                #将IMU的角速度转换为弧度
                all_imu[:,1:4] *= 180/math.pi
                # 将IMU根据时间戳进行排序，以此避免时间戳不连续的问题
                # 按照时间戳（第0列）排序
                all_imu = all_imu[all_imu[:, 0].argsort()]

                traj_est, tstamps, flowdata, avg_fps = run_DEIO2(datapath_val, cfg, args.network, viz=args.viz, 
                                        iterator=davis240c_evs_iterator(datapath_val, side=args.side, stride=args.stride, timing=False, H=180, W=240),
                                        _all_imu=all_imu,
                                        _all_gt=all_gt,
                                        _all_gt_keys=all_gt_keys,
                                        timing=args.timing, H=180, W=240, viz_flow=False)

            else:
                # 报错
                # raise NotImplementedError("No loop closure and no IMU, please check the config file")
                traj_est, tstamps, flowdata, avg_fps = EVO_run(datapath_val, cfg, args.network, viz=args.viz, 
                                        iterator=davis240c_evs_iterator(datapath_val, side=args.side, stride=args.stride, timing=False, H=180, W=240),
                                        timing=args.timing, H=180, W=240, viz_flow=False)

            # do evaluation （进行验证）
            data = (traj_hf, tss_traj_us, traj_est, tstamps)
            hyperparam = (None, args.network, dataset_name, scene, trial, cfg, args)
            # 通过log_results函数来记录结果(用evo评估定位的精度)
            all_results, results_dict_scene, figures, outfolder = log_results(data, hyperparam, all_results, results_dict_scene, figures, 
                                                                   plot=True, save=True, return_figure=False, stride=args.stride,
                                                                   expname=scene,
                                                                   _n_to_align=1000,
                                                                   avg_fps=avg_fps
                                                                   )
            
            gwp_debug=1;            

        print(scene, sorted(results_dict_scene[scene]))
    
    results_dict = compute_median_results(results_dict_scene, all_results, dataset_name,outfolder=outfolder)

    for k in results_dict:
        print(k, results_dict[k])

    print("Done!")

    

    
