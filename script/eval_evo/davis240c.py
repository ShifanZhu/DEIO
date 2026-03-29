import glob
import os
# from multiprocessing import Process, Queue
from pathlib import Path

import cv2
import evo.main_ape as main_ape
import numpy as np
import torch

# 处理服务器中evo的可视化问题
import evo
from evo.tools.settings import SETTINGS
SETTINGS['plot_backend'] = 'Agg'

from evo.core import sync
from evo.core.metrics import PoseRelation
from evo.core.trajectory import PoseTrajectory3D
from evo.tools import file_interface

# import sys
# sys.path.append('/home/gwp/DEIO2')#要导入

from devo.config import cfg # 这里的cfg是从devo.config中导入的
# from dpvo.dpvo import DPVO
# from dpvo.plot_utils import plot_trajectory
# from dpvo.stream import image_stream
from dpvo.utils import Timer

from utils.load_utils import load_gt_us,davis240c_evs_iterator
from utils.eval_utils import log_results,compute_median_results,VO_run,EVO_run,EVO_run_GBA

SKIP = 0

def show_image(image, t=0):
    image = image.permute(1, 2, 0).cpu().numpy()
    cv2.imshow('image', image / 255.0)
    cv2.waitKey(t)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--inputdir', default="datasets") # 数据集的路径
    parser.add_argument('--network', type=str, default='dpvo.pth') # 网络的路径
    parser.add_argument('--val_split', type=str, default="splits") # 验证集的路径,有它来决定验证的序列
    parser.add_argument('--config', default="config/infer_base.yaml")
    parser.add_argument('--stride', type=int, default=1)
    parser.add_argument('--viz', action="store_true")
    parser.add_argument('--enable_event', action="store_true")#是否启用事件,启用了后就不会再使用图像了
    parser.add_argument('--show_img', action="store_true")
    parser.add_argument('--trials', type=int, default=1)
    parser.add_argument('--backend_thresh', type=float, default=64.0) # 用于判断是否使用后端优化的阈值，原本为64.0
    parser.add_argument('--plot', action="store_true")
    parser.add_argument('--opts', nargs='+', default=[])
    parser.add_argument('--save_trajectory', action="store_true")
    parser.add_argument('--side', type=str, default="left")
    args = parser.parse_args()

    cfg.merge_from_file(args.config)
    # cfg.BACKEND_THRESH = args.backend_thresh
    cfg.merge_from_list(args.opts)

    print("\033[42m Running EVO with config...\033[0m ")
    print(cfg, "\n")

    # torch.manual_seed(1234) #反而不利于同时测试多个数据集

    if cfg.LOOP_CLOSURE or cfg.CLASSIC_LOOP_CLOSURE:
        print("\033[41m open the LOOP_CLOSURE with Global BA \033[0m ")

    # 读取场景的名称    
    test_scenes = open(args.val_split).read().split()
    print("the number of scenes is", len(test_scenes),"the input scenes are: ", test_scenes)

    dataset_name = "davis240c"
    if args.enable_event:
        dataset_name += "/EVO"
    else:
        dataset_name += "/VO"

    results_dict_scene, figures = {}, {}
    all_results = []
    for i, scene in enumerate(test_scenes):
        print(f"Eval on {scene}")
        results_dict_scene[scene] = []

        imagedir = os.path.join(args.inputdir, scene, "images_undistorted_left")
        groundtruth = os.path.join(args.inputdir, scene, "gt_stamped_left.txt")
        calibdir = os.path.join(args.inputdir, scene, "calib_undist_left.txt")
        image_timestamps = os.path.join(args.inputdir, scene, "tss_imgs_us_left.txt")

        for trial in range(args.trials):
            print(f"\nRunning trial {trial} of {scene}...")
            
            # 运行dpvo主程序
            if not args.enable_event:
                traj_est, timestamps = VO_run(cfg, args.network, imagedir, calibdir, args.stride, args.viz, args.show_img)

                images_list = sorted(glob.glob(os.path.join(imagedir, "*.png")))[::args.stride]
                # tstamps = [float(x.split('/')[-1][:-4]) for x in images_list]#获取时间戳
                tstamps=np.loadtxt(image_timestamps)#注意时间戳的单位
                # 按照args.stride来取时间戳
                tstamps = tstamps[::args.stride]
            else:

                datapath_val = os.path.join(args.inputdir, scene)
                traj_est, tstamps, flowdata = EVO_run_GBA(datapath_val, cfg, args.network, viz=args.viz, 
                                          iterator=davis240c_evs_iterator(datapath_val, side=args.side, stride=args.stride, timing=False, H=180, W=240),
                                          timing=False, H=180, W=240, viz_flow=False)

            # load  traj（这应该是获取gt trajectory的值,从txt文件中读取）
            tss_traj_us, traj_hf = load_gt_us(groundtruth)

            # do evaluation （进行验证）
            data = (traj_hf, tss_traj_us, traj_est, tstamps)
            hyperparam = (None, args.network, dataset_name, scene, trial, cfg, args)
            # 通过log_results函数来记录结果(用evo评估定位的精度)
            all_results, results_dict_scene, figures, outfolder = log_results(data, hyperparam, all_results, results_dict_scene, figures, 
                                                                   plot=True, save=True, return_figure=False, stride=args.stride,
                                                                   _n_to_align=1000,
                                                                   expname=scene#args.expname
                                                                   )
            

        print(scene, sorted(results_dict_scene[scene]))

    results_dict = compute_median_results(results_dict_scene, all_results, dataset_name)

    for k in results_dict:
        print(k, results_dict[k])

    # print("AVG: ", np.mean(results_dict))

    

    
