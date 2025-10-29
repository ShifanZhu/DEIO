
<!-- # [Website](https://kwanwaipang.github.io/DEIO) -->
[comment]: <> (# DEIO)

<!-- PROJECT LOGO -->

<p align="center">

  <h1 align="center"> DEIO: Deep Event Inertial Odometry
  </h1>

[comment]: <> (  <h2 align="center">PAPER</h2>)
  <h3 align="center">
  <a href="https://arxiv.org/abs/2411.03928">Paper</a> 
  | <a href="https://kwanwaipang.github.io/DEIO">Website</a> 
  | <a href="https://www.youtube.com/watch?v=gs_LLOh3AsQ">Demo</a> 
  </h3>
  
  <!-- <div align="center"></div> -->

<div align="center">
  <img src="./Figs/cover_figure.png" width="90%" />
</div>

  <!-- <br> -->

## Abstract
<!-- <p style="text-align: justify;">  -->
<div align="justify">
Event cameras show great potential for visual odometry (VO) in handling challenging situations, such as fast motion and high dynamic range. Despite this promise, the sparse and motion-dependent characteristics of event data continue to limit the performance of feature-based or direct-based data association methods in practical applications. To address these limitations, we propose Deep Event Inertial Odometry (DEIO), the first monocular learning-based event-inertial framework, which combines a learning-based method with traditional nonlinear graph-based optimization. Specifically, an event-based recurrent network is adopted to provide accurate and sparse associations of event patches over time. DEIO further integrates it with the IMU to recover up-to-scale pose and provide robust state estimation. The Hessian information derived from the learned differentiable bundle adjustment (DBA) is utilized to optimize the co-visibility factor graph, which tightly incorporates event patch correspondences and IMU pre-integration within a keyframe-based sliding window. Comprehensive validations demonstrate that DEIO achieves superior performance on 10 challenging public benchmarks compared with more than 20 state-of-the-art methods.
<!-- </p> -->
</div>

## Update log
- [x] README Upload (2024/10/28)
- [x] Paper Upload (2024/11/06)
- [x] Estimated Trajectories Upload (2024/11/07)
- [x] Code Upload (2025/07/19)
- [x] More Raw Results of VECtor Dataset (2025/07/20) 

## Setup and Installation

```sh
# for cuda 11.7
conda env create -f environment.yml   
conda activate DEIO
# conda remove --name DEIO --all

pip install .
pip install numpy-quaternion==2022.4.3

# install GTSAM
cd thirdparty/gtsam
mkdir build
cd build
cmake .. -DGTSAM_BUILD_PYTHON=1 -DGTSAM_PYTHON_VERSION=3.10.11
make python-install
```

## Run an Example


```sh
conda activate DEIO

# example for the davia240c
CUDA_VISIBLE_DEVICES=0 PYTHONPATH={YOUR_PATH}/DEIO python script/eval_deio/davis240c.py \
    --inputdir=/media/lfl-data2/davis240c \
    --config=config/davis240c.yaml \
    --val_split=script/splits/davis240c/davis240c_val.txt \
    --enable_event \
    --network={YOUR_PATH}/DEVO/DEVO.pth  \
    --plot \
    --save_trajectory \
    --trials=5

```

<!--
For self-used:

CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/home/gwp/DEIO python script/eval_deio/davis240c.py \
    --inputdir=/media/lfl-data2/davis240c \
    --config=config/davis240c.yaml \
    --val_split=script/splits/davis240c/davis240c_val.txt \
    --enable_event \
    --network=/home/gwp/DEVO/DEVO.pth  \
    --plot \
    --save_trajectory \
    --trials=5

CUDA_VISIBLE_DEVICES=1 PYTHONPATH=/home/gwp/DEIO python script/eval_deio/uzh-fpv.py \
    --inputdir=/media/lfl-data2/UZH-FPV \
    --config=config/uzhfpv.yaml \
    --val_split=script/splits/fpv/fpv_val.txt \
    --enable_event \
    --network=/home/gwp/DEVO/DEVO.pth  \
    --plot \
    --save_trajectory \
    --trials=5 

-->

The Results will save in path: `results`.

## Using Our Results as Comparison
<div align="justify">
For the convenience of the comparison, we release the estimated trajectories of DEIO in <code>tum</code> format in the dir of <code>estimated_trajectories</code>.
What's more, we also give the <a href="./estimated_trajectories/evo_evaluation_trajectory.ipynb">sample code</a> for the quantitative and qualitative evaluation using <a href="https://github.com/MichaelGrupp/evo">evo package</a>
</div>
<!-- [sample code](../estimated_trajectories/evo_evaluation_trajectory.ipynb) -->

* [DAVIS240c](./estimated_trajectories/evo_evaluation_davis240c.ipynb)
* [Mono-HKU](./estimated_trajectories/evo_evaluation_monohku.ipynb)
* [Stereo-HKU](./estimated_trajectories/evo_evaluation_stereohku.ipynb)
* [UZH-PFV](./estimated_trajectories/evo_evaluation_fpv.ipynb)
* [VECtor](./estimated_trajectories/evo_evaluation_vector.ipynb)
* [MVSEC](./estimated_trajectories/evo_evaluation_mvsec.ipynb)
* [EDS](./estimated_trajectories/evo_evaluation_eds.ipynb)
* [TUM-VIE](./estimated_trajectories/evo_evaluation_tumvie.ipynb)
* [DSEC](./estimated_trajectories/evo_evaluation_dsec.ipynb)


## Acknowledgement
* This work is based on [DPVO](https://github.com/princeton-vl/DPVO), [DEVO](https://github.com/tum-vision/DEVO), [DROID-SLAM](https://github.com/princeton-vl/DROID-SLAM), [DBA-Fusion](https://github.com/GREAT-WHU/DBA-Fusion), and [GTSAM](https://github.com/borglab/gtsam)
* If you find this work helpful in your research, a simple star or citation of our work should be the best affirmation for us. :blush: 

~~~
@inproceedings{GWPHKU:DEIO,
  title={Deio: Deep event inertial odometry},
  author={Guan, Weipeng and Lin, Fuling and Chen, Peiyu and Lu, Peng},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={4606--4615},
  year={2025}
}
~~~
