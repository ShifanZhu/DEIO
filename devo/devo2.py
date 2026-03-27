import torch
import numpy as np
import torch.nn.functional as F

from dpvo import fastba
from dpvo import altcorr
from dpvo import lietorch
from dpvo.lietorch import SE3
from dpvo.patchgraph import PatchGraph

# from .net import VONet # TODO add net.py
from .enet import eVONet
from .utils import *
from . import projective_ops as pops

autocast = torch.cuda.amp.autocast
Id = SE3.Identity(1, device="cuda")

from utils.viz_utils import visualize_voxel


class DEVO:
    def __init__(self, cfg, network, evs=False, ht=480, wd=640, viz=False, viz_flow=False, dim_inet=384, dim_fnet=128, dim=32):
        self.cfg = cfg
        self.evs = evs # 是否使用事件

        self.dim_inet = dim_inet
        self.dim_fnet = dim_fnet
        self.dim = dim
        # # TODO add patch_selector

        self.args = cfg#参数文件传入
        
        self.load_weights(network)#读取网络的权重
        self.is_initialized = False
        self.enable_timing = False # TODO timing in param

        torch.set_num_threads(2)

        self.viz_flow = viz_flow
        
        self.M = self.cfg.PATCHES_PER_FRAME     # (default: 96) patch的数目
        self.N = self.cfg.BUFFER_SIZE           # max number of keyframes (default: 2048)，buffer的数目

        self.ht = ht    # image height
        self.wd = wd    # image width

        RES = self.RES

        ### state attributes ###
        self.tlist = []
        self.counter = 0 # how often this network is called __call__()

        # keep track of global-BA calls
        self.ran_global_ba = np.zeros(100000, dtype=bool)

        self.flow_data = {}

        # dummy image for visualization
        self.image_ = torch.zeros(self.ht, self.wd, 3, dtype=torch.uint8, device="cpu")#原图的大小

        self.patches_gt_ = torch.zeros(self.N, self.M, 3, self.P, self.P, dtype=torch.float, device="cuda")

        ### network attributes ###
        if self.cfg.MIXED_PRECISION:
            self.kwargs = kwargs = {"device": "cuda", "dtype": torch.half}
        else:
            self.kwargs = kwargs = {"device": "cuda", "dtype": torch.float}

        ### frame memory size ###
        self.pmem = self.mem = 36 #32
        if self.cfg.LOOP_CLOSURE:
            self.last_global_ba = -1000 # keep track of time since last global opt
            self.pmem = self.cfg.MAX_EDGE_AGE # patch memory
        
        self.imap_ = torch.zeros(self.pmem, self.M, self.dim_inet, **kwargs)
        self.gmap_ = torch.zeros(self.pmem, self.M, self.dim_fnet, self.P, self.P, **kwargs)

        ht = int(ht // RES)
        wd = int(wd // RES)

        # 定义一个PatchGraph类的实例
        self.pg = PatchGraph(self.cfg, self.P, self.dim_inet, self.pmem, **kwargs)

        # classic backend
        if self.cfg.CLASSIC_LOOP_CLOSURE:
            self.load_long_term_loop_closure()

        self.fmap1_ = torch.zeros(1, self.mem, self.dim_fnet, int(ht // 1), int(wd // 1), **kwargs)
        self.fmap2_ = torch.zeros(1, self.mem, self.dim_fnet, int(ht // 4), int(wd // 4), **kwargs)

        # feature pyramid
        self.pyramid = (self.fmap1_, self.fmap2_)

        self.viewer = None
        if viz:
            self.start_viewer()
    
    def load_long_term_loop_closure(self):
        try:
            from dpvo.loop_closure.long_term import LongTermLoopClosure
            self.long_term_lc = LongTermLoopClosure(self.cfg, self.pg)#初始化LongTermLoopClosure类
        except ModuleNotFoundError as e:
            self.cfg.CLASSIC_LOOP_CLOSURE = False
            print(f"WARNING: {e}")

    def load_weights(self, network):
        # load network from checkpoint file
        if isinstance(network, str):
            print(f"Loading from {network}")
            checkpoint = torch.load(network, weights_only=False)
            # TODO infer dim_inet=self.dim_inet, dim_fnet=self.dim_fnet, dim=self.dim
            self.network = eVONet(self.args,dim_inet=self.dim_inet, dim_fnet=self.dim_fnet, dim=self.dim, patch_selector=self.cfg.PATCH_SELECTOR)
            if 'model_state_dict' in checkpoint:
                self.network.load_state_dict(checkpoint['model_state_dict'])
            else:
                # legacy
                from collections import OrderedDict
                new_state_dict = OrderedDict()
                for k, v in checkpoint.items():
                    if "update.lmbda" not in k:
                        new_state_dict[k.replace('module.', '')] = v
                self.network.load_state_dict(new_state_dict)

        else:
            self.network = network

        # steal network attributes
        self.dim_inet = self.network.dim_inet
        self.dim_fnet = self.network.dim_fnet
        self.dim = self.network.dim
        self.RES = self.network.RES
        self.P = self.network.P

        self.network.cuda()
        self.network.eval()

        # if self.cfg.MIXED_PRECISION:
        #     self.network.half()


    def start_viewer(self):
        from dpviewer import Viewer

        intrinsics_ = torch.zeros(1, 4, dtype=torch.float32, device="cuda")

        self.viewer = Viewer(
            self.image_,
            self.poses_,
            self.points_,
            self.colors_,
            intrinsics_)

    @property
    def poses(self):
        return self.pg.poses_.view(1, self.N, 7)

    @property
    def patches(self):
        return self.pg.patches_.view(1, self.N*self.M, 3, 3, 3)

    @property
    def intrinsics(self):
        return self.pg.intrinsics_.view(1, self.N, 4)

    @property
    def ix(self):
        return self.pg.index_.view(-1)

    @property
    def imap(self):
        return self.imap_.view(1, self.pmem * self.M, self.dim_inet)

    @property
    def gmap(self):
        return self.gmap_.view(1, self.pmem * self.M, self.dim_fnet, 3, 3)

    @property
    def n(self):
        return self.pg.n

    @n.setter
    def n(self, val):
        self.pg.n = val

    @property
    def m(self):
        return self.pg.m

    @m.setter
    def m(self, val):
        self.pg.m = val
    
    @property
    def patches_gt(self):
        return self.patches_gt_.view(1, self.N*self.M, 3, 3, 3)

    def get_pose(self, t):
        if t in self.traj:
            return SE3(self.traj[t])

        t0, dP = self.pg.delta[t]
        return dP * self.get_pose(t0)

    def terminate(self):

        if self.cfg.CLASSIC_LOOP_CLOSURE:
            self.long_term_lc.terminate(self.n)

        if self.cfg.LOOP_CLOSURE:
            self.append_factors(*self.pg.edges_loop())

        for _ in range(12):
            self.ran_global_ba[self.n] = False
            self.update()

        """ interpolate missing poses """
        print("keyframes", self.n)
        self.traj = {}
        for i in range(self.n):
            self.traj[self.pg.tstamps_[i]] = self.pg.poses_[i]

        poses = [self.get_pose(t) for t in range(self.counter)]
        poses = lietorch.stack(poses, dim=0)
        poses = poses.inv().data.cpu().numpy()
        tstamps = np.array(self.tlist, dtype=np.float64)
        if self.viewer is not None:
            self.viewer.join()

        return poses, tstamps
    
    def corr(self, coords, indicies=None):
        """ local correlation volume """
        ii, jj = indicies if indicies is not None else (self.pg.kk, self.pg.jj)
        ii1 = ii % (self.M * self.pmem)
        jj1 = jj % (self.mem)
        corr1 = altcorr.corr(self.gmap, self.pyramid[0], coords / 1, ii1, jj1, 3)
        corr2 = altcorr.corr(self.gmap, self.pyramid[1], coords / 4, ii1, jj1, 3)
        return torch.stack([corr1, corr2], -1).view(1, len(ii), -1)

    def reproject(self, indicies=None):
        """ reproject patch k from i -> j """
        (ii, jj, kk) = indicies if indicies is not None else (self.pg.ii, self.pg.jj, self.pg.kk)
        coords = pops.transform(SE3(self.poses), self.patches, self.intrinsics, ii, jj, kk)
        return coords.permute(0, 1, 4, 2, 3).contiguous()

    def append_factors(self, ii, jj):
        self.pg.jj = torch.cat([self.pg.jj, jj])
        self.pg.kk = torch.cat([self.pg.kk, ii])#插入的ii其实就是patch的索引，kk
        self.pg.ii = torch.cat([self.pg.ii, self.ix[ii]]) #self.ix[ii]也就是self.ix[kk]才是ii的索引

        net = torch.zeros(1, len(ii), self.dim_inet, **self.kwargs)
        self.pg.net = torch.cat([self.pg.net, net], dim=1)

    def remove_factors(self, m, store: bool):
        assert self.pg.ii.numel() == self.pg.weight.shape[1]
        if store:#如果store为True，则将要删除的边存储到inactive edge中
            self.pg.ii_inac = torch.cat((self.pg.ii_inac, self.pg.ii[m]))
            self.pg.jj_inac = torch.cat((self.pg.jj_inac, self.pg.jj[m]))
            self.pg.kk_inac = torch.cat((self.pg.kk_inac, self.pg.kk[m]))
            self.pg.weight_inac = torch.cat((self.pg.weight_inac, self.pg.weight[:,m]), dim=1)
            self.pg.target_inac = torch.cat((self.pg.target_inac, self.pg.target[:,m]), dim=1)
        self.pg.weight = self.pg.weight[:,~m]
        self.pg.target = self.pg.target[:,~m]

        self.pg.ii = self.pg.ii[~m]
        self.pg.jj = self.pg.jj[~m]
        self.pg.kk = self.pg.kk[~m]
        self.pg.net = self.pg.net[:,~m]
        assert self.pg.ii.numel() == self.pg.weight.shape[1]

    def motion_probe(self):
        """ kinda hacky way to ensure enough motion for initialization """
        kk = torch.arange(self.m-self.M, self.m, device="cuda")
        jj = self.n * torch.ones_like(kk)
        ii = self.ix[kk]

        net = torch.zeros(1, len(ii), self.dim_inet, **self.kwargs)
        coords = self.reproject(indicies=(ii, jj, kk))

        with autocast(enabled=self.cfg.MIXED_PRECISION):
            corr = self.corr(coords, indicies=(kk, jj))
            ctx = self.imap[:,kk % (self.M * self.pmem)]
            net, (delta, weight, _) = \
                self.network.update(net, ctx, corr, None, ii, jj, kk)

        return torch.quantile(delta.norm(dim=-1).float(), 0.5)

    def motionmag(self, i, j):
        k = (self.pg.ii == i) & (self.pg.jj == j)
        ii = self.pg.ii[k]
        jj = self.pg.jj[k]
        kk = self.pg.kk[k]

        # flow, _ = pops.flow_mag(SE3(self.poses), self.patches, self.intrinsics, ii, jj, kk, beta=0.5)
        flow = pops.flow_mag(SE3(self.poses), self.patches, self.intrinsics, ii, jj, kk, beta=0.5)
        return flow.mean().item()

    def keyframe(self):
        
        i = self.n - self.cfg.KEYFRAME_INDEX - 1
        j = self.n - self.cfg.KEYFRAME_INDEX + 1
        m = self.motionmag(i, j) + self.motionmag(j, i)

        # print(f'the mition between {i} and {j} is {m/2}')

        if m / 2 < self.cfg.KEYFRAME_THRESH:
            k = self.n - self.cfg.KEYFRAME_INDEX
            t0 = self.pg.tstamps_[k-1]
            t1 = self.pg.tstamps_[k]

            dP = SE3(self.pg.poses_[k]) * SE3(self.pg.poses_[k-1]).inv()
            self.pg.delta[t1] = (t0, dP)

            to_remove = (self.pg.ii == k) | (self.pg.jj == k)
            self.remove_factors(to_remove, store=False)#此处是不会存的，因为运动不够，不是关键帧

            self.pg.kk[self.pg.ii > k] -= self.M
            self.pg.ii[self.pg.ii > k] -= 1
            self.pg.jj[self.pg.jj > k] -= 1

            for i in range(k, self.n-1):
                self.pg.tstamps_[i] = self.pg.tstamps_[i+1]
                self.pg.colors_[i] = self.pg.colors_[i+1]
                self.pg.poses_[i] = self.pg.poses_[i+1]
                self.pg.patches_[i] = self.pg.patches_[i+1]
                self.patches_gt_[i] = self.patches_gt_[i+1]
                self.pg.intrinsics_[i] = self.pg.intrinsics_[i+1]

                self.imap_[i % self.pmem] = self.imap_[(i+1) % self.pmem]
                self.gmap_[i % self.pmem] = self.gmap_[(i+1) % self.pmem]
                self.fmap1_[0,i%self.mem] = self.fmap1_[0,(i+1)%self.mem]
                self.fmap2_[0,i%self.mem] = self.fmap2_[0,(i+1)%self.mem]

            self.n -= 1
            self.m-= self.M

            if self.cfg.CLASSIC_LOOP_CLOSURE:
                self.long_term_lc.keyframe(k)

        to_remove = self.ix[self.pg.kk] < self.n - self.cfg.REMOVAL_WINDOW # Remove edges falling outside the optimization window
        if self.cfg.LOOP_CLOSURE:
            # ...unless they are being used for loop closure
            lc_edges = ((self.pg.jj - self.pg.ii) > 30) & (self.pg.jj > (self.n - self.cfg.OPTIMIZATION_WINDOW))
            to_remove = to_remove & ~lc_edges
        self.remove_factors(to_remove, store=True) #此处则是要存的，因为是关键帧，只是滑动出了窗口

    # 全局的BA优化
    def __run_global_BA(self):
        """ Global bundle adjustment
         Includes both active and inactive edges """
        full_target = torch.cat((self.pg.target_inac, self.pg.target), dim=1)
        full_weight = torch.cat((self.pg.weight_inac, self.pg.weight), dim=1)
        full_ii = torch.cat((self.pg.ii_inac, self.pg.ii))
        full_jj = torch.cat((self.pg.jj_inac, self.pg.jj))
        full_kk = torch.cat((self.pg.kk_inac, self.pg.kk))

        self.pg.normalize()#! 归一化,目的是？
        lmbda = torch.as_tensor([1e-4], device="cuda") #给定值，不像droid那样需要计算
        t0 = self.pg.ii.min().item()
        # 似乎只是加入了全局的边，和targer weight等信息，然后进行全局的BA优化，并无太大区别？
        # 主要区别应该就是前面用的eff_impl=False，这里用的是True
        fastba.BA(self.poses, self.patches, self.intrinsics,
            full_target, full_weight, lmbda, full_ii, full_jj, full_kk, t0, self.n, M=self.M, iterations=2, eff_impl=True)
        self.ran_global_ba[self.n] = True #标记当前帧已经运行过全局BA优化

    def update(self):
        coords = self.reproject()

        with autocast(enabled=True):
            corr = self.corr(coords) #计算相关性，获取当前帧与上一帧之间的特征匹配信息。
            ctx = self.imap[:, self.pg.kk % (self.M * self.pmem)]
            self.pg.net, (delta, weight, _) = \
                self.network.update(self.pg.net, ctx, corr, None, self.pg.ii, self.pg.jj, self.pg.kk)

        lmbda = torch.as_tensor([1e-4], device="cuda")
        weight = weight.float()
        target = coords[...,self.P//2,self.P//2] + delta.float()

        self.pg.target = target
        self.pg.weight = weight

        # Bundle adjustment进行BA优化
        with Timer("BA", enabled=self.enable_timing):
            t0 = self.n - self.cfg.OPTIMIZATION_WINDOW if self.is_initialized else 1
            t0 = max(t0, 1)

            try:
                #运行全局BA优化
                # run global bundle adjustment if there exist long-range edges
                if (self.pg.ii < self.n - self.cfg.REMOVAL_WINDOW - 1).any() and not self.ran_global_ba[self.n]:
                    # 如果ii中有小于n-REMOVAL_WINDOW-1的值（就是有回环匹配了），且当前帧没有运行过全局BA优化，则运行全局BA优化
                    self.__run_global_BA()
                else:#运行局部BA优化
                    t0 = self.n - self.cfg.OPTIMIZATION_WINDOW if self.is_initialized else 1
                    t0 = max(t0, 1)
                    fastba.BA(self.poses, self.patches, self.intrinsics, 
                        target, weight, lmbda, self.pg.ii, self.pg.jj, self.pg.kk, t0, self.n, M=self.M, iterations=2, eff_impl=False)
            except:
                print("Warning BA failed...")
            
            points = pops.point_cloud(SE3(self.poses), self.patches[:, :self.m], self.intrinsics, self.ix[:self.m])
            points = (points[...,1,1,:3] / points[...,1,1,3:]).reshape(-1, 3)
            self.pg.points_[:len(points)] = points[:]

    def flow_viz_step(self):
        # [DEBUG]
        # dij = (self.ii - self.jj).abs()
        # assert (dij==0).sum().item() == len(torch.unique(self.kk)) 
        # [DEBUG]

        coords_est = pops.transform(SE3(self.poses), self.patches, self.intrinsics, self.ii, self.jj, self.kk) # p_ij (B,close_edges,P,P,2)
        self.flow_data[self.counter-1] = {"ii": self.ii, "jj": self.jj, "kk": self.kk,\
                                          "coords_est": coords_est, "img": self.image_, "n": self.n}

        # import matplotlib.pyplot as plt
        # plt.figure()
        # plt.imshow(self.image_)
        # plt.show()
                
    def __edges_all(self):
        return flatmeshgrid(
            torch.arange(0, self.m, device="cuda"),
            torch.arange(0, self.n, device="cuda"), indexing='ij')

    def __edges_forw(self):
        r=self.cfg.PATCH_LIFETIME  # default: 13
        t0 = self.M * max((self.n - r), 0)
        t1 = self.M * max((self.n - 1), 0)
        return flatmeshgrid(
            torch.arange(t0, t1, device="cuda"),
            torch.arange(self.n-1, self.n, device="cuda"), indexing='ij')

    def __edges_back(self):
        r=self.cfg.PATCH_LIFETIME  # default: 13
        t0 = self.M * max((self.n - 1), 0)
        t1 = self.M * max((self.n - 0), 0)
        return flatmeshgrid(
            torch.arange(t0, t1, device="cuda"),
            torch.arange(max(self.n-r, 0), self.n, device="cuda"), indexing='ij')

    def __call__(self, tstamp, image, intrinsics, scale=1.0):
        """ track new frame """

        if self.cfg.CLASSIC_LOOP_CLOSURE:#如果开启了经典的闭环检测
            self.long_term_lc(image, self.n)

        if (self.n+1) >= self.N:
            raise Exception(f'The buffer size is too small. You can increase it using "--buffer {self.N*2}"')

        if self.viewer is not None:
            self.viewer.update_image(image)

        if self.viz_flow:
            self.image_ = image.detach().cpu().permute((1, 2, 0)).numpy()

        if not self.evs:#如果不使用事件,就是正常的图像归一化操作
            image = 2 * (image[None,None] / 255.0) - 0.5 
        else:
            image = image[None,None]
            
            if self.n == 0:
                nonzero_ev = (image != 0.0)
                zero_ev = ~nonzero_ev
                num_nonzeros = nonzero_ev.sum().item()
                num_zeros = zero_ev.sum().item()
                # [DEBUG]
                # print("nonzero-zero-ratio", num_nonzeros, num_zeros, num_nonzeros / (num_zeros + num_nonzeros))
                if num_nonzeros / (num_zeros + num_nonzeros) < 2e-2: # TODO eval hyperparam (add to config.py)
                    print(f"skip voxel at {tstamp} due to lack of events!")
                    return

            b, n, v, h, w = image.shape
            flatten_image = image.view(b,n,-1)
            
            if self.cfg.NORM.lower() == 'none':
                pass
            elif self.cfg.NORM.lower() == 'rescale' or self.cfg.NORM.lower() == 'norm':
                # Normalize (rescaling) neg events into [-1,0) and pos events into (0,1] sequence-wise
                # Preserve pos-neg inequality (quantity only)
                pos = flatten_image > 0.0
                neg = flatten_image < 0.0
                vx_max = torch.Tensor([1]).to("cuda") if pos.sum().item() == 0 else flatten_image[pos].max(dim=-1, keepdim=True)[0]
                vx_min = torch.Tensor([1]).to("cuda") if neg.sum().item() == 0 else flatten_image[neg].min(dim=-1, keepdim=True)[0]
                # [DEBUG]
                # print("vx_max", vx_max.item())
                # print("vx_min", vx_min.item())
                if vx_min.item() == 0.0 or vx_max.item() == 0.0:
                    # no information for at least one polarity
                    print(f"empty voxel at {tstamp}!")
                    return
                flatten_image[pos] = flatten_image[pos] / vx_max
                flatten_image[neg] = flatten_image[neg] / -vx_min
            elif self.cfg.NORM.lower() == 'standard' or self.cfg.NORM.lower() == 'std':
                # Data standardization of events only
                # Does not preserve pos-neg inequality
                # see https://github.com/uzh-rpg/rpg_e2depth/blob/master/utils/event_tensor_utils.py#L52
                nonzero_ev = (flatten_image != 0.0)
                num_nonzeros = nonzero_ev.sum(dim=-1)
                if torch.all(num_nonzeros > 0):
                    # compute mean and stddev of the **nonzero** elements of the event tensor
                    # we do not use PyTorch's default mean() and std() functions since it's faster
                    # to compute it by hand than applying those funcs to a masked array

                    mean = torch.sum(flatten_image, dim=-1, dtype=torch.float32) / num_nonzeros  # force torch.float32 to prevent overflows when using 16-bit precision
                    stddev = torch.sqrt(torch.sum(flatten_image ** 2, dim=-1, dtype=torch.float32) / num_nonzeros - mean ** 2)
                    mask = nonzero_ev.type_as(flatten_image)
                    flatten_image = mask * (flatten_image - mean[...,None]) / stddev[...,None]
            else:
                print(f"{self.cfg.NORM} not implemented")
                raise NotImplementedError

            image = flatten_image.view(b,n,v,h,w)

        if image.shape[-1] == 346:
            image = image[..., 1:-1] # hack for MVSEC, FPV,...

        # TODO patches with depth is available (val)
        with autocast(enabled=self.cfg.MIXED_PRECISION):
            fmap, gmap, imap, patches, _, clr = \
                self.network.patchify(image,
                    patches_per_image=self.cfg.PATCHES_PER_FRAME, 
                    return_color=True,
                    scorer_eval_mode=self.cfg.SCORER_EVAL_MODE,
                    scorer_eval_use_grid=self.cfg.SCORER_EVAL_USE_GRID)

        self.patches_gt_[self.n] = patches.clone()

        ### update state attributes ###
        self.tlist.append(tstamp)
        self.pg.tstamps_[self.n] = self.counter#也只是数字
        self.pg.intrinsics_[self.n] = intrinsics / self.RES
        
        # color info for visualization
        if not self.evs:
            clr = (clr[0,:,[2,1,0]] + 0.5) * (255.0 / 2)
            self.pg.colors_[self.n] = clr.to(torch.uint8)
        else:
            clr = (clr[0,:,[0,0,0]] + 0.5) * (255.0 / 2)
            self.pg.colors_[self.n] = clr.to(torch.uint8)
            

        self.pg.index_[self.n + 1] = self.n + 1
        self.pg.index_map_[self.n + 1] = self.m + self.M 

        if self.n > 1:
            if self.cfg.MOTION_MODEL == 'DAMPED_LINEAR':
                P1 = SE3(self.pg.poses_[self.n-1])
                P2 = SE3(self.pg.poses_[self.n-2])

                # To deal with varying camera hz
                *_, a,b,c = [1]*3 + self.tlist
                fac = (c-b) / (b-a)

                xi = self.cfg.MOTION_DAMPING * fac * (P1 * P2.inv()).log()
                tvec_qvec = (SE3.exp(xi) * P1).data
                self.pg.poses_[self.n] = tvec_qvec
            else:
                tvec_qvec = self.poses[self.n-1]
                self.poses_[self.n] = tvec_qvec

        # TODO better depth initialization
        patches[:,:,2] = torch.rand_like(patches[:,:,2,0,0,None,None])
        if self.is_initialized:
            s = torch.median(self.pg.patches_[self.n-3:self.n,:,2])
            patches[:,:,2] = s

        self.pg.patches_[self.n] = patches

        ### update network attributes ###
        self.imap_[self.n % self.pmem] = imap.squeeze()
        self.gmap_[self.n % self.pmem] = gmap.squeeze()
        self.fmap1_[:, self.n % self.mem] = F.avg_pool2d(fmap[0], 1, 1)
        self.fmap2_[:, self.n % self.mem] = F.avg_pool2d(fmap[0], 4, 4)

        self.counter += 1

        if self.n > 0 and not self.is_initialized:
            thres = 2.0 if scale == 1.0 else scale ** 2 # TODO adapt thres for lite version
            if self.motion_probe() < thres: # TODO: replace by 8 pixels flow criterion (as described in 3.3 Initialization)
                self.pg.delta[self.counter - 1] = (self.counter - 2, Id[0])
                return

        self.n += 1 # add one (key)frame
        self.m += self.M # add patches per (key)frames to patch number

        if self.cfg.LOOP_CLOSURE: #如果开启了闭环检测（这应该是DPVO中实现的闭环检测）
            if self.n - self.last_global_ba >= self.cfg.GLOBAL_OPT_FREQ:
                """ Add loop closure factors """
                lii, ljj = self.pg.edges_loop() #获取闭环检测的边
                if lii.numel() > 0:
                    self.last_global_ba = self.n #标记上一次全局BA优化的帧数，用于控制全局BA优化的频率
                    self.append_factors(lii, ljj)#添加闭环检测的边

        # relative pose
        self.append_factors(*self.__edges_forw())
        self.append_factors(*self.__edges_back())

        if self.n == 8 and not self.is_initialized:
            self.is_initialized = True            

            for itr in range(12):
                self.update()
        
        elif self.is_initialized:
            self.update()
            self.keyframe()

        if self.cfg.CLASSIC_LOOP_CLOSURE:#如果开启了经典的闭环检测
            self.long_term_lc.attempt_loop_closure(self.n)#尝试进行闭环检测
            self.long_term_lc.lc_callback()

        if self.viz_flow:
            self.flow_viz_step()