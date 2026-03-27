import numpy as np
import torch
import glob
import cv2
import os
import os.path as osp
import functools
import operator
import h5py
import hdf5plugin

# from ..lietorch import SE3
from dpvo.lietorch import SE3
from .base import RGBDDataset, EVSDDataset
from .utils import is_converted, scene_in_split

class TartanAir(RGBDDataset):
    """ Derived class for TartanAir RGBD dataset """
    # scale depths to balance rot & trans
    DEPTH_SCALE = 5.0

    def __init__(self, mode='training', **kwargs):
        self.mode = mode
        self.n_frames = 2
        super(TartanAir, self).__init__(name='TartanAir', **kwargs)

    def _build_dataset(self):
        from tqdm import tqdm
        print("Building TartanAir RGBD dataset")

        scene_info = {}
        scenes = glob.glob(osp.join(self.root, '*/*/image_left'))
        scenes = [glob.glob(osp.join(s, '*/*/*/*')) for s in scenes]
        scenes = functools.reduce(operator.concat, scenes)
        for scene in tqdm(sorted(scenes)):
            if not scene_in_split(scene, self.train_split):
                continue
            
            images = sorted(glob.glob(osp.join(scene, 'imgs/*.png')))
            assert len(images) > 0
            depths = sorted(glob.glob(osp.join(scene.replace("image_left", "depth_left"), 'depth_left/*.npy')))
            assert len(images) == len(depths)

            poses = np.loadtxt(osp.join(scene, 'pose_left.txt'), delimiter=' ')
            poses = poses[:, [1, 2, 0, 4, 5, 3, 6]] # NED (z,x,y) to (x,y,z) camera frame
            poses[:,:3] /= TartanAir.DEPTH_SCALE
            intrinsics = [TartanAir.calib_read()] * len(images)
            assert poses.shape[0] == len(images)

            # graph of co-visible frames based on flow
            graph = self.build_frame_graph(poses, depths, intrinsics) # graph is dict of {frameIdx: (co-visible frames, distance)}

            scene = '/'.join(scene.split('/'))
            scene_info[scene] = {'images': images, 'depths': depths, 
                'poses': poses, 'intrinsics': intrinsics, 'graph': graph}

            print(f"Added {scene} to TartanAir RGBD dataset")

        return scene_info

    @staticmethod
    def calib_read():
        return np.array([320.0, 320.0, 320.0, 240.0])

    @staticmethod
    def image_read(image_file):
        return cv2.imread(image_file)

    @staticmethod
    def depth_read(depth_file):
        depth = np.load(depth_file) / TartanAir.DEPTH_SCALE
        depth[depth==np.nan] = 1.0
        depth[depth==np.inf] = 1.0
        # visualize_depth_map(depth)
        return depth


class TartanAirE2VID(RGBDDataset):
    """ Derived class for TartanAir e2v dataset """
    # scale depths to balance rot & trans
    DEPTH_SCALE = 5.0

    def __init__(self, mode='training', **kwargs):
        self.mode = mode
        self.n_frames = 2
        super(TartanAirE2VID, self).__init__(name='TartanAirE2VID', **kwargs)

    def _build_dataset(self):
        from tqdm import tqdm
        print("Building TartanAirE2VID dataset")

        scene_info = {}
        scenes = glob.glob(osp.join(self.root, '*/*/e2v'))
        scenes = [glob.glob(osp.join(s, '*/*/*/*')) for s in scenes]
        scenes = functools.reduce(operator.concat, scenes)
        for scene in tqdm(sorted(scenes)):
            if not scene_in_split(scene, self.train_split):
                continue

            images = sorted(glob.glob(osp.join(scene, 'e2calib/*.png')))
            assert len(images) > 0
            depthdir = scene.replace("/e2v/", "/depth_left/").replace("/datasets/tartan-e2v/", "/datasets/tartan/")
            depths = sorted(glob.glob(osp.join(depthdir, 'depth_left/*.npy')))[1:]
            assert len(images) == len(depths)

            scene_tartan = scene.replace("/e2v/", "/image_left/").replace("/datasets/tartan-e2v/", "/datasets/tartan/")
            poses = np.loadtxt(osp.join(scene_tartan, 'pose_left.txt'), delimiter=' ')
            poses = poses[1:, [1, 2, 0, 4, 5, 3, 6]] # NED (z,x,y) to (x,y,z) camera frame
            poses[:,:3] /= TartanAir.DEPTH_SCALE
            intrinsics = [TartanAir.calib_read()] * len(images)
            assert poses.shape[0] == len(images)

            # graph of co-visible frames based on flow
            graph = self.build_frame_graph(poses, depths, intrinsics) # graph is dict of {frameIdx: (co-visible frames, distance)}

            scene = '/'.join(scene.split('/'))
            scene_info[scene] = {'images': images, 'depths': depths,
                'poses': poses, 'intrinsics': intrinsics, 'graph': graph}

            print(f"Added {scene} to TartanAir RGBD dataset")

        return scene_info

    @staticmethod
    def calib_read():
        return np.array([320.0, 320.0, 320.0, 240.0])

    @staticmethod
    def image_read(image_file):
        return cv2.imread(image_file)

    @staticmethod
    def depth_read(depth_file):
        depth = np.load(depth_file) / TartanAir.DEPTH_SCALE
        depth[depth==np.nan] = 1.0
        depth[depth==np.inf] = 1.0
        # visualize_depth_map(depth)
        return depth


class TartanAirEVS(EVSDDataset):
    """ Derived class for TartanAir event + depth dataset (EVSD) """
    # scale depths to balance rot & trans
    DEPTH_SCALE = 5.0

    def __init__(self, mode='training', **kwargs):
        self.mode = mode
        self.n_frames = 2
        super(TartanAirEVS, self).__init__(name='TartanAirEVS', **kwargs)

    def _build_dataset(self):
        from tqdm import tqdm
        print("Building TartanAir EVSD dataset")

        scene_info = {}

        # --- nested structure: root/scene/difficulty/evs_left/scene/scene/difficulty/seqnum ---
        nested_evs = glob.glob(osp.join(self.root, '*/*/evs_left'))
        nested_scenes = [glob.glob(osp.join(s, '*/*/*/*')) for s in nested_evs]
        nested_scenes = functools.reduce(operator.concat, nested_scenes, [])

        for scene in tqdm(sorted(nested_scenes)):
            if not is_converted(scene):
                print(f"Skipping {scene}. Not fully converted")
                continue

            if not scene_in_split(scene, self.train_split):
                continue

            voxels = sorted(glob.glob(osp.join(scene, 'h5/*.h5')))
            assert len(voxels) > 0
            depths = sorted(glob.glob(osp.join(scene.replace("evs_left", "depth_left"), 'depth_left/*.npy')))[1:]
            assert len(voxels) == len(depths)

            poses = np.loadtxt(osp.join(scene.replace('evs_left', 'image_left'), 'pose_left.txt'), delimiter=' ')[1:]
            poses = poses[:, [1, 2, 0, 4, 5, 3, 6]]
            poses[:,:3] /= TartanAirEVS.DEPTH_SCALE
            intrinsics = [TartanAirEVS.calib_read()] * len(voxels)
            assert poses.shape[0] == len(voxels)

            graph = self.build_frame_graph(poses, depths, intrinsics)
            scene_info[scene] = {'voxels': voxels, 'depths': depths,
                'poses': poses, 'intrinsics': intrinsics, 'graph': graph}
            print(f"Added {scene} to TartanAir EVDS dataset")

        # --- flat structure: root/seqname/evs_left (single-level sample sequences) ---
        flat_evs = glob.glob(osp.join(self.root, '*/evs_left'))

        for evs_dir in tqdm(sorted(flat_evs)):
            seq_dir = osp.dirname(evs_dir)

            if not scene_in_split(seq_dir, self.train_split):
                continue

            voxels = sorted(glob.glob(osp.join(evs_dir, 'h5/*.h5')))
            if not voxels:
                continue

            depths = sorted(glob.glob(osp.join(seq_dir, 'depth_left/*.npy')))[1:]
            if len(voxels) != len(depths):
                print(f"Skipping {seq_dir}: voxel/depth count mismatch ({len(voxels)} vs {len(depths)})")
                continue

            poses = np.loadtxt(osp.join(seq_dir, 'pose_left.txt'), delimiter=' ')[1:]
            poses = poses[:, [1, 2, 0, 4, 5, 3, 6]]
            poses[:,:3] /= TartanAirEVS.DEPTH_SCALE
            intrinsics = [TartanAirEVS.calib_read()] * len(voxels)
            if poses.shape[0] != len(voxels):
                print(f"Skipping {seq_dir}: pose/voxel count mismatch")
                continue

            graph = self.build_frame_graph(poses, depths, intrinsics)
            scene_info[evs_dir] = {'voxels': voxels, 'depths': depths,
                'poses': poses, 'intrinsics': intrinsics, 'graph': graph}
            print(f"Added {evs_dir} to TartanAir EVDS dataset")

        return scene_info

    @staticmethod
    def calib_read():
        return np.array([320.0, 320.0, 320.0, 240.0])

    @staticmethod
    def voxel_read(voxel_file):
        h5 = h5py.File(voxel_file, 'r')
        voxel = h5['voxel'][:]
        # assert voxel.dtype == np.float32 # (5, 480, 640)
        h5.close()
        return voxel

    @staticmethod
    def depth_read(depth_file):
        depth = np.load(depth_file) / TartanAirEVS.DEPTH_SCALE
        depth[depth==np.nan] = 1.0
        depth[depth==np.inf] = 1.0
        # visualize_depth_map(depth)
        return depth