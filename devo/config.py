from yacs.config import CfgNode as CN

_C = CN()

# max number of keyframes
_C.BUFFER_SIZE = 2048*2

# bias patch selection towards high gradient regions?
_C.GRADIENT_BIAS = False
# Select between random, gradient, scorer
_C.PATCH_SELECTOR = "scorer"
# Eval mode of patch selector (random, topk, multinomial)
_C.SCORER_EVAL_MODE = "multi"
_C.SCORER_EVAL_USE_GRID = True
# Normalizer (only evs): norm, standard
_C.NORM = "std"
_C.ENABLE_DBA_TRACK_FILTERS = True
_C.ENABLE_EVENT_SUPPORT_FILTER = True
_C.MIN_EVENT_SUPPORT = 4.0
_C.MIN_EVENT_PIXELS = 4.0
_C.ENABLE_RANSAC_FILTER = True
_C.RANSAC_REPROJ_THRESH = 3e-3
_C.RANSAC_CONFIDENCE = 0.99
_C.RANSAC_MIN_POINTS = 8
_C.RANSAC_MIN_INLIERS = 8

# VO config (increase for better accuracy)
_C.PATCHES_PER_FRAME = 80
_C.REMOVAL_WINDOW = 20
_C.OPTIMIZATION_WINDOW = 12
_C.PATCH_LIFETIME = 12

# threshold for keyframe removal
_C.KEYFRAME_INDEX = 4
_C.KEYFRAME_THRESH = 12.5

# camera motion model
_C.MOTION_MODEL = 'DAMPED_LINEAR'
_C.MOTION_DAMPING = 0.5

_C.MIXED_PRECISION = True

# Optional ResNet backbone for event feature extraction
_C.resnet = False
_C.block_dims = [64, 128, 256]
_C.initial_dim = 64
_C.pretrain = "resnet18"

# Loop closure
_C.LOOP_CLOSURE = False
_C.BACKEND_THRESH = 64.0
_C.MAX_EDGE_AGE = 1000
_C.GLOBAL_OPT_FREQ = 15

# Classic loop closure
_C.CLASSIC_LOOP_CLOSURE = False
_C.LOOP_CLOSE_WINDOW_SIZE = 3
_C.LOOP_RETR_THRESH = 0.04


# For IMU integration
_C.ENALBE_IMU = False
_C.Ti1c=[1.0, 0.0, 0.0, 0.0,      
         0.0, 1.0, 0.0, 0.0,
         0.0, 0.0, 1.0, 0.0,
         0.0, 0.0, 0.0, 1.0]
_C.ENALBE_INV = False
_C.accel_noise_sigma=2.0000e-3
_C.gyro_noise_sigma=1.6968e-04
_C.accel_bias_sigma=3.0000e-3
_C.gyro_bias_sigma=1.9393e-05

cfg = _C
