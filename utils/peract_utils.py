# Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the NVIDIA Source Code License [see LICENSE for details].

import torch
from omegaconf import OmegaConf


from torch.optim.lr_scheduler import CosineAnnealingLR

# Contants
# TODO: Unclear about the best way to handle them
CAMERAS = ["front", "left_shoulder", "right_shoulder", "wrist"]
SCENE_BOUNDS = [
    0.1,
    -0.4,
    -0.2,
    0.5,
    0.4,
    0.8,
]  # [x_min, y_min, z_min, x_max, y_max, z_max] - the metric volume to be voxelized
IMAGE_SIZE = 128
VOXEL_SIZES = [100]  # 100x100x100 voxels
LOW_DIM_SIZE = 4  # {left_finger_joint, right_finger_joint, gripper_open, timestep}

DATA_FOLDER = "data"
EPISODE_FOLDER = "episode%d"
VARIATION_DESCRIPTIONS_PKL = "variation_descriptions.pkl"  # the pkl file that contains language goals for each demonstration
DEMO_AUGMENTATION_EVERY_N = 10  # sample n-th frame in demo
ROTATION_RESOLUTION = 5  # degree increments per axis
# settings
NUM_LATENTS = 512  # PerceiverIO latents


def _norm_rgb(x):
    return (x.float() / 255.0) * 2.0 - 1.0


