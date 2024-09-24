# Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the NVIDIA Source Code License [see LICENSE for details].



import model.mvt.config as default_mvt_cfg
import model.rvt_agent as rvt_agent
import config as default_exp_cfg

from utils.rvt_utils import load_agent as load_agent_state



# Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the NVIDIA Source Code License [see LICENSE for details].

import os
import time
import tqdm
import random
import yaml
import argparse

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from collections import defaultdict
from contextlib import redirect_stdout

import torch


os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["BITSANDBYTES_NOWELCOME"] = "1"

import config as exp_cfg_mod
import model.rvt_agent as rvt_agent
import model.mvt.config as mvt_cfg_mod

from utils.dataset_utils import get_dataset

from model.mvt.mvt import MVT
from utils.rvt_utils import (
    TensorboardManager,
    short_name,
    get_num_feat,
    RLBENCH_TASKS,
    get_eval_parser
)
from utils.peract_utils import (
    CAMERAS,
    SCENE_BOUNDS,
    IMAGE_SIZE,
)
import pyrealsense2 as rs
import numpy as np
try:
    from xarm.wrapper import XArmAPI
    enable_xarm = True
except:
    enable_xarm = False
    print('no xarm, simulator')

class Camera:
    def __init__(self) -> None:
        self.cameras = []
        self.pipeline = []
        self.align = []
        self.intrinsics = []
        self.extrinsics = []
        self.num_cameras = 1

        self.x_bounds = (0.0, 0.6)
        self.y_bounds = (-0.35, 0.25)
        self.z_bounds = (-0.1, 0.5)

        pipeline = rs.pipeline()
        config = rs.config()
        # config.enable_stream(rs.stream.depth, 1024, 768, rs.format.z16, 30)
        # config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, 960, 540, rs.format.bgr8, 30)
        pipeline.start(config)
        self.pipeline.append(pipeline)
        self.align.append(rs.align(rs.stream.color))
        self.extrinsics.append(np.load('extrinsic.npy'))

    def _capture_frame(self):
        frames_data = []
        for i, (pipeline, align) in enumerate(zip(self.pipeline, self.align)):
            frames = pipeline.wait_for_frames()
            aligned_frames = align.process(frames)
            depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()
            
            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())
            
            if len(self.intrinsics) < self.num_cameras:
                intrinsics = depth_frame.profile.as_video_stream_profile().intrinsics
                self.intrinsics.append(np.array([[intrinsics.fx, 0, intrinsics.ppx],
                                                 [0, intrinsics.fy, intrinsics.ppy],
                                                 [0, 0, 1]]))
            
            frames_data.append({
                "color": color_image,
                "depth": depth_image,
                "intrinsics": self.intrinsics[i],
                "extrinsics": self.extrinsics[i]
            })
        
        return frames_data[0]
    
    def _rgbd2pc(self, color: np.ndarray, depth: np.ndarray, intrinsic: np.ndarray, extrinsic: np.ndarray):
        '''
        color: 3, h, w
        depth: h, w
        '''
        h, w = depth.shape
        i, j = np.meshgrid(np.arange(w), np.arange(h), indexing='xy')
        z = depth * 1.
        valid_mask = z > 0  # Check for valid depth values

        x = (i - intrinsic[0, 2]) * z / intrinsic[0, 0]
        y = (j - intrinsic[1, 2]) * z / intrinsic[1, 1]
        points = np.stack((x, y, z), axis=-1).reshape(-1, 3)
        colors = color.reshape(3, -1).T  # npts, 3

        mask = valid_mask.reshape(-1) & (points[:, 2] <= 1)
        points = points[mask]
        colors = colors[mask]

        R = extrinsic[:3, :3] * 1.
        t = extrinsic[:3, 3] * 1.
        points_transformed = (R @ points.T).T + t

        crop_mask = (points_transformed[:, 0] >= self.x_bounds[0]) & (points_transformed[:, 0] <= self.x_bounds[1]) & \
                    (points_transformed[:, 1] >= self.y_bounds[0]) & (points_transformed[:, 1] <= self.y_bounds[1]) & \
                    (points_transformed[:, 2] >= self.z_bounds[0]) & (points_transformed[:, 2] <= self.z_bounds[1])
        cropped_points = points_transformed[crop_mask]
        cropped_colors = colors[crop_mask]

        return cropped_points, cropped_colors
    
    def get_pc(self):
        frame_data = self._capture_frame()
        pts, cols = self._rgbd2pc(color=frame_data['color'][..., ::-1].transpose(2, 0, 1) / 255.,
                                  depth=frame_data['depth'] * 0.00025,
                                  intrinsic=frame_data["intrinsics"],
                                  extrinsic=frame_data['extrinsics'])
        return pts, cols

class Arm:
    def __init__(self) -> None:
        ip = '192.168.1.197'  
        self.arm = XArmAPI(ip)
        self.arm.motion_enable(enable=True)

        self.arm.set_mode(0)
        self.arm.set_state(0)
        time.sleep(0.1)

        self.arm.set_gripper_mode(0)
        self.arm.set_gripper_enable(True)

    def control(self, pos, rot, grip):
        pos = pos * 1000
        pos[2] = pos[2] + 180
        self.arm.set_position(*pos, *rot, is_radian=False, wait=True, speed=100)
        grip = float(input('input gripper: '))
        if grip > 0.5:
            self.arm.set_gripper_position(800, wait=True)
        else:
            self.arm.set_gripper_position(50, wait=True)

def load_agent(
    model_path=None,
    peract_official=False,
    peract_model_dir=None,
    exp_cfg_path=None,
    mvt_cfg_path=None,
    eval_log_dir="",
    device=0,
    use_input_place_with_mean=False,
):
    device = f"cuda:{device}"

    if not (peract_official):
        assert model_path is not None

        # load exp_cfg
        model_folder = os.path.join(os.path.dirname(model_path))

        exp_cfg = default_exp_cfg.get_cfg_defaults()
        if exp_cfg_path != None:
            exp_cfg.merge_from_file(exp_cfg_path)
        else:
            exp_cfg.merge_from_file(os.path.join(model_folder, "exp_cfg.yaml"))

        # NOTE: to not use place_with_mean in evaluation
        # needed for rvt-1 but not rvt-2
        if not use_input_place_with_mean:
            # for backward compatibility
            old_place_with_mean = exp_cfg.rvt.place_with_mean
            exp_cfg.rvt.place_with_mean = True

        exp_cfg.freeze()


        if exp_cfg.agent == "our":
            mvt_cfg = default_mvt_cfg.get_cfg_defaults()
            if mvt_cfg_path != None:
                mvt_cfg.merge_from_file(mvt_cfg_path)
            else:
                mvt_cfg.merge_from_file(os.path.join(model_folder, "mvt_cfg.yaml"))

            mvt_cfg.freeze()

            # for rvt-2 we do not change place_with_mean regardless of the arg
            # done this way to ensure backward compatibility and allow the
            # flexibility for rvt-1
            if mvt_cfg.stage_two:
                exp_cfg.defrost()
                exp_cfg.rvt.place_with_mean = old_place_with_mean
                exp_cfg.freeze()

            rvt = MVT(
                renderer_device=device,
                **mvt_cfg,
            )

            agent = rvt_agent.RVTAgent(
                network=rvt.to(device),
                image_resolution=[IMAGE_SIZE, IMAGE_SIZE],
                add_lang=mvt_cfg.add_lang,
                stage_two=mvt_cfg.stage_two,
                rot_ver=mvt_cfg.rot_ver,
                scene_bounds=SCENE_BOUNDS,
                cameras=CAMERAS,
                log_dir=f"{eval_log_dir}/eval_run",
                **exp_cfg.peract,
                **exp_cfg.rvt,
            )
        else:
            raise NotImplementedError

        agent.build(training=False, device=device)
        load_agent_state(model_path, agent)
        agent.eval()


    print("Agent Information")
    print(agent)
    return agent


@torch.no_grad()
def eval(
    agent,
):
    agent.eval()
    if isinstance(agent, rvt_agent.RVTAgent):
        agent.load_clip()

    # dataset = get_dataset(bs=1)
    # for batch in dataset:
    #     agent.act(batch)

    cam = Camera()
    if enable_xarm:
        arm = Arm()
    pts, cols = cam.get_pc()
    print(pts.shape, cols.shape)
    pts = torch.tensor(pts).cuda().float()
    cols = torch.tensor(cols).cuda().float()
    trans, rot, gripper = agent.act({
        'current_pts': [pts],
        'current_cols': [cols],
        'instruction': 'place mango on the plate',
    })
    if enable_xarm:
        arm.control(trans, rot, gripper)


    # set agent to back train mode
    agent.train()

    # unloading clip to save memory
    if isinstance(agent, rvt_agent.RVTAgent):
        agent.unload_clip()
        agent._network.free_mem()



def get_model_index(filename):
    """
    :param filenam: path of file of format /.../model_idx.pth
    :return: idx or None
    """
    if len(filename) >= 9 and filename[-4:] == ".pth":
        try:
            index = int(filename[:-4].split("_")[-1])
        except:
            index = None
    else:
        index = None
    return index

def _eval(args):

    model_paths = []
    if not (args.peract_official):
        assert args.model_name is not None
        model_paths.append(os.path.join(args.model_folder, args.model_name))
    else:
        model_paths.append(None)

    for model_path in model_paths:

        if args.peract_official:
            model_idx = 0
        else:
            model_idx = get_model_index(model_path)
            if model_idx is None:
                model_idx = 0

        agent = load_agent(
            model_path=model_path,
            exp_cfg_path=args.exp_cfg_path,
            mvt_cfg_path=args.mvt_cfg_path,
            eval_log_dir=args.eval_log_dir,
            device=args.device,
            use_input_place_with_mean=args.use_input_place_with_mean,
        )
        eval(agent)





if __name__ == "__main__":
    parser = get_eval_parser()

    args = parser.parse_args()

    if args.log_name is None:
        args.log_name = "none"

    if not (args.peract_official):
        args.eval_log_dir = os.path.join(args.model_folder, "eval", args.log_name)
    else:
        args.eval_log_dir = os.path.join(args.peract_model_dir, "eval", args.log_name)

    os.makedirs(args.eval_log_dir, exist_ok=True)

    # save the arguments for future reference
    with open(os.path.join(args.eval_log_dir, "eval_config.yaml"), "w") as fp:
        yaml.dump(args.__dict__, fp)


    _eval(args)
