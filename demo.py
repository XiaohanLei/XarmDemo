import pyrealsense2 as rs
import numpy as np
import cv2
import os
import json
from datetime import datetime
import keyboard
from pathlib import Path
from scipy.spatial.transform import Rotation as R
from xarm.wrapper import XArmAPI
import time

class DataCollector:
    def __init__(self):

        self.num_cameras = 1
        self.cameras = []
        self.pipeline = []
        self.align = []
        self.intrinsics = []
        self.extrinsics = []
        
        # 初始化相机
        for i in range(self.num_cameras):
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
    

        
        self.task_dir = Path("data")
        self.episode_count = 0
        self.frame_count = 0
        self.current_episode = {}

        self.arm = None
        self.init_xarm()
        
    def start_task(self, task_instruction):
        self.c_task_dir = self.task_dir / task_instruction
        self.c_task_dir.mkdir(parents=True, exist_ok=True)
        
        self.current_episode = {
            "instruction": task_instruction,
            "frames": []
        }
        self.episode_count += 1
        self.frame_count = 0
        
        print(f"开始任务：{task_instruction}")
        print("按 'F' 键记录关键帧，按 'Q' 键结束当前episode")
    
    def capture_frame(self):
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
        
        return frames_data
    
    def record_keyframe(self):
        frames_data = self.capture_frame()
        
        # 这里需要实现获取机械臂gripper的6-DoF位姿和夹爪开合情况
        gripper_pose = self.get_gripper_pose()  # 需要实现这个函数
        gripper_state = self.get_gripper_state()  # 需要实现这个函数
        
        # ignore_collision = input("是否忽略碰撞？(y/n): ").lower() == 'y'
        ignore_collision = 0
        
        self.frame_count += 1
        frame_data = {
            "frame_id": self.frame_count,
            "camera_data": frames_data,
            "gripper_pose": gripper_pose,
            "gripper_state": gripper_state,
            "ignore_collision": ignore_collision
        }
        
        self.current_episode["frames"].append(frame_data)
        print(f"记录关键帧 {self.frame_count}")
    
    def save_episode(self):
        episode_filename = self.c_task_dir / f"episode_{self.episode_count:03d}.npz"
        np.savez_compressed(episode_filename, **self.current_episode)
        print(f"保存episode：{episode_filename}")

    def init_xarm(self, ip="192.168.1.197"):
        self.arm = XArmAPI(ip, is_radian=False)
        print('机械臂初始化完成')
    
    def get_gripper_pose(self):
        # 返回一个包含位置和旋转的字典
        _ , pos = self.arm.get_position(is_radian=False)
        return {"position": np.array(pos[:3]) / 1000., "rotation": np.array(pos[-3:])}  # 示例数据
    
    def get_gripper_state(self):
        # 这里需要实现获取夹爪开合情况
        # 返回一个表示夹爪开合程度的浮点数（0.0表示完全闭合，1.0表示完全打开）
        _, p = self.arm.get_gripper_position()
        return p / 850  # 示例数据
    
    def run(self):
        while True:
            task_instruction = input("请输入任务指令（或按Enter结束）：")
            if task_instruction != 'c':
                pre_task_instruction = task_instruction
            if not task_instruction:
                break
            
            self.start_task(pre_task_instruction)
            
            while True:
                if keyboard.is_pressed('f'):
                    self.record_keyframe()
                    time.sleep(0.2)
                elif keyboard.is_pressed('q'):
                    self.save_episode()
                    time.sleep(0.2)
                    break
        
        print("数据收集完成")

if __name__ == "__main__":
    collector = DataCollector()
    collector.run()
