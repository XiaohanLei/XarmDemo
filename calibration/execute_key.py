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


        self.task_dir = Path("data/keys")
        self.task_dir.mkdir(parents=True, exist_ok=True)
        self.frame_count = 0

        self.arm = None
        self.init_xarm()
        

    def execute_keyframe(self, keyframe):

        key_frames = os.listdir(self.task_dir)
        key_frames.sort()
        for key_frame in key_frames:
            key_frame_path = self.task_dir / key_frame
            key_frame_data = np.load(key_frame_path)
            gripper_pose = key_frame_data[:6]
            gripper_state = key_frame_data[6]
            self.execute_gripper_pose(gripper_pose.tolist())
            self.execute_gripper_state(gripper_state)
            print(f"执行关键帧 {key_frame}")

    def execute_gripper_pose(self, gripper_pose):
        # 这里需要实现将机械臂gripper移动到指定的6-DoF位姿
        # gripper_pose: 6-DoF位姿，包括位置和旋转
        # 位置单位为m，旋转单位为弧度
        self.arm.set_position(*gripper_pose, is_radian=False, wait=True, speed=100)
    
    def execute_gripper_state(self, gripper_state):
        # 这里需要实现将机械臂gripper移动到指定的夹爪开合程度
        # gripper_state: 夹爪开合程度，0.0表示完全闭合，1.0表示完全打开
        self.arm.set_gripper_position(gripper_state, wait=True)
    
    

    def init_xarm(self, ip="192.168.1.197"):
        self.arm = XArmAPI(ip, is_radian=False)
        self.arm.motion_enable(enable=True)
        self.arm.set_mode(0)
        self.arm.set_state(state=0)
        self.arm.set_gripper_mode(0)
        self.arm.set_gripper_enable(True)
        print('机械臂初始化完成')
    
    def get_gripper_pose(self):
        # 返回一个包含位置和旋转的字典
        _ , pos = self.arm.get_position(is_radian=False)
        # return {"position": np.array(pos[:3]) / 1000., "rotation": np.array(pos[-3:])}  # 示例数据
        return np.array(pos)
    
    def get_gripper_state(self):
        # 这里需要实现获取夹爪开合情况
        # 返回一个表示夹爪开合程度的浮点数（0.0表示完全闭合，1.0表示完全打开）
        _, p = self.arm.get_gripper_position()
        return np.array(p)
    
    def run(self):
        time.sleep(5)
        self.execute_keyframe(self.task_dir)

if __name__ == "__main__":
    collector = DataCollector()
    collector.run()
