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
        

    
    def record_keyframe(self):
        
        # 这里需要实现获取机械臂gripper的6-DoF位姿和夹爪开合情况
        gripper_pose = self.get_gripper_pose()  # 需要实现这个函数
        gripper_state = self.get_gripper_state()  # 需要实现这个函数
        print('gripper_pose shape: ', gripper_pose.shape)
        print('gripper_state shape: ', gripper_state.shape)
        save_array = np.concatenate([gripper_pose, gripper_state])
        np.save(self.task_dir / f"frame_{self.frame_count}.npy", save_array)
        self.frame_count += 1
        print(f"记录关键帧 {self.frame_count}")
    

    def init_xarm(self, ip="192.168.1.197"):
        self.arm = XArmAPI(ip, is_radian=False)
        print('机械臂初始化完成')
    
    def get_gripper_pose(self):
        # 返回一个包含位置和旋转的字典
        _ , pos = self.arm.get_position(is_radian=False)
        # return {"position": np.array(pos[:3]) / 1000., "rotation": np.array(pos[-3:])}  # 示例数据
        return np.array(pos)
    
    def get_gripper_state(self, self_define=False):
        # 这里需要实现获取夹爪开合情况
        # 返回一个表示夹爪开合程度的浮点数（0.0表示完全闭合，1.0表示完全打开）
        _, p = self.arm.get_gripper_position()
        if self_define:
            if input("open?: ").lower() == 'y':
                p = 850
            else:
                p = 50
        return np.array([p])
    
    def run(self):
            
        while True:
            if keyboard.is_pressed('f'):
                self.record_keyframe()
                time.sleep(1)
            elif keyboard.is_pressed('q'):
                break
        
        print("数据收集完成")

if __name__ == "__main__":
    collector = DataCollector()
    collector.run()
