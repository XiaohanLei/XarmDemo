#!/usr/bin/env python3

import os
import sys
import time
import pygame

sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

from xarm.wrapper import XArmAPI

# 初始化pygame和游戏手柄
pygame.init()
pygame.joystick.init()
joystick = pygame.joystick.Joystick(0)
joystick.init()

# 初始化机械臂
ip = '192.168.1.197'  # 替换为实际的IP地址
arm = XArmAPI(ip)
arm.motion_enable(enable=True)
arm.set_mode(5)  # 设置为笛卡尔速度控制模式
arm.set_state(state=0)
angle_speed = 20
trans_speed = 100
time.sleep(1)

arm.set_gripper_mode(0)
arm.set_gripper_enable(True)

trans_acc = 5000 * 5  # mm/s^2
trans_jerk = 10000 * 5  # mm/s^3
# arm.set_tcp_acc(trans_acc)
arm.set_tcp_maxacc(trans_acc)
arm.set_tcp_jerk(trans_jerk)

# 设置死区
DEADZONE = 0.2

def apply_deadzone(value, deadzone):
    if abs(value) < deadzone:
        return 0
    return value

try:
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                break

        # 读取左摇杆输入 (X和Y轴移动)
        x_speed = apply_deadzone(joystick.get_axis(1), DEADZONE) * trans_speed  # 左右移动
        y_speed = apply_deadzone(joystick.get_axis(0), DEADZONE) * trans_speed  # 前后移动

        # 读取L2和R2按钮输入 (Z轴移动)
        z_speed = (joystick.get_button(11) - joystick.get_button(12)) * trans_speed  # 上下移动

        # 读取右摇杆输入 (Roll和Pitch旋转)
        roll = apply_deadzone(joystick.get_axis(2), DEADZONE) * angle_speed  # 绕X轴旋转
        pitch = apply_deadzone(-joystick.get_axis(3), DEADZONE) * angle_speed  # 绕Y轴旋转

        # 读取L1和R1按钮输入 (Yaw旋转)
        yaw = (joystick.get_button(3) - joystick.get_button(0)) * angle_speed   # 绕Z轴旋转


        # 读取数字键盘输入
        gripper_action = joystick.get_button(9) - joystick.get_button(10)  # 0键开启夹爪，1键关闭夹爪

        # 控制机械臂移动
        # print([x_speed, y_speed, z_speed, roll, pitch, yaw])
        arm.vc_set_cartesian_velocity([x_speed, y_speed, z_speed, roll, pitch, yaw])

        # 控制夹爪
        if gripper_action == 1:
            arm.set_gripper_position(800, wait=True)  # 开启夹爪
        elif gripper_action == -1:
            arm.set_gripper_position(0, wait=True)  # 关闭夹爪

        time.sleep(0.01)  # 稍微延迟以防止过快发送命令

except KeyboardInterrupt:
    print("程序已终止")
finally:
    # 停止机械臂移动
    arm.vc_set_cartesian_velocity([0, 0, 0, 0, 0, 0])
    # 清理并断开连接
    arm.disconnect()
    pygame.quit()