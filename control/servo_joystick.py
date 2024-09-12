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

# 设置为伺服模式
arm.set_mode(1)
arm.set_state(0)
time.sleep(0.1)


# 设置夹爪模式并使能
arm.set_gripper_mode(0)
arm.set_gripper_enable(True)

# 设置死区
DEADZONE = 0.1

def apply_deadzone(value, deadzone):
    if abs(value) < deadzone:
        return 0
    return value

# 获取当前位置
current_position = arm.get_position()[1]

try:
    while arm.connected and arm.state != 4:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                break

        # 读取左摇杆输入 (X和Y轴移动)
        dx = apply_deadzone(joystick.get_axis(1), DEADZONE) * 1  # 左右移动
        dy = apply_deadzone(joystick.get_axis(0), DEADZONE) * 1  # 前后移动

        # 读取L1和L2按钮输入 (Z轴移动)
        dz = (joystick.get_button(11) - joystick.get_button(12)) * 1  # 上下移动

        # test
        # print('joystick button: ', joystick.get_button(0), joystick.get_button(1), joystick.get_button(2), joystick.get_button(3), joystick.get_button(4), joystick.get_button(5), joystick.get_button(6), joystick.get_button(7))
        # print('joystick axis', joystick.get_axis(0), joystick.get_axis(1), joystick.get_axis(2), joystick.get_axis(3), joystick.get_axis(4), joystick.get_axis(5))
        # print(joystick.get_button(8), joystick.get_button(9), joystick.get_button(10), joystick.get_button(11), joystick.get_button(12), joystick.get_button(13), joystick.get_button(14), joystick.get_button(15))
        # print('button 4: ', joystick.get_button(4), 'button 6: ', joystick.get_button(6))
        # dz = 0

        # 读取右摇杆输入 (Roll和Pitch旋转)
        droll = apply_deadzone(joystick.get_axis(2), DEADZONE) * 0.5  # 绕X轴旋转
        dpitch = apply_deadzone(-joystick.get_axis(3), DEADZONE) * 0.5  # 绕Y轴旋转

        # 读取R1和R2按钮输入 (Yaw旋转)
        dyaw = (joystick.get_button(3) - joystick.get_button(0)) * 0.5  # 绕Z轴旋转
        # print('button 5: ', joystick.get_button(5), 'button 7: ', joystick.get_button(7))
        # dyaw = 0

        # 更新位置
        new_position = [
            current_position[0] + dx,
            current_position[1] + dy,
            current_position[2] + dz,
            current_position[3] + droll,
            current_position[4] + dpitch,
            current_position[5] + dyaw
        ]

        # 控制机械臂移动
        ret = arm.set_servo_cartesian(new_position, speed=30, mvacc=500)
        # print('set_servo_cartesian, ret={}'.format(ret))

        # 更新当前位置
        current_position = new_position

        # 读取三角形和X按钮输入
        gripper_action = joystick.get_button(9) - joystick.get_button(10)  # 三角形键开启夹爪，X键关闭夹爪
        # print('button 3: ', joystick.get_button(3), 'button 2: ', joystick.get_button(2))

        # 控制夹爪
        if gripper_action == 1:
            arm.set_gripper_position(800, wait=True)  # 开启夹爪
        elif gripper_action == -1:
            arm.set_gripper_position(0, wait=True)  # 关闭夹爪

        time.sleep(0.01)  # 小延迟以防止过快发送命令

except KeyboardInterrupt:
    print("程序已终止")
finally:
    # 清理并断开连接
    arm.set_mode(0)
    arm.disconnect()
    pygame.quit()