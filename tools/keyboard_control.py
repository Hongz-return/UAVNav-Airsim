"""
全局键盘控制 - 在UE窗口中直接按键控制无人机
需要管理员权限运行
"""
import cosysairsim as airsim
import time
import ctypes
from ctypes import wintypes
import threading

# Windows API
user32 = ctypes.windll.user32
kernel32 = ctypes.windll.kernel32

# Key codes
VK_W, VK_S, VK_A, VK_D = 0x57, 0x53, 0x41, 0x44
VK_Q, VK_E, VK_P = 0x51, 0x45, 0x50
VK_J, VK_L = 0x4A, 0x4C  # 左转/右转
VK_ESCAPE = 0x1B

def is_key_pressed(vk_code):
    return user32.GetAsyncKeyState(vk_code) & 0x8000

# 连接 AirSim
client = airsim.MultirotorClient()
client.confirmConnection()
client.enableApiControl(True)
client.armDisarm(True)
client.takeoffAsync().join()

def get_pos():
    p = client.simGetVehiclePose().position
    return p.x_val, p.y_val, p.z_val

print("=" * 50)
print("全局键盘控制 - 直接在UE中按键!")
print("=" * 50)
print("W/S - 前进/后退")
print("A/D - 左移/右移")
print("Q/E - 上升/下降")
print("J/L - 左转/右转")
print("P   - 打印坐标")
print("ESC - 退出")
print("=" * 50)
x, y, z = get_pos()
print(f"当前位置: X={x:.1f}, Y={y:.1f}, Z={z:.1f}")
print("\n>>> 现在切换到UE窗口，直接按WASD控制 <<<")

speed = 3.0
running = True
last_p = 0

yaw_rate = 45  # 度/秒

while running:
    vx, vy, vz, yaw = 0, 0, 0, 0

    if is_key_pressed(VK_W):
        vx = speed
        print("W", end=" ", flush=True)
    if is_key_pressed(VK_S):
        vx = -speed
        print("S", end=" ", flush=True)
    if is_key_pressed(VK_A):
        vy = -speed
        print("A", end=" ", flush=True)
    if is_key_pressed(VK_D):
        vy = speed
        print("D", end=" ", flush=True)
    if is_key_pressed(VK_Q):
        vz = -speed
        print("Q", end=" ", flush=True)
    if is_key_pressed(VK_E):
        vz = speed
        print("E", end=" ", flush=True)
    if is_key_pressed(VK_J):
        yaw = -yaw_rate
        print("J", end=" ", flush=True)
    if is_key_pressed(VK_L):
        yaw = yaw_rate
        print("L", end=" ", flush=True)

    if is_key_pressed(VK_P):
        now = time.time()
        if now - last_p > 0.5:
            x, y, z = get_pos()
            print(f"\n位置: X={x:.1f}, Y={y:.1f}, Z={z:.1f}")
            last_p = now

    if is_key_pressed(VK_ESCAPE):
        running = False

    # 发送速度命令
    if vx != 0 or vy != 0 or vz != 0 or yaw != 0:
        client.moveByVelocityAsync(
            vx, vy, vz, 0.3,
            yaw_mode=airsim.YawMode(is_rate=True, yaw_or_rate=yaw)
        ).join()

    time.sleep(0.02)

x, y, z = get_pos()
print(f"\n最终位置: X={x:.1f}, Y={y:.1f}, Z={z:.1f}")
client.landAsync().join()
client.enableApiControl(False)
print("完成!")
