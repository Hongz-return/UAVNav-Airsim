"""离线模拟 AirSim 数据源（``--simulate``），接口形状与 cosysairsim 只读子集一致。

用途：没有仿真器 / 不想触碰训练端口时，端到端验证前端渲染、WebSocket 二进制帧与全部交互。
模拟无人机沿一条带障碍的"顶棚隧道"飞行，通过与真实采集器完全相同的 :class:`AirSimTap` 管线
（反投影、限速、坐标转换）产生数据；每圈结束时模拟一次碰撞与"位姿跳变重置"，开局若干帧返回
``[0, 1]`` 未编译着色器亮度帧以演示 ``Shader Warming`` 提示。

场景构造用车辆局部 ENU（X 前 / Y 左 / Z 上）；对外按 AirSim 约定转换为 NED。
"""

from __future__ import annotations

import math
import struct
import time
import zlib
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Optional

import numpy as np

from .coords import camera_ray_factors, quat_to_matrix
from .geometry_loader import DEFAULT_FLOOR_Z


class _ImageType:
    Scene = 0
    DepthPlanar = 1


@dataclass
class _ImageRequest:
    camera_name: str
    image_type: int
    pixels_as_float: bool = False
    compress: bool = True


SIM_API = SimpleNamespace(ImageRequest=_ImageRequest, ImageType=_ImageType)

_W, _H, _FOV = 320, 180, 90.0


def _v3(x: float, y: float, z: float) -> SimpleNamespace:
    return SimpleNamespace(x_val=x, y_val=y, z_val=z)


def _q(w: float, x: float, y: float, z: float) -> SimpleNamespace:
    return SimpleNamespace(w_val=w, x_val=x, y_val=y, z_val=z)


def _png(rgb: np.ndarray) -> bytes:
    """最小 PNG 编码（标准库 zlib），用于模拟 Scene 图。"""
    h, w, _ = rgb.shape
    raw = b"".join(b"\x00" + rgb[row].tobytes() for row in range(h))

    def chunk(tag: bytes, data: bytes) -> bytes:
        body = tag + data
        return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body) & 0xFFFFFFFF)

    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw, 3)) + chunk(b"IEND", b""))


class SimulatedAirSim:
    """只读 AirSim 客户端子集的模拟实现。"""

    def __init__(self, *, warmup_frames: int = 3, lap_seconds: float = 24.0, floor_z: float = DEFAULT_FLOOR_Z,
                 spawn_ned_z: float = 1.0, latency_s: float = 0.0) -> None:
        self.floor_z = floor_z
        self.spawn_ned_z = spawn_ned_z
        self.lap_seconds = lap_seconds
        self.warmup_frames = warmup_frames
        self.latency_s = latency_s
        self._t0 = time.monotonic()
        self._frames = 0
        self._collision_ts = 0
        self.calls: list[str] = []
        self._rays = tuple(ray.astype(np.float32) for ray in camera_ray_factors(_W, _H, _FOV))
        roof_z = 3.6
        zc = (floor_z + roof_z) / 2.0
        # (key, center_enu, size) —— 局部 ENU
        self.boxes: list[tuple[str, tuple[float, float, float], tuple[float, float, float]]] = [
            ("wall_min_y", (45.0, 6.2, zc), (90.0, 0.3, roof_z - floor_z)),
            ("wall_max_y", (45.0, -6.2, zc), (90.0, 0.3, roof_z - floor_z)),
            ("corridor_roof", (45.0, 0.0, roof_z + 0.18), (90.0, 12.7, 0.36)),
            ("goal_end_wall", (89.8, 0.0, zc), (0.3, 12.7, roof_z - floor_z)),
            ("over_0", (30.0, 0.0, floor_z + 1.6), (3.0, 4.0, 3.2)),
            ("under_1", (58.0, 0.0, roof_z - 0.8), (3.0, 6.0, 1.6)),
            ("floor_slab", (45.0, 0.0, floor_z - 0.15), (90.0, 12.7, 0.3)),
        ]
        self._names = {key: f"Ch4Stage3V2_1_{key}" for key, _, _ in self.boxes}

    # --- 场景对象（对象系 = 局部 ENU 转 NED，z 偏移 spawn_ned_z） ---
    def simListSceneObjects(self, name_regex: str = ".*") -> list[str]:
        self.calls.append("simListSceneObjects")
        return list(self._names.values())

    def _find(self, name: str):
        for key, center, size in self.boxes:
            if self._names[key] == name:
                return center, size
        raise KeyError(name)

    def simGetObjectPose(self, name: str, *_a: Any) -> SimpleNamespace:
        self.calls.append("simGetObjectPose")
        (cx, cy, cz), _ = self._find(name)
        return SimpleNamespace(position=_v3(cx, -cy, self.spawn_ned_z - cz), orientation=_q(1, 0, 0, 0))

    def simGetObjectScale(self, name: str, *_a: Any) -> SimpleNamespace:
        self.calls.append("simGetObjectScale")
        _, size = self._find(name)
        return _v3(*size)

    def ping(self) -> bool:
        return True

    # --- 飞行轨迹（局部 ENU） ---
    def _state_at(self, t: float) -> tuple[np.ndarray, np.ndarray, int, np.ndarray]:
        lap = t / self.lap_seconds
        lap_idx = int(lap)
        s = lap - lap_idx
        x = 2.0 + s * 84.0
        y = 3.2 * math.sin(2 * math.pi * 1.5 * s)
        z = self.floor_z + 1.8 + 1.1 * math.sin(2 * math.pi * 2.0 * s + 0.6)
        pos = np.array([x, y, z])
        dy = 3.2 * math.cos(2 * math.pi * 1.5 * s) * 2 * math.pi * 1.5
        dz = 1.1 * math.cos(2 * math.pi * 2.0 * s + 0.6) * 2 * math.pi * 2.0
        vel = np.array([84.0, dy, dz]) / self.lap_seconds
        yaw = math.atan2(vel[1], vel[0])
        pitch = math.atan2(vel[2], math.hypot(vel[0], vel[1]))  # 抬头为正
        roll = -0.25 * math.cos(2 * math.pi * 1.5 * s)  # 右翼下沉为正
        # ENU/FLU：R = Rz(yaw) · Ry(-pitch) · Rx(roll)（绕 Y(左) 正向 = 低头，故抬头取负）
        cr, sr = math.cos(roll / 2), math.sin(roll / 2)
        cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
        cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
        q = np.array([
            cy * cp * cr - sy * sp * sr,
            cy * cp * sr + sy * sp * cr,
            -cy * sp * cr + sy * cp * sr,
            sy * cp * cr + cy * sp * sr,
        ])
        return pos, vel, lap_idx, q

    def _kin(self):
        t = time.monotonic() - self._t0
        pos, vel, lap_idx, q_enu = self._state_at(t)
        collided = (t / self.lap_seconds) % 1.0 > 0.985  # 圈末撞端墙
        if collided:
            self._collision_ts = lap_idx + 1
        return t, pos, vel, q_enu, collided
    def _ned_pose(self, pos: np.ndarray, q_enu: np.ndarray):
        w, x, y, z = q_enu
        q_ned = (w, x, -y, -z)  # 共轭 180° 绕 X，自逆
        return SimpleNamespace(position=_v3(float(pos[0]), float(-pos[1]), float(-pos[2])), orientation=_q(*q_ned))

    def simGetVehiclePose(self, *_a: Any) -> SimpleNamespace:
        self.calls.append("simGetVehiclePose")
        self._lag()
        _, pos, _, q_enu, _ = self._kin()
        return self._ned_pose(pos, q_enu)

    def getMultirotorState(self, *_a: Any, **_k: Any) -> SimpleNamespace:
        self.calls.append("getMultirotorState")
        self._lag()
        t, pos, vel, q_enu, _ = self._kin()
        pose = self._ned_pose(pos, q_enu)
        kin = SimpleNamespace(position=pose.position, orientation=pose.orientation,
                              linear_velocity=_v3(float(vel[0]), float(-vel[1]), float(-vel[2])))
        return SimpleNamespace(timestamp=int((t + 1.0) * 1e9), kinematics_estimated=kin)

    def simGetCollisionInfo(self, *_a: Any) -> SimpleNamespace:
        self.calls.append("simGetCollisionInfo")
        _, _, _, _, collided = self._kin()
        return SimpleNamespace(has_collided=collided, object_name="Ch4Stage3V2_1_goal_end_wall" if collided else "",
                               time_stamp=self._collision_ts * 1_000_000)

    def _lag(self) -> None:
        if self.latency_s > 0:
            time.sleep(self.latency_s)

    # --- 相机 ---
    def _render_depth(self, pos: np.ndarray, q_enu: np.ndarray) -> np.ndarray:
        right, down = self._rays
        rotation = quat_to_matrix(q_enu).astype(np.float32)
        forward, left, up = rotation.T[:, :, None, None]
        direction = forward - left * right - up * down
        camera = pos.astype(np.float32)
        best = np.full(_W * _H, np.inf, dtype=np.float32)
        with np.errstate(divide="ignore", invalid="ignore"):
            inverse = np.reciprocal(direction.reshape(3, -1))
            for _key, center, size in self.boxes:
                relative = np.asarray(center, dtype=np.float32) - camera
                half = np.asarray(size, dtype=np.float32) / 2.0
                t1 = (relative - half)[:, None] * inverse
                t2 = (relative + half)[:, None] * inverse
                near = np.fmin(t1, t2)
                far = np.fmax(t1, t2)
                t_near = np.fmax(np.fmax(near[0], near[1]), near[2])
                t_far = np.fmin(np.fmin(far[0], far[1]), far[2])
                hit = (t_far >= np.maximum(t_near, 0.05)) & np.isfinite(t_near)
                t_hit = np.where(t_near > 0.05, t_near, t_far)
                best = np.where(hit & (t_hit < best), t_hit, best)
        best[~np.isfinite(best)] = 100.0
        return best.reshape(_H, _W)  # forward 分量为 1 → t 即沿光轴深度

    def simGetImages(self, requests: list, *_a: Any, **_k: Any) -> list:
        self.calls.append("simGetImages")
        self._lag()
        t, pos, _vel, q_enu, _ = self._kin()
        self._frames += 1
        pose = self._ned_pose(pos, q_enu)
        out = []
        for req in requests:
            if req.image_type == _ImageType.DepthPlanar:
                if self._frames <= self.warmup_frames:
                    depth = np.random.default_rng(self._frames).random((_H, _W), dtype=np.float32)  # [0,1) 亮度帧
                else:
                    depth = self._render_depth(pos, q_enu)
                out.append(SimpleNamespace(
                    width=_W, height=_H, image_data_float=depth.ravel(), image_data_uint8=b"",
                    camera_position=pose.position, camera_orientation=pose.orientation, image_type=req.image_type))
            else:
                yy, xx = np.mgrid[0:90, 0:160]
                rgb = np.stack([(xx * 255 // 160), (yy * 255 // 90), np.full_like(xx, 80 + int(60 * math.sin(t)))],
                               axis=-1).astype(np.uint8)
                out.append(SimpleNamespace(
                    width=160, height=90, image_data_float=[], image_data_uint8=_png(rgb),
                    camera_position=pose.position, camera_orientation=pose.orientation, image_type=req.image_type))
        return out

    # 任何写接口：模拟客户端也不实现 → AttributeError，进一步佐证只读


