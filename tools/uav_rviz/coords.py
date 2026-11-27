"""坐标系与针孔反投影（纯 numpy，无 AirSim 依赖，可独立单元测试）。

统一显示 / 传输坐标系 = **ENU 风格 Z-up 右手工程坐标系**：``X 前 / Y 左 / Z 上``。

AirSim 世界系是 NED：``X 前(北) / Y 右(东) / Z 下``。两者相差绕 X 轴旋转 180°（``D = diag(1, -1, -1)``，
det = +1，仍是右手系）：

- 点 / 向量：``p_enu = D @ p_ned = (x, -y, -z)``
- 机体 / 相机系 FRD(前右下) → FLU(前左上) 同样是 ``D``
- 姿态四元数 ``(w, x, y, z)``：``R_enu = D R_ned D``，等价于四元数 ``(w, x, -y, -z)``
  （共轭绕 X 轴 180° 旋转，只翻转 y、z 向量分量）

由此：AirSim ``yaw=+90°``（机头转向东 / 右）在本坐标系下机头指向 ``-Y``；
AirSim ``pitch>0``（抬头）机头 ``z>0``；AirSim ``roll>0``（右翼下沉）机体左向量 ``z>0``。
"""

from __future__ import annotations

import math
from functools import lru_cache
from typing import Any, Iterable

import numpy as np

# ---------------------------------------------------------------------------
# 基础读取：兼容 cosysairsim 对象 (x_val / w_val) 与普通序列 / 字典
# ---------------------------------------------------------------------------


def vec3(v: Any) -> tuple[float, float, float]:
    if hasattr(v, "x_val"):
        return float(v.x_val), float(v.y_val), float(v.z_val)
    if isinstance(v, dict):
        return float(v["x"]), float(v["y"]), float(v["z"])
    x, y, z = v
    return float(x), float(y), float(z)


def quat_wxyz(q: Any) -> tuple[float, float, float, float]:
    if hasattr(q, "w_val"):
        return float(q.w_val), float(q.x_val), float(q.y_val), float(q.z_val)
    if isinstance(q, dict):
        return float(q["w"]), float(q["x"]), float(q["y"]), float(q["z"])
    w, x, y, z = q
    return float(w), float(x), float(y), float(z)


def normalize_quat(q: Iterable[float]) -> tuple[float, float, float, float]:
    w, x, y, z = (float(c) for c in q)
    n = math.sqrt(w * w + x * x + y * y + z * z)
    if not math.isfinite(n) or n < 1e-9:
        return 1.0, 0.0, 0.0, 0.0
    return w / n, x / n, y / n, z / n


def quat_is_valid(q: Iterable[float], tol: float = 0.05) -> bool:
    w, x, y, z = (float(c) for c in q)
    n = math.sqrt(w * w + x * x + y * y + z * z)
    return math.isfinite(n) and abs(n - 1.0) <= tol


# ---------------------------------------------------------------------------
# NED -> ENU / Z-up
# ---------------------------------------------------------------------------


def ned_to_enu_point(p: Iterable[float]) -> tuple[float, float, float]:
    x, y, z = (float(c) for c in p)
    return x, -y, -z


def ned_to_enu_quat(q: Iterable[float]) -> tuple[float, float, float, float]:
    w, x, y, z = (float(c) for c in q)
    return w, x, -y, -z


def object_ned_to_local_enu(p: Iterable[float], spawn_ned_z: float = 1.0) -> tuple[float, float, float]:
    """AirSim 场景对象位姿（PlayerStart 原点）→ 车辆局部 ENU。

    车辆 / 相机位姿以初始车辆原点为基准，该原点在 PlayerStart 下方 ``spawn_ned_z`` 米
    （本仓库 ``SPAWN_NED = (0, 0, 1)``），因此对象 ``z_up_local = spawn_ned_z - z_ned_object``。
    """
    x, y, z = (float(c) for c in p)
    return x, -y, spawn_ned_z - z


def quat_to_matrix(q: Iterable[float]) -> np.ndarray:
    """单位四元数 (w, x, y, z) → 3x3 旋转矩阵（列向量右乘约定）。"""
    w, x, y, z = normalize_quat(q)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def euler_from_enu_quat(q_enu: Iterable[float]) -> dict[str, float]:
    """从 ENU/FLU 姿态提取易读欧拉角（度），由旋转矩阵列向量直接推导，避免手性歧义。

    - ``yaw``  : 机头在水平面内的方位，``atan2(f_y, f_x)``，逆时针（向左）为正（ENU 约定）
    - ``pitch``: 抬头为正（与 AirSim 一致），``asin(f_z)``
    - ``roll`` : 右翼下沉为正（与 AirSim 一致），``atan2(left_z, up_z)``
    """
    r = quat_to_matrix(q_enu)
    f, left, up = r[:, 0], r[:, 1], r[:, 2]
    return {
        "yaw": math.degrees(math.atan2(f[1], f[0])),
        "pitch": math.degrees(math.asin(max(-1.0, min(1.0, f[2])))),
        "roll": math.degrees(math.atan2(left[2], up[2])),
    }


# ---------------------------------------------------------------------------
# 针孔反投影
# ---------------------------------------------------------------------------


def focal_length_px(width: int, fov_deg: float) -> float:
    """水平 FOV → 像素焦距：``fx = fy = W / (2 tan(FOV/2))``（方形像素）。"""
    return width / (2.0 * math.tan(math.radians(fov_deg) / 2.0))


@lru_cache(maxsize=16)
def sample_grid(width: int, height: int, grid_w: int, grid_h: int) -> tuple[np.ndarray, np.ndarray]:
    """在每个网格单元中心取样，每个维度最多使用原图像素数。"""
    nx, ny = min(width, grid_w), min(height, grid_h)
    cols = ((np.arange(nx) + 0.5) * width / nx).astype(np.intp)
    rows = ((np.arange(ny) + 0.5) * height / ny).astype(np.intp)
    cols.setflags(write=False)
    rows.setflags(write=False)
    return rows, cols


@lru_cache(maxsize=16)
def camera_ray_factors(width: int, height: int, fov_deg: float) -> tuple[np.ndarray, np.ndarray]:
    """整幅图像每个像素的相机系侧向 / 竖向比例（前向深度为 1 时的 FRD 分量）。

    返回 ``(right_ratio, down_ratio)``，形状均为 (H, W)::

        right_ratio = (u - cx) / fx      # 图像向右 → 相机 +Y(右)
        down_ratio  = (v - cy) / fy      # 图像向下 → 相机 +Z(下)

    主点取像素中心 ``cx = (W-1)/2, cy = (H-1)/2``。
    """
    f = focal_length_px(width, fov_deg)
    cx, cy = (width - 1) / 2.0, (height - 1) / 2.0
    u = (np.arange(width, dtype=np.float64) - cx) / f
    v = (np.arange(height, dtype=np.float64) - cy) / f
    right = np.broadcast_to(u[None, :], (height, width)).copy()
    down = np.broadcast_to(v[:, None], (height, width)).copy()
    right.setflags(write=False)
    down.setflags(write=False)
    return right, down


def depth_to_camera_flu(depth_grid: np.ndarray, right_ratio: np.ndarray, down_ratio: np.ndarray) -> np.ndarray:
    """沿光轴深度 ``d`` (DepthPlanar) → 相机系 FLU 点 (H, W, 3)。

    FRD 点为 ``(d, d·right, d·down)``；转 FLU(前左上) 得 ``(d, -d·right, -d·down)``。
    """
    d = depth_grid.astype(np.float64, copy=False)
    return np.stack((d, -d * right_ratio, -d * down_ratio), axis=-1)


def rotate_points(points: np.ndarray, rot: np.ndarray) -> np.ndarray:
    """``points @ rot.T``（逐列广播乘加；避免对 (n,3) 大数组走 BLAS 小矩阵乘的额外开销与依赖）。"""
    return points[:, 0:1] * rot[:, 0] + points[:, 1:2] * rot[:, 1] + points[:, 2:3] * rot[:, 2]


def camera_to_world_enu(points_flu: np.ndarray, cam_pos_enu: Iterable[float], cam_quat_enu: Iterable[float]) -> np.ndarray:
    """相机系 FLU 点 → ENU 世界点：``p_w = p_cam_pos + R_enu @ p_flu``。"""
    rot = quat_to_matrix(cam_quat_enu)
    shape = points_flu.shape
    world = rotate_points(points_flu.reshape(-1, 3), rot) + np.asarray(tuple(cam_pos_enu), dtype=np.float64)
    return world.reshape(shape)


def depth_to_world_cloud(
    depth: np.ndarray,
    cam_pos_ned: Iterable[float],
    cam_quat_ned: Iterable[float],
    fov_deg: float = 90.0,
    grid: tuple[int, int] = (80, 45),
    min_range: float = 0.1,
    max_range: float = 25.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """DepthPlanar 全分辨率深度(米) → ENU 世界点云。

    Parameters
    ----------
    depth: (H, W) float，沿光轴深度；
    cam_pos_ned / cam_quat_ned: 相机在 AirSim NED 世界系下的位置与姿态 (w, x, y, z)；
    grid: 目标采样网格 (grid_w, grid_h)，原图较小时按原图像素数采样。

    Returns
    -------
    (points_enu (n,3) float32, depth_grid (gh,gw) float32, valid (gh,gw) bool)
    """
    height, width = depth.shape
    rows, cols = sample_grid(width, height, int(grid[0]), int(grid[1]))
    right, down = camera_ray_factors(width, height, float(fov_deg))
    grid_depth = depth[np.ix_(rows, cols)].astype(np.float32, copy=False)
    right_g = right[np.ix_(rows, cols)]
    down_g = down[np.ix_(rows, cols)]
    valid = np.isfinite(grid_depth) & (grid_depth >= min_range) & (grid_depth <= max_range)
    cam_flu = depth_to_camera_flu(np.where(valid, grid_depth, 0.0), right_g, down_g)
    world = camera_to_world_enu(cam_flu, ned_to_enu_point(cam_pos_ned), ned_to_enu_quat(cam_quat_ned))
    return world[valid].astype(np.float32), grid_depth, valid


