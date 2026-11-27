"""紧凑二进制遥测帧（WebSocket binary message）编解码。

遥测帧 ``UVRT`` 布局（全部小端）::

    offset  size  field
    0       4     magic = b"UVRT"
    4       2     version (=1)
    6       2     flags   (bit0 shader_warming, bit1 no_return)
    8       4     seq
    12      8     t (unix seconds, float64)
    20      4     meta_len  (JSON 字节数，已补空格对齐到 4 的倍数)
    24      4     n_points
    28      2     depth_w
    30      2     depth_h
    32      1     depth_enc (0=无, 1=uint8, 2=uint16)
    33      1     reserved
    34      2     reserved
    36      -     meta JSON (UTF-8)
    ...     n*12  点云 XYZ float32（统一 ENU / Z-up 世界系，米）
    ...     w*h*k 深度图（uint8: 米*255/depth_max；uint16: 毫米），行优先

RGB 帧 ``UVRG``::

    0  4  magic = b"UVRG"
    4  4  seq
    8  2  width
    10 2  height
    12 4  png_len
    16 -  PNG 字节

头部 36 字节、meta 对齐 4 字节，保证浏览器端 ``Float32Array`` 直接按偏移映射，无需拷贝。
"""

from __future__ import annotations

import json
import struct
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

TELEMETRY_MAGIC = b"UVRT"
RGB_MAGIC = b"UVRG"
PROTOCOL_VERSION = 1

FLAG_SHADER_WARMING = 0x1
FLAG_NO_RETURN = 0x2

DEPTH_NONE = 0
DEPTH_U8 = 1
DEPTH_U16 = 2

_TEL_HEADER = struct.Struct("<4sHHIdIIHHBBH")
_RGB_HEADER = struct.Struct("<4sIHHI")
TELEMETRY_HEADER_SIZE = _TEL_HEADER.size  # 36
assert TELEMETRY_HEADER_SIZE == 36


@dataclass
class TelemetryFrame:
    seq: int
    t: float
    flags: int
    meta: dict[str, Any]
    points: np.ndarray  # (n, 3) float32
    depth: Optional[np.ndarray] = None  # (h, w) uint8 / uint16
    depth_enc: int = DEPTH_NONE

    @property
    def shader_warming(self) -> bool:
        return bool(self.flags & FLAG_SHADER_WARMING)


def quantize_depth_u8(depth_m: np.ndarray, depth_max_m: float) -> np.ndarray:
    """深度(米) → uint8；非有限值 / 超量程映射为 255（最远 / 无回波）。"""
    scaled = np.where(np.isfinite(depth_m), depth_m, depth_max_m) * (255.0 / depth_max_m)
    return np.clip(np.rint(scaled), 0, 255).astype(np.uint8)


def quantize_depth_u16(depth_m: np.ndarray, depth_max_m: float) -> np.ndarray:
    """深度(米) → uint16 毫米；非有限值映射为 depth_max。"""
    clean = np.where(np.isfinite(depth_m), depth_m, depth_max_m)
    return np.clip(np.rint(clean * 1000.0), 0, 65535).astype(np.uint16)


def encode_telemetry(
    seq: int,
    t: float,
    meta: dict[str, Any],
    points: Optional[np.ndarray] = None,
    depth: Optional[np.ndarray] = None,
    flags: int = 0,
) -> bytes:
    """把一帧遥测打包成单个二进制消息。``depth`` 必须已是 uint8 或 uint16 的 (h, w) 数组。"""
    pts = np.ascontiguousarray(points if points is not None else np.empty((0, 3)), dtype="<f4")
    if pts.size and (pts.ndim != 2 or pts.shape[1] != 3):
        raise ValueError("points 必须是 (n, 3)")
    n_points = int(pts.shape[0]) if pts.size else 0
    if depth is None:
        enc, dw, dh, depth_bytes = DEPTH_NONE, 0, 0, b""
    else:
        if depth.dtype == np.uint8:
            enc = DEPTH_U8
        elif depth.dtype == np.uint16:
            enc = DEPTH_U16
        else:
            raise ValueError("depth 必须是 uint8 或 uint16")
        dh, dw = depth.shape
        depth_bytes = np.ascontiguousarray(depth).astype(depth.dtype.newbyteorder("<"), copy=False).tobytes()
    meta_bytes = json.dumps(meta, separators=(",", ":"), allow_nan=False).encode("utf-8")
    meta_bytes += b" " * ((-len(meta_bytes)) % 4)
    header = _TEL_HEADER.pack(
        TELEMETRY_MAGIC, PROTOCOL_VERSION, flags, seq & 0xFFFFFFFF, t,
        len(meta_bytes), n_points, dw, dh, enc, 0, 0,
    )
    return b"".join((header, meta_bytes, pts.tobytes(), depth_bytes))


def decode_telemetry(data: bytes) -> TelemetryFrame:
    if len(data) < TELEMETRY_HEADER_SIZE:
        raise ValueError("遥测帧过短")
    (magic, version, flags, seq, t, meta_len, n_points, dw, dh, enc, _r1, _r2) = _TEL_HEADER.unpack_from(data, 0)
    if magic != TELEMETRY_MAGIC:
        raise ValueError(f"错误的 magic: {magic!r}")
    if version != PROTOCOL_VERSION:
        raise ValueError(f"不支持的协议版本: {version}")
    offset = TELEMETRY_HEADER_SIZE
    meta = json.loads(data[offset:offset + meta_len].decode("utf-8"))
    offset += meta_len
    pts_bytes = n_points * 12
    points = np.frombuffer(data, dtype="<f4", count=n_points * 3, offset=offset).reshape(n_points, 3)
    offset += pts_bytes
    depth = None
    if enc == DEPTH_U8:
        depth = np.frombuffer(data, dtype=np.uint8, count=dw * dh, offset=offset).reshape(dh, dw)
    elif enc == DEPTH_U16:
        depth = np.frombuffer(data, dtype="<u2", count=dw * dh, offset=offset).reshape(dh, dw)
    return TelemetryFrame(seq=seq, t=t, flags=flags, meta=meta, points=points, depth=depth, depth_enc=enc)


def encode_rgb(seq: int, width: int, height: int, png_bytes: bytes) -> bytes:
    return _RGB_HEADER.pack(RGB_MAGIC, seq & 0xFFFFFFFF, width, height, len(png_bytes)) + png_bytes


def decode_rgb(data: bytes) -> tuple[int, int, int, bytes]:
    magic, seq, width, height, length = _RGB_HEADER.unpack_from(data, 0)
    if magic != RGB_MAGIC:
        raise ValueError(f"错误的 magic: {magic!r}")
    start = _RGB_HEADER.size
    return seq, width, height, data[start:start + length]


def frame_kind(data: bytes) -> str:
    """返回 'telemetry' / 'rgb' / 'unknown'（按 magic 判断）。"""
    magic = bytes(data[:4])
    if magic == TELEMETRY_MAGIC:
        return "telemetry"
    if magic == RGB_MAGIC:
        return "rgb"
    return "unknown"
