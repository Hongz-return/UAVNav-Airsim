"""只读 AirSim 采集器（Tap）。

关键保证
--------
1. **100% 只读**：所有 RPC 都经过 :class:`ReadOnlyClient` 白名单代理，仅允许
   ``ping / getSettingsString / getMultirotorState / simGetVehiclePose / simGetImages /
   simListSceneObjects / simGetObjectPose / simGetObjectScale``。``enableApiControl / armDisarm /
   simPause / reset / moveByVelocity*`` 等任何写接口都会在客户端侧直接抛 :class:`ReadOnlyViolation`，
   根本不会发往仿真器。
2. **训练旁路模式 (bypass)**：4–6 Hz（默认 5 Hz）、点云 80×45、仅 DepthPlanar、
   ``timeout=2.0 s``；任一次采集 RPC 耗时 > 250 ms 自动退避到 2.5 Hz，
   连续 10 次 RPC 均 < 120 ms 才恢复；无任何网页观众时完全暂停采集（零 RPC）。
3. **直连模式 (direct)**：20 Hz、点云 160×90、可同步请求 RGB。
4. **断线 / UE 重置**：指数退避自动重连，任何异常都不会使线程退出。
5. **未编译着色器防护**：DepthPlanar ``max_depth <= 1.2 m``（UE 延迟编译产出 [0,1] 亮度帧）→
   ``shader_warming=true`` 并发送空点云，避免渲染虚假"贴脸墙"。
"""

from __future__ import annotations

import asyncio
import collections
import json
import logging
import math
import random
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Optional

import numpy as np

from .coords import (
    camera_ray_factors,
    depth_to_world_cloud,
    euler_from_enu_quat,
    ned_to_enu_point,
    ned_to_enu_quat,
    normalize_quat,
    quat_is_valid,
    quat_to_matrix,
    quat_wxyz,
    rotate_points,
    vec3,
)
from .geometry_loader import DEFAULT_FLOOR_Z, DEFAULT_SPAWN_NED_Z, GeometryLoader
from .protocol import (
    FLAG_NO_RETURN,
    FLAG_SHADER_WARMING,
    encode_rgb,
    encode_telemetry,
    quantize_depth_u16,
    quantize_depth_u8,
)

LOG = logging.getLogger("uav_rviz.tap")

# ---------------------------------------------------------------------------
# 只读保护
# ---------------------------------------------------------------------------

READ_ONLY_METHODS = frozenset(
    {
        "ping",
        "getSettingsString",
        "getMultirotorState",
        "simGetVehiclePose",
        "simGetImages",
        "simListSceneObjects",
        "simGetObjectPose",
        "simGetObjectScale",
    }
)


class ReadOnlyViolation(PermissionError):
    """尝试调用白名单之外的 AirSim 接口。"""


class ReadOnlyClient:
    """AirSim 客户端的只读白名单代理；同时统计各接口调用次数（便于审计）。"""

    def __init__(self, client: Any) -> None:
        object.__setattr__(self, "_client", client)
        object.__setattr__(self, "calls", collections.Counter())

    def __getattr__(self, name: str) -> Any:
        if name not in READ_ONLY_METHODS:
            raise ReadOnlyViolation(f"uav_rviz 严格只读：禁止调用 AirSim 接口 {name!r}")
        method = getattr(self._client, name)

        def _call(*args: Any, **kwargs: Any) -> Any:
            self.calls[name] += 1
            return method(*args, **kwargs)

        return _call

    def __setattr__(self, name: str, value: Any) -> None:
        raise ReadOnlyViolation("ReadOnlyClient 不可修改")

    def close(self) -> None:
        inner = getattr(self._client, "client", None)  # cosysairsim: msgpackrpc.Client
        for target in (inner, self._client):
            closer = getattr(target, "close", None)
            if callable(closer):
                try:
                    closer()
                except Exception:  # 关闭失败不影响重连
                    pass
                return


def connect_airsim(host: str, port: int, timeout: float) -> ReadOnlyClient:
    """连接真实 AirSim（cosysairsim）。不调用 confirmConnection / enableApiControl 等任何写接口。"""
    import cosysairsim as airsim  # 延迟导入：单元测试与 --simulate 不需要

    client = airsim.MultirotorClient(ip=host, port=int(port), timeout_value=float(timeout))
    wrapped = ReadOnlyClient(client)
    try:
        wrapped.ping()
    except Exception:
        wrapped.close()
        raise
    return wrapped


# ---------------------------------------------------------------------------
# 模式配置 / 限速 / 退避
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ModeProfile:
    name: str
    label: str
    hz: float
    degraded_hz: float
    grid: tuple[int, int]
    timeout: float
    adaptive: bool
    slow_rpc_ms: float
    recover_rpc_ms: float
    recover_ticks: int
    scene_scan_s: float
    depth_enc: str  # "u8" | "u16"
    max_range: float = 25.0


PROFILES: dict[str, ModeProfile] = {
    "bypass": ModeProfile(
        name="bypass", label="训练旁路 (低开销)", hz=5.0, degraded_hz=2.5, grid=(80, 45), timeout=2.0,
        adaptive=True, slow_rpc_ms=250.0, recover_rpc_ms=120.0, recover_ticks=10,
        scene_scan_s=45.0, depth_enc="u8",
    ),
    "direct": ModeProfile(
        name="direct", label="演示直连 (20Hz)", hz=20.0, degraded_hz=20.0, grid=(160, 90), timeout=5.0,
        adaptive=False, slow_rpc_ms=1e9, recover_rpc_ms=0.0, recover_ticks=1,
        scene_scan_s=5.0, depth_enc="u16",
    ),
}


class RateGovernor:
    """旁路模式自适应让行：慢 RPC → 降频；连续快 RPC → 恢复。"""

    def __init__(self, profile: ModeProfile) -> None:
        self.profile = profile
        self.degraded = False
        self._fast_streak = 0

    def set_profile(self, profile: ModeProfile) -> None:
        self.profile = profile
        self.degraded = False
        self._fast_streak = 0

    @property
    def hz(self) -> float:
        return self.profile.degraded_hz if self.degraded else self.profile.hz

    @property
    def interval(self) -> float:
        return 1.0 / max(self.hz, 0.1)

    def observe(self, max_rpc_ms: float) -> bool:
        """喂入本 tick 内最慢的单次 RPC 耗时(ms)；返回 degraded 状态是否发生变化。"""
        p = self.profile
        if not p.adaptive:
            return False
        before = self.degraded
        if max_rpc_ms > p.slow_rpc_ms:
            self.degraded = True
            self._fast_streak = 0
        elif self.degraded:
            if max_rpc_ms <= p.recover_rpc_ms:
                self._fast_streak += 1
                if self._fast_streak >= p.recover_ticks:
                    self.degraded = False
                    self._fast_streak = 0
            else:
                self._fast_streak = 0
        return before != self.degraded


class Backoff:
    """指数退避（带抖动），上限 ``maximum`` 秒。"""

    def __init__(self, base: float = 0.5, factor: float = 2.0, maximum: float = 10.0, jitter: float = 0.2,
                 rng: Optional[random.Random] = None) -> None:
        self.base, self.factor, self.maximum, self.jitter = base, factor, maximum, jitter
        self._delay = base
        self._rng = rng or random.Random()

    def next(self) -> float:
        delay = self._delay * (1.0 + self._rng.uniform(-self.jitter, self.jitter))
        self._delay = min(self.maximum, self._delay * self.factor)
        return min(self.maximum, delay)

    def reset(self) -> None:
        self._delay = self.base


# ---------------------------------------------------------------------------
# 深度分析
# ---------------------------------------------------------------------------

SHADER_WARMING_MAX_DEPTH_M = 1.2


@dataclass
class DepthStats:
    finite_count: int
    min_depth: Optional[float]
    center_depth: Optional[float]
    max_depth: Optional[float]
    min_range: Optional[float]  # 欧氏距离最小值（考虑像素射线倾角）
    shader_warming: bool
    no_return: bool


def analyze_depth(depth: np.ndarray, fov_deg: float = 90.0) -> DepthStats:
    """基于全分辨率深度计算 min/center/max，并判定未编译着色器的 [0,1] 亮度帧。"""
    finite = np.isfinite(depth)
    count = int(finite.sum())
    if count == 0:
        return DepthStats(0, None, None, None, None, False, True)
    h, w = depth.shape
    right, down = camera_ray_factors(w, h, float(fov_deg))
    positive = finite & (depth > 0.01)
    d_max = float(depth[finite].max())
    if positive.any():
        d_min = float(depth[positive].min())
        rng = depth[positive] * np.sqrt(1.0 + right[positive] ** 2 + down[positive] ** 2)
        r_min: Optional[float] = float(rng.min())
    else:
        d_min, r_min = 0.0, 0.0
    cy, cx = h // 2, w // 2
    block = depth[max(0, cy - 2):cy + 3, max(0, cx - 2):cx + 3]
    block = block[np.isfinite(block) & (block > 0.01)]
    center = float(block.mean()) if block.size else None
    warming = d_max <= SHADER_WARMING_MAX_DEPTH_M
    return DepthStats(count, d_min, center, d_max, r_min, warming, False)


def _num(value: Optional[float], digits: int = 4) -> Optional[float]:
    if value is None:
        return None
    value = float(value)
    return round(value, digits) if math.isfinite(value) else None


def _vec(values: Any, digits: int = 4) -> list[Optional[float]]:
    return [_num(v, digits) for v in values]


# ---------------------------------------------------------------------------
# 采集器
# ---------------------------------------------------------------------------


@dataclass
class StepResult:
    telemetry: bytes
    rgb: Optional[bytes]
    meta: dict[str, Any]
    max_rpc_ms: float


class AirSimTap:
    """只读采集线程；``step()`` 可在单元测试中同步调用。"""

    def __init__(
        self,
        *,
        host: str = "127.0.0.1",
        port: int = 42072,
        mode: str = "bypass",
        client_factory: Optional[Callable[[str, int, float], Any]] = None,
        api: Any = None,
        camera: str = "0",
        fov_deg: float = 90.0,
        geometry: Optional[GeometryLoader] = None,
        floor_z: float = DEFAULT_FLOOR_Z,
        spawn_ned_z: float = DEFAULT_SPAWN_NED_Z,
        depth_max_m: float = 25.0,
        always_on: bool = False,
        on_telemetry: Optional[Callable[[bytes], None]] = None,
        on_rgb: Optional[Callable[[bytes], None]] = None,
        on_scene: Optional[Callable[[dict], None]] = None,
        on_status: Optional[Callable[[dict], None]] = None,
        on_event: Optional[Callable[[dict], None]] = None,
        clock: Callable[[], float] = time.perf_counter,
        backoff: Optional[Backoff] = None,
    ) -> None:
        if mode not in PROFILES:
            raise ValueError(f"未知模式 {mode!r}")
        self.host = host
        self.port = int(port)
        self.profile = PROFILES[mode]
        self.governor = RateGovernor(self.profile)
        self.client_factory = client_factory or connect_airsim
        self._api = api
        self.camera = camera
        self.fov_deg = float(fov_deg)
        self.floor_z = floor_z
        self.spawn_ned_z = spawn_ned_z
        self.depth_max_m = depth_max_m
        self.always_on = always_on
        self.geometry = geometry or GeometryLoader(None, spawn_ned_z=spawn_ned_z, floor_z=floor_z)
        self.on_telemetry = on_telemetry
        self.on_rgb = on_rgb
        self.on_scene = on_scene
        self.on_status = on_status
        self.on_event = on_event
        self._clock = clock
        self._backoff = backoff or Backoff()

        self.rgb_enabled = False
        self.episode = 0
        self.state = "idle"
        self.last_error: Optional[str] = None
        self.retry_in = 0.0
        self._retry_at = 0.0
        self.shader_warming = False
        self.consecutive_failures = 0

        self._client: Optional[ReadOnlyClient] = None
        self._expected_navrl_launch_id: Optional[str] = None
        self._reported_navrl_launch_id: Optional[str] = None
        self._seq = 0
        self._viewers = 0
        self._last_pos: Optional[tuple[float, float, float]] = None
        self._last_sim_ts: Optional[int] = None
        self._last_collision_ts: Optional[int] = None
        self._pub_times: collections.deque[float] = collections.deque(maxlen=30)
        self._last_run_poll = 0.0
        self._pending: dict[str, Any] = {}
        self._pending_lock = threading.Lock()
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    # -- 公共线程安全接口 ------------------------------------------------
    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="uav-rviz-tap", daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 5.0) -> None:
        self._stop.set()
        self._wake.set()
        if self._thread:
            self._thread.join(timeout)

    def _post(self, **items: Any) -> None:
        with self._pending_lock:
            self._pending.update(items)
        self._wake.set()

    def set_port(self, port: int) -> None:
        self._post(port=int(port))

    def set_mode(self, mode: str) -> None:
        if mode not in PROFILES:
            raise ValueError(f"未知模式 {mode!r}")
        self._post(mode=mode)

    def set_rgb(self, enabled: bool) -> None:
        self._post(rgb=bool(enabled))

    def refresh_scene(self) -> None:
        self._post(refresh_scene=True)

    def clear_trail(self) -> None:
        self._post(clear_trail=True)

    def set_viewers(self, count: int) -> None:
        self._viewers = max(0, int(count))
        self._wake.set()

    # -- 状态 --------------------------------------------------------------
    def status(self) -> dict[str, Any]:
        p = self.profile
        calls = dict(self._client.calls) if self._client is not None else {}
        return {
            "type": "status",
            "state": self.state,
            "connected": self.state == "connected",
            "airsim_host": self.host,
            "port": self.port,
            "mode": p.name,
            "mode_label": p.label,
            "hz_target": round(self.governor.hz, 2),
            "degraded": self.governor.degraded,
            "grid": list(p.grid),
            "rgb": self.rgb_enabled,
            "viewers": self._viewers,
            "error": self.last_error,
            "retry_in": round(self.retry_in, 1),
            "episode": self.episode,
            "shader_warming": self.shader_warming,
            "rpc_calls": calls,
            "read_only": True,
        }

    def _emit_status(self) -> None:
        if self.on_status:
            self.on_status(self.status())

    def _set_state(self, state: str, error: Optional[str] = None) -> None:
        changed = state != self.state or error != self.last_error
        self.state, self.last_error = state, error
        if changed:
            self._emit_status()

    # -- 连接管理 ----------------------------------------------------------
    def _connect(self) -> bool:
        self._set_state("connecting")
        try:
            client = self.client_factory(self.host, self.port, self.profile.timeout)
            if not isinstance(client, ReadOnlyClient):
                client = ReadOnlyClient(client)
        except Exception as exc:
            self._on_connect_failed(f"{type(exc).__name__}: {exc}")
            return False
        self._client = client
        self.consecutive_failures = 0
        self.retry_in = 0.0
        self._retry_at = 0.0
        self._last_pos = None
        self._last_sim_ts = None
        self.geometry.attach(client)
        self._reported_navrl_launch_id = None
        self.geometry.poll_run_dir(rpc_port=self.port)
        self._expected_navrl_launch_id = self.geometry.navrl_launch_id
        if self._expected_navrl_launch_id is not None:
            self._reported_navrl_launch_id = self._read_navrl_launch_id(client)
            self._poll_run_dir()
        self.geometry.scan_live()
        self._emit_scene()
        self._set_state("connected")
        LOG.info("已连接 AirSim %s:%s (%s)", self.host, self.port, self.profile.name)
        return True

    def _read_navrl_launch_id(self, client: ReadOnlyClient) -> Optional[str]:
        try:
            settings = json.loads(client.getSettingsString())
        except Exception:
            return None  # Settings are optional for pose telemetry.
        if not isinstance(settings, dict) or settings.get("ApiServerPort") != self.port:
            return None
        launch_id = settings.get("TrainEnvLaunchId")
        return launch_id if isinstance(launch_id, str) and launch_id else None

    def _poll_run_dir(self) -> bool:
        return self.geometry.poll_run_dir(rpc_port=self.port, launch_id=self._reported_navrl_launch_id)

    def _on_connect_failed(self, error: str) -> None:
        self._schedule_reconnect(error)
        LOG.warning("AirSim 连接失败 (%s)，%.1fs 后重试", error, self.retry_in)

    def _schedule_reconnect(self, error: str) -> None:
        self._close_client()
        self.retry_in = self._backoff.next()
        self._retry_at = self._clock() + self.retry_in
        self._set_state("reconnecting", error)

    def _close_client(self) -> None:
        client, self._client = self._client, None
        if client is not None:
            client.close()
            if self.geometry.is_navrl_run:
                self.geometry.attach(None)
                self._emit_scene()

    def _emit_scene(self) -> None:
        if self.on_scene:
            self.on_scene(self.geometry.scene_message())

    def _apply_pending(self) -> None:
        with self._pending_lock:
            pending, self._pending = self._pending, {}
        reconnect = False
        if "port" in pending and pending["port"] != self.port:
            self.port = pending["port"]
            reconnect = True
        if "mode" in pending and pending["mode"] != self.profile.name:
            old_timeout = self.profile.timeout
            self.profile = PROFILES[pending["mode"]]
            self.governor.set_profile(self.profile)
            reconnect = reconnect or self.profile.timeout != old_timeout
        if "rgb" in pending:
            self.rgb_enabled = pending["rgb"] and self.profile.name == "direct"
        if self.profile.name == "bypass":
            self.rgb_enabled = False
        if "rgb" in pending and not reconnect:
            self._emit_status()
        if pending.get("clear_trail"):
            self.episode += 1
            if self.on_event:
                self.on_event({"type": "event", "event": "clear_trail", "episode": self.episode})
        if reconnect:
            self._close_client()
            self._backoff.reset()
            self.retry_in = 0.0
            self._retry_at = 0.0
            self._pub_times.clear()
            self._last_collision_ts = None
            self.shader_warming = False
            self.state, self.last_error = "idle", None
            self._emit_status()
        if pending.get("refresh_scene") and self._client is not None:
            self.geometry.force_rescan()
            self._poll_run_dir()
            self._emit_scene()

    # -- 主循环 --------------------------------------------------------------
    def _run(self) -> None:
        # msgpackrpc / tornado 需要线程内存在 asyncio 事件循环
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        next_sample = 0.0
        try:
            while not self._stop.is_set():
                try:
                    self._apply_pending()
                    if self._viewers == 0 and not self.always_on:
                        self._set_state("paused")
                        self._last_pos = None
                        self._pub_times.clear()
                        next_sample = 0.0
                        self._sleep(0.25)
                        continue
                    if self._client is None:
                        remaining = self._retry_at - self._clock()
                        if remaining > 0:
                            self._sleep(remaining)
                            continue
                        if not self._connect():
                            continue
                        next_sample = 0.0
                    elif self.state == "paused":
                        self._set_state("connected")
                    started = self._clock()
                    if started < next_sample:
                        self._sleep(next_sample - started)
                        continue
                    self._tick(started)
                    interval = self.governor.interval
                    if self.profile.name == "direct" and started - next_sample < interval:
                        next_sample += interval  # 固定时间基准，避免 Windows 等待误差逐帧累加。
                    else:
                        next_sample = started + interval
                except Exception as exc:
                    LOG.exception("采集循环异常: %s", exc)
                    self._schedule_reconnect(f"{type(exc).__name__}: {exc}")
        finally:
            self._close_client()
            loop.close()

    def _sleep(self, seconds: float) -> None:
        if seconds > 0:
            self._wake.wait(seconds)
        self._wake.clear()

    def _tick(self, now: float) -> None:
        poll_interval = 0.5 if self.geometry.is_navrl_run else 2.0
        if now - self._last_run_poll >= poll_interval:
            self._last_run_poll = now
            scene_changed = self._poll_run_dir()
            launch_id = self.geometry.navrl_launch_id
            if launch_id is not None and launch_id != self._expected_navrl_launch_id:
                self._schedule_reconnect("NavRL 训练实例已切换")
                return
            if scene_changed:
                self._emit_scene()
        try:
            result = self.step()
        except Exception as exc:
            self.consecutive_failures += 1
            error = f"{type(exc).__name__}: {exc}"
            LOG.warning("采集失败 #%d: %s", self.consecutive_failures, error)
            if self.consecutive_failures >= 3:
                self._schedule_reconnect(error)
            else:
                self._set_state(self.state, error)
            return
        self.consecutive_failures = 0
        self._backoff.reset()
        if self.last_error:
            self._set_state(self.state, None)
        if self.governor.observe(result.max_rpc_ms):
            LOG.info("旁路自适应退避 degraded=%s hz=%.1f", self.governor.degraded, self.governor.hz)
            self._emit_status()
        reset = bool(result.meta.get("reset"))
        scan = self.geometry.client is not None and (reset or now - self.geometry.last_scan_time > self.profile.scene_scan_s)
        if scan:
            self.geometry.scan_live()
        goal_changed = self.geometry.update_live_pose(tuple(result.meta["pos"]), tuple(result.meta["vel"]))
        if scan or goal_changed:
            self._emit_scene()

    # -- 单次采样 ------------------------------------------------------------
    def _api_module(self) -> Any:
        if self._api is None:
            import cosysairsim as airsim

            self._api = airsim
        return self._api

    def step(self) -> StepResult:
        """执行一次只读采样并（若设置了回调）发布遥测帧。"""
        client = self._client
        if client is None:
            raise RuntimeError("未连接")
        api = self._api_module()
        prof = self.profile
        rpc: dict[str, float] = {}

        def timed(name: str, fn: Callable[..., Any], *args: Any) -> Any:
            t0 = self._clock()
            value = fn(*args)
            rpc[name] = (self._clock() - t0) * 1000.0
            return value

        pose = timed("pose", client.simGetVehiclePose)
        state = timed("state", client.getMultirotorState)
        # CosysAirSim's simGetCollisionInfo consumes the flag used by the training task.
        collision = getattr(state, "collision", None)
        want_rgb = prof.name == "direct" and self.rgb_enabled
        requests = [api.ImageRequest(self.camera, api.ImageType.DepthPlanar, True, False)]
        if want_rgb:
            requests.append(api.ImageRequest(self.camera, api.ImageType.Scene, False, True))
        responses = timed("images", client.simGetImages, requests)

        # --- 车辆状态 (NED → ENU) ---
        pos_ned = vec3(pose.position)
        q_ned = normalize_quat(quat_wxyz(pose.orientation))
        pos = ned_to_enu_point(pos_ned)
        q_enu = ned_to_enu_quat(q_ned)
        kin = state.kinematics_estimated
        vel = ned_to_enu_point(vec3(kin.linear_velocity))
        rot = quat_to_matrix(q_enu)
        vel_body = tuple(float(v) for v in rotate_points(np.asarray([vel]), rot.T)[0])  # = Rᵀ·v（世界→机体 FLU）
        speed = math.sqrt(vel[0] ** 2 + vel[1] ** 2 + vel[2] ** 2)
        sim_ts = int(getattr(state, "timestamp", 0) or 0)

        reset = False
        if self._last_pos is not None:
            jump = math.dist(pos, self._last_pos)
            if jump > 3.0 or (self._last_sim_ts and sim_ts and sim_ts < self._last_sim_ts - 1_000_000):
                reset = True
                self.episode += 1
        self._last_pos, self._last_sim_ts = pos, sim_ts

        # --- 相机位姿（优先使用响应内的真实相机位姿，缺失则回退车辆位姿） ---
        depth_resp = responses[0] if responses else None
        cam_pos_ned, cam_q_ned = pos_ned, q_ned
        if depth_resp is not None:
            try:
                cp, cq = vec3(depth_resp.camera_position), normalize_quat(quat_wxyz(depth_resp.camera_orientation))
                raw_q = quat_wxyz(depth_resp.camera_orientation)
                if quat_is_valid(raw_q) and all(math.isfinite(c) for c in cp) and math.dist(cp, pos_ned) < 2.0:
                    cam_pos_ned, cam_q_ned = cp, cq
            except (AttributeError, TypeError, ValueError):
                pass
        cam_pos_enu = ned_to_enu_point(cam_pos_ned)
        cam_q_enu = ned_to_enu_quat(cam_q_ned)

        # --- 深度 → 点云 ---
        flags = 0
        points = np.empty((0, 3), dtype=np.float32)
        depth_img: Optional[np.ndarray] = None
        depth_meta: dict[str, Any] = {"w": 0, "h": 0, "enc": "none", "max_m": self.depth_max_m}
        stats = DepthStats(0, None, None, None, None, False, False)
        depth_valid = False
        if depth_resp is not None and depth_resp.width > 0 and depth_resp.height > 0:
            w, h = int(depth_resp.width), int(depth_resp.height)
            data = np.asarray(depth_resp.image_data_float, dtype=np.float32)
            if data.size == w * h:
                depth_valid = True
                depth_full = data.reshape(h, w)
                stats = analyze_depth(depth_full, self.fov_deg)
                if stats.no_return:
                    flags |= FLAG_NO_RETURN
                elif not stats.shader_warming:
                    points, grid_depth, valid = depth_to_world_cloud(
                        depth_full, cam_pos_ned, cam_q_ned, self.fov_deg, prof.grid,
                        min_range=0.1, max_range=prof.max_range,
                    )
                    shown = np.where(valid, grid_depth, np.nan)
                    if prof.depth_enc == "u16":
                        depth_img = quantize_depth_u16(shown, self.depth_max_m)
                    else:
                        depth_img = quantize_depth_u8(shown, self.depth_max_m)
                    depth_meta.update(w=int(depth_img.shape[1]), h=int(depth_img.shape[0]), enc=prof.depth_enc)
        warming = (not depth_valid) or stats.shader_warming
        if warming:
            flags |= FLAG_SHADER_WARMING
        self.shader_warming = warming
        depth_meta.update(
            min=_num(stats.min_depth, 3), center=_num(stats.center_depth, 3), max=_num(stats.max_depth, 3),
            min_range=_num(stats.min_range, 3),
        )

        # --- 碰撞 / 目标误差 ---
        col_ts = int(getattr(collision, "time_stamp", 0) or 0)
        has_collided = bool(getattr(collision, "has_collided", False))
        new_collision = has_collided and col_ts != self._last_collision_ts
        if has_collided:
            self._last_collision_ts = col_ts
        goal = self.geometry.effective_goal()
        goal_err = None
        if goal is not None:
            goal_err = {
                "horizontal": _num(math.hypot(goal.position[0] - pos[0], goal.position[1] - pos[1]), 3),
                "vertical": _num(goal.position[2] - pos[2], 3),
            }

        self._seq += 1
        now = time.time()
        self._pub_times.append(self._clock())
        fps = 0.0
        if len(self._pub_times) >= 2 and self._pub_times[-1] > self._pub_times[0]:
            fps = (len(self._pub_times) - 1) / (self._pub_times[-1] - self._pub_times[0])
        max_rpc = max(rpc.values()) if rpc else 0.0
        meta: dict[str, Any] = {
            "seq": self._seq,
            "t": round(now, 3),
            "sim_ts": sim_ts,
            "mode": prof.name,
            "port": self.port,
            "hz": round(self.governor.hz, 2),
            "degraded": self.governor.degraded,
            "fps": round(fps, 2),
            "rpc_ms": round(max_rpc, 1),
            "rpc_detail": {k: round(v, 1) for k, v in rpc.items()},
            "episode": self.episode,
            "reset": reset,
            "pos": _vec(pos),
            "agl": _num(pos[2] - self.floor_z),
            "quat": _vec(q_enu, 6),
            "euler": {k: _num(v, 2) for k, v in euler_from_enu_quat(q_enu).items()},
            "vel": _vec(vel),
            "vel_body": _vec(vel_body),
            "speed": _num(speed),
            "camera": {"pos": _vec(cam_pos_enu), "quat": _vec(cam_q_enu, 6)},
            "collision": {
                "available": has_collided or col_ts > 0,
                "has_collided": has_collided,
                "new": new_collision,
                "object": str(getattr(collision, "object_name", "") or ""),
                "ts": col_ts,
            },
            "shader_warming": warming,
            "no_return": bool(stats.no_return),
            "n_points": int(points.shape[0]),
            "depth": depth_meta,
            "goal_err": goal_err,
            "rgb": bool(want_rgb),
        }
        telemetry = encode_telemetry(self._seq, now, meta, points, depth_img, flags)

        rgb_bytes: Optional[bytes] = None
        if want_rgb and len(responses) > 1:
            rgb_resp = responses[1]
            png = bytes(rgb_resp.image_data_uint8)
            if png.startswith(b"\x89PNG") and rgb_resp.width > 0:
                rgb_bytes = encode_rgb(self._seq, int(rgb_resp.width), int(rgb_resp.height), png)

        if self.on_telemetry:
            self.on_telemetry(telemetry)
        if rgb_bytes is not None and self.on_rgb:
            self.on_rgb(rgb_bytes)
        return StepResult(telemetry=telemetry, rgb=rgb_bytes, meta=meta, max_rpc_ms=max_rpc)


