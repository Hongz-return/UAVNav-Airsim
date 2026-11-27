"""uav_rviz 单元 / 集成测试：坐标系、针孔反投影、着色器防护、RFC6455、二进制帧、旁路降频、只读保护。

全部使用模拟 AirSim（``tools.uav_rviz.simulated``），不连接任何真实端口，不触碰正在运行的训练。
"""

from __future__ import annotations

import asyncio
import json
import math
import os
import random
import re
import tempfile
import time
import unittest
from collections import Counter
from pathlib import Path
from unittest.mock import patch
from uuid import NAMESPACE_URL, uuid5

import numpy as np

from tools.uav_rviz import airsim_tap, coords, geometry_loader, protocol, ws_server
from tools.uav_rviz.airsim_tap import (
    PROFILES, READ_ONLY_METHODS, AirSimTap, Backoff, RateGovernor, ReadOnlyClient, ReadOnlyViolation, analyze_depth,
)
from tools.uav_rviz.simulated import SIM_API, SimulatedAirSim

PACKAGE_DIR = Path(airsim_tap.__file__).resolve().parent


def airsim_quat(roll: float, pitch: float, yaw: float) -> tuple[float, float, float, float]:
    """与 gym_env.envs.dynamics.multirotor_airsim.to_quaternion 相同的 AirSim NED 欧拉→四元数 (w, x, y, z)。"""
    cy, sy = math.cos(yaw * 0.5), math.sin(yaw * 0.5)
    cr, sr = math.cos(roll * 0.5), math.sin(roll * 0.5)
    cp, sp = math.cos(pitch * 0.5), math.sin(pitch * 0.5)
    return (cy * cr * cp + sy * sr * sp, cy * sr * cp - sy * cr * sp, cy * cr * sp + sy * sr * cp,
            sy * cr * cp - cy * sr * sp)


def det3(m: np.ndarray) -> float:
    """3x3 行列式（显式公式，避免依赖 numpy.linalg）。"""
    return float(m[0, 0] * (m[1, 1] * m[2, 2] - m[1, 2] * m[2, 1]) - m[0, 1] * (m[1, 0] * m[2, 2] - m[1, 2] * m[2, 0])
                 + m[0, 2] * (m[1, 0] * m[2, 1] - m[1, 1] * m[2, 0]))


def enu_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    return coords.quat_to_matrix(coords.ned_to_enu_quat(airsim_quat(roll, pitch, yaw)))


class CoordinateTests(unittest.TestCase):
    def test_ned_to_enu_point(self):
        self.assertEqual(coords.ned_to_enu_point((1.0, 2.0, 3.0)), (1.0, -2.0, -3.0))

    def test_enu_frame_is_right_handed_rotation(self):
        d = np.diag([1.0, -1.0, -1.0])
        self.assertAlmostEqual(det3(d), 1.0)
        rng = random.Random(7)
        for _ in range(20):
            q = coords.normalize_quat([rng.uniform(-1, 1) for _ in range(4)])
            r_ned = coords.quat_to_matrix(q)
            r_enu = coords.quat_to_matrix(coords.ned_to_enu_quat(q))
            np.testing.assert_allclose(r_enu, d @ r_ned @ d, atol=1e-9)
            np.testing.assert_allclose(det3(r_enu), 1.0, atol=1e-9)

    def test_yaw_plus_90_nose_points_right(self):
        # AirSim yaw=+90°：机头转向 NED +Y(右/东) → 本坐标系 -Y；机体左向量指向 +X
        r = enu_matrix(0.0, 0.0, math.radians(90))
        np.testing.assert_allclose(r[:, 0], [0, -1, 0], atol=1e-9)
        np.testing.assert_allclose(r[:, 1], [1, 0, 0], atol=1e-9)
        np.testing.assert_allclose(r[:, 2], [0, 0, 1], atol=1e-9)

    def test_pitch_positive_nose_up(self):
        r = enu_matrix(0.0, math.radians(15), 0.0)
        np.testing.assert_allclose(r[:, 0], [math.cos(math.radians(15)), 0, math.sin(math.radians(15))], atol=1e-9)
        self.assertGreater(r[2, 0], 0.0)

    def test_roll_positive_right_wing_down(self):
        r = enu_matrix(math.radians(20), 0.0, 0.0)
        # 右翼下沉 ⇒ 机体左向量上抬，机体上向量朝右(−Y)倾
        self.assertAlmostEqual(r[2, 1], math.sin(math.radians(20)), places=9)
        self.assertLess(r[1, 2], 0.0)

    def test_euler_round_trip(self):
        rng = random.Random(3)
        for _ in range(50):
            roll, pitch, yaw = (math.radians(rng.uniform(-60, 60)), math.radians(rng.uniform(-60, 60)),
                                math.radians(rng.uniform(-170, 170)))
            q_enu = coords.ned_to_enu_quat(airsim_quat(roll, pitch, yaw))
            e = coords.euler_from_enu_quat(q_enu)
            self.assertAlmostEqual(e["roll"], math.degrees(roll), places=6)
            self.assertAlmostEqual(e["pitch"], math.degrees(pitch), places=6)
            self.assertAlmostEqual(e["yaw"], -math.degrees(yaw), places=6)  # NED 顺时针 → ENU 逆时针

    def test_object_pose_offset(self):
        self.assertEqual(coords.object_ned_to_local_enu((10.0, 2.0, 0.2), spawn_ned_z=1.0), (10.0, -2.0, 0.8))


class PinholeTests(unittest.TestCase):
    W, H = 320, 180

    def plane(self, d=5.0):
        return np.full((self.H, self.W), d, dtype=np.float32)

    def cloud(self, yaw=0.0, pitch=0.0, pos_ned=(0.0, 0.0, -2.0), depth=None):
        return coords.depth_to_world_cloud(
            self.plane() if depth is None else depth, pos_ned, airsim_quat(0.0, math.radians(pitch), math.radians(yaw)),
            90.0, (80, 45),
        )[0]

    def test_focal_length_for_90_degree_fov(self):
        self.assertAlmostEqual(coords.focal_length_px(320, 90.0), 160.0)

    def test_grid_caps_nondivisible_resolution(self):
        rows, cols = coords.sample_grid(256, 144, 160, 90)
        self.assertEqual((len(cols), len(rows)), (160, 90))
        self.assertTrue(np.all(np.diff(rows) > 0))
        self.assertTrue(np.all(np.diff(cols) > 0))

    def test_forward_plane_axes_are_x_front_y_left_z_up(self):
        pts = self.cloud()
        self.assertEqual(pts.shape[0], 80 * 45)
        np.testing.assert_allclose(pts[:, 0], 5.0, atol=1e-5)  # 全部在前方 5 m 平面
        self.assertGreater(pts[0, 1], 0.0)  # 第一个点 = 图像左上角：左(+Y)
        self.assertGreater(pts[0, 2], 2.0)  # 且位于相机上方(Z-up)
        self.assertLess(pts[-1, 1], 0.0)  # 右下角：右(−Y)
        self.assertLess(pts[-1, 2], 2.0)
        # 90° FOV：边缘像素中心 (cx - 2)/fx*d
        self.assertAlmostEqual(float(pts[:, 1].max()), 5.0 * (159.5 - 2) / 160.0, places=4)
        self.assertAlmostEqual(float(pts[:, 2].max()), 2.0 + 5.0 * (89.5 - 2) / 160.0, places=4)
        np.testing.assert_allclose(pts.mean(axis=0), [5.0, 0.0, 2.0], atol=0.05)

    def test_yaw_plus_90_plane_lands_on_negative_y(self):
        pts = self.cloud(yaw=90.0)
        np.testing.assert_allclose(pts[:, 1], -5.0, atol=1e-4)
        self.assertGreater(pts[0, 0], 0.0)  # 图像左上角 → 机体左 → 世界 +X
        self.assertLess(pts[-1, 0], 0.0)
        np.testing.assert_allclose(pts.mean(axis=0), [0.0, -5.0, 2.0], atol=0.05)

    def test_pitch_up_tilts_cloud_upwards(self):
        pts = self.cloud(pitch=15.0)
        expected = [5.0 * math.cos(math.radians(15)), 0.0, 2.0 + 5.0 * math.sin(math.radians(15))]
        np.testing.assert_allclose(pts.mean(axis=0), expected, atol=0.05)

    def test_nan_inf_and_out_of_range_dropped(self):
        depth = self.plane()
        depth[:90, :] = np.nan
        depth[90:120, :] = np.inf
        depth[120:150, :] = 0.01
        depth[150:, :] = 30.0  # 超过 max_range=25
        pts = self.cloud(depth=depth)
        self.assertEqual(pts.shape[0], 0)

    def test_camera_frame_flu_center_ray(self):
        right, down = coords.camera_ray_factors(self.W, self.H, 90.0)
        flu = coords.depth_to_camera_flu(self.plane(4.0), right, down)
        np.testing.assert_allclose(flu[90, 160], [4.0, -4.0 * right[90, 160], -4.0 * down[90, 160]])
        self.assertAlmostEqual(float(flu[90, 160, 1]), 0.0, delta=0.02)
        self.assertGreater(float(flu[0, 0, 1]), 0.0)  # 左上：左
        self.assertGreater(float(flu[0, 0, 2]), 0.0)  # 左上：上


class ShaderWarmingTests(unittest.TestCase):
    def test_uncompiled_unit_brightness_frame_flagged(self):
        depth = np.random.default_rng(0).random((180, 320), dtype=np.float32)  # [0,1) 亮度帧
        stats = analyze_depth(depth)
        self.assertTrue(stats.shader_warming)
        self.assertFalse(stats.no_return)
        self.assertLessEqual(stats.max_depth, 1.2)

    def test_real_depth_not_flagged(self):
        depth = np.full((180, 320), 8.0, dtype=np.float32)
        depth[80:100, 150:170] = 2.5
        stats = analyze_depth(depth)
        self.assertFalse(stats.shader_warming)
        self.assertAlmostEqual(stats.min_depth, 2.5, places=4)
        self.assertAlmostEqual(stats.center_depth, 2.5, places=3)
        self.assertAlmostEqual(stats.max_depth, 8.0, places=4)
        self.assertGreaterEqual(stats.min_range, stats.min_depth - 1e-6)  # 欧氏距离 ≥ 平面深度

    def test_all_infinite_is_no_return_not_warming(self):
        stats = analyze_depth(np.full((180, 320), np.inf, dtype=np.float32))
        self.assertTrue(stats.no_return)
        self.assertFalse(stats.shader_warming)

    def test_tap_sends_empty_cloud_while_warming_then_real_cloud(self):
        sim = SimulatedAirSim(warmup_frames=2)
        tap = AirSimTap(client_factory=lambda *_: sim, api=SIM_API, always_on=True)
        self.assertTrue(tap._connect())
        for _ in range(2):
            frame = protocol.decode_telemetry(tap.step().telemetry)
            self.assertTrue(frame.meta["shader_warming"])
            self.assertTrue(frame.shader_warming)
            self.assertEqual(frame.points.shape[0], 0)
            self.assertIsNone(frame.depth)
        frame = protocol.decode_telemetry(tap.step().telemetry)
        self.assertFalse(frame.meta["shader_warming"])
        self.assertGreater(frame.points.shape[0], 500)
        self.assertEqual(frame.depth.shape, (45, 80))
        self.assertEqual(frame.depth.dtype, np.uint8)


class WebSocketCodecTests(unittest.IsolatedAsyncioTestCase):
    def test_rfc6455_sample_accept_key(self):
        self.assertEqual(ws_server.compute_accept_key("dGhlIHNhbXBsZSBub25jZQ=="), "s3pPLMBiTxaQ9kYGzzhZRbK+xOo=")

    async def _read(self, raw: bytes, **kw):
        reader = asyncio.StreamReader()
        reader.feed_data(raw)
        reader.feed_eof()
        return await ws_server.read_frame(reader, **kw)

    async def test_masked_client_frames_roundtrip_all_length_classes(self):
        rng = random.Random(1)
        for size in (0, 1, 5, 125, 126, 1001, 65535, 65536, 70000):
            payload = bytes(rng.getrandbits(8) for _ in range(size))
            key = bytes(rng.getrandbits(8) for _ in range(4))
            fin, opcode, decoded = await self._read(ws_server.encode_frame(ws_server.OP_BINARY, payload, mask_key=key))
            self.assertEqual((fin, opcode, decoded), (True, ws_server.OP_BINARY, payload), size)

    async def test_server_frames_unmasked_and_length_headers(self):
        for size, header_len in ((5, 2), (200, 4), (70000, 10)):
            raw = ws_server.encode_frame(ws_server.OP_TEXT, b"x" * size)
            self.assertEqual(len(raw), header_len + size)
            fin, opcode, payload = await self._read(raw, require_mask=False)
            self.assertEqual((fin, opcode, len(payload)), (True, ws_server.OP_TEXT, size))

    async def test_unmasked_client_frame_rejected(self):
        with self.assertRaises(ws_server.ProtocolError):
            await self._read(ws_server.encode_frame(ws_server.OP_TEXT, b"hi"))

    async def test_ping_and_close_frames(self):
        key = b"\x01\x02\x03\x04"
        self.assertEqual((await self._read(ws_server.encode_frame(ws_server.OP_PING, b"hb", mask_key=key)))[1:],
                         (ws_server.OP_PING, b"hb"))
        fin, op, payload = await self._read(ws_server.encode_frame(ws_server.OP_CLOSE, b"\x03\xe8", mask_key=key))
        self.assertEqual((op, payload), (ws_server.OP_CLOSE, b"\x03\xe8"))

    def test_static_path_traversal_blocked(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "dist"
            root.mkdir()
            (root / "index.html").write_text("ok", encoding="utf-8")
            (Path(tmp) / "secret.txt").write_text("secret", encoding="utf-8")
            self.assertEqual(ws_server.resolve_static_path(root, "/"), (root / "index.html").resolve())
            for evil in ("/../secret.txt", "/%2e%2e/secret.txt", "/..%5csecret.txt", "/a/../../secret.txt"):
                self.assertIsNone(ws_server.resolve_static_path(root, evil), evil)


class BinaryFrameTests(unittest.TestCase):
    def test_roundtrip_u8_with_alignment(self):
        rng = np.random.default_rng(5)
        pts = rng.normal(size=(321, 3)).astype(np.float32)
        depth = rng.integers(0, 256, size=(45, 80), dtype=np.uint8)
        meta = {"hello": "世界", "pos": [1.5, -2.0, 3.25]}
        data = protocol.encode_telemetry(42, 1234.5, meta, pts, depth, flags=protocol.FLAG_SHADER_WARMING)
        meta_len = int.from_bytes(data[20:24], "little")
        self.assertEqual((protocol.TELEMETRY_HEADER_SIZE + meta_len) % 4, 0)  # 点云 Float32 4 字节对齐
        self.assertEqual(len(data), protocol.TELEMETRY_HEADER_SIZE + meta_len + 321 * 12 + 45 * 80)
        frame = protocol.decode_telemetry(data)
        self.assertEqual((frame.seq, frame.t, frame.meta), (42, 1234.5, meta))
        self.assertTrue(frame.shader_warming)
        np.testing.assert_array_equal(frame.points, pts)
        np.testing.assert_array_equal(frame.depth, depth)
        self.assertEqual(protocol.frame_kind(data), "telemetry")

    def test_roundtrip_u16_and_empty(self):
        depth = (np.arange(90 * 160, dtype=np.uint32) % 25000).astype(np.uint16).reshape(90, 160)
        frame = protocol.decode_telemetry(protocol.encode_telemetry(1, 0.0, {}, np.zeros((0, 3), np.float32), depth))
        self.assertEqual(frame.depth_enc, protocol.DEPTH_U16)
        np.testing.assert_array_equal(frame.depth, depth)
        empty = protocol.decode_telemetry(protocol.encode_telemetry(2, 0.0, {"a": 1}))
        self.assertEqual(empty.points.shape, (0, 3))
        self.assertIsNone(empty.depth)

    def test_quantization(self):
        d = np.array([[0.0, 12.5, 25.0, np.nan, np.inf, 40.0]], dtype=np.float32)
        np.testing.assert_array_equal(protocol.quantize_depth_u8(d, 25.0), [[0, 128, 255, 255, 255, 255]])
        np.testing.assert_array_equal(protocol.quantize_depth_u16(d, 25.0), [[0, 12500, 25000, 25000, 25000, 40000]])

    def test_rgb_frame(self):
        png = b"\x89PNG\r\n\x1a\nxxxx"
        seq, w, h, body = protocol.decode_rgb(protocol.encode_rgb(9, 160, 90, png))
        self.assertEqual((seq, w, h, body), (9, 160, 90, png))
        self.assertEqual(protocol.frame_kind(protocol.encode_rgb(1, 1, 1, png)), "rgb")

    def test_nan_in_meta_rejected(self):
        with self.assertRaises(ValueError):
            protocol.encode_telemetry(1, 0.0, {"x": float("nan")})


class BypassGovernorTests(unittest.TestCase):
    def test_profiles_match_spec(self):
        b, d = PROFILES["bypass"], PROFILES["direct"]
        self.assertTrue(4.0 <= b.hz <= 6.0)
        self.assertEqual(b.grid, (80, 45))
        self.assertEqual(b.timeout, 2.0)
        self.assertTrue(2.0 <= b.degraded_hz <= 3.0)
        self.assertEqual(b.slow_rpc_ms, 250.0)
        self.assertEqual((d.hz, d.grid), (20.0, (160, 90)))

    def test_slow_rpc_backs_off_then_recovers(self):
        gov = RateGovernor(PROFILES["bypass"])
        self.assertEqual(gov.hz, 5.0)
        self.assertFalse(gov.observe(100.0))
        self.assertTrue(gov.observe(251.0))  # > 250 ms → 降频
        self.assertTrue(gov.degraded)
        self.assertTrue(2.0 <= gov.hz <= 3.0)
        for _ in range(9):  # 需要连续 10 次快速 RPC 才恢复
            gov.observe(50.0)
        self.assertTrue(gov.degraded)
        gov.observe(180.0)  # 介于恢复阈值与慢阈值之间 → 不恢复且清零计数
        for _ in range(9):
            gov.observe(50.0)
        self.assertTrue(gov.degraded)
        self.assertTrue(gov.observe(50.0))
        self.assertFalse(gov.degraded)
        self.assertEqual(gov.hz, 5.0)

    def test_direct_mode_never_degrades(self):
        gov = RateGovernor(PROFILES["direct"])
        self.assertFalse(gov.observe(5000.0))
        self.assertEqual(gov.hz, 20.0)

    def test_tap_degrades_on_mock_slow_airsim_and_recovers(self):
        clock = [0.0]

        class TimedAirSim(SimulatedAirSim):
            image_latency = 0.270

            def simGetImages(self, requests, *args, **kwargs):
                clock[0] += self.image_latency
                return super().simGetImages(requests, *args, **kwargs)

        sim = TimedAirSim(warmup_frames=0)
        tap = AirSimTap(client_factory=lambda *_: sim, api=SIM_API, mode="bypass", clock=lambda: clock[0])
        self.assertTrue(tap._connect())
        tap._tick(clock[0])
        self.assertTrue(tap.status()["degraded"])
        self.assertLessEqual(tap.status()["hz_target"], 3.0)
        sim.image_latency = 0.040
        for _ in range(10):
            tap._tick(clock[0])
        self.assertFalse(tap.status()["degraded"])
        self.assertEqual(tap.status()["hz_target"], 5.0)

    def test_backoff_exponential_and_capped(self):
        b = Backoff(base=0.5, factor=2.0, maximum=10.0, jitter=0.0)
        self.assertEqual([b.next() for _ in range(7)], [0.5, 1.0, 2.0, 4.0, 8.0, 10.0, 10.0])
        b.reset()
        self.assertEqual(b.next(), 0.5)

    def test_reconnect_on_connect_failure_never_raises(self):
        attempts = []

        def factory(host, port, timeout):
            attempts.append(port)
            raise ConnectionRefusedError("UE 正在重置")

        tap = AirSimTap(client_factory=factory, api=SIM_API, backoff=Backoff(jitter=0.0))
        for _ in range(3):
            self.assertFalse(tap._connect())
        self.assertEqual(tap.state, "reconnecting")
        self.assertIn("ConnectionRefusedError", tap.status()["error"])
        self.assertEqual(tap.retry_in, 2.0)  # 0.5 → 1.0 → 2.0

    def test_mid_stream_failures_trigger_reconnect_without_crash(self):
        sim = SimulatedAirSim(warmup_frames=0)
        tap = AirSimTap(client_factory=lambda *_: sim, api=SIM_API, always_on=True)
        tap._connect()

        def boom(*_a, **_k):
            raise TimeoutError("RPC timeout")

        sim.simGetImages = boom
        for _ in range(3):
            tap._tick(0.0)
        self.assertIsNone(tap._client)
        self.assertEqual(tap.state, "reconnecting")

    def test_episode_counter_increments_on_pose_jump(self):
        sim = SimulatedAirSim(warmup_frames=0)
        tap = AirSimTap(client_factory=lambda *_: sim, api=SIM_API, always_on=True)
        tap._connect()
        first = tap.step().meta
        pose = sim.simGetVehiclePose()
        pose.position.x_val = first["pos"][0] + 10
        with patch.object(sim, "simGetVehiclePose", return_value=pose):
            meta = tap.step().meta
        self.assertTrue(meta["reset"])
        self.assertEqual(meta["episode"], 1)


class ReadOnlyGuardTests(unittest.TestCase):
    FORBIDDEN = ("enableApiControl", "armDisarm", "simPause", "reset", "moveByVelocityAsync",
                 "moveByVelocityZAsync", "moveByVelocityBodyFrameAsync", "takeoffAsync", "landAsync",
                 "simSetVehiclePose", "simSpawnObject", "simDestroyObject", "simContinueForTime", "hoverAsync",
                 "simSetObjectPose", "simSetCameraPose", "simRunConsoleCommand", "simGetCollisionInfo")

    def test_every_write_api_blocked_before_reaching_simulator(self):
        sim = SimulatedAirSim()
        guarded = ReadOnlyClient(sim)
        for name in self.FORBIDDEN:
            with self.assertRaises(ReadOnlyViolation, msg=name):
                getattr(guarded, name)
        self.assertEqual(sim.calls, [])
        with self.assertRaises(ReadOnlyViolation):
            guarded.anything = 1
        self.assertTrue(READ_ONLY_METHODS.isdisjoint(self.FORBIDDEN))

    def test_full_pipeline_only_issues_whitelisted_rpcs(self):
        sim = SimulatedAirSim(warmup_frames=0)
        tap = AirSimTap(client_factory=lambda *_: sim, api=SIM_API, mode="direct", always_on=True)
        tap.set_rgb(True)
        tap._apply_pending()
        tap._connect()
        for _ in range(5):
            tap.step()
        tap.geometry.force_rescan()
        issued = set(sim.calls)
        self.assertTrue(issued <= READ_ONLY_METHODS, issued - READ_ONLY_METHODS)
        self.assertIn("simGetImages", issued)
        self.assertNotIn("getSettingsString", issued)
        self.assertEqual(set(tap._client.calls) - READ_ONLY_METHODS, set())

    def test_bypass_only_requests_depth_even_when_rgb_is_requested(self):
        sim = SimulatedAirSim()
        tap = AirSimTap(client_factory=lambda *_: sim, api=SIM_API, mode="bypass")
        tap.set_rgb(True)
        tap._apply_pending()
        self.assertTrue(tap._connect())
        with patch.object(sim, "simGetImages", wraps=sim.simGetImages) as images:
            self.assertIsNone(tap.step().rgb)
        self.assertEqual([req.image_type for req in images.call_args.args[0]], [SIM_API.ImageType.DepthPlanar])

    def test_source_contains_no_write_rpc_calls(self):
        pattern = re.compile(r"client\.(%s)\(" % "|".join(self.FORBIDDEN))
        for path in PACKAGE_DIR.glob("*.py"):
            self.assertIsNone(pattern.search(path.read_text(encoding="utf-8")), path.name)


class GeometryTests(unittest.TestCase):
    def test_plan_goal_is_read_in_vehicle_ned(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            (folder / "plan.json").write_text('{"scene_goal_position": [162, 3, -1.5]}', encoding="utf-8")
            loader = geometry_loader.GeometryLoader(run_dir=folder)
            self.assertTrue(loader.poll_run_dir())
            goal = loader.scene_message()["goal"]
            self.assertEqual(goal["p"], [162, -3, 1.5])
            self.assertEqual(goal["radii"], [2.0, 1.6])
            self.assertEqual(goal["source"], "plan")

    def test_live_boxes_roundtrip_to_local_enu(self):
        sim = SimulatedAirSim()
        loader = geometry_loader.GeometryLoader(ReadOnlyClient(sim))
        self.assertTrue(loader.scan_live())
        by_key = {b.key: b for b in loader.boxes}
        self.assertEqual(set(by_key), {key for key, _, _ in sim.boxes})
        for key, center, size in sim.boxes:
            np.testing.assert_allclose(by_key[key].center, center, atol=1e-3)
            np.testing.assert_allclose(by_key[key].size, size, atol=1e-3)
        self.assertEqual(by_key["corridor_roof"].kind, "roof")
        self.assertEqual(by_key["wall_min_y"].kind, "wall")
        self.assertEqual(by_key["floor_slab"].kind, "floor")
        self.assertEqual(by_key["over_0"].kind, "block")

    def test_incremental_scan_does_not_requery_known_objects(self):
        sim = SimulatedAirSim()
        loader = geometry_loader.GeometryLoader(ReadOnlyClient(sim))
        loader.scan_live()
        before = Counter(sim.calls)
        self.assertFalse(loader.scan_live())
        after = Counter(sim.calls)
        self.assertEqual(after["simListSceneObjects"], before["simListSceneObjects"] + 1)
        self.assertEqual(after["simGetObjectPose"], before["simGetObjectPose"])

    def test_scan_failure_is_contained(self):
        class Broken:
            def simListSceneObjects(self, *_):
                raise TimeoutError("boom")

        loader = geometry_loader.GeometryLoader(Broken())
        self.assertFalse(loader.scan_live())
        self.assertIn("TimeoutError", loader.last_scan_error)

    def test_static_mesh_actors_included_and_floor_slab_ignored_without_requery(self):
        class FakeTrainEnvClient:
            def __init__(self):
                self.calls = Counter()
                self.poses = {
                    "StaticMeshActor_0": ((12.0, 33.0, 1.5), (3.9, 3.3, 4.2)),
                    "StaticMeshActor_12": ((171.0, 0.0, 1.35), (0.75, 120.0, 4.5)),
                    "StaticMeshActor_178": ((81.0, 0.0, 3.75), (180.0, 120.0, 0.3)),
                    "Ch4Stage3V2_1_tunnel_roof": ((81.0, 0.0, -0.8), (179.6, 119.0, 0.36)),
                }

            def simListSceneObjects(self, *_):
                self.calls["list"] += 1
                return list(self.poses)

            def simGetObjectPose(self, name):
                self.calls["pose"] += 1
                x, y, z = self.poses[name][0]
                return type("Pose", (), {"position": (x, y, z), "orientation": (0, 0, 0, 1)})()

            def simGetObjectScale(self, name):
                self.calls["scale"] += 1
                return self.poses[name][1]

        client = FakeTrainEnvClient()
        loader = geometry_loader.GeometryLoader(client)
        self.assertTrue(loader.scan_live())
        by_name = {b.name: b for b in loader.boxes}
        self.assertIn("StaticMeshActor_0", by_name)
        self.assertEqual(by_name["StaticMeshActor_0"].kind, "block")
        self.assertEqual(by_name["StaticMeshActor_12"].kind, "wall")
        self.assertEqual(by_name["Ch4Stage3V2_1_tunnel_roof"].kind, "roof")
        self.assertNotIn("StaticMeshActor_178", by_name)
        self.assertFalse(loader.scan_live())
        self.assertEqual(client.calls["pose"], 4)  # 第二次扫描不再重复查静态块和已忽略地板

    def test_infer_live_goal_from_corridor_and_roofed_route(self):
        # 1) 动态走廊：goal_end_wall 在 xmax 处，应精确推算出 corridor 终点而非串用其他进程日志
        corridor_boxes = {
            "Ch4Stage3V2_9_wall_min_y": geometry_loader.SceneBox("wall_min_y", "Ch4Stage3V2_9_wall_min_y", (117.5, -53.35 - 5.0, 0.5), (58.0, 0.32, 6.2)),
            "Ch4Stage3V2_9_wall_max_y": geometry_loader.SceneBox("wall_max_y", "Ch4Stage3V2_9_wall_max_y", (117.5, -53.35 + 5.0, 0.5), (58.0, 0.32, 6.2)),
            "Ch4Stage3V2_9_goal_end_wall": geometry_loader.SceneBox("goal_end_wall", "Ch4Stage3V2_9_goal_end_wall", (146.32, -53.35, 0.5), (0.36, 10.4, 6.2)),
            "Ch4Stage3V2_9_under_0": geometry_loader.SceneBox("under_0", "Ch4Stage3V2_9_under_0", (120.0, -53.35, 2.0), (2.5, 10.4, 2.0)),
        }
        g_corridor = geometry_loader.infer_live_goal((92.0, -53.35, 1.5), (3.0, 0.0, 0.0), corridor_boxes)
        self.assertIsNotNone(g_corridor)
        self.assertEqual(g_corridor.source, "live_corridor")
        self.assertAlmostEqual(g_corridor.position[0], 146.14 - 7.9, places=2)
        self.assertAlmostEqual(g_corridor.position[1], -53.35, places=2)

        # 2) 原场带顶棚路线：从 x=5 飞往 x=162 的 Y=+15 跑道
        roofed_boxes = {
            "Ch4Stage3V2_10_tunnel_roof": geometry_loader.SceneBox("tunnel_roof", "Ch4Stage3V2_10_tunnel_roof", (81.0, 0.0, 1.0), (179.6, 119.0, 0.36)),
        }
        g_route = geometry_loader.infer_live_goal((5.0, 1.0, -1.0), (3.5, 0.32, 0.0), roofed_boxes)
        self.assertIsNotNone(g_route)
        self.assertEqual(g_route.source, "live_route")
        self.assertEqual(g_route.position, (162.0, 15.0, -1.0))

    def test_course_episode_goal_and_boxes(self):
        record = {
            "goal_xyz_agl_m": [78.4, 53.36, 2.709],
            "family": "side",
            "temporary_obstacles": [
                {"key": "wall_min_y", "x": [27.0, 85.8], "y": [47.03, 47.35], "z": [-2.78, 3.42]},
                {"key": "corridor_roof", "x": [27.0, 85.8], "y": [47.16, 59.54], "z": [3.27, 3.63]},
                {"broken": True},
            ],
        }
        goal = geometry_loader.goal_from_episode_record(record, floor_z=-2.6)
        np.testing.assert_allclose(goal.position, (78.4, -53.36, 0.109), atol=1e-9)
        self.assertEqual(goal.radii, (2.0, 1.6))
        boxes = geometry_loader.boxes_from_episode_record(record)
        self.assertEqual([b.key for b in boxes], ["wall_min_y", "corridor_roof"])
        np.testing.assert_allclose(boxes[0].center, (56.4, -47.19, 0.32), atol=1e-6)
        np.testing.assert_allclose(boxes[0].size, (58.8, 0.32, 6.2), atol=1e-6)

    def test_run_dir_tail_parsing_and_change_detection(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            log = run / "course_episodes.jsonl"
            good = {"goal_xyz_agl_m": [10, 2, 1.5], "temporary_obstacles": []}
            log.write_text(json.dumps({"goal_xyz_agl_m": [1, 1, 1]}) + "\n" + json.dumps(good) + "\n{\"trunc", encoding="utf-8")
            self.assertEqual(geometry_loader.read_last_json_line(log)["goal_xyz_agl_m"], [10, 2, 1.5])
            loader = geometry_loader.GeometryLoader(None, run_dir=run, floor_z=-2.6)
            self.assertTrue(loader.poll_run_dir())
            self.assertFalse(loader.poll_run_dir())  # 文件未变
            np.testing.assert_allclose(loader.effective_goal().position, (10, -2, -1.1), atol=1e-9)
            with log.open("a", encoding="utf-8") as f:
                f.write("\n" + json.dumps({"goal_xyz_agl_m": [20, 0, 2.0], "temporary_obstacles": []}) + "\n")
            self.assertTrue(loader.poll_run_dir())
            self.assertEqual(loader.scene_message()["goal"]["p"][0], 20.0)
            self.assertEqual(geometry_loader.resolve_run_dir(None), None)

    def test_resolve_run_dir_auto_picks_latest(self):
        with tempfile.TemporaryDirectory() as tmp:
            runs = Path(tmp) / "Logs" / "TrainData" / "Runs" / "trainenv"
            (runs / "a").mkdir(parents=True)
            (runs / "b").mkdir(parents=True)
            (runs / "a" / "course_episodes.jsonl").write_text("{}\n", encoding="utf-8")
            time.sleep(0.05)
            (runs / "b" / "course_episodes.jsonl").write_text("{}\n", encoding="utf-8")
            self.assertEqual(geometry_loader.resolve_run_dir("auto", root=Path(tmp)), runs / "b")


class NavRLGoalTests(unittest.TestCase):
    class SimulatedLaunch(SimulatedAirSim):
        def __init__(self, port: int, launch_id: str):
            super().__init__(warmup_frames=0)
            self.settings = {"ApiServerPort": port, "TrainEnvLaunchId": launch_id}

        def getSettingsString(self) -> str:
            return json.dumps(self.settings)

    @staticmethod
    def _curriculum(root: Path, timestamp: int = 1) -> Path:
        root.mkdir(parents=True)
        marker = root / "curriculum.json"
        marker.write_text("{}", encoding="utf-8")
        os.utime(marker, (timestamp, timestamp))
        return root

    @staticmethod
    def _launch_id(directory: Path) -> str:
        return uuid5(NAMESPACE_URL, directory.as_posix()).hex

    def _launch(self, root: Path, stage: int, *ports: int) -> Path:
        directory = root / f"stage_{stage}"
        directory.mkdir(parents=True, exist_ok=True)
        launch_id = self._launch_id(directory)
        manifest = {
            "map": f"/Game/Training/Maps/L_Train_NavRL_Forest_Stage{stage}",
            "instances": [
                {"port": port, "settings": {"ApiServerPort": port, "TrainEnvLaunchId": launch_id}}
                for port in ports
            ],
        }
        (directory / "launch.json").write_text(json.dumps(manifest), encoding="utf-8")
        return directory

    def _poll(self, loader: geometry_loader.GeometryLoader, launch: Path) -> bool:
        return loader.poll_run_dir(rpc_port=41811, launch_id=self._launch_id(launch))

    @staticmethod
    def _publish(directory: Path, port: int, **fields) -> None:
        snapshot = {"stage": int(directory.name.split("_")[-1]), "rpc_port": port,
                    "episode": 1, "goal_position": [42, 7, -0.75], "published_at": 1, **fields}
        (directory / f"live_goal_{port}.json").write_text(json.dumps(snapshot), encoding="utf-8")

    def test_exact_snapshot_for_explicit_stage_and_no_tunnel_guess(self):
        with tempfile.TemporaryDirectory() as tmp:
            stage = self._launch(Path(tmp), 1, 41811, 41812)
            (stage / "course_episodes.jsonl").write_text('{"goal_position":[0,0,0]}\n', encoding="utf-8")
            loader = geometry_loader.GeometryLoader(run_dir=stage)
            self._publish(stage, 41812)
            self.assertTrue(self._poll(loader, stage))
            self.assertTrue(loader.scene_message()["navrl"])
            self.assertIsNone(loader.effective_goal())
            with patch.object(geometry_loader, "infer_live_goal") as infer:
                loader.update_live_pose((0, 0, 0))
                infer.assert_not_called()
            self._publish(stage, 41811, rpc_port=41812)
            self.assertFalse(self._poll(loader, stage))
            self._publish(stage, 41811, episode=3)
            self.assertTrue(self._poll(loader, stage))
            goal = loader.scene_message()["goal"]
            self.assertEqual(goal, {"p": [42.0, -7.0, 0.75], "radii": [0.5, 0.5], "alt_tol": 0.5,
                                    "source": "navrl_live", "family": None, "stage": 1, "episode": 3})
            self._publish(stage, 41811, episode=4, stage=2)
            self.assertTrue(self._poll(loader, stage))
            self.assertIsNone(loader.effective_goal())

    def test_stage_directory_follows_curriculum_for_its_port(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = self._curriculum(Path(tmp) / "run")
            stage1 = self._launch(run, 1, 41811)
            self._publish(stage1, 41811)
            loader = geometry_loader.GeometryLoader(run_dir=stage1)
            self.assertTrue(self._poll(loader, stage1))
            self._launch(run, 2, 41812)
            self.assertFalse(self._poll(loader, stage1))
            stage2 = self._launch(run, 2, 41811, 41812)
            self.assertTrue(self._poll(loader, stage1))
            self.assertIsNone(loader.effective_goal())
            self._publish(stage2, 41811, goal_position=[16, -24, 0.1])
            self.assertFalse(self._poll(loader, stage1))
            self.assertTrue(self._poll(loader, stage2))
            self.assertEqual(loader.scene_message()["goal"]["stage"], 2)
            self.assertEqual(loader.effective_goal().position, (16, 24, -0.1))
            (stage2 / "launch.json").unlink()
            self.assertTrue(self._poll(loader, stage2))
            self.assertIsNone(loader.effective_goal())

    def test_follow_navrl_selects_new_run_for_port_and_clears_completed_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            runs = Path(tmp)
            stage1 = self._launch(self._curriculum(runs / "first", 1), 1, 41811)
            self._publish(stage1, 41811)
            unrelated = self._launch(self._curriculum(runs / "second", 2), 1, 41812)
            self._publish(unrelated, 41812)
            loader = geometry_loader.GeometryLoader(navrl_runs_root=runs, infer_goals=False)
            self.assertTrue(self._poll(loader, stage1))
            self.assertEqual(loader.scene_message()["goal"]["p"], [42.0, -7.0, 0.75])
            new_stage = self._launch(self._curriculum(runs / "third", 3), 1, 41811)
            self.assertTrue(self._poll(loader, stage1))
            self.assertIsNone(loader.effective_goal())
            self._publish(new_stage, 41811, goal_position=[16, 24, 1])
            self.assertTrue(self._poll(loader, new_stage))
            self.assertEqual(loader.scene_message()["goal"]["p"], [16.0, -24.0, -1.0])
            (new_stage / "decision.json").write_text("{}", encoding="utf-8")
            self.assertTrue(self._poll(loader, new_stage))
            self.assertIsNone(loader.effective_goal())
            (new_stage / "launch.json").unlink()
            self.assertFalse(self._poll(loader, new_stage))
            self.assertIsNone(loader.effective_goal())

    def test_tap_polls_new_episode_and_clears_marker_on_port_switch(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = self._curriculum(Path(tmp) / "run")
            stage = self._launch(run, 1, 41811, 41812)
            self._publish(stage, 41811)
            messages, frames = [], []
            loader = geometry_loader.GeometryLoader(run_dir=run, infer_goals=False)
            factory = lambda _host, port, _timeout: self.SimulatedLaunch(port, self._launch_id(stage))
            tap = AirSimTap(port=41811, client_factory=factory, api=SIM_API,
                            geometry=loader, on_scene=messages.append, on_telemetry=frames.append)
            try:
                self.assertTrue(tap._connect())
                self.assertEqual(messages[-1]["goal"]["episode"], 1)
                self._publish(stage, 41811, episode=2, goal_position=[14, -24, 2])
                tap._tick(0.5)
                self.assertEqual(messages[-1]["goal"]["episode"], 2)
                self.assertIsNotNone(protocol.decode_telemetry(frames[-1]).meta["goal_err"])
                tap.set_port(41812)
                tap._apply_pending()
                self.assertIsNone(messages[-1]["goal"])
                self.assertTrue(tap._connect())
                self.assertIsNone(messages[-1]["goal"])
                self._publish(stage, 41812)
                tap._tick(1.0)
                self.assertEqual(messages[-1]["goal"]["episode"], 1)
                tap._schedule_reconnect("simulated disconnect")
                self.assertIsNone(messages[-1]["goal"])
                loader.set_manual_goal((1, 2, 3))
                self.assertTrue(tap._connect())
                self.assertEqual(messages[-1]["goal"]["source"], "manual")
            finally:
                tap._close_client()

    def test_same_port_other_launch_has_telemetry_and_no_stale_goal(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = self._curriculum(Path(tmp) / "run")
            stage = self._launch(run, 1, 41811)
            self._publish(stage, 41811)
            messages, frames = [], []
            loader = geometry_loader.GeometryLoader(run_dir=run)
            factory = lambda _host, port, _timeout: self.SimulatedLaunch(port, "other-launch")
            tap = AirSimTap(port=41811, client_factory=factory, api=SIM_API, geometry=loader,
                            on_scene=messages.append, on_telemetry=frames.append)
            try:
                self.assertTrue(tap._connect())
                self.assertTrue(messages[-1]["navrl"])
                self.assertIsNone(messages[-1]["goal"])
                tap._tick(0.5)
                tap._tick(1.0)
                self.assertEqual(tap.state, "connected")
                self.assertEqual(tap.status()["rpc_calls"]["getSettingsString"], 1)
                self.assertTrue(frames)
                self.assertIsNone(protocol.decode_telemetry(frames[-1]).meta["goal_err"])
                self.assertIsNone(messages[-1]["goal"])
            finally:
                tap._close_client()

    def test_tap_verifies_new_stage_once_at_reconnect(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = self._curriculum(Path(tmp) / "run")
            active_stage = self._launch(run, 1, 41811)
            self._publish(active_stage, 41811)
            messages = []
            loader = geometry_loader.GeometryLoader(run_dir=run)
            factory = lambda _host, port, _timeout: self.SimulatedLaunch(port, self._launch_id(active_stage))
            tap = AirSimTap(port=41811, client_factory=factory, api=SIM_API,
                            geometry=loader, on_scene=messages.append)
            try:
                self.assertTrue(tap._connect())
                self.assertEqual(messages[-1]["goal"]["stage"], 1)
                active_stage = self._launch(run, 2, 41811)
                self._publish(active_stage, 41811)
                tap._tick(0.5)
                self.assertEqual(tap.state, "reconnecting")
                self.assertIsNone(messages[-1]["goal"])
                self.assertTrue(tap._connect())
                self.assertEqual(messages[-1]["goal"]["stage"], 2)
                tap._tick(1.0)
                self.assertEqual(tap.state, "connected")
                self.assertEqual(tap.status()["rpc_calls"]["getSettingsString"], 1)
            finally:
                tap._close_client()


class CommandRouterTests(unittest.TestCase):
    class StubTap:
        def __init__(self):
            self.calls = []

        def __getattr__(self, name):
            return lambda *a: self.calls.append((name, a))

        def status(self):
            return {"state": "stub"}

    def test_validation(self):
        tap = self.StubTap()
        router = ws_server.CommandRouter(tap, lambda: 8765)
        for port in (41982, 42072, 42073, 41451, 12345):
            self.assertTrue(router({"cmd": "set_port", "port": port})["ok"])
        for bad in (0, 70000, "42072", True, None, 8765):
            self.assertFalse(router({"cmd": "set_port", "port": bad})["ok"], bad)
        self.assertTrue(router({"cmd": "set_mode", "mode": "direct"})["ok"])
        self.assertFalse(router({"cmd": "set_mode", "mode": "turbo"})["ok"])
        self.assertTrue(router({"cmd": "set_color_mode", "value": "height"})["ok"])
        self.assertEqual(router.color_mode, "height")
        self.assertFalse(router({"cmd": "set_color_mode", "value": "rainbow"})["ok"])
        self.assertTrue(router({"cmd": "set_rgb", "enabled": True})["ok"])
        self.assertTrue(router({"cmd": "clear_trail"})["ok"])
        self.assertTrue(router({"cmd": "refresh_scene"})["ok"])
        self.assertFalse(router({"cmd": "enableApiControl"})["ok"])
        self.assertIn(("set_port", (42073,)), tap.calls)
        self.assertNotIn(("set_port", (8765,)), tap.calls)


class ServerIntegrationTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        static = Path(self.tmp.name)
        (static / "index.html").write_text("<!doctype html><title>t</title>", encoding="utf-8")
        (static / "app.js").write_text("export {}", encoding="utf-8")
        self.server = ws_server.RvizServer(port=0, static_dir=static)
        self.geometry = geometry_loader.GeometryLoader(None)
        self.tap = AirSimTap(
            client_factory=lambda *_: ReadOnlyClient(SimulatedAirSim(warmup_frames=1)), api=SIM_API, mode="bypass",
            geometry=self.geometry,
            on_telemetry=lambda d: self.server.publish_binary(d, "telemetry"),
            on_rgb=lambda d: self.server.publish_binary(d, "rgb"),
            on_scene=self.server.publish_json, on_status=self.server.publish_json, on_event=self.server.publish_json,
        )
        router = ws_server.CommandRouter(self.tap, lambda: self.server.port, broadcast=self.server.publish_json)
        self.server.command_handler = router
        self.server.status_provider = self.tap.status
        self.server.on_clients_changed = self.tap.set_viewers
        self.server.initial_messages = lambda: [self.tap.status(), self.geometry.scene_message()]
        await self.server.start()
        self.tap.start()

    async def asyncTearDown(self):
        self.tap.stop()
        await self.server.stop()
        self.tmp.cleanup()

    async def http_get(self, path, headers=""):
        reader, writer = await asyncio.open_connection("127.0.0.1", self.server.port)
        writer.write(f"GET {path} HTTP/1.1\r\nHost: x\r\n{headers}\r\n".encode())
        await writer.drain()
        raw = await asyncio.wait_for(reader.read(), 5)
        writer.close()
        await writer.wait_closed()
        head, _, body = raw.partition(b"\r\n\r\n")
        return int(head.split()[1]), head.decode("latin-1"), body

    async def ws_connect(self, origin=None):
        reader, writer = await asyncio.open_connection("127.0.0.1", self.server.port)
        key = "dGhlIHNhbXBsZSBub25jZQ=="
        extra = f"Origin: {origin}\r\n" if origin else ""
        writer.write((f"GET /ws HTTP/1.1\r\nHost: x\r\nUpgrade: websocket\r\nConnection: Upgrade\r\n"
                      f"Sec-WebSocket-Key: {key}\r\nSec-WebSocket-Version: 13\r\n{extra}\r\n").encode())
        await writer.drain()
        head = await asyncio.wait_for(reader.readuntil(b"\r\n\r\n"), 5)
        return reader, writer, head.decode()

    async def next_frame(self, reader, timeout=10.0):
        _fin, opcode, payload = await asyncio.wait_for(ws_server.read_frame(reader, require_mask=False), timeout)
        return opcode, payload

    async def send_cmd(self, writer, message):
        writer.write(ws_server.encode_frame(ws_server.OP_TEXT, json.dumps(message).encode(), mask_key=b"\x11\x22\x33\x44"))
        await writer.drain()

    async def test_http_static_and_status(self):
        status, head, body = await self.http_get("/")
        self.assertEqual(status, 200)
        self.assertIn("text/html", head)
        status, head, body = await self.http_get("/app.js")
        self.assertIn("text/javascript", head)
        self.assertEqual((await self.http_get("/%2e%2e/%2e%2e/secret"))[0], 404)
        status, _, body = await self.http_get("/api/status")
        info = json.loads(body)
        self.assertEqual((status, info["read_only"], info["mode"], info["web_clients"]), (200, True, "bypass", 0))
        self.assertEqual((await self.http_get("/healthz"))[0], 200)

    async def test_forbidden_origin_and_missing_upgrade(self):
        _, writer, head = await self.ws_connect(origin="http://evil.example")
        self.assertIn(" 403 ", head.splitlines()[0])
        writer.close()
        status, _, _ = await self.http_get("/ws")
        self.assertEqual(status, 426)

    async def test_end_to_end_binary_frames_and_controls(self):
        reader, writer, head = await self.ws_connect(origin=f"http://127.0.0.1:{self.server.port}")
        self.assertIn("101", head.splitlines()[0])
        self.assertIn("s3pPLMBiTxaQ9kYGzzhZRbK+xOo=", head)
        texts, frames = [], []
        deadline = time.monotonic() + 20
        while time.monotonic() < deadline and not any(not f.shader_warming and f.points.shape[0] > 0 for f in frames):
            opcode, payload = await self.next_frame(reader)
            if opcode == ws_server.OP_TEXT:
                texts.append(json.loads(payload))
            elif protocol.frame_kind(payload) == "telemetry":
                frames.append(protocol.decode_telemetry(payload))
        self.assertEqual(texts[0]["type"], "hello")
        self.assertTrue(any(t["type"] == "scene" and len(t["boxes"]) == 7 for t in texts))
        good = [f for f in frames if not f.shader_warming and f.points.shape[0] > 0][0]
        self.assertTrue(frames[0].shader_warming)  # warmup_frames=1：首帧被拦截
        self.assertEqual(good.meta["mode"], "bypass")
        self.assertLessEqual(good.meta["hz"], 5.0)
        self.assertEqual(good.depth.shape, (45, 80))
        self.assertEqual(len(good.meta["pos"]), 3)

        await self.send_cmd(writer, {"cmd": "set_port", "port": 99999, "id": 7})
        await self.send_cmd(writer, {"cmd": "set_mode", "mode": "direct", "id": 8})
        acks, direct_frame = {}, None
        deadline = time.monotonic() + 25
        while time.monotonic() < deadline and not (len(acks) == 2 and direct_frame is not None):
            opcode, payload = await self.next_frame(reader)
            if opcode == ws_server.OP_TEXT:
                msg = json.loads(payload)
                if msg["type"] == "ack":
                    acks[msg["id"]] = msg
            elif protocol.frame_kind(payload) == "telemetry":
                f = protocol.decode_telemetry(payload)
                if f.meta["mode"] == "direct" and not f.shader_warming:
                    direct_frame = f
        self.assertFalse(acks[7]["ok"])
        self.assertTrue(acks[8]["ok"])
        self.assertIsNotNone(direct_frame)
        self.assertEqual(direct_frame.depth.shape, (90, 160))
        self.assertEqual(direct_frame.depth_enc, protocol.DEPTH_U16)
        self.assertEqual(direct_frame.meta["hz"], 20.0)

        await self.send_cmd(writer, {"cmd": "set_rgb", "enabled": True, "id": 9})
        got_rgb = False
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline and not got_rgb:
            opcode, payload = await self.next_frame(reader)
            if opcode == ws_server.OP_BINARY and protocol.frame_kind(payload) == "rgb":
                _seq, w, h, png = protocol.decode_rgb(payload)
                got_rgb = png.startswith(b"\x89PNG") and (w, h) == (160, 90)
        self.assertTrue(got_rgb)
        writer.write(ws_server.encode_frame(ws_server.OP_CLOSE, b"\x03\xe8", mask_key=b"abcd"))
        await writer.drain()
        writer.close()

    async def test_tap_pauses_without_viewers(self):
        await asyncio.sleep(1.0)
        self.assertEqual(self.tap.status()["state"], "paused")
        self.assertEqual(self.tap.status()["rpc_calls"], {})


if __name__ == "__main__":
    unittest.main()



