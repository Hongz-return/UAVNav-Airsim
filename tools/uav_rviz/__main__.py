"""``python -m tools.uav_rviz`` 启动入口。

示例::

    pixi run python -m tools.uav_rviz --port 42072 --mode bypass --run-dir auto --app
    pixi run python -m tools.uav_rviz --port 41451 --mode direct --open
    pixi run python -m tools.uav_rviz --demo --open

严格只读：本进程不会向任何 AirSim 端口发送 enableApiControl / armDisarm / simPause / reset /
moveByVelocity* 等写请求（见 ``airsim_tap.READ_ONLY_METHODS``）。
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import shutil
import subprocess
import sys
import webbrowser
from pathlib import Path
from typing import Optional

from .airsim_tap import PROFILES, AirSimTap, ReadOnlyClient
from .coords import ned_to_enu_point
from .geometry_loader import DEFAULT_FLOOR_Z, DEFAULT_SPAWN_NED_Z, GeometryLoader, resolve_run_dir
from .ws_server import CommandRouter, RvizServer

PACKAGE_DIR = Path(__file__).resolve().parent
DEFAULT_STATIC_DIR = PACKAGE_DIR / "web" / "dist"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m tools.uav_rviz", description="UAV AirSim 实时 RViz 监控台（严格只读）")
    p.add_argument("--port", type=int, default=42072, help="要监视的 AirSim RPC 端口（可在网页中热切换，默认 42072）")
    p.add_argument("--airsim-host", default="127.0.0.1", help="AirSim 主机地址（默认 127.0.0.1）")
    p.add_argument("--web-port", type=int, default=8765, help="本监控台 HTTP/WS 端口（0=自动分配，默认 8765）")
    p.add_argument("--host", default="127.0.0.1", help="监听地址（默认仅本机回环）")
    p.add_argument("--mode", choices=sorted(PROFILES), default="bypass",
                   help="bypass=训练旁路(低开销,默认) / direct=演示直连(20Hz)")
    p.add_argument("--run-dir", default=None,
                   help="训练运行目录（读取 course_episodes.jsonl 尾部的目标点/回合障碍）；auto=自动取 Logs/TrainData/Runs 下最新")
    p.add_argument("--goal-position", type=float, nargs=3, metavar=("X", "Y", "Z"),
                   help="手动提供 AirSim 车辆局部 NED 目标，单位米")
    p.add_argument("--no-infer-goal", action="store_true", help="只显示已知目标，不从旧隧道几何推断终点")
    p.add_argument("--open", action="store_true", help="启动后用默认浏览器打开")
    p.add_argument("--app", action="store_true", help="尝试用 Edge --app 窗口模式打开，缺失则退回默认浏览器")
    p.add_argument("--demo", "--simulate", dest="simulate", action="store_true",
                   help="使用内置模拟数据源（不连接任何 AirSim 端口）")
    p.add_argument("--always-on", action="store_true", help="没有网页观众时也持续采集（默认无人观看即暂停，零 RPC）")
    p.add_argument("--camera", default="0", help="相机名（训练使用 '0'）")
    p.add_argument("--fov", type=float, default=90.0, help="水平视场角（默认 90°）")
    p.add_argument("--floor-z", type=float, default=DEFAULT_FLOOR_Z, help="地面在车辆局部 z-up 中的高度，用于 AGL（默认 -2.6）")
    p.add_argument("--spawn-ned-z", type=float, default=DEFAULT_SPAWN_NED_Z, help="车辆原点相对 PlayerStart 的 NED z 偏移（默认 1.0）")
    p.add_argument("--static-dir", default=str(DEFAULT_STATIC_DIR), help="前端静态目录（默认 web/dist）")
    p.add_argument("--log-level", default="INFO")
    return p


def find_edge() -> Optional[str]:
    found = shutil.which("msedge") or shutil.which("msedge.exe")
    if found:
        return found
    for base in (os.environ.get("ProgramFiles(x86)"), os.environ.get("ProgramFiles"), os.environ.get("LOCALAPPDATA")):
        if base:
            candidate = Path(base) / "Microsoft" / "Edge" / "Application" / "msedge.exe"
            if candidate.exists():
                return str(candidate)
    return None


def open_ui(url: str, app: bool) -> Optional[str]:
    """打开界面；返回实际使用的方式（edge-app / browser）。"""
    if app:
        edge = find_edge()
        if edge:
            try:
                subprocess.Popen([edge, f"--app={url}", "--window-size=1600,940"],
                                 stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                return "edge-app"
            except OSError:
                pass
    return "browser" if webbrowser.open(url) else None


async def serve(args: argparse.Namespace) -> None:
    run_dir = resolve_run_dir(args.run_dir)
    geometry = GeometryLoader(
        None, spawn_ned_z=args.spawn_ned_z, floor_z=args.floor_z,
        run_dir=run_dir, infer_goals=not args.no_infer_goal,
    )
    if args.goal_position:
        geometry.set_manual_goal(ned_to_enu_point(args.goal_position))
    elif args.simulate and not run_dir:
        geometry.set_manual_goal((85.0, 0.0, args.floor_z + 2.0))
    client_factory = None
    api = None
    if args.simulate:
        from .simulated import SIM_API, SimulatedAirSim

        def client_factory(_host: str, _port: int, _timeout: float) -> ReadOnlyClient:
            return ReadOnlyClient(SimulatedAirSim(floor_z=args.floor_z, spawn_ned_z=args.spawn_ned_z))

        api = SIM_API

    server = RvizServer(host=args.host, port=args.web_port, static_dir=Path(args.static_dir))
    status_details = {"run_dir": str(run_dir) if run_dir else None, "simulate": args.simulate}
    tap = AirSimTap(
        host=args.airsim_host, port=args.port, mode=args.mode, client_factory=client_factory, api=api,
        camera=args.camera, fov_deg=args.fov, geometry=geometry, floor_z=args.floor_z,
        spawn_ned_z=args.spawn_ned_z, always_on=args.always_on,
        on_telemetry=lambda data: server.publish_binary(data, "telemetry"),
        on_rgb=lambda data: server.publish_binary(data, "rgb"),
        on_scene=server.publish_json,
        on_status=lambda message: server.publish_json({**message, **status_details}),
        on_event=server.publish_json,
    )
    router = CommandRouter(tap, lambda: server.port, broadcast=server.publish_json)
    server.command_handler = router
    server.status_provider = lambda: {**tap.status(), **status_details, "color_mode": router.color_mode}
    server.on_clients_changed = tap.set_viewers
    server.initial_messages = lambda: [geometry.scene_message(),
                                       {"type": "event", "event": "color_mode", "value": router.color_mode}]

    await server.start()
    url = f"http://{'127.0.0.1' if args.host in ('0.0.0.0', '') else args.host}:{server.port}/"
    tap.start()
    dist_ok = Path(args.static_dir).joinpath("index.html").exists()
    print(f"[uav_rviz] 监控台: {url}  (模式={args.mode}, AirSim={args.airsim_host}:{args.port}"
          f"{', 模拟数据源' if args.simulate else ''}, 只读)", flush=True)
    if not dist_ok:
        print("[uav_rviz] 未找到前端构建产物，请在 tools/uav_rviz/web 中执行 npm ci 和 npm run build", flush=True)
    if run_dir:
        print(f"[uav_rviz] 运行目录: {run_dir}", flush=True)
    try:
        if args.open or args.app:
            opener = open_ui(url, args.app)
            print(f"[uav_rviz] 已用 {opener} 打开" if opener else f"[uav_rviz] 请在浏览器打开 {url}", flush=True)
        await asyncio.Event().wait()
    finally:
        tap.stop()
        await server.stop()


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not 1 <= args.port <= 65535 or not 0 <= args.web_port <= 65535:
        parser.error("AirSim 端口须在 1–65535 之间，Web 端口须在 0–65535 之间")
    if not args.simulate and args.port == args.web_port and args.airsim_host == args.host:
        parser.error("AirSim 端口与 Web 端口须不同")
    if not 0 < args.fov < 180:
        parser.error("水平视场角须在 0° 到 180° 之间")
    logging.basicConfig(level=getattr(logging, str(args.log_level).upper(), logging.INFO),
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    try:
        asyncio.run(serve(args))
    except KeyboardInterrupt:
        print("\n[uav_rviz] 已退出", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
