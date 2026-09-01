#!/usr/bin/env python3
"""Launch packaged UE windows in small waves so screen recording stays smooth.

Six simultaneous windows overload a single GPU: the chase view stutters and
AirSim RPC connections drop. Default is two windows per wave.

Example:
  pixi run python -m scripts.evaluation.record_parallel --scene tunnel_obs --max-parallel 2
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import socket
import subprocess
import sys
import time
from ctypes import wintypes
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.training.train_parallel import SCENES, read_start_position
from tools.airsim_instances import (
    auto_pin_p_core,
    load_settings,
    resolve_runtime_executable,
    terminate_processes,
    validate_ports_available,
    write_instance_settings,
)
from tools.experiment_paths import DEMO_RECORDINGS_ROOT, TRAIN_RUNS_ROOT

# Cooked map names inside D:\UEBuilds\TunnelEnv (not the editor aliases).
PACKAGED_MAPS = {
    "tunnel": "/Game/Tunnel_Col_Vol1/Tunnel",
    "tunnel_obs": "/Game/Tunnel_Col_Vol1/Maps/Tunnel_Obs",
    "tunnel_activate": "/Game/Tunnel_Col_Vol1/Maps/Tunnel_Activate",
    "tunnel_mixed": "/Game/Tunnel_Col_Vol1/Maps/Tunnel_AO",
}

# Paper-selected seed per method: highest 100-ep success, then reward, then smoothness.
BEST_JOBS = {
    "tunnel": [
        {"method": "ma", "seed": 0, "model": "best", "label": "MA-SAC"},
        {"method": "mamba", "seed": 0, "model": "best", "label": "Mamba-SAC"},
        {"method": "lstm", "seed": 1, "model": "best", "label": "LSTM-SAC"},
        {"method": "coordatt", "seed": 0, "model": "best", "label": "CoordAtt-SAC"},
        {"method": "transformer", "seed": 2, "model": "best", "label": "Transformer-SAC"},
        {"method": "sac", "seed": 0, "model": "best", "label": "SAC"},
    ],
    "tunnel_obs": [
        {"method": "ma", "seed": 0, "model": "best", "label": "MA-SAC"},
        {"method": "mamba", "seed": 2, "model": "best", "label": "Mamba-SAC"},
        {"method": "lstm", "seed": 1, "model": "checkpoint", "step": 60_000, "label": "LSTM-SAC"},
        {"method": "coordatt", "seed": 0, "model": "checkpoint", "step": 40_000, "label": "CoordAtt-SAC"},
        {"method": "transformer", "seed": 1, "model": "best", "label": "Transformer-SAC"},
        {"method": "sac", "seed": 2, "model": "final", "label": "SAC"},
    ],
    "tunnel_activate": [
        {"method": "ma", "seed": 2, "model": "checkpoint", "step": 200_000, "label": "MA-SAC"},
        {"method": "mamba", "seed": 2, "model": "best", "label": "Mamba-SAC"},
        {"method": "lstm", "seed": 2, "model": "checkpoint", "step": 130_000, "label": "LSTM-SAC"},
        {"method": "coordatt", "seed": 2, "model": "checkpoint", "step": 140_000, "label": "CoordAtt-SAC"},
        {"method": "transformer", "seed": 1, "model": "checkpoint", "step": 180_000, "label": "Transformer-SAC"},
        {"method": "sac", "seed": 0, "model": "best", "label": "SAC"},
    ],
    "tunnel_mixed": [
        {"method": "ma", "seed": 2, "model": "best", "label": "MA-SAC"},
        {"method": "mamba", "seed": 2, "model": "best", "label": "Mamba-SAC"},
        {"method": "lstm", "seed": 1, "model": "checkpoint", "step": 100_000, "label": "LSTM-SAC"},
        {"method": "coordatt", "seed": 0, "model": "best", "label": "CoordAtt-SAC"},
        {"method": "transformer", "seed": 1, "model": "best", "label": "Transformer-SAC"},
        {"method": "sac", "seed": 1, "model": "checkpoint", "step": 170_000, "label": "SAC"},
    ],
}

DEFAULT_PORTS = [41451, 41461, 41471, 41491, 41501, 41511]
DEFAULT_UE_EXE = Path(r"D:\UEBuilds\TunnelEnv\Windows\Tunnel.exe")


def user32():
    """Window placement needs Win32; keep the import lazy so the module loads anywhere."""
    return ctypes.windll.user32


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", choices=SCENES, default="tunnel_obs")
    parser.add_argument("-n", "--episodes", type=int, default=8)
    parser.add_argument("--ports", nargs="+", type=int, default=DEFAULT_PORTS)
    parser.add_argument("--ue-exe", type=Path, default=DEFAULT_UE_EXE)
    parser.add_argument("--res-x", type=int, default=None)
    parser.add_argument("--res-y", type=int, default=None)
    parser.add_argument("--startup-wait", type=int, default=70)
    parser.add_argument("--clock-speed", type=float, default=1.0)
    parser.add_argument("--airsim-ip", default="127.0.0.1")
    parser.add_argument("--scenario-seed", type=int, default=2026)
    parser.add_argument("--keep-airsim", action="store_true")
    parser.add_argument(
        "--max-parallel",
        type=int,
        default=2,
        help="UE windows per recording wave (default 2). Six at once overloads the GPU.",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=None,
        help="Subset of methods to record, e.g. ma mamba",
    )
    parser.add_argument(
        "--ready-wait",
        type=int,
        default=15,
        help="Seconds after windows appear before evaluation starts, so recording can begin.",
    )
    parser.add_argument(
        "--wave-gap",
        type=int,
        default=8,
        help="Seconds to wait after closing a wave so GPU and RPC ports can recover.",
    )
    return parser.parse_args()


def default_resolution(max_parallel: int) -> tuple[int, int]:
    if max_parallel <= 1:
        return 1280, 720
    if max_parallel == 2:
        return 960, 540
    if max_parallel == 3:
        return 640, 540
    return 640, 480


def build_waves(jobs: list[dict], width: int) -> list[list[dict]]:
    if width < 1:
        raise ValueError("--max-parallel must be at least 1")
    return [jobs[offset : offset + width] for offset in range(0, len(jobs), width)]


def layout_origin(index: int, count: int, res_x: int, res_y: int) -> tuple[int, int]:
    if count <= 3:
        return index * res_x, 0
    cols = 3
    return (index % cols) * res_x, (index // cols) * res_y


def find_run_dir(scene: str, method: str, seed: int) -> Path:
    """Resolve a run directory without walking checkpoint zips."""
    training = TRAIN_RUNS_ROOT
    seed_name = f"seed_{seed}"
    candidates: list[Path] = []
    direct = training / scene / method / seed_name
    if direct.is_dir():
        candidates.extend(path for path in direct.iterdir() if path.is_dir())
    run_dirs = [path for path in candidates if (path / "run.json").is_file()]
    if not run_dirs:
        raise FileNotFoundError(f"No training run for {scene}/{method}/seed_{seed}")
    return sorted(run_dirs)[-1]


def recording_settings_template(path: Path) -> Path:
    settings = {
        "SettingsVersion": 2.0,
        "SimMode": "Multirotor",
        "ViewMode": "SpringArmChase",
        "ClockSpeed": 1.0,
        "LocalHostIp": "127.0.0.1",
        "Vehicles": {
            "SimpleFlight": {
                "VehicleType": "SimpleFlight",
                "AutoCreate": True,
            }
        },
        "CameraDefaults": {
            "CaptureSettings": [
                {"ImageType": 0, "Width": 640, "Height": 480, "FOV_Degrees": 90},
                {"ImageType": 1, "Width": 90, "Height": 60, "FOV_Degrees": 120},
                {"ImageType": 2, "Width": 90, "Height": 60, "FOV_Degrees": 120},
            ]
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(settings, indent=2), encoding="utf-8")
    return path


def build_windowed_command(
    ue_exe: Path,
    map_path: str,
    settings_path: Path,
    res_x: int,
    res_y: int,
    win_x: int,
    win_y: int,
) -> list[str]:
    return [
        str(ue_exe),
        map_path,
        "-windowed",
        f"-ResX={res_x}",
        f"-ResY={res_y}",
        f"-WinX={win_x}",
        f"-WinY={win_y}",
        "-NoVSync",
        "-NoSound",
        f"-settings={settings_path.resolve()}",
    ]


def child_pids(pid: int) -> list[int]:
    try:
        result = subprocess.run(
            [
                "powershell",
                "-NoProfile",
                "-Command",
                f"(Get-CimInstance Win32_Process | Where-Object {{ $_.ParentProcessId -eq {pid} }}).ProcessId",
            ],
            capture_output=True,
            text=True,
            timeout=8,
        )
    except Exception:
        return []
    pids = []
    for line in result.stdout.split():
        if line.strip().isdigit():
            pids.append(int(line.strip()))
    return pids


def windows_for_pids(pids: set[int]) -> list[int]:
    api = user32()
    hits: list[int] = []

    @ctypes.WINFUNCTYPE(ctypes.c_bool, wintypes.HWND, wintypes.LPARAM)
    def callback(hwnd, _lparam):
        if not api.IsWindowVisible(hwnd):
            return True
        proc = wintypes.DWORD()
        api.GetWindowThreadProcessId(hwnd, ctypes.byref(proc))
        if proc.value in pids:
            hits.append(int(hwnd))
        return True

    api.EnumWindows(callback, 0)
    return hits


def title_and_place(hwnd: int, title: str, x: int, y: int, w: int, h: int) -> None:
    api = user32()
    api.SetWindowTextW(hwnd, title)
    api.MoveWindow(hwnd, x, y, w, h, True)


def build_eval_command(job: dict, port: int, args: argparse.Namespace) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "scripts.evaluation.evaluate_single",
        "-m",
        str(job["run_dir"]),
        "-n",
        str(args.episodes),
        "--airsim-ip",
        args.airsim_ip,
        "--airsim-port",
        str(port),
        "--scenario-seed",
        str(args.scenario_seed),
    ]
    if job["model"] == "best":
        command.append("--best")
    elif job["model"] == "checkpoint":
        command.extend(["--checkpoint-step", str(job["step"])])
    return command


def terminate_tree(processes: list[subprocess.Popen]) -> None:
    if os.name == "nt":
        for process in processes:
            subprocess.run(
                ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                capture_output=True,
                text=True,
                timeout=15,
            )
    terminate_processes(processes)


def wait_for_ports_free(host: str, ports: list[int], timeout: int) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            validate_ports_available(host, ports)
            return
        except RuntimeError:
            time.sleep(1)
    validate_ports_available(host, ports)


def port_is_open(host: str, port: int) -> bool:
    try:
        with socket.create_connection((host, port), timeout=0.5):
            return True
    except OSError:
        return False


def wait_until_airsim_ready(
    host: str,
    ports: list[int],
    processes: list[subprocess.Popen],
    timeout: int,
) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        dead = [process.pid for process in processes if process.poll() is not None]
        if dead:
            raise RuntimeError(
                f"Packaged UE exited during startup (pids {dead}). "
                "Check Tunnel.log for ViewportClosed / map errors."
            )
        if all(port_is_open(host, port) for port in ports):
            print("AirSim RPC is listening.", flush=True)
            return
        time.sleep(2)
    raise RuntimeError(f"AirSim RPC did not become ready on {ports} within {timeout}s")


def wait_for_recorder(seconds: int, wave_label: str) -> None:
    print(f"Start recording now: {wave_label}", flush=True)
    if seconds <= 0:
        return
    print(f"Evaluation starts in {seconds}s.", flush=True)
    for remaining in range(seconds, 0, -1):
        print(f"  {remaining}...", flush=True)
        time.sleep(1)


def resolve_jobs(scene: str, methods: list[str] | None) -> list[dict]:
    jobs = []
    wanted = set(methods) if methods else None
    for spec in BEST_JOBS[scene]:
        if wanted is not None and spec["method"] not in wanted:
            continue
        run_dir = find_run_dir(scene, spec["method"], spec["seed"])
        job = dict(spec)
        job["run_dir"] = run_dir
        jobs.append(job)
    if wanted is not None:
        missing = wanted - {job["method"] for job in jobs}
        if missing:
            raise ValueError(f"Unknown or unavailable methods: {sorted(missing)}")
    if not jobs:
        raise ValueError("No recording jobs selected")
    return jobs


def run_wave(
    *,
    wave_index: int,
    wave_count: int,
    jobs: list[dict],
    ports: list[int],
    args: argparse.Namespace,
    runtime_exe: Path,
    template_data: dict,
    start_position: list[float],
    run_dir: Path,
) -> list[str]:
    labels = " | ".join(job["label"] for job in jobs)
    print("=" * 72, flush=True)
    print(f"Wave {wave_index}/{wave_count}: {labels}", flush=True)
    for index, (job, port) in enumerate(zip(jobs, ports)):
        x, y = layout_origin(index, len(jobs), args.res_x, args.res_y)
        ckpt = job["model"] if job["model"] != "checkpoint" else f"ckpt_{job['step']}"
        print(
            f"  [{index + 1}] {job['label']:16s}  seed {job['seed']}  {ckpt:12s}  "
            f"port {port}  window ({x},{y})"
        )
        print(f"      {job['run_dir'].relative_to(ROOT)}")

    validate_ports_available(args.airsim_ip, ports)
    airsim_processes: list[subprocess.Popen] = []
    eval_handles = []
    try:
        for index, (job, port) in enumerate(zip(jobs, ports)):
            x, y = layout_origin(index, len(jobs), args.res_x, args.res_y)
            settings_path = write_instance_settings(
                template_data,
                run_dir / "airsim_settings",
                port,
                (wave_index - 1) * len(ports) + index,
                args.clock_speed,
                start_position,
            )
            command = build_windowed_command(
                runtime_exe,
                PACKAGED_MAPS[args.scene],
                settings_path,
                args.res_x,
                args.res_y,
                x,
                y,
            )
            process = subprocess.Popen(command, cwd=runtime_exe.parent, env=None)
            airsim_processes.append(process)
            print(f"[UE] {job['label']} port={port} pid={process.pid} pos=({x},{y})")

        print(f"Waiting up to {args.startup_wait}s for packaged AirSim...")
        wait_until_airsim_ready(args.airsim_ip, ports, airsim_processes, args.startup_wait)
        auto_pin_p_core([process.pid for process in airsim_processes])

        for index, (job, port, process) in enumerate(zip(jobs, ports, airsim_processes)):
            x, y = layout_origin(index, len(jobs), args.res_x, args.res_y)
            pids = {process.pid, *child_pids(process.pid)}
            hwnds = windows_for_pids(pids)
            title = f"{job['label']}  seed{job['seed']}"
            if hwnds:
                title_and_place(hwnds[0], title, x, y, args.res_x, args.res_y)
                print(f"[WINDOW] {title}")
            else:
                print(f"[WINDOW] not found for {title}; check port {port}")

        wait_for_recorder(args.ready_wait, labels)
        print("Starting evaluation. Record this wave.", flush=True)
        for job, port in zip(jobs, ports):
            stem = f"{args.scene}_{job['method']}_seed_{job['seed']}_port_{port}"
            stdout = (run_dir / f"{stem}.out.log").open("w", encoding="utf-8")
            stderr = (run_dir / f"{stem}.err.log").open("w", encoding="utf-8")
            process = subprocess.Popen(
                build_eval_command(job, port, args),
                cwd=ROOT,
                stdout=stdout,
                stderr=stderr,
                env=os.environ.copy(),
            )
            auto_pin_p_core([process.pid])
            eval_handles.append((job, port, process, stdout, stderr))
            print(f"[EVAL] {job['label']} seed={job['seed']} port={port} pid={process.pid}")

        failures = []
        for job, port, process, stdout, stderr in eval_handles:
            code = process.wait()
            stdout.close()
            stderr.close()
            label = f"{job['label']} seed={job['seed']} port={port}"
            if code:
                failures.append(f"{label}: exit {code}")
            print(f"[{'DONE' if code == 0 else 'FAIL'}] {label}")
        return failures
    finally:
        last_wave = wave_index == wave_count
        if not (args.keep_airsim and last_wave):
            terminate_tree(airsim_processes)


def main() -> int:
    sys.stdout.reconfigure(line_buffering=True)
    args = parse_args()
    if args.scene not in BEST_JOBS:
        raise ValueError(f"No recording seed table for scene {args.scene}")
    if args.max_parallel < 1:
        raise ValueError("--max-parallel must be at least 1")
    if args.res_x is None or args.res_y is None:
        res_x, res_y = default_resolution(args.max_parallel)
        if args.res_x is None:
            args.res_x = res_x
        if args.res_y is None:
            args.res_y = res_y
    if len(args.ports) < args.max_parallel:
        raise ValueError("--ports must provide at least --max-parallel values")
    if len(set(args.ports[: args.max_parallel])) != args.max_parallel:
        raise ValueError("Wave ports must be unique")

    jobs = resolve_jobs(args.scene, args.methods)
    waves = build_waves(jobs, args.max_parallel)
    start_position = read_start_position(jobs[0]["run_dir"] / "config" / "config.ini")
    run_id = datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%f")
    run_dir = DEMO_RECORDINGS_ROOT / run_id
    template = recording_settings_template(run_dir / "settings_template.json")
    template_data = load_settings(template)
    runtime_exe = resolve_runtime_executable(args.ue_exe)
    if not runtime_exe.is_file():
        raise FileNotFoundError(runtime_exe)

    print("=" * 72, flush=True)
    print(f"Scene: {args.scene}  map: {PACKAGED_MAPS[args.scene]}", flush=True)
    print(
        f"Waves: {len(waves)} x {args.max_parallel} window(s) at {args.res_x}x{args.res_y}"
    )
    print("=" * 72)

    failures: list[str] = []
    for wave_index, wave in enumerate(waves, start=1):
        ports = args.ports[: len(wave)]
        failures.extend(
            run_wave(
                wave_index=wave_index,
                wave_count=len(waves),
                jobs=wave,
                ports=ports,
                args=args,
                runtime_exe=runtime_exe,
                template_data=template_data,
                start_position=start_position,
                run_dir=run_dir,
            )
        )
        if wave_index < len(waves):
            print(f"Closing wave {wave_index}. Stop recording; next wave in {args.wave_gap}s.")
            wait_for_ports_free(args.airsim_ip, ports, timeout=max(args.wave_gap, 20))
            extra = args.wave_gap - 1
            if extra > 0:
                time.sleep(extra)
    if failures:
        print("Failures:")
        for item in failures:
            print(f"  {item}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
