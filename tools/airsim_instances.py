"""Minimal helpers for launching independent packaged AirSim instances."""

from __future__ import annotations

import copy
import ctypes
import json
import os
import re
import socket
import subprocess
import time
from pathlib import Path


def load_settings(path: Path) -> dict:
    if not path.is_file():
        return {"SettingsVersion": 2.0, "SimMode": "Multirotor"}
    settings = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(settings, dict):
        raise ValueError(f"AirSim settings must be a JSON object: {path}")
    return settings


def write_instance_settings(
    template: dict,
    output_dir: Path,
    port: int,
    index: int,
    clock_speed: float,
    start_position: list[float],
) -> Path:
    settings = copy.deepcopy(template)
    settings["SettingsVersion"] = settings.get("SettingsVersion", 2.0)
    settings["SimMode"] = "Multirotor"
    settings["ViewMode"] = settings.get("ViewMode", "NoDisplay")
    settings["ClockSpeed"] = clock_speed
    settings["ApiServerPort"] = port
    settings["LocalHostIp"] = settings.get("LocalHostIp", "127.0.0.1")

    vehicles = settings.get("Vehicles")
    if not isinstance(vehicles, dict) or not vehicles:
        vehicles = {"SimpleFlight": {"VehicleType": "SimpleFlight", "AutoCreate": True}}
        settings["Vehicles"] = vehicles
    for vehicle in vehicles.values():
        if isinstance(vehicle, dict) and vehicle.get("AutoCreate", True):
            vehicle.update(
                {
                    "X": float(start_position[0]),
                    "Y": float(start_position[1]),
                    "Z": float(start_position[2]),
                }
            )
            vehicle.setdefault("Yaw", 0)

    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir / f"settings_instance_{index}_port_{port}.json"
    target.write_text(json.dumps(settings, indent=2), encoding="utf-8")
    return target


def build_ue_command(
    ue_exe: Path,
    map_path: str,
    settings_path: Path,
    res_x: int,
    res_y: int,
) -> list[str]:
    return [
        str(ue_exe),
        map_path,
        "-windowed",
        f"-ResX={res_x}",
        f"-ResY={res_y}",
        "-NoVSync",
        "-NoSound",
        f"-settings={settings_path.resolve()}",
    ]


def resolve_runtime_executable(ue_exe: Path) -> Path:
    """Prefer the packaged game binary over its short-lived bootstrap launcher."""
    packaged_binary = (
        ue_exe.parent
        / ue_exe.stem
        / "Binaries"
        / "Win64"
        / ue_exe.name
    )
    return packaged_binary if packaged_binary.is_file() else ue_exe


def validate_ports_available(host: str, ports: list[int]) -> None:
    occupied = []
    for port in ports:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            try:
                probe.bind((host, port))
            except OSError:
                occupied.append(port)
    if occupied:
        raise RuntimeError(f"AirSim RPC ports already occupied: {occupied}")


# ------------------------------------------------------------------ #
# CPU affinity: pin processes to P-cores on Intel hybrid architectures #
# ------------------------------------------------------------------ #

_cpu_topology_cache: tuple[int | None, int | None] | None = None


def _get_cpu_topology() -> tuple[int | None, int | None]:
    """Return (physical_cores, logical_processors). Cached after first call."""
    global _cpu_topology_cache
    if _cpu_topology_cache is not None:
        return _cpu_topology_cache
    logical = os.cpu_count()
    if logical is None or os.name != "nt":
        _cpu_topology_cache = (None, logical)
        return _cpu_topology_cache
    try:
        result = subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             "(Get-CimInstance Win32_Processor).NumberOfCores"],
            capture_output=True, text=True, timeout=5,
        )
        physical = int(result.stdout.strip())
    except Exception:
        physical = None
    _cpu_topology_cache = (physical, logical)
    return _cpu_topology_cache


def _compute_p_core_mask(physical: int | None, logical: int | None) -> int | None:
    """Compute the P-core logical-processor bitmask.

    Intel hybrid CPUs have P-cores (2 threads each) and E-cores (1 thread
    each).  P-core count = logical - physical; P-core logicals = that * 2;
    they occupy the first N logical processors.

    On a symmetric SMT CPU (physical * 2 == logical) returns ``None``
    (no pinning needed).
    """
    if physical is None or logical is None:
        return None
    if physical * 2 == logical:
        return None  # symmetric SMT, no E-cores
    p_core_logical = (logical - physical) * 2
    if p_core_logical <= 0 or p_core_logical >= logical:
        return None  # unexpected topology
    return (1 << p_core_logical) - 1


def auto_pin_p_core(pids: list[int]) -> list[int]:
    """Pin processes to P-cores on Intel hybrid CPUs (no-op otherwise).

    On Intel 12th-14th gen P/E-core CPUs the Windows Thread Director may
    schedule UE4 render/physics threads onto E-cores, increasing per-step
    latency.  This moves the given processes to P-core logical processors.

    On symmetric SMT CPUs (no E-cores) this is a no-op.

    Returns the list of PIDs that were successfully pinned.
    """
    if os.name != "nt":
        return []
    physical, logical = _get_cpu_topology()
    mask = _compute_p_core_mask(physical, logical)
    if mask is None:
        return []
    kernel32 = ctypes.windll.kernel32
    PROCESS_SET_INFORMATION = 0x0100
    pinned: list[int] = []
    for pid in pids:
        try:
            handle = kernel32.OpenProcess(PROCESS_SET_INFORMATION, False, pid)
            if handle:
                if kernel32.SetProcessAffinityMask(
                    handle, ctypes.c_size_t(mask)
                ):
                    pinned.append(pid)
                kernel32.CloseHandle(handle)
        except Exception:
            pass
    if pinned:
        print(
            f"[CPU] Pinned {len(pinned)} process(es) to P-core "
            f"(physical={physical}, logical={logical}, mask=0x{mask:X})"
        )
    return pinned


def launch_instances(
    *,
    ue_exe: Path,
    map_path: str,
    settings_template: Path,
    output_dir: Path,
    ports: list[int],
    start_position: list[float],
    clock_speed: float,
    res_x: int,
    res_y: int,
    wait_seconds: int,
) -> list[subprocess.Popen]:
    if not ue_exe.is_file():
        raise FileNotFoundError(f"Unreal executable not found: {ue_exe}")
    runtime_exe = resolve_runtime_executable(ue_exe)
    template = load_settings(settings_template)
    host = str(template.get("LocalHostIp") or "127.0.0.1")
    validate_ports_available(host, ports)

    processes = []
    for index, port in enumerate(ports):
        settings_path = write_instance_settings(
            template, output_dir, port, index, clock_speed, start_position
        )
        command = build_ue_command(runtime_exe, map_path, settings_path, res_x, res_y)
        processes.append(
            subprocess.Popen(command, cwd=runtime_exe.parent, env=None)
        )
        print(f"[AirSim] instance={index} port={port} pid={processes[-1].pid}")

    if wait_seconds > 0:
        print(f"Waiting {wait_seconds}s for AirSim startup...")
        time.sleep(wait_seconds)

    auto_pin_p_core([p.pid for p in processes])
    return processes


def terminate_processes(processes: list[subprocess.Popen]) -> None:
    for process in processes:
        if process.poll() is None:
            process.terminate()
    for process in processes:
        if process.poll() is None:
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
