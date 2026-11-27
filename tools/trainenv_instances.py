"""Owned, depth-only UE 5.4 instances for local TrainEnv training."""
from contextlib import contextmanager
from copy import deepcopy
import ctypes
from ctypes import wintypes
import json
import os
from pathlib import Path
import socket
import subprocess
from tempfile import TemporaryDirectory
import time
from uuid import uuid4

import cosysairsim as airsim
import numpy as np

from tools.airsim_instances import (
    load_settings, resolve_runtime_executable, validate_ports_available, write_instance_settings,
)


HOST = "127.0.0.1"
TRAINENV_MAP = "/Game/UAVTraining/Maps/TrainEnv"
TRAINENV_3D_MAP = "/Game/UAVTraining/Maps/TrainEnv_3D"
DEFAULT_EXE = Path(r"D:\UEBuilds\TrainEnv\Windows\TrainEnvAirSim.exe")
DEFAULT_3D_EXE = Path(r"D:\UEBuilds\TrainEnv3D\Windows\TrainEnvAirSim.exe")
DEFAULT_EDITOR = Path("D:/software/Epic Games/UE/UE_5.4/Engine/Binaries/Win64/UnrealEditor.exe")
DEFAULT_TEMPLATE = Path.home() / "Documents/AirSim/settings.json"
DEPTH_SIZE = (320, 180)
MIN_READY_DEPTH_M = 1.2
RENDER_COMMANDS = (
    "t.MaxFPS 60",
    "sg.ShadowQuality 0",
    "sg.GlobalIlluminationQuality 0",
    "sg.ReflectionQuality 0",
    "sg.PostProcessQuality 0",
    "sg.EffectsQuality 0",
    "sg.AntiAliasingQuality 0",
    "r.ScreenPercentage 25",
)


def trainenv_settings(template, *, show_viewport=False):
    settings = deepcopy(template)
    settings.update(ViewMode="Fpv" if show_viewport else "NoDisplay", LocalHostIp=HOST, SubWindows=[])
    global_camera = settings.setdefault("CameraDefaults", {})
    for camera in [
        global_camera,
        *settings.get("ExternalCameras", {}).values(),
        *(camera for vehicle in settings.get("Vehicles", {}).values()
          for camera in vehicle.get("Cameras", {}).values()),
    ]:
        capture = [item for item in camera.get("CaptureSettings", [])
                   if item.get("ImageType") == airsim.ImageType.DepthPlanar]
        if camera is global_camera and not capture:
            capture = [{"ImageType": airsim.ImageType.DepthPlanar}]
        for item in capture:
            item["Width"], item["Height"] = DEPTH_SIZE
        if capture or "CaptureSettings" in camera:
            camera["CaptureSettings"] = capture
    return settings


def trainenv_command(executable, settings_path, log_path, map_path=TRAINENV_MAP, editor_project=None, *,
                     show_viewport=False):
    target = [str(executable), map_path]
    if editor_project is not None:
        target = [str(executable), str(editor_project), map_path, "-game", "-DisablePlugins=FunplayMCP"]
    render_commands = RENDER_COMMANDS
    if show_viewport:
        render_commands += ("sg.PostProcessQuality 2", "r.ScreenPercentage 100")
    return [
        *target, "-windowed", "-ResX=160", "-ResY=120",
        "-NoSound", "-unattended", "-nosplash", f"-settings={settings_path}",
        f"-AbsLog={log_path}", "-ExecCmds=" + ",".join(render_commands),
    ]


class _BasicJobLimits(ctypes.Structure):
    _fields_ = [
        ("per_process_time", ctypes.c_int64), ("per_job_time", ctypes.c_int64),
        ("flags", wintypes.DWORD), ("min_working_set", ctypes.c_size_t),
        ("max_working_set", ctypes.c_size_t), ("active_processes", wintypes.DWORD),
        ("affinity", ctypes.c_size_t), ("priority", wintypes.DWORD),
        ("scheduling_class", wintypes.DWORD),
    ]


class _JobLimits(ctypes.Structure):
    _fields_ = [
        ("basic", _BasicJobLimits), ("io_counters", ctypes.c_uint64 * 6),
        ("process_memory_limit", ctypes.c_size_t), ("job_memory_limit", ctypes.c_size_t),
        ("peak_process_memory", ctypes.c_size_t), ("peak_job_memory", ctypes.c_size_t),
    ]


class _UnrealJob:
    """Windows closes the job and reaps its Unreal process trees if the owner exits."""

    def __enter__(self):
        if os.name != "nt":
            raise RuntimeError("The packaged TrainEnv launcher requires Windows")
        self.api = ctypes.WinDLL("kernel32", use_last_error=True)
        self.api.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
        self.api.CreateJobObjectW.restype = wintypes.HANDLE
        self.api.CloseHandle.argtypes = [wintypes.HANDLE]
        self.api.SetInformationJobObject.argtypes = [
            wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD,
        ]
        self.api.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
        self.api.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        self.api.OpenProcess.restype = wintypes.HANDLE

        self.simulators = {}
        self.handle = self.api.CreateJobObjectW(None, None)
        if not self.handle:
            raise ctypes.WinError(ctypes.get_last_error())
        limits = _JobLimits()
        limits.basic.flags = 0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        if not self.api.SetInformationJobObject(self.handle, 9, ctypes.byref(limits), ctypes.sizeof(limits)):
            error = ctypes.get_last_error()
            self.api.CloseHandle(self.handle)
            raise ctypes.WinError(error)
        return self

    def add(self, process):
        # AssignProcessToJobObject requires PROCESS_SET_QUOTA | PROCESS_TERMINATE.
        handle = self.api.OpenProcess(0x0101, False, process.pid)
        if not handle:
            raise ctypes.WinError(ctypes.get_last_error())
        try:
            if not self.api.AssignProcessToJobObject(self.handle, handle):
                raise ctypes.WinError(ctypes.get_last_error())
        finally:
            self.api.CloseHandle(handle)

    def __exit__(self, *exc):
        self.api.CloseHandle(self.handle)

    def stop_simulator(self, port):
        process = self.simulators[port]
        if process.poll() is None:
            process.kill()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                # The rest of this batch shares the job, and launch_trainenv
                # reaps every child again after closing that job.
                print(f"[TrainEnv] Awaiting job cleanup for port={port} pid={process.pid}", flush=True)


def _probe_depth_ready(client, vehicle_name):
    """Ensure DepthPlanar has compiled its post-process shader (not [0, 1] RGB fallback)."""
    responses = client.simGetImages(
        [airsim.ImageRequest("0", airsim.ImageType.DepthPlanar, True, False)],
        vehicle_name=vehicle_name,
    )
    if not responses:
        return None
    image = responses[0]
    if (int(image.width), int(image.height)) != DEPTH_SIZE:
        return None
    values = np.asarray(image.image_data_float, dtype=np.float32)
    if values.size != DEPTH_SIZE[0] * DEPTH_SIZE[1]:
        return None
    valid = values[np.isfinite(values) & (values > 0)]
    if valid.size < 0.99 * values.size:
        return None
    max_depth = float(valid.max())
    if max_depth <= MIN_READY_DEPTH_M:
        return None
    return max_depth


def _wait_ready(processes, launch_id, timeout):
    pending = dict(processes)
    errors = {}
    deadline = time.monotonic() + timeout
    while pending and time.monotonic() < deadline:
        for port, process in list(pending.items()):
            if process.poll() is not None:
                raise RuntimeError(f"TrainEnv PID {process.pid} on port {port} exited with code {process.returncode}")
            try:
                with socket.create_connection((HOST, port), timeout=.25):
                    pass
                client = airsim.MultirotorClient(ip=HOST, port=port, timeout_value=5)
                try:
                    vehicles = client.listVehicles() if client.ping() else []
                    settings = json.loads(client.getSettingsString()) if vehicles else {}
                    if vehicles:
                        if settings.get("TrainEnvLaunchId") != launch_id or settings.get("ApiServerPort") != port:
                            raise RuntimeError(f"RPC port {port} answered with settings from a different launch")
                        max_depth = _probe_depth_ready(client, vehicles[0])
                    else:
                        max_depth = None
                finally:
                    client.client.close()
            except RuntimeError:
                raise
            except Exception as exc:
                errors[port] = str(exc)
                continue
            if vehicles and max_depth is not None:
                del pending[port]
                print(
                    f"[TrainEnv] RPC + DepthPlanar ready port={port} pid={process.pid} max_m={max_depth:.2f}",
                    flush=True,
                )
            elif vehicles:
                errors[port] = f"DepthPlanar shader not ready (max_depth <= {MIN_READY_DEPTH_M:g}m)"
        if pending:
            time.sleep(.25)
    if pending:
        raise TimeoutError(f"TrainEnv RPC not ready on ports {list(pending)} after {timeout:g}s. Last errors {errors}")


@contextmanager
def launch_trainenv(
    *, executable, template, ports, start_position, clock_speed, run_dir, timeout, map_path=TRAINENV_MAP,
    editor_project=None, show_viewport=False, clock_type=None,
):
    """Launch TrainEnv instances and bind their lifetime to the caller."""
    if not executable.is_file():
        raise FileNotFoundError(f"TrainEnv executable not found: {executable}")
    if editor_project is not None and not editor_project.is_file():
        raise FileNotFoundError(f"Unreal source project not found: {editor_project}")
    runtime_exe = (executable if editor_project else resolve_runtime_executable(executable)).resolve()
    validate_ports_available(HOST, ports)
    launch_id = uuid4().hex
    settings = trainenv_settings(load_settings(template), show_viewport=show_viewport)
    settings["TrainEnvLaunchId"] = launch_id
    if clock_type is not None:
        settings["ClockType"] = clock_type
    processes = {}
    manifest = {
        "launcher_pid": os.getpid(), "map": map_path, "clock_speed": clock_speed,
        "host": HOST, "settings_template": str(template), "capture_size": DEPTH_SIZE,
        "instances": [],
    }
    if editor_project:
        manifest["editor_project"] = str(editor_project)
    with TemporaryDirectory(prefix="trainenv-") as temporary:
        try:
            with _UnrealJob() as job:
                job.simulators = processes
                for index, port in enumerate(ports):
                    instance_settings = write_instance_settings(
                        settings, Path(temporary), port, index, clock_speed, start_position,
                    )
                    log_path = run_dir / f"ue_{port}.log"
                    command = trainenv_command(runtime_exe, instance_settings, log_path, map_path, editor_project,
                                               show_viewport=show_viewport)
                    process = subprocess.Popen(command, cwd=runtime_exe.parent, creationflags=subprocess.CREATE_NO_WINDOW)
                    try:
                        job.add(process)
                    except BaseException:
                        process.terminate()
                        process.wait()
                        raise
                    processes[port] = process
                    manifest["instances"].append({
                        "port": port, "pid": process.pid, "command": command,
                        "settings": json.loads(instance_settings.read_text(encoding="utf-8")),
                    })
                    print(f"[TrainEnv] Starting port={port} pid={process.pid}", flush=True)
                (run_dir / "launch.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
                _wait_ready(processes, launch_id, timeout)
                yield job
        finally:
            for process in processes.values():
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    # A standalone editor can linger while tearing down shader workers.
                    # The handle belongs to this launch, even if its RPC port is already closed.
                    process.kill()
                    process.wait(timeout=10)
