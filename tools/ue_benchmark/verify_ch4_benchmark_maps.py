"""Verify Chapter 4 maps with physical AirSim flights in standalone UE 5.4.

Run from the repository root with pixi run python. The UE build script writes
the manifest used here. Use --verify or --evaluate to run only one phase.
"""

import argparse
import configparser
import hashlib
import json
import math
import re
import socket
import subprocess
import sys
import time
import traceback
from contextlib import contextmanager
from pathlib import Path

import cosysairsim as airsim
import cv2
import numpy as np

from tools.airsim_instances import validate_ports_available


ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / "reports/ch4_benchmark_rebuild_20261125"
PROJECT = Path("D:/software/Epic Games/UE/UE_5.4/Env/Tunnel/Tunnel.uproject")
UNREAL = PROJECT.parents[2] / "Engine/Binaries/Win64/UnrealEditor.exe"
MODEL_RUN = ROOT / (
    "Logs/TrainData/Runs/trainenv/"
    "stage2_alt_ceil_20261125_seed0_cs2_4env_1000000_restart_022838/run"
)
CONFIGS = {
    "t1": "ch4_eval_t1_double_static.ini",
    "t2": "ch4_eval_t2_double_dynamic.ini",
    "t3": "ch4_eval_t3_single_scurve.ini",
}
PORTS = {"t1": 41961, "t2": 41962, "t3": 41963}
RANGES = {"t1": (70, 90), "t2": (50, 75), "t3": (45, 70)}
HOST = "127.0.0.1"
VEHICLE = "SimpleFlight"
MAX_DEPTH = 25.0
COLLISION_TOKENS = ("Ch4_", "ParkedTrain_")


def write_json(path, val):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(val, indent=2, allow_nan=False), encoding="utf-8")
    temp.replace(path)


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def read_ini(path):
    config = configparser.ConfigParser()
    with path.open(encoding="utf-8-sig") as handle:
        config.read_file(handle)
    return config


def check_map_identity(spec):
    """Tie simulator results to both the locked source and the saved copy."""
    for key in ("source", "map"):
        package = spec[key]
        if not package.startswith("/Game/RailwayTunnelPrototype/Maps/"):
            raise ValueError(f"Unexpected {key} package: {package}")
        file = PROJECT.parent / "Content" / (package.removeprefix("/Game/") + ".umap")
        if hashlib.sha256(file.read_bytes()).hexdigest() != spec[f"{key}_sha256"]:
            raise RuntimeError(f"{file} changed since the manifest was built")


def active_games():
    cmd = (
        "Get-CimInstance Win32_Process -Filter \"Name LIKE 'UnrealEditor%'\" | "
        "Where-Object { $_.CommandLine -match '(?i)(^|\\s)-game(\\s|$)' } | "
        "ForEach-Object { $_.ProcessId }"
    )
    result = subprocess.run(["powershell", "-NoProfile", "-Command", cmd],
                            capture_output=True, text=True, timeout=12, check=True)
    return [int(line.strip()) for line in result.stdout.splitlines() if line.strip()]


@contextmanager
def game(scene, manifest, report, hd):
    port = PORTS[scene]
    occupied = active_games()
    if occupied:
        raise RuntimeError(f"An independent Unreal -game already exists: PID {occupied}")
    validate_ports_available(HOST, [port])
    width, height = (640, 360) if hd else (320, 180)
    captures = [airsim.ImageType.DepthPlanar]
    if hd:
        captures.append(airsim.ImageType.Scene)
    settings = {
        "SettingsVersion": 2.0, "SimMode": "Multirotor", "ViewMode": "Fpv" if hd else "NoDisplay",
        "ClockSpeed": 1.0, "LocalHostIp": HOST, "ApiServerPort": port,
        "Vehicles": {VEHICLE: {"VehicleType": "SimpleFlight", "AutoCreate": True,
                               "X": 0, "Y": 0, "Z": 0, "Yaw": 0,
                               "EnableCollisionPassthrough": False}},
        "CameraDefaults": {"CaptureSettings": [
            {"ImageType": kind, "Width": width, "Height": height, "FOV_Degrees": 90,
             **({"MotionBlurAmount": 0.0} if kind == airsim.ImageType.Scene else {})}
            for kind in captures
        ]},
    }
    settings_path = report / "settings.json"
    write_json(settings_path, settings)
    log_path = (report / "unreal.log").resolve()
    window_size = (width, height) if hd else (320, 240)
    cmd = [
        str(UNREAL), str(PROJECT), manifest["map"], "-game", "-windowed",
        f"-ResX={window_size[0]}", f"-ResY={window_size[1]}", "-NoSound", "-unattended", "-nosplash", "-NoVSync",
        f"-settings={settings_path.resolve()}", f"-abslog={log_path}",
    ]
    if hd:
        cmd.append("-ExecCmds=r.SkyAtmosphere.EditorNotifications 0,r.ScreenPercentage 100")
    process = subprocess.Popen(cmd, cwd=UNREAL.parent)
    started = time.monotonic()
    print(f"{scene} {'640x360 verification' if hd else '320x180 evaluation'} PID={process.pid} port={port}",
          flush=True)
    try:
        last_failure = None
        while time.monotonic() - started < 180:
            if process.poll() is not None:
                raise RuntimeError(f"Unreal exited with code {process.returncode}. See {log_path}")
            try:
                with socket.create_connection((HOST, port), timeout=1):
                    probe = airsim.MultirotorClient(ip=HOST, port=port, timeout_value=5)
                    if probe.ping() and VEHICLE in probe.listVehicles():
                        break
            except Exception as error:
                last_failure = str(error)
            time.sleep(.5)
        else:
            raise TimeoutError(f"AirSim RPC did not start on {port}: {last_failure}")
        client = airsim.MultirotorClient(ip=HOST, port=port, timeout_value=15)
        yield client, round(time.monotonic() - started, 3), settings, (width, height)
    finally:
        if process.poll() is None:
            result = subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"],
                                    capture_output=True, timeout=20, text=True)
            if result.returncode and process.poll() is None:
                raise RuntimeError(f"Could not terminate benchmark PID {process.pid}: {result.stderr}")
        process.wait(timeout=25)
        print(f"{scene} standalone PID={process.pid} ended", flush=True)


def depth_response(response):
    depth = np.asarray(response.image_data_float, dtype=np.float32)
    if response.height <= 0 or response.width <= 0 or depth.size != response.width * response.height:
        raise ValueError(f"Invalid DepthPlanar response {response.width}x{response.height}: {depth.size} floats")
    return depth.reshape(response.height, response.width)


def depth_stats(depth):
    finite = np.isfinite(depth)
    valid = depth[finite & (depth > 0) & (depth <= MAX_DEPTH)]
    quantiles = np.percentile(valid, [0, 50, 100]).tolist() if len(valid) else [None] * 3
    quantized = np.floor(valid * (255 / MAX_DEPTH)).astype(np.uint8)
    return {
        "finite_ratio": float(finite.mean()), "in_range_ratio": len(valid) / depth.size,
        "distinct_valid_depth_levels": len(np.unique(quantized)),
        "min_m": quantiles[0], "median_m": quantiles[1], "max_m": quantiles[2],
    }


def fetch_depth(client):
    (response,) = client.simGetImages(
        [airsim.ImageRequest("0", airsim.ImageType.DepthPlanar, True, False)], VEHICLE
    )
    return depth_response(response)


def depth_ready(client, size, timeout=55):
    end = time.monotonic() + timeout
    stats = None
    while time.monotonic() < end:
        try:
            depth = fetch_depth(client)
            stats = depth_stats(depth)
            if (depth.shape == (size[1], size[0]) and stats["finite_ratio"] == 1.0
                    and stats["in_range_ratio"] >= .05 and stats["distinct_valid_depth_levels"] >= 16):
                return stats
        except ValueError:
            pass
        time.sleep(.35)
    raise RuntimeError(f"AirSim depth did not become ready at {size}: {stats}")


def vector(value):
    return np.array([value.x_val, value.y_val, value.z_val], dtype=float)


def position(client):
    return vector(client.simGetVehiclePose(VEHICLE).position)


def collision(client):
    hit = client.simGetCollisionInfo(VEHICLE)
    return {
        "has_collided": bool(hit.has_collided), "object_name": str(hit.object_name),
        "impact_ned_m": vector(hit.impact_point).tolist(), "timestamp": int(hit.time_stamp),
    }


def pose_at(client, point, yaw_deg):
    state = airsim.KinematicsState()
    state.position = airsim.Vector3r(*point)
    half = math.radians(yaw_deg) / 2
    state.orientation = airsim.Quaternionr(0, 0, math.sin(half), math.cos(half))
    state.linear_velocity = state.angular_velocity = airsim.Vector3r()
    state.linear_acceleration = state.angular_acceleration = airsim.Vector3r()
    client.simSetKinematics(state, True, VEHICLE)


def command_velocity(client, velocity, duration, yaw):
    return client.moveByVelocityAsync(
        *velocity, duration, drivetrain=airsim.DrivetrainType.MaxDegreeOfFreedom,
        yaw_mode=airsim.YawMode(False, yaw), vehicle_name=VEHICLE,
    ).get()


def fly(client, path, deadline_s, size, sample_depth):
    """Track the route through velocity commands. Teleport only during reset."""
    path = np.asarray(path, dtype=float)
    client.reset()
    client.enableApiControl(True, VEHICLE)
    if not client.armDisarm(True, VEHICLE):
        raise RuntimeError("AirSim failed to arm for physical verification")
    heading = math.degrees(math.atan2(*(path[1, :2] - path[0, :2])[::-1]))
    pose_at(client, path[0], heading)
    command_velocity(client, (0, 0, 0), 1.0, heading)
    if sample_depth:
        depth_ready(client, size)
    initial = position(client)
    hit = collision(client)
    if hit["has_collided"] or float(np.linalg.norm(initial - path[0])) > .25:
        raise RuntimeError(f"Invalid physical flight reset at {initial.tolist()}: {hit}")
    started = time.monotonic()
    samples = []
    depth_min, depth_count = 1.0, 0
    step, best = 0, 0
    distance_to_goal = math.inf
    while time.monotonic() - started < deadline_s:
        loc = position(client)
        hit = collision(client)
        end = min(len(path), best + 15)
        closest = best + int(np.argmin(np.linalg.norm(path[best:end] - loc, axis=1)))
        best = max(best, closest)
        distance_to_goal = float(np.linalg.norm(loc - path[-1]))
        if step % 3 == 0 or hit["has_collided"] or distance_to_goal < .45:
            row = {"elapsed_s": round(time.monotonic() - started, 3),
                   "ned_m": loc.tolist(), "route_index": best}
            if sample_depth:
                info = depth_stats(fetch_depth(client))
                depth_min = min(depth_min, info["finite_ratio"])
                depth_count += 1
                row["depth"] = info
            samples.append(row)
        if hit["has_collided"] or distance_to_goal < .45:
            break
        ahead = min(len(path) - 1, best + 4)
        error = path[ahead] - loc
        horizontal_error = float(np.linalg.norm(error[:2]))
        velocity = np.zeros(3)
        if horizontal_error > .001:
            velocity[:2] = error[:2] * min(1.32, 2.24 / horizontal_error)
            heading_diff = path[ahead, :2] - path[best, :2]
            if np.linalg.norm(heading_diff) > .01:
                heading = math.degrees(math.atan2(heading_diff[1], heading_diff[0]))
        velocity[2] = np.clip(1.37 * error[2], -.65, .65)
        if not command_velocity(client, velocity, .29, heading):
            raise RuntimeError("AirSim rejected the verification velocity command")
        step += 1
        if step % 100 == 0:
            print(f"Verification physical flight: node {best}/{len(path)-1} "
                  f"remaining={distance_to_goal:.1f}m", flush=True)
    return {
        "reached": distance_to_goal < .45 and not hit["has_collided"],
        "remaining_goal_m": distance_to_goal, "collision": hit,
        "depth_finite_ratio_min": depth_min if depth_count else None,
        "depth_frames": depth_count, "trajectory": samples,
    }


def verify_motion(client, dynamic):
    if not dynamic:
        return []
    # Record a complete cycle plus a six-second window. For each actor, retain
    # four consecutive 2 s samples in a monotone part of that same observation.
    start = time.monotonic()
    count = math.ceil((max(item["period_s"] for item in dynamic) + 6) * 2) + 1
    samples = {item["rpc_tag"]: [] for item in dynamic}
    for i in range(count):
        time.sleep(max(0, start + i / 2 - time.monotonic()))
        for tag, rows in samples.items():
            p = vector(client.simGetObjectPose(tag).position)
            if not np.isfinite(p).all():
                raise RuntimeError(f"Dynamic actor tag missing in standalone: {tag}")
            rows.append({"elapsed_s": round(time.monotonic() - start, 4), "ned_m": p.tolist()})
    results = []
    for item in dynamic:
        tag = item["rpc_tag"]
        positions = np.array([row["ned_m"] for row in samples[tag]])
        axis = np.array(item["projected_axis_ue"]) * [1, 1, -1]
        axis /= np.linalg.norm(axis)
        track = positions @ axis
        lateral_axis = np.array(item["lateral_axis_ue"]) * [1, 1, -1]
        lateral = float(np.ptp(positions @ lateral_axis))
        span = float(np.ptp(track))
        index = max(range(len(track) - 12),
                    key=lambda k: np.ptp(track[k:k+13:4]))
        window = samples[tag][index:index+13:4]
        shift = float(np.ptp(track[index:index+13:4]))
        delta = np.diff(track)
        # Both directions must be observable over the full runtime sample.
        reversed_direction = bool(np.any(delta > .06) and np.any(delta < -.06))
        minimum = item["require_m"]
        lateral_minimum = item["require_lateral_m"]
        elapsed = np.array([row["elapsed_s"] for row in samples[tag]])
        intervals = np.diff(elapsed)
        # After an RPC stall the scheduler can take several catch-up samples
        # together. Those timestamps cannot estimate speed.
        distinct = intervals > .05
        speeds = np.linalg.norm(np.diff(positions, axis=0)[distinct], axis=1) / intervals[distinct]
        results.append({
            "tag": tag, "required_displacement_m": minimum,
            "required_lateral_displacement_m": lateral_minimum,
            "observed_lateral_span_m": lateral,
            "median_speed_mps": float(np.median(speeds)) if len(speeds) else None,
            "design_speed_mps": item["speed_mps"],
            "six_second_axis_displacement_m": shift,
            "total_observed_axis_span_m": span,
            "both_directions_observed": reversed_direction,
            "samples_at_0_2_4_6_s": [
                {**row, "nominal_t_s": i * 2, "elapsed_from_t0_s": round(
                    row["elapsed_s"] - window[0]["elapsed_s"], 4
                )} for i, row in enumerate(window)
            ],
            "full_sampling": samples[tag],
            "passed": bool(len(speeds)) and span >= minimum and lateral >= lateral_minimum and reversed_direction,
        })
    return results


def png(path, frame):
    if not cv2.imwrite(str(path), frame):
        raise RuntimeError(f"Could not write {path}")


def viewport_still(client, path, width, height):
    pending = path.with_name(f"{path.stem}_{time.time_ns()}.png")
    if not client.simRunConsoleCommand(
        f'HighResShot {width}x{height} filename="{pending.resolve().as_posix()}"'
    ):
        raise RuntimeError(f"Unreal rejected the screenshot command for {path}")
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        if pending.is_file():
            frame = cv2.imdecode(np.frombuffer(pending.read_bytes(), dtype=np.uint8), cv2.IMREAD_COLOR)
            if frame is not None and frame.shape[:2] == (height, width):
                pending.replace(path)
                return frame
        time.sleep(.15)
    raise TimeoutError(f"Unreal viewport screenshot was not saved at {width}x{height}: {path}")


def make_stills(client, scene, spec, report, size):
    rows, stats = [], []
    width, height = size
    client.reset()
    client.enableApiControl(True, VEHICLE)
    if not client.armDisarm(True, VEHICLE):
        raise RuntimeError("AirSim failed to arm for image positioning")
    if not client.simRunConsoleCommand("r.SkyAtmosphere.EditorNotifications 0"):
        raise RuntimeError("Could not disable the Unreal editor's sky diagnostics")
    # TSR retained a border from a previous view after abrupt camera changes.
    # TAA lets the FPV viewport settle while preserving the level's lighting.
    client.simRunConsoleCommand("viewmode lit")
    client.simRunConsoleCommand("r.AntiAliasingMethod 2")
    client.simRunConsoleCommand("r.MotionBlurQuality 0")
    for i, view in enumerate(spec["stills"], 1):
        pose_at(client, view["ned_m"], view["yaw_deg"])
        command_velocity(client, (0, 0, 0), 1.15, view["yaw_deg"])
        rgb = viewport_still(client, report / f"{i:02d}_rgb.png", width, height)
        depth = fetch_depth(client)
        if depth.shape != (height, width):
            raise RuntimeError(f"Expected {width}x{height} RGB and depth for view {i}")
        depth_record = depth_stats(depth)
        grey = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
        stats.append({
            "station_m": view["station_m"], "rgb_mean": float(grey.mean()),
            "rgb_p90": float(np.percentile(grey, 90)),
            "rgb_white_clip_ratio": float(np.all(rgb >= 249, axis=2).mean()),
            "actual_position_ned_m": position(client).tolist(), **depth_record,
        })
        safe_depth = np.nan_to_num(depth, posinf=MAX_DEPTH, neginf=0)
        normalized = np.rint(np.clip(safe_depth / MAX_DEPTH, 0, 1) * 255).astype(np.uint8)
        depth_color = cv2.applyColorMap(normalized, cv2.COLORMAP_TURBO)
        png(report / f"{i:02d}_depth.png", depth_color)
        caption = np.full((30, 2*width, 3), (24, 25, 26), dtype=np.uint8)
        cv2.putText(caption, f"{scene.upper()}   {view['station_m']:5.1f} m   RGB",
                    (12, 21), cv2.FONT_HERSHEY_SIMPLEX, .55, (230, 233, 235), 1, cv2.LINE_AA)
        cv2.putText(caption, "DEPTH PLANAR   0 - 25 m", (width+12, 21),
                    cv2.FONT_HERSHEY_SIMPLEX, .55, (230, 233, 235), 1, cv2.LINE_AA)
        rows.extend((caption, np.hstack((rgb, depth_color))))
    png(report.parent / f"{scene}_overview.png", np.vstack(rows))
    return stats


def stills_pass(stills):
    return 4 <= len(stills) <= 6 and all(
        5 < item["rgb_mean"] < 220 and item["rgb_white_clip_ratio"] < .15
        and item["finite_ratio"] == 1.0 for item in stills
    )


def widths_pass(scene, spec):
    ranges = {"t1": (2.5, 3.5), "t2": (2.5, math.inf), "t3": (2.1, 3.0)}
    low, high = ranges[scene]
    checks = []
    for row in spec["static_openings"]:
        width = row["effective_width_m"]
        hint = row["label"]
        passed = width is not None and low <= width <= high
        if "Gantry" in hint or "HangingServiceFrame" in hint:
            vertical = row["vertical_opening_m"]
            roof = (row["ceiling"] or {}).get("label", "")
            passed &= vertical is not None and vertical >= 1.6 and roof.endswith("_Crossbeam")
        checks.append({"label": hint, "width_m": width, "passed": passed})
    for detail in spec["dynamic_openings"]:
        width = detail["min_width_m"]
        dynamic_low = 2.4 if scene == "t2" else low
        checks.append({"label": detail["tag"], "width_m": width, "passed": dynamic_low <= width <= high})
    return checks


def verify(scene, spec, report):
    out = report / scene
    results = {}
    with game(scene, spec, out, hd=True) as (client, boot_s, settings, size):
        ready = depth_ready(client, size)
        anchor = vector(client.simGetObjectPose("Ch4BenchOrigin").position)
        if not np.isfinite(anchor).all() or np.linalg.norm(anchor) > .10:
            raise RuntimeError(f"AirSim NED frame did not match the manifest origin: {anchor}")
        results.update(boot_s=boot_s, map=spec["map"], map_sha256=spec["map_sha256"],
                       source=spec["source"], source_sha256=spec["source_sha256"],
                       clock_speed=settings["ClockSpeed"],
                       depth_ready=ready, origin_ned_m=anchor.tolist())
        results["motion"] = verify_motion(client, spec["dynamic"])
        print(f"{scene} motion spans: " + str([
            round(item["six_second_axis_displacement_m"], 2) for item in results["motion"]
        ]), flush=True)
        results["stills"] = make_stills(client, scene, spec, out, size)
        results["widths"] = widths_pass(scene, spec)
        write_json(out / "verification.json", results)
        # The RPC/render loop can run slower than simulation time when the
        # desktop editor is also open. The wall-clock limit is a watchdog.
        results["detour"] = fly(client, spec["detour_ned_m"], 2 * spec["route_length_m"] + 45, size, True)
        write_json(out / "verification.json", results)
        results["centerline"] = fly(client, spec["centerline_ned_m"], spec["route_length_m"] + 40,
                                    size, False)
        blocked = results["centerline"]["collision"]
        obstacles = {item["actor_name"] for item in spec["obstacles"]}
        obstacle_hit = blocked["has_collided"] and (
            blocked["object_name"] in obstacles
            or blocked["object_name"].startswith(COLLISION_TOKENS)
        )
        results["gates"] = {
            "physical_detour": results["detour"]["reached"]
            and results["detour"]["depth_finite_ratio_min"] == 1.0,
            "centerline_blocked_by_benchmark_obstacle": obstacle_hit,
            "runtime_motion": len(results["motion"]) == {"t1": 0, "t2": 3, "t3": 1}[scene]
            and all(item["passed"] for item in results["motion"]),
            "rgb_depth_stills": stills_pass(results["stills"]),
            "openings": bool(results["widths"]) and all(item["passed"] for item in results["widths"]),
        }
        results["passed"] = all(results["gates"].values())
        write_json(out / "verification.json", results)
        print(f"{scene} gates: {results['gates']}", flush=True)
    return results


def compare_images(report):
    sources = [report / f"{key}_overview.png" for key in CONFIGS]
    if all(p.is_file() for p in sources):
        panels = [cv2.imread(str(p)) for p in sources]
        height = max(panel.shape[0] for panel in panels)
        panels = [cv2.copyMakeBorder(panel, 0, height - panel.shape[0], 0, 0,
                                     cv2.BORDER_CONSTANT, value=(24, 25, 26)) for panel in panels]
        png(report / "all_3_scenes_comparison.png", np.hstack(panels))


def sync_config(scene, spec):
    filename = ROOT / "configs" / CONFIGS[scene]
    text = filename.read_text(encoding="utf-8-sig")
    updates = {
        "start_position": spec["start_ned_m"], "goal_position": spec["goal_ned_m"],
        "work_space_x": spec["workspace_x"], "work_space_y": spec["workspace_y"],
        "work_space_z": spec["workspace_z_enu"],
    }
    text, map_count = re.subn(r"^# UE Map: .+$", f"# UE Map: {spec['map']}", text, count=1, flags=re.M)
    if map_count != 1:
        raise RuntimeError(f"Missing UE Map header in {filename}")
    for name, value in updates.items():
        text, count = re.subn(rf"^{name}\s*=.*$", f"{name} = {json.dumps(value)}",
                              text, count=1, flags=re.M)
        if count != 1:
            raise RuntimeError(f"Missing {name} in {filename}")
    filename.write_text(text, encoding="utf-8")
    return filename


def evaluation_child(telemetry_path, arguments):
    """Run the normal evaluator and record contacts before DummyVecEnv resets."""
    from scripts.evaluation import evaluate_single as evaluator

    build_runtime_env = evaluator.build_runtime_env

    def with_terminal_telemetry(config):
        if config.getfloat("perception", "virtual_ceiling_prob") != 0:
            raise ValueError("Virtual ceilings must be disabled in the effective config")
        wrapped = build_runtime_env(config)
        base = evaluator.get_base_env(wrapped)
        step = base.step

        def record(action):
            observation, reward, terminated, truncated, info = step(action)
            if terminated or truncated:
                row = {
                    "position_enu_m": [float(v) for v in info["terminal_position"]],
                    "success": bool(info["is_success"]),
                    "out_of_bounds": bool(info["is_not_in_workspace"]),
                    "collision": collision(base.client) if info["is_crash"] else None,
                }
                with telemetry_path.open("a", encoding="utf-8") as log:
                    log.write(json.dumps(row, allow_nan=False) + "\n")
            return observation, reward, terminated, truncated, info

        base.step = record
        return wrapped

    evaluator.build_runtime_env = with_terminal_telemetry
    sys.argv = [str(ROOT / "scripts/evaluation/evaluate_single.py"), *arguments]
    evaluator.main()


def evaluate(scene, spec, report, model_run, episodes):
    scene_config = ROOT / "configs" / CONFIGS[scene]
    checkpoint = model_run / "models/best_model.zip"
    # The MambaSAC checkpoint is a torch.save archive. Release the checker
    # process before starting Unreal so it does not occupy simulation memory.
    metadata = subprocess.check_output([
        sys.executable, "-c",
        "import sys, torch; "
        "print(torch.load(sys.argv[1], map_location='cpu', weights_only=False)['num_timesteps'])",
        str(checkpoint),
    ], cwd=ROOT, text=True, timeout=45)
    step = int(metadata.strip())
    if step != 495000:
        raise ValueError(f"Expected best@495k, found {step} steps in {checkpoint}")
    config = read_ini(scene_config)
    frozen = read_ini(model_run / "config/config.ini")
    if (config.getfloat("perception", "virtual_ceiling_prob") != 0
            or config.getfloat("environment", "max_vertical_difference") != 2
            or config.getfloat("multirotor", "v_z_max") != .8
            or frozen.getfloat("multirotor", "v_z_max") != .8):
        raise ValueError(f"Unexpected flight limits or virtual ceilings in {scene_config}")
    for name, expected in (("start_position", "start_ned_m"), ("goal_position", "goal_ned_m"),
                           ("work_space_x", "workspace_x"), ("work_space_y", "workspace_y"),
                           ("work_space_z", "workspace_z_enu")):
        if json.loads(config.get("environment", name)) != spec[expected]:
            raise ValueError(f"{scene_config} {name} is out of sync with the benchmark")
    out = report / "eval_stage2_495k" / scene
    existing_metrics = set(out.rglob("summary_metrics.json"))
    with game(scene, spec, out, hd=False) as (client, boot_s, settings, size):
        ready = depth_ready(client, size)
        print(f"{scene} depth ready: {ready}", flush=True)
        telemetry = out / "terminal_telemetry.jsonl"
        telemetry.write_text("", encoding="utf-8")
        command = [
            sys.executable, "-u", "-m", "tools.ue_benchmark.verify_ch4_benchmark_maps",
            "--evaluation-child", str(telemetry.resolve()),
            "--model_path", str(model_run), "--episodes", str(episodes), "--best",
            "--scene-config", str(scene_config), "--airsim-ip", HOST, "--airsim-port", str(PORTS[scene]),
            "--scenario-seed", "2026", "--yaw-range-deg", "10.0",
            "--save", "--output-root", str(out.resolve()), "--save-tag", scene,
        ]
        # The repository's evaluator runs deterministic policy inference with its
        # full history. Both streams are captured as the episodes finish.
        with (out / "evaluation.log").open("w", encoding="utf-8") as log:
            with subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE,
                                  stderr=subprocess.STDOUT, text=True, encoding="utf-8",
                                  errors="replace") as process:
                for line in process.stdout:
                    log.write(line)
                    log.flush()
                    if line.startswith(("Ep ", "History:", "Results saved", "Traceback", "RuntimeError")):
                        print(f"{scene} {line.rstrip()}", flush=True)
                if process.wait() != 0:
                    raise RuntimeError(f"The evaluation failed. See {out / 'evaluation.log'}")
        new_metrics = set(out.rglob("summary_metrics.json")) - existing_metrics
        if len(new_metrics) != 1:
            raise RuntimeError(f"Expected one new evaluation summary under {out}, found {len(new_metrics)}")
        (metrics_path,) = new_metrics
        summary = read_json(metrics_path)
        evaluation_protocol = read_json(metrics_path.with_name("evaluation_manifest.json"))
        if not evaluation_protocol["use_history"] or summary["episodes"] != episodes:
            raise RuntimeError("Evaluation history or episode count did not match the requested protocol")
        terminal = [json.loads(line) for line in telemetry.read_text(encoding="utf-8").splitlines()]
        if len(terminal) != episodes:
            raise RuntimeError(f"Expected {episodes} terminal telemetry rows, found {len(terminal)}")
        visible = spec.get("visible_scene_actors", {})
        names = []
        unidentified = []
        floor, ceiling = spec["workspace_z_enu"]
        bounds = {"below_floor": 0, "above_ceiling": 0, "xy_only": 0}
        for row in terminal:
            if row["collision"]:
                name = row["collision"]["object_name"]
                row["collision"]["editor_label"] = visible.get(name)
                names.append(visible.get(name, name))
                if name not in visible:
                    unidentified.append(name)
            if row["out_of_bounds"]:
                z = row["position_enu_m"][2]
                key = "below_floor" if z < floor else "above_ceiling" if z > ceiling else "xy_only"
                bounds[key] += 1
        target_low, target_high = RANGES[scene]
        data = {
            "map": spec["map"], "map_sha256": spec["map_sha256"],
            "source": spec["source"], "source_sha256": spec["source_sha256"],
            "checkpoint": str(checkpoint), "model_step": step,
            "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            "clock_speed": settings["ClockSpeed"], "seed": 2026, "yaw_range_deg": 10.0,
            "use_history": True, "virtual_ceiling_prob": 0.0, "depth_ready": ready,
            "boot_s": boot_s, "target_success_pct": [target_low, target_high],
            "target_met": target_low <= summary["success_rate_pct"] <= target_high,
            "summary": summary, "terminal_events": terminal,
            "oob_breakdown": bounds, "collision_objects": names,
            "unmatched_collision_objects": unidentified,
            "artifacts": str(metrics_path.parent),
        }
        write_json(out / "result.json", data)
    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", choices=CONFIGS, action="append", help="May be supplied repeatedly.")
    parser.add_argument("--report-dir", type=Path, default=REPORT)
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--sync-configs", action="store_true")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--model-run", type=Path, default=MODEL_RUN)
    args = parser.parse_args()
    scenes = args.scene or list(CONFIGS)
    manifest = read_json(args.report_dir / "benchmark_manifest.json")
    failures = []
    run_verification = args.verify or not (args.evaluate or args.sync_configs)
    run_evaluation = args.evaluate or not (args.verify or args.sync_configs)
    aggregate = args.report_dir / "eval_stage2_495k/overall_summary.json"
    for scene in scenes:
        try:
            check_map_identity(manifest[scene])
            if run_verification and not verify(scene, manifest[scene], args.report_dir)["passed"]:
                failures.append(scene)
                continue
            if args.sync_configs:
                sync_config(scene, manifest[scene])
            if run_evaluation:
                data = read_json(aggregate) if aggregate.is_file() else {}
                data[scene] = evaluate(scene, manifest[scene], args.report_dir, args.model_run, args.episodes)
                write_json(aggregate, data)
        except Exception:
            traceback.print_exc()
            failures.append(scene)
    if run_verification:
        compare_images(args.report_dir)
    if failures:
        raise SystemExit(f"Unsuccessful scenes: {', '.join(failures)}")


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == "--evaluation-child":
        evaluation_child(Path(sys.argv[2]), sys.argv[3:])
    else:
        main()
