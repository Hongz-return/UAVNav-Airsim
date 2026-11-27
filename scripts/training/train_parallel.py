#!/usr/bin/env python3
"""Run independent single-environment training jobs on separate AirSim ports.

Default schedule: method-major job order, up to 3 parallel envs.
With seeds 0 1 2 and max-parallel 3 each wave is one algorithm's three seeds
(few stragglers). With a single seed, waves pack different methods in parallel.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import subprocess
import sys
from configparser import ConfigParser
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.airsim_instances import auto_pin_p_core, launch_instances, terminate_processes
from tools.experiment_paths import PARALLEL_RUNS_ROOT, TRAIN_RUNS_ROOT


METHOD_CONFIGS = {
    "sac": Path("configs/config_sac_framestack4.ini"),
    "coordatt": Path("configs/config_coordatt_sac.ini"),
    "lstm": Path("configs/config_lstm.ini"),
    "mamba": Path("configs/config_mamba.ini"),
    "ma": Path("configs/config_mamba_coordatt.ini"),
    "transformer": Path("configs/config_transformer.ini"),
    "ma_motion": Path("configs/config_ma_motion.ini"),
    "ma_aux": Path("configs/config_ma_aux.ini"),
    "ma_motion_aux": Path("configs/config_ma_motion_aux.ini"),
}

SCENES = {
    "tunnel": {
        "env_name": "Tunnel",
        "map_path": "/Game/Tunnel_Col_Vol1/CyTunnel",
        "timesteps": 80_000,
    },
    "tunnel_obs": {
        "env_name": "Obs",
        "map_path": "/Game/Tunnel_Col_Vol1/Maps/Tunnel",
        # The completed ENV2 matrix used 150,000 steps; see RETRAINING.md.
        "timesteps": 150_000,
    },
    "tunnel_activate": {
        "env_name": "Obs",
        "map_path": "/Game/Tunnel_Col_Vol1/Maps/Tunnel_Activate",
        "timesteps": 200_000,
    },
    "tunnel_mixed": {
        "env_name": "Obs",
        "map_path": "/Game/Tunnel_Col_Vol1/Maps/Tunnel_AO",
        "timesteps": 200_000,
    },
    "rail_single_dynamic": {
        "env_name": "Obs",
        "map_path": "/Game/RailwayTunnelPrototype/Maps/02_Conditions/SingleTrack/L_SingleTrack_SCurve_DynamicMaintenance",
        "timesteps": 200_000,
        # Local NED coordinates calibrated with the same -1.5 m vehicle spawn setting.
        "goal_position": [108.029625, 24.986841, -1.5],
        "requires_existing_airsim": True,
    },
}

DEFAULT_METHODS = ["sac", "coordatt", "lstm", "mamba", "ma", "transformer"]
DEFAULT_SEEDS = [0, 1, 2]
DEFAULT_MAX_PARALLEL = 3


@dataclass(frozen=True)
class Job:
    method: str
    seed: int
    port: int
    source_config: Path
    runtime_config: Path
    log_root: Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_manifest() -> dict[str, str]:
    """Hash executable experiment sources without copying the working tree."""
    roots = [ROOT / name for name in ("configs", "gym_env", "scripts", "tools")]
    files = [path for root in roots for path in root.rglob("*") if path.is_file()]
    files.extend(path for path in (ROOT / "setup.py", ROOT / "pixi.toml") if path.exists())
    return {
        path.relative_to(ROOT).as_posix(): sha256(path)
        for path in sorted(files)
        if "__pycache__" not in path.parts and path.suffix != ".pyc"
    }


def git_state() -> dict:
    """Record the base commit and dirty-state digest for auditability."""
    def run(*args):
        try:
            result = subprocess.run(
                ["git", *args],
                cwd=ROOT,
                capture_output=True,
                text=True,
                check=False,
            )
        except FileNotFoundError:
            return None
        return result.stdout.strip() if result.returncode == 0 else None

    status = run("status", "--porcelain=v1") or ""
    return {
        "commit": run("rev-parse", "HEAD"),
        "dirty": bool(status),
        "status_sha256": hashlib.sha256(status.encode("utf-8")).hexdigest(),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Launch independent one-env training jobs on separate AirSim instances. "
            "Default: 3 parallel envs, method-major order (3 seeds of one algorithm "
            "per wave; single-seed still packs methods in parallel)."
        )
    )
    parser.add_argument("--scene", choices=SCENES, required=True)
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=METHOD_CONFIGS,
        default=list(DEFAULT_METHODS),
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=list(DEFAULT_SEEDS),
        help="Random seeds (default: 0 1 2). Listed method-major before batching.",
    )
    parser.add_argument("--ports", nargs="+", type=int)
    parser.add_argument(
        "--max-parallel",
        type=int,
        default=DEFAULT_MAX_PARALLEL,
        help="Max concurrent AirSim/train processes (default: 3).",
    )
    parser.add_argument("--timesteps", type=int)
    parser.add_argument("--log-root", type=Path, default=TRAIN_RUNS_ROOT)
    parser.add_argument("--ue-exe", type=Path, default=Path(r"D:\UEBuilds\TunnelEnv\Windows\Tunnel.exe"))
    parser.add_argument("--map-path")
    parser.add_argument(
        "--settings-template",
        type=Path,
        default=Path(r"C:\Users\31732\Documents\AirSim\settings.json"),
    )
    parser.add_argument("--airsim-ip", default="127.0.0.1")
    parser.add_argument("--clock-speed", type=float, default=1.0)
    parser.add_argument("--res-x", type=int, default=640)
    parser.add_argument("--res-y", type=int, default=480)
    parser.add_argument("--startup-wait", type=int, default=60)
    parser.add_argument("--no-launch-airsim", action="store_true")
    parser.add_argument("--keep-airsim", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def default_ports(count: int) -> list[int]:
    return [41451 + 10 * index for index in range(count)]


def read_start_position(config_path: Path) -> list[float]:
    cfg = ConfigParser()
    cfg.read(config_path, encoding="utf-8")
    value = ast.literal_eval(cfg.get("environment", "start_position"))
    if not isinstance(value, list) or len(value) != 3:
        raise ValueError(f"Invalid start_position in {config_path}: {value!r}")
    return [float(item) for item in value]


def write_runtime_config(
    source: Path,
    target: Path,
    *,
    env_name: str,
    airsim_ip: str,
    port: int,
    goal_position: list[float] | None = None,
) -> None:
    cfg = ConfigParser()
    cfg.read(source, encoding="utf-8")
    if not cfg.has_section("options"):
        raise ValueError(f"Invalid config: {source}")
    cfg.set("options", "env_name", env_name)
    if goal_position is not None:
        cfg.set("environment", "goal_position", repr(goal_position))
    if not cfg.has_section("airsim"):
        cfg.add_section("airsim")
    cfg.set("airsim", "ip", airsim_ip)
    cfg.set("airsim", "rpc_port", str(port))
    cfg.set("airsim", "timeout_value", "3600")
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as stream:
        cfg.write(stream)


def build_batches(jobs: list[Job], max_parallel: int) -> list[list[Job]]:
    """Slice method-major jobs into waves of at most max_parallel.

    Jobs must already be ordered method-major (all seeds of A, then B, ...).
    Then:
    - seeds 0 1 2 + max_parallel 3 → each wave is one method's three seeds
    - seeds 0 only + max_parallel 3 → each wave packs up to three methods
    """
    if max_parallel < 1:
        raise ValueError("--max-parallel must be at least 1")
    return [jobs[offset : offset + max_parallel] for offset in range(0, len(jobs), max_parallel)]


def build_jobs(args: argparse.Namespace, run_id: str) -> list[Job]:
    if len(set(args.methods)) != len(args.methods):
        raise ValueError(f"Methods must be unique: {args.methods}")
    if len(set(args.seeds)) != len(args.seeds):
        raise ValueError(f"Seeds must be unique: {args.seeds}")
    if args.max_parallel < 1:
        raise ValueError("--max-parallel must be at least 1")

    # Method-major: keeps same-method seeds adjacent for clean waves when
    # len(seeds) == max_parallel (the default 3×3 case).
    combinations = [(method, seed) for method in args.methods for seed in args.seeds]
    if not combinations:
        raise ValueError("Need at least one method and one seed")

    parallelism = min(args.max_parallel, len(combinations))
    ports = args.ports or default_ports(parallelism)
    if len(ports) < parallelism:
        raise ValueError(f"Need {parallelism} ports, got {ports}")
    ports = ports[:parallelism]
    if len(set(ports)) != len(ports):
        raise ValueError(f"Ports must be unique: {ports}")

    run_dir = PARALLEL_RUNS_ROOT / run_id
    jobs = []
    for index, (method, seed) in enumerate(combinations):
        source = ROOT / METHOD_CONFIGS[method]
        port = ports[index % parallelism]
        jobs.append(
            Job(
                method=method,
                seed=seed,
                port=port,
                source_config=source,
                runtime_config=ROOT / run_dir / "configs" / f"{method}_seed_{seed}_port_{port}.ini",
                log_root=ROOT / args.log_root / run_id / args.scene / method / f"seed_{seed}",
            )
        )
    return jobs


def build_train_command(job: Job, timesteps: int) -> list[str]:
    return [
        sys.executable,
        "-m",
        "scripts.training.train_single",
        "-c",
        str(job.runtime_config),
        "--timesteps",
        str(timesteps),
        "--seed",
        str(job.seed),
        "--logdir",
        str(job.log_root),
    ]


def printable(command: list[str]) -> str:
    return " ".join(f'"{part}"' if " " in part else part for part in command)


def print_plan(jobs: list[Job], batches: list[list[Job]], active_ports: list[int], timesteps: int, scene: str, run_id: str) -> None:
    print(f"Run: {run_id}")
    print(f"Scene: {scene}; timesteps/job: {timesteps}; max-parallel: {len(active_ports)}")
    print(f"Schedule: method_major ({len(batches)} waves); AirSim ports: {active_ports}")
    for wave_index, batch in enumerate(batches, start=1):
        methods = list(dict.fromkeys(job.method for job in batch))
        label = methods[0] if len(methods) == 1 else "+".join(methods)
        print(f"  wave {wave_index}/{len(batches)} [{label}]:")
        for job in batch:
            print(f"    {job.method} seed={job.seed} port={job.port}")


def main() -> int:
    args = parse_args()
    run_id = datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%f")
    scene = SCENES[args.scene]
    if scene.get("requires_existing_airsim") and not args.no_launch_airsim:
        raise ValueError(f"{args.scene} requires prestarted UE game instances and --no-launch-airsim")
    timesteps = args.timesteps or scene["timesteps"]
    jobs = build_jobs(args, run_id)
    batches = build_batches(jobs, args.max_parallel)
    active_ports = list(dict.fromkeys(job.port for batch in batches for job in batch))
    map_path = args.map_path or scene["map_path"]

    start_positions = {tuple(read_start_position(job.source_config)) for job in jobs}
    if len(start_positions) != 1:
        raise ValueError(f"All parallel jobs must share one start position: {start_positions}")
    start_position = list(next(iter(start_positions)))

    print_plan(jobs, batches, active_ports, timesteps, args.scene, run_id)

    if args.dry_run:
        for wave_index, batch in enumerate(batches, start=1):
            print(f"# wave {wave_index}")
            for job in batch:
                print(printable(build_train_command(job, timesteps)))
        return 0

    for job in jobs:
        write_runtime_config(
            job.source_config,
            job.runtime_config,
            env_name=scene["env_name"],
            airsim_ip=args.airsim_ip,
            port=job.port,
            goal_position=scene.get("goal_position"),
        )

    run_dir = PARALLEL_RUNS_ROOT / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    source_files = source_manifest()
    source_digest = hashlib.sha256(
        "".join(f"{name}:{digest}\n" for name, digest in source_files.items()).encode("utf-8")
    ).hexdigest()
    manifest = {
        "run_id": run_id,
        "scene": args.scene,
        "map_path": map_path,
        "scene_goal_position": scene.get("goal_position"),
        "timesteps_per_job": timesteps,
        "topology": "independent_single_env_processes",
        "schedule": "method_major",
        "max_parallel": args.max_parallel,
        "waves": len(batches),
        "python_executable": sys.executable,
        "git": git_state(),
        "source_tree_sha256": source_digest,
        "source_files": source_files,
        "batches": [
            [{"method": job.method, "seed": job.seed, "port": job.port} for job in batch]
            for batch in batches
        ],
        "jobs": [
            {
                "method": job.method,
                "seed": job.seed,
                "port": job.port,
                "source_config": str(job.source_config.relative_to(ROOT)),
                "source_config_sha256": sha256(job.source_config),
                "runtime_config": str(job.runtime_config.relative_to(ROOT)),
                "runtime_config_sha256": sha256(job.runtime_config),
                "log_root": str(job.log_root.relative_to(ROOT)),
            }
            for job in jobs
        ],
    }
    (run_dir / "plan.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    airsim_processes = []
    try:
        if not args.no_launch_airsim:
            airsim_processes = launch_instances(
                ue_exe=args.ue_exe,
                map_path=map_path,
                settings_template=args.settings_template,
                output_dir=run_dir / "airsim_settings",
                ports=active_ports,
                start_position=start_position,
                clock_speed=args.clock_speed,
                res_x=args.res_x,
                res_y=args.res_y,
                wait_seconds=args.startup_wait,
            )

        failures = []
        for wave_index, batch in enumerate(batches, start=1):
            methods = list(dict.fromkeys(job.method for job in batch))
            label = methods[0] if len(methods) == 1 else "+".join(methods)
            print(f"[WAVE {wave_index}/{len(batches)}] {label} x{len(batch)}")
            running = []
            for job in batch:
                stdout_path = run_dir / f"{job.method}_seed_{job.seed}_port_{job.port}.out.log"
                stderr_path = run_dir / f"{job.method}_seed_{job.seed}_port_{job.port}.err.log"
                stdout = stdout_path.open("w", encoding="utf-8")
                stderr = stderr_path.open("w", encoding="utf-8")
                process = subprocess.Popen(
                    build_train_command(job, timesteps),
                    cwd=ROOT,
                    stdout=stdout,
                    stderr=stderr,
                    env=os.environ.copy(),
                )
                running.append((job, process, stdout, stderr))
                auto_pin_p_core([process.pid])
                print(f"[START] {job.method} seed={job.seed} port={job.port} pid={process.pid}")
            for job, process, stdout, stderr in running:
                code = process.wait()
                stdout.close()
                stderr.close()
                if code:
                    failures.append(f"{job.method}/seed_{job.seed}: exit {code}")
                print(f"[{'DONE' if code == 0 else 'FAIL'}] {job.method} seed={job.seed}")
            if failures:
                break
        if failures:
            print("\n".join(failures), file=sys.stderr)
            return 1
        return 0
    finally:
        if airsim_processes and not args.keep_airsim:
            terminate_processes(airsim_processes)


if __name__ == "__main__":
    raise SystemExit(main())
