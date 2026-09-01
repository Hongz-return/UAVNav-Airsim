#!/usr/bin/env python3
"""Evaluate every artifact-complete revision run with its saved best model.

The launcher groups runs by scene, starts up to three independent AirSim
instances for that map, and reuses those instances across evaluation waves.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from configparser import ConfigParser
from dataclasses import dataclass, replace
from datetime import datetime
from itertools import groupby
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.training.train_parallel import SCENES, default_ports
from scripts.evaluation.rebuild_summary import reconstruct
from tools.airsim_instances import (
    auto_pin_p_core,
    launch_instances,
    terminate_processes,
)
from tools.experiment_paths import (
    EVALUATION_RESULTS_ROOT,
    FORMAL_EVALUATION_BATCHES_ROOT,
    TRAIN_RUNS_ROOT,
)


DEFAULT_EPISODES = 100
DEFAULT_MAX_PARALLEL = 3


@dataclass(frozen=True)
class EvaluationJob:
    scene: str
    method: str
    seed: int
    run_dir: Path
    target_steps: int
    port: int = 0


def discover_completed_runs(training_root: Path) -> list[EvaluationJob]:
    """Return runs with target checkpoint, final model, and best model."""
    jobs: list[EvaluationJob] = []
    if not training_root.is_dir():
        return jobs

    for metadata_path in training_root.rglob("run.json"):
        run_dir = metadata_path.parent
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            target_steps = int(metadata["total_timesteps"])
            seed_dir = run_dir.parent
            method_dir = seed_dir.parent
            scene_dir = method_dir.parent
            seed = int(seed_dir.name.removeprefix("seed_"))
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            continue

        models = run_dir / "models"
        required = (
            models / f"checkpoint_{target_steps}_steps.zip",
            models / "model_sb3.zip",
            models / "best_model.zip",
        )
        if all(path.is_file() for path in required):
            jobs.append(
                EvaluationJob(
                    scene=scene_dir.name,
                    method=method_dir.name,
                    seed=seed,
                    run_dir=run_dir,
                    target_steps=target_steps,
                )
            )

    return sorted(jobs, key=lambda job: (job.scene, job.method, job.seed))


def build_evaluation_command(
    job: EvaluationJob,
    *,
    episodes: int,
    airsim_ip: str,
    scenario_seed: int,
    yaw_range_deg: float,
) -> list[str]:
    """Build one reproducible best-model evaluation command."""
    return [
        sys.executable,
        "-m",
        "scripts.evaluation.evaluate_single",
        "-m",
        str(job.run_dir),
        "-n",
        str(episodes),
        "--best",
        "--save",
        "--output-root",
        str(EVALUATION_RESULTS_ROOT),
        "--airsim-ip",
        airsim_ip,
        "--airsim-port",
        str(job.port),
        "--scenario-seed",
        str(scenario_seed),
        "--yaw-range-deg",
        str(yaw_range_deg),
    ]


def build_scene_batches(
    jobs: list[EvaluationJob], ports: list[int]
) -> list[tuple[str, list[list[EvaluationJob]]]]:
    """Group jobs by map/scene and assign one reusable port per parallel slot."""
    if not ports:
        raise ValueError("At least one AirSim port is required")

    result: list[tuple[str, list[list[EvaluationJob]]]] = []
    ordered = sorted(jobs, key=lambda job: (job.scene, job.method, job.seed))
    for scene, scene_jobs_iter in groupby(ordered, key=lambda job: job.scene):
        scene_jobs = list(scene_jobs_iter)
        waves: list[list[EvaluationJob]] = []
        for offset in range(0, len(scene_jobs), len(ports)):
            wave = [
                replace(job, port=ports[index])
                for index, job in enumerate(scene_jobs[offset : offset + len(ports)])
            ]
            waves.append(wave)
        result.append((scene, waves))
    return result


def read_start_position(run_dir: Path) -> list[float]:
    """Read the frozen takeoff position from one training run."""
    cfg = ConfigParser()
    config_path = run_dir / "config" / "config.ini"
    if not cfg.read(config_path, encoding="utf-8"):
        raise FileNotFoundError(f"Training config not found: {config_path}")
    value = json.loads(cfg.get("environment", "start_position").replace("'", '"'))
    if not isinstance(value, list) or len(value) != 3:
        raise ValueError(f"Invalid start_position in {config_path}: {value!r}")
    return [float(item) for item in value]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Discover all completed revision runs and evaluate best_model.zip "
            "with reusable parallel AirSim instances."
        )
    )
    parser.add_argument(
        "--training-root",
        type=Path,
        default=TRAIN_RUNS_ROOT,
    )
    parser.add_argument(
        "--evaluation-root",
        type=Path,
        default=FORMAL_EVALUATION_BATCHES_ROOT,
    )
    parser.add_argument("-n", "--episodes", type=int, default=DEFAULT_EPISODES)
    parser.add_argument("--scenes", nargs="+", choices=SCENES)
    parser.add_argument("--methods", nargs="+")
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument(
        "--training-run-ids",
        nargs="+",
        help=(
            "Optional top-level training batch IDs to include. This keeps "
            "independent repeats separate from the main experiment matrix."
        ),
    )
    parser.add_argument("--max-parallel", type=int, default=DEFAULT_MAX_PARALLEL)
    parser.add_argument("--ports", nargs="+", type=int)
    parser.add_argument("--scenario-seed", type=int, default=2026)
    parser.add_argument("--yaw-range-deg", type=float, default=10.0)
    parser.add_argument("--airsim-ip", default="127.0.0.1")
    parser.add_argument(
        "--ue-exe",
        type=Path,
        default=Path(r"D:\UEBuilds\TunnelEnv\Windows\Tunnel.exe"),
    )
    parser.add_argument(
        "--settings-template",
        type=Path,
        default=Path(r"C:\Users\31732\Documents\AirSim\settings.json"),
    )
    parser.add_argument("--clock-speed", type=float, default=1.0)
    parser.add_argument("--res-x", type=int, default=640)
    parser.add_argument("--res-y", type=int, default=480)
    parser.add_argument("--startup-wait", type=int, default=60)
    parser.add_argument(
        "--no-launch-airsim",
        action="store_true",
        help="Use already-running AirSim instances on --ports",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def printable(command: list[str]) -> str:
    return subprocess.list2cmdline(command)


def filter_jobs(jobs: list[EvaluationJob], args: argparse.Namespace) -> list[EvaluationJob]:
    """Apply optional scene/method/seed selectors."""
    selected = jobs
    if args.scenes:
        selected = [job for job in selected if job.scene in args.scenes]
    if args.methods:
        selected = [job for job in selected if job.method in args.methods]
    if args.seeds is not None:
        selected = [job for job in selected if job.seed in args.seeds]
    if args.training_run_ids:
        accepted = set(args.training_run_ids)
        selected = [
            job
            for job in selected
            if len(job.run_dir.parents) > 3 and job.run_dir.parents[3].name in accepted
        ]
    return selected


def write_plan(
    path: Path,
    *,
    run_id: str,
    scene_groups: list[tuple[str, list[list[EvaluationJob]]]],
    args: argparse.Namespace,
) -> None:
    jobs = [job for _, waves in scene_groups for wave in waves for job in wave]
    payload = {
        "run_id": run_id,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "model_selection": "best_model.zip",
        "episodes_per_model": args.episodes,
        "scenario_seed": args.scenario_seed,
        "yaw_range_deg": args.yaw_range_deg,
        "max_parallel": args.max_parallel,
        "jobs": [
            {
                "scene": job.scene,
                "method": job.method,
                "seed": job.seed,
                "port": job.port,
                "target_steps": job.target_steps,
                "run_dir": str(job.run_dir.resolve()),
                "model": str((job.run_dir / "models" / "best_model.zip").resolve()),
                "command": build_evaluation_command(
                    job,
                    episodes=args.episodes,
                    airsim_ip=args.airsim_ip,
                    scenario_seed=args.scenario_seed,
                    yaw_range_deg=args.yaw_range_deg,
                ),
            }
            for job in jobs
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def run_scene(
    *,
    scene: str,
    waves: list[list[EvaluationJob]],
    run_dir: Path,
    args: argparse.Namespace,
) -> list[str]:
    """Launch one map, evaluate all its waves, and return failure descriptions."""
    failures: list[str] = []
    active_ports = list(dict.fromkeys(job.port for wave in waves for job in wave))
    airsim_processes = []
    try:
        if not args.no_launch_airsim:
            airsim_processes = launch_instances(
                ue_exe=args.ue_exe,
                map_path=SCENES[scene]["map_path"],
                settings_template=args.settings_template,
                output_dir=run_dir / "airsim_settings" / scene,
                ports=active_ports,
                start_position=read_start_position(waves[0][0].run_dir),
                clock_speed=args.clock_speed,
                res_x=args.res_x,
                res_y=args.res_y,
                wait_seconds=args.startup_wait,
            )

        for wave_index, wave in enumerate(waves, start=1):
            print(f"[{scene}] wave {wave_index}/{len(waves)}: {len(wave)} model(s)")
            running = []
            for job in wave:
                stem = f"{scene}_{job.method}_seed_{job.seed}_port_{job.port}"
                stdout = (run_dir / f"{stem}.out.log").open("w", encoding="utf-8")
                stderr = (run_dir / f"{stem}.err.log").open("w", encoding="utf-8")
                command = build_evaluation_command(
                    job,
                    episodes=args.episodes,
                    airsim_ip=args.airsim_ip,
                    scenario_seed=args.scenario_seed,
                    yaw_range_deg=args.yaw_range_deg,
                )
                process = subprocess.Popen(
                    command,
                    cwd=ROOT,
                    stdout=stdout,
                    stderr=stderr,
                    env=os.environ.copy(),
                )
                auto_pin_p_core([process.pid])
                running.append((job, process, stdout, stderr))
                print(
                    f"[START] {job.scene}/{job.method}/seed_{job.seed} "
                    f"port={job.port} pid={process.pid}"
                )

            for job, process, stdout, stderr in running:
                code = process.wait()
                stdout.close()
                stderr.close()
                label = f"{job.scene}/{job.method}/seed_{job.seed}"
                if code:
                    failures.append(f"{label}: exit {code}")
                print(f"[{'DONE' if code == 0 else 'FAIL'}] {label}")
    finally:
        if airsim_processes:
            terminate_processes(airsim_processes)
    return failures


def main() -> int:
    args = parse_args()
    if args.episodes < 1:
        raise ValueError("--episodes must be at least 1")
    if args.max_parallel < 1:
        raise ValueError("--max-parallel must be at least 1")

    ports = args.ports or default_ports(args.max_parallel)
    if len(ports) < args.max_parallel:
        raise ValueError("--ports must provide at least --max-parallel values")
    ports = ports[: args.max_parallel]
    if len(set(ports)) != len(ports):
        raise ValueError("--ports must be unique")

    jobs = filter_jobs(discover_completed_runs(args.training_root), args)
    if not jobs:
        raise RuntimeError("No completed runs with best_model.zip were found")
    scene_groups = build_scene_batches(jobs, ports)

    run_id = datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%f")
    run_dir = args.evaluation_root / run_id
    write_plan(run_dir / "plan.json", run_id=run_id, scene_groups=scene_groups, args=args)

    print(
        f"Discovered {len(jobs)} completed model(s), "
        f"{sum(len(waves) for _, waves in scene_groups)} wave(s), "
        f"{args.episodes} episodes each, best_model.zip"
    )
    for scene, waves in scene_groups:
        print(f"  {scene}: {sum(len(wave) for wave in waves)} model(s)")
        for wave in waves:
            for job in wave:
                print(f"    {job.method} seed={job.seed} port={job.port}")
                if args.dry_run:
                    print(
                        "      "
                        + printable(
                            build_evaluation_command(
                                job,
                                episodes=args.episodes,
                                airsim_ip=args.airsim_ip,
                                scenario_seed=args.scenario_seed,
                                yaw_range_deg=args.yaw_range_deg,
                            )
                        )
                    )

    print(f"Evaluation plan: {run_dir / 'plan.json'}")
    if args.dry_run:
        return 0

    failures: list[str] = []
    for scene, waves in scene_groups:
        failures.extend(run_scene(scene=scene, waves=waves, run_dir=run_dir, args=args))

    if failures:
        print("Evaluation failures:", file=sys.stderr)
        for failure in failures:
            print(f"  {failure}", file=sys.stderr)
        return 1
    results = reconstruct(run_dir)
    if results["failed"]:
        print(
            f"Summary reconstruction found {results['failed']} incomplete job(s).",
            file=sys.stderr,
        )
        return 1
    print(f"All {len(jobs)} best-model evaluations completed.")
    print(
        f"Batch summary: completed={results['completed']} "
        f"failed={results['failed']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
