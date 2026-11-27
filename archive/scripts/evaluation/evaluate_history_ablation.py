#!/usr/bin/env python3
"""Run the ENV3 closed-loop Full-vs-Trunc history ablation."""

from __future__ import annotations

import argparse
import ast
import json
import os
import shutil
import subprocess
import sys
import time
from configparser import ConfigParser
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.airsim_instances import launch_instances, terminate_processes
from tools.experiment_paths import EVALUATION_RESULTS_ROOT, TRAIN_RUNS_ROOT


DEFAULT_MODEL_DIR = TRAIN_RUNS_ROOT / (
    "tunnel_activate/ma/seed_0/2026_07_28_16_54_21_375604_Multirotor_cnn_SAC"
)
DEFAULT_OUTPUT_DIR = Path(
    r"D:\code\ZipFIle\paper\experiments\history-ablation-obstunnel"
)
DEFAULT_UE_EXE = Path(r"D:\UEBuilds\TunnelEnv\Windows\Tunnel.exe")
DEFAULT_SETTINGS = Path(r"C:\Users\31732\Documents\AirSim\settings.json")
MAP_PATH = "/Game/Tunnel_Col_Vol1/Maps/ObsTunnel"


def default_evaluation_python() -> Path:
    """Prefer the project-managed runtime, which contains the AirSim stack."""
    pixi_python = ROOT / ".pixi" / "envs" / "default" / "python.exe"
    if pixi_python.is_file():
        return pixi_python
    return Path(sys.executable)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate frozen MA-SAC best_model with and without inference history."
    )
    parser.add_argument("-n", "--episodes", type=int, default=30)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--scenario-seed", type=int, default=2026)
    parser.add_argument("--yaw-range-deg", type=float, default=10.0)
    parser.add_argument("--airsim-ip", default="127.0.0.1")
    parser.add_argument("--airsim-port", type=int, default=41451)
    parser.add_argument("--ue-exe", type=Path, default=DEFAULT_UE_EXE)
    parser.add_argument("--settings-template", type=Path, default=DEFAULT_SETTINGS)
    parser.add_argument(
        "--python-exe",
        type=Path,
        default=default_evaluation_python(),
        help=(
            "Python used for scripts.evaluation.evaluate_single "
            "(default: project Pixi environment when available)."
        ),
    )
    parser.add_argument("--startup-wait", type=int, default=60)
    parser.add_argument("--clock-speed", type=float, default=1.0)
    parser.add_argument("--res-x", type=int, default=640)
    parser.add_argument("--res-y", type=int, default=480)
    parser.add_argument(
        "--no-launch-airsim",
        action="store_true",
        help="Use an already-running ObsTunnel instance on --airsim-port.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def read_config(model_dir: Path) -> ConfigParser:
    config_path = model_dir / "config" / "config.ini"
    cfg = ConfigParser()
    if not cfg.read(config_path, encoding="utf-8"):
        raise FileNotFoundError(f"Training config not found: {config_path}")
    return cfg


def read_start_position(cfg: ConfigParser) -> list[float]:
    value = ast.literal_eval(cfg.get("environment", "start_position"))
    if not isinstance(value, list) or len(value) != 3:
        raise ValueError(f"Invalid start_position: {value!r}")
    return [float(item) for item in value]


def build_eval_command(
    args: argparse.Namespace,
    *,
    tag: str,
    no_history: bool,
    scenario_manifest: Path | None = None,
) -> list[str]:
    command = [
        str(args.python_exe),
        "-m",
        "scripts.evaluation.evaluate_single",
        "-m",
        str(args.model_dir),
        "-n",
        str(args.episodes),
        "--best",
        "--save",
        "--output-root",
        str(EVALUATION_RESULTS_ROOT),
        "--save_tag",
        tag,
        "--airsim-ip",
        args.airsim_ip,
        "--airsim-port",
        str(args.airsim_port),
        "--scenario-seed",
        str(args.scenario_seed),
        "--yaw-range-deg",
        str(args.yaw_range_deg),
    ]
    if scenario_manifest is not None:
        command.extend(["--scenario-manifest", str(scenario_manifest)])
    if no_history:
        command.append("--no-history")
    return command


def printable(command: list[str]) -> str:
    return subprocess.list2cmdline(command)


def evaluation_dirs(model_dir: Path) -> set[Path]:
    return {
        path.parent.resolve()
        for path in model_dir.glob("eval_*/evaluation_manifest.json")
    }


def run_condition(
    args: argparse.Namespace,
    *,
    label: str,
    tag: str,
    no_history: bool,
    scenario_manifest: Path | None = None,
) -> Path:
    command = build_eval_command(
        args,
        tag=tag,
        no_history=no_history,
        scenario_manifest=scenario_manifest,
    )
    before = evaluation_dirs(args.model_dir)
    log_path = args.output_dir / f"eval_{label}.log"
    print(f"[{label.upper()}] {printable(command)}")
    with log_path.open("w", encoding="utf-8") as log_handle:
        completed = subprocess.run(
            command,
            cwd=ROOT,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            env=os.environ.copy(),
            check=False,
        )
    if completed.returncode != 0:
        tail = log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-30:]
        raise RuntimeError(
            f"{label} evaluation failed with exit {completed.returncode}.\n"
            + "\n".join(tail)
        )

    created = evaluation_dirs(args.model_dir) - before
    if not created:
        raise RuntimeError(f"{label} evaluation succeeded but no result directory was created")
    result_dir = max(created, key=lambda path: path.stat().st_mtime_ns)
    print(f"[{label.upper()}] results: {result_dir}")
    return result_dir


def load_outcomes(result_dir: Path) -> tuple[dict, list[int]]:
    summary = json.loads((result_dir / "summary_metrics.json").read_text(encoding="utf-8"))
    payload = np.load(result_dir / "results.npy", allow_pickle=True).item()
    outcomes = [int(bool(value)) for value in payload["successes"]]
    return summary, outcomes


def build_comparison(
    args: argparse.Namespace,
    cfg: ConfigParser,
    full_dir: Path,
    trunc_dir: Path,
) -> dict:
    full_summary, full_outcomes = load_outcomes(full_dir)
    trunc_summary, trunc_outcomes = load_outcomes(trunc_dir)
    if len(full_outcomes) != args.episodes or len(trunc_outcomes) != args.episodes:
        raise RuntimeError("Saved outcome counts do not match --episodes")

    full_scenarios = json.loads(
        (full_dir / "scenario_manifest.json").read_text(encoding="utf-8")
    )["scenarios"]
    trunc_scenarios = json.loads(
        (trunc_dir / "scenario_manifest.json").read_text(encoding="utf-8")
    )["scenarios"]
    if full_scenarios != trunc_scenarios:
        raise RuntimeError("Full and Trunc did not use identical scenario manifests")

    full_successes = sum(full_outcomes)
    trunc_successes = sum(trunc_outcomes)
    full_rate = 100.0 * full_successes / args.episodes
    trunc_rate = 100.0 * trunc_successes / args.episodes
    paired = {
        "both_success": sum(f and t for f, t in zip(full_outcomes, trunc_outcomes)),
        "full_only_success": sum(f and not t for f, t in zip(full_outcomes, trunc_outcomes)),
        "trunc_only_success": sum(t and not f for f, t in zip(full_outcomes, trunc_outcomes)),
        "both_failure": sum(not f and not t for f, t in zip(full_outcomes, trunc_outcomes)),
    }
    model_s = str(args.model_dir).replace("\\", "/").lower()
    if "tunnel_obs" in model_s:
        scene, scene_key, map_path = "ENV2 / Tunnel (static obstacles)", "tunnel_obs", "/Game/.../Tunnel"
    elif "tunnel_activate" in model_s or "obstunnel" in model_s:
        scene, scene_key, map_path = "ENV3 / ObsTunnel", "tunnel_activate", MAP_PATH
    else:
        scene, scene_key, map_path = "unknown", "unknown", MAP_PATH

    return {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "experiment": "closed_loop_inference_history_ablation",
        "scene": scene,
        "scene_key": scene_key,
        "map_path": map_path,
        "model_dir": str(args.model_dir.resolve()),
        "model_file": str((args.model_dir / "models" / "best_model.zip").resolve()),
        "model_selection": "best_model.zip",
        "training_seed": 0,
        "episodes_per_condition": args.episodes,
        "scenario_seed": args.scenario_seed,
        "yaw_range_deg": args.yaw_range_deg,
        "inference_seq_len_full": cfg.getint("mamba", "inference_seq_len"),
        "inference_seq_len_trunc": 1,
        "same_scenario_manifest": True,
        "full": {
            "successes": full_successes,
            "episodes": args.episodes,
            "success_rate_pct": full_rate,
            "summary": full_summary,
            "result_dir": str(full_dir),
            "episode_successes": full_outcomes,
        },
        "trunc": {
            "successes": trunc_successes,
            "episodes": args.episodes,
            "success_rate_pct": trunc_rate,
            "summary": trunc_summary,
            "result_dir": str(trunc_dir),
            "episode_successes": trunc_outcomes,
        },
        "success_rate_difference_percentage_points": full_rate - trunc_rate,
        "paired_outcome_counts": paired,
    }


def main() -> int:
    args = parse_args()
    args.model_dir = args.model_dir.resolve()
    args.output_dir = args.output_dir.resolve()
    args.python_exe = args.python_exe.resolve()
    if args.episodes < 1:
        raise ValueError("--episodes must be at least 1")
    if args.startup_wait < 0:
        raise ValueError("--startup-wait must be non-negative")
    if not (args.model_dir / "models" / "best_model.zip").is_file():
        raise FileNotFoundError(args.model_dir / "models" / "best_model.zip")
    if not args.python_exe.is_file():
        raise FileNotFoundError(f"Evaluation Python not found: {args.python_exe}")

    cfg = read_config(args.model_dir)
    if not cfg.getboolean("options", "use_mamba", fallback=False):
        raise ValueError("History ablation requires a MambaSAC training run")

    full_command = build_eval_command(
        args,
        tag="history_full",
        no_history=False,
    )
    trunc_command = build_eval_command(
        args,
        tag="history_trunc",
        no_history=True,
        scenario_manifest=Path("<full-scenario-manifest>"),
    )
    if args.dry_run:
        print(printable(full_command))
        print(printable(trunc_command))
        return 0

    args.output_dir.mkdir(parents=True, exist_ok=True)
    airsim_processes = []
    started_at = time.time()
    try:
        if not args.no_launch_airsim:
            airsim_processes = launch_instances(
                ue_exe=args.ue_exe,
                map_path=MAP_PATH,
                settings_template=args.settings_template,
                output_dir=args.output_dir / "airsim_settings",
                ports=[args.airsim_port],
                start_position=read_start_position(cfg),
                clock_speed=args.clock_speed,
                res_x=args.res_x,
                res_y=args.res_y,
                wait_seconds=args.startup_wait,
            )
            dead = [process for process in airsim_processes if process.poll() is not None]
            if dead:
                raise RuntimeError(f"AirSim exited during startup: {[p.returncode for p in dead]}")

        full_dir = run_condition(
            args,
            label="full",
            tag="history_full",
            no_history=False,
        )
        full_manifest = full_dir / "scenario_manifest.json"
        trunc_dir = run_condition(
            args,
            label="trunc",
            tag="history_trunc",
            no_history=True,
            scenario_manifest=full_manifest,
        )
        comparison = build_comparison(args, cfg, full_dir, trunc_dir)
        comparison["wall_time_sec"] = time.time() - started_at

        comparison_path = args.output_dir / "comparison.json"
        comparison_path.write_text(
            json.dumps(comparison, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        shutil.copy2(full_manifest, args.output_dir / "scenario_manifest.json")
        print(json.dumps(comparison, ensure_ascii=False, indent=2))
        print(f"Comparison saved to: {comparison_path}")
        return 0
    finally:
        if airsim_processes:
            terminate_processes(airsim_processes)


if __name__ == "__main__":
    raise SystemExit(main())
