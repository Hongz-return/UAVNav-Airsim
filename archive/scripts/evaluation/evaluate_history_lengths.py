#!/usr/bin/env python3
"""ENV2 history-length sweep with 3 parallel AirSim instances.

Runs L=2,4,6 concurrently (one length per port). Reuses existing L=1 / L=8
results from history-ablation-tunnel-obs/comparison.json when present.
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import subprocess
import sys
import time
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from configparser import ConfigParser
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.training.train_parallel import SCENES, default_ports
from tools.experiment_paths import EVALUATION_RESULTS_ROOT, TRAIN_RUNS_ROOT
from tools.airsim_instances import launch_instances, terminate_processes

DEFAULT_MODEL_DIR = TRAIN_RUNS_ROOT / (
    "tunnel_obs/ma/seed_0/2026_07_27_16_05_31_659067_Multirotor_cnn_SAC"
)
DEFAULT_OUTPUT_DIR = Path(
    r"D:\code\ZipFIle\paper\experiments\history-length-sweep-tunnel-obs"
)
DEFAULT_PRIOR = Path(
    r"D:\code\ZipFIle\paper\experiments\history-ablation-tunnel-obs\comparison.json"
)
DEFAULT_MANIFEST = Path(
    r"D:\code\ZipFIle\paper\experiments\history-ablation-tunnel-obs\scenario_manifest.json"
)
DEFAULT_UE_EXE = Path(r"D:\UEBuilds\TunnelEnv\Windows\Tunnel.exe")
DEFAULT_SETTINGS = Path(r"C:\Users\31732\Documents\AirSim\settings.json")
SWEEP_LENGTHS = (2, 4, 6)


def default_evaluation_python() -> Path:
    pixi_python = ROOT / ".pixi" / "envs" / "default" / "python.exe"
    if pixi_python.is_file():
        return pixi_python
    return Path(sys.executable)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Parallel ENV2 MA-SAC inference history-length sweep (L=2/4/6)."
    )
    parser.add_argument("-n", "--episodes", type=int, default=50)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--prior-comparison", type=Path, default=DEFAULT_PRIOR)
    parser.add_argument("--scenario-manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--scenario-seed", type=int, default=2026)
    parser.add_argument("--yaw-range-deg", type=float, default=10.0)
    parser.add_argument("--airsim-ip", default="127.0.0.1")
    parser.add_argument("--ports", nargs="+", type=int)
    parser.add_argument("--max-parallel", type=int, default=3)
    parser.add_argument("--ue-exe", type=Path, default=DEFAULT_UE_EXE)
    parser.add_argument("--settings-template", type=Path, default=DEFAULT_SETTINGS)
    parser.add_argument("--python-exe", type=Path, default=default_evaluation_python())
    parser.add_argument("--startup-wait", type=int, default=60)
    parser.add_argument("--clock-speed", type=float, default=1.0)
    parser.add_argument("--res-x", type=int, default=640)
    parser.add_argument("--res-y", type=int, default=480)
    parser.add_argument("--no-launch-airsim", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--lengths",
        nargs="+",
        type=int,
        default=list(SWEEP_LENGTHS),
        help="History lengths to evaluate (default: 2 4 6).",
    )
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


def evaluation_dirs(model_dir: Path) -> set[Path]:
    return {
        path.parent.resolve()
        for path in model_dir.glob("eval_*/evaluation_manifest.json")
    }


def build_eval_command(
    args: argparse.Namespace,
    *,
    seq_len: int,
    port: int,
    tag: str,
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
        str(port),
        "--scenario-seed",
        str(args.scenario_seed),
        "--yaw-range-deg",
        str(args.yaw_range_deg),
        "--scenario-manifest",
        str(args.scenario_manifest),
        "--inference-seq-len",
        str(seq_len),
    ]
    return command


def run_one_length(
    args: argparse.Namespace,
    *,
    seq_len: int,
    port: int,
) -> dict:
    tag = f"history_L{seq_len}"
    label = f"L{seq_len}"
    command = build_eval_command(args, seq_len=seq_len, port=port, tag=tag)
    before = evaluation_dirs(args.model_dir)
    log_path = args.output_dir / f"eval_{label}.log"
    print(f"[{label}] port={port} {subprocess.list2cmdline(command)}", flush=True)
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
        tail = log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-40:]
        raise RuntimeError(
            f"{label} failed exit={completed.returncode}\n" + "\n".join(tail)
        )
    created = evaluation_dirs(args.model_dir) - before
    if not created:
        raise RuntimeError(f"{label} succeeded but no result directory found")
    result_dir = max(created, key=lambda path: path.stat().st_mtime_ns)
    summary = json.loads((result_dir / "summary_metrics.json").read_text(encoding="utf-8"))
    payload = np.load(result_dir / "results.npy", allow_pickle=True).item()
    outcomes = [int(bool(value)) for value in payload["successes"]]
    rate = 100.0 * sum(outcomes) / len(outcomes)
    print(f"[{label}] success={rate:.1f}% dir={result_dir}", flush=True)
    return {
        "seq_len": seq_len,
        "successes": sum(outcomes),
        "episodes": len(outcomes),
        "success_rate_pct": rate,
        "summary": summary,
        "result_dir": str(result_dir),
        "episode_successes": outcomes,
        "port": port,
        "log": str(log_path),
    }


def load_prior_endpoints(prior_path: Path) -> dict[int, dict]:
    """Map seq_len -> result dict from existing Full/Trunc comparison."""
    if not prior_path.is_file():
        return {}
    prior = json.loads(prior_path.read_text(encoding="utf-8"))
    out: dict[int, dict] = {}
    trunc = prior.get("trunc")
    full = prior.get("full")
    if trunc:
        out[1] = {
            "seq_len": 1,
            "successes": trunc["successes"],
            "episodes": trunc["episodes"],
            "success_rate_pct": trunc["success_rate_pct"],
            "summary": trunc.get("summary"),
            "result_dir": trunc.get("result_dir"),
            "episode_successes": trunc.get("episode_successes"),
            "source": "prior_trunc",
        }
    if full:
        out[8] = {
            "seq_len": 8,
            "successes": full["successes"],
            "episodes": full["episodes"],
            "success_rate_pct": full["success_rate_pct"],
            "summary": full.get("summary"),
            "result_dir": full.get("result_dir"),
            "episode_successes": full.get("episode_successes"),
            "source": "prior_full",
        }
    return out


def main() -> int:
    args = parse_args()
    args.model_dir = args.model_dir.resolve()
    args.output_dir = args.output_dir.resolve()
    args.python_exe = args.python_exe.resolve()
    args.scenario_manifest = args.scenario_manifest.resolve()
    lengths = sorted({int(x) for x in args.lengths})
    if any(x < 1 for x in lengths):
        raise ValueError("lengths must be >= 1")
    if args.max_parallel < 1:
        raise ValueError("--max-parallel must be >= 1")
    ports = list(args.ports or default_ports(args.max_parallel))
    if len(ports) < min(args.max_parallel, len(lengths)):
        raise ValueError("Need enough ports for parallel lengths")
    ports = ports[: args.max_parallel]
    if len(set(ports)) != len(ports):
        raise ValueError("ports must be unique")
    if not (args.model_dir / "models" / "best_model.zip").is_file():
        raise FileNotFoundError(args.model_dir / "models" / "best_model.zip")
    if not args.scenario_manifest.is_file():
        raise FileNotFoundError(args.scenario_manifest)
    if not args.python_exe.is_file():
        raise FileNotFoundError(args.python_exe)

    cfg = read_config(args.model_dir)
    scene = SCENES["tunnel_obs"]
    map_path = scene["map_path"]

    plan = [
        {"seq_len": L, "port": ports[i % len(ports)]}
        for i, L in enumerate(lengths)
    ]
    print("History-length sweep plan:")
    for item in plan:
        print(f"  L={item['seq_len']} -> port {item['port']}")
    if args.dry_run:
        for item in plan:
            print(
                subprocess.list2cmdline(
                    build_eval_command(
                        args,
                        seq_len=item["seq_len"],
                        port=item["port"],
                        tag=f"history_L{item['seq_len']}",
                    )
                )
            )
        return 0

    args.output_dir.mkdir(parents=True, exist_ok=True)
    protocol = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "scene": "ENV2 / tunnel_obs",
        "map_path": map_path,
        "model_dir": str(args.model_dir),
        "episodes": args.episodes,
        "scenario_manifest": str(args.scenario_manifest),
        "lengths_to_run": lengths,
        "ports": ports,
        "reuse_prior": str(args.prior_comparison),
    }
    (args.output_dir / "protocol.json").write_text(
        json.dumps(protocol, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    airsim_processes: list = []
    started = time.time()
    results_by_L: dict[int, dict] = load_prior_endpoints(args.prior_comparison)
    try:
        # Run in waves of max_parallel
        for offset in range(0, len(plan), len(ports)):
            wave = plan[offset : offset + len(ports)]
            wave_ports = [item["port"] for item in wave]
            if not args.no_launch_airsim:
                if airsim_processes:
                    terminate_processes(airsim_processes)
                    airsim_processes = []
                    time.sleep(5)
                airsim_processes = launch_instances(
                    ue_exe=args.ue_exe,
                    map_path=map_path,
                    settings_template=args.settings_template,
                    output_dir=args.output_dir / "airsim_settings",
                    ports=wave_ports,
                    start_position=read_start_position(cfg),
                    clock_speed=args.clock_speed,
                    res_x=args.res_x,
                    res_y=args.res_y,
                    wait_seconds=args.startup_wait,
                )
                dead = [p for p in airsim_processes if p.poll() is not None]
                if dead:
                    raise RuntimeError(f"AirSim exited during startup: {[p.returncode for p in dead]}")

            with ThreadPoolExecutor(max_workers=len(wave)) as pool:
                futures = {
                    pool.submit(
                        run_one_length,
                        args,
                        seq_len=item["seq_len"],
                        port=item["port"],
                    ): item["seq_len"]
                    for item in wave
                }
                for future in as_completed(futures):
                    L = futures[future]
                    results_by_L[L] = future.result()
                    results_by_L[L]["source"] = "this_sweep"

        comparison = {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "experiment": "closed_loop_inference_history_length_sweep",
            "scene": "ENV2 / Tunnel (static obstacles)",
            "scene_key": "tunnel_obs",
            "map_path": map_path,
            "model_dir": str(args.model_dir),
            "episodes_per_condition": args.episodes,
            "scenario_seed": args.scenario_seed,
            "scenario_manifest": str(args.scenario_manifest),
            "lengths": sorted(results_by_L),
            "by_length": {str(k): results_by_L[k] for k in sorted(results_by_L)},
            "success_rate_curve": [
                {
                    "seq_len": k,
                    "success_rate_pct": results_by_L[k]["success_rate_pct"],
                    "source": results_by_L[k].get("source"),
                }
                for k in sorted(results_by_L)
            ],
            "wall_time_sec": time.time() - started,
        }
        out_path = args.output_dir / "comparison.json"
        out_path.write_text(
            json.dumps(comparison, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(comparison["success_rate_curve"], ensure_ascii=False, indent=2))
        print(f"Saved {out_path}")
        return 0
    finally:
        if airsim_processes:
            terminate_processes(airsim_processes)


if __name__ == "__main__":
    raise SystemExit(main())
