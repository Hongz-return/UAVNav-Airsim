#!/usr/bin/env python3
"""Rebuild batch-level evaluation summaries from completed per-model artifacts."""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from tools.experiment_paths import evaluation_parent_for_run


ROOT = Path(__file__).resolve().parents[2]
SUMMARY_FIELDS = (
    "episodes",
    "success_rate_pct",
    "collision_rate_pct",
    "timeout_rate_pct",
    "out_of_bounds_rate_pct",
    "avg_reward",
    "reward_std",
    "spl",
    "avg_path_length_m",
    "avg_path_length_success_m",
    "avg_steps",
    "avg_steps_success",
    "avg_time_sec",
    "avg_time_success_sec",
    "avg_speed_mps",
    "avg_speed_success_mps",
    "max_speed_mps",
    "avg_min_obstacle_dist_m",
    "smoothness",
    "control_effort",
    "mean_yaw_rate",
    "latency_ms",
)
CSV_FIELDS = (
    "scene",
    "method",
    "seed",
    "model_run",
    "evaluation_dir",
    *SUMMARY_FIELDS,
)


def relative_text(path: Path) -> str:
    """Return a repository-relative Windows-style path when possible."""
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(ROOT))
    except ValueError:
        return str(resolved)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def recorded_path(raw_path: str | Path) -> Path:
    path = Path(raw_path)
    return (path if path.is_absolute() else ROOT / path).resolve()


def find_evaluation(
    run_dir: Path,
    *,
    episodes: int,
    best_model: Path,
) -> tuple[Path, dict[str, Any]] | None:
    """Find the newest complete evaluation matching this plan job."""
    matches: list[tuple[Path, dict[str, Any]]] = []
    try:
        result_parent = evaluation_parent_for_run(run_dir)
    except ValueError:
        result_parent = run_dir
    for evaluation_dir in sorted(
        result_parent.glob(f"eval_{episodes}_best_*"), reverse=True
    ):
        summary_path = evaluation_dir / "summary_metrics.json"
        manifest_path = evaluation_dir / "evaluation_manifest.json"
        if not summary_path.is_file() or not manifest_path.is_file():
            continue
        try:
            summary = load_json(summary_path)
            manifest = load_json(manifest_path)
            manifest_model = Path(manifest["model_file"]).resolve()
        except (KeyError, TypeError, ValueError, json.JSONDecodeError, OSError):
            continue
        if int(summary.get("episodes", -1)) != episodes:
            continue
        if manifest.get("model_type") != "best":
            continue
        if str(manifest_model).casefold() != str(best_model.resolve()).casefold():
            continue
        if any(field not in summary for field in SUMMARY_FIELDS):
            continue
        matches.append((evaluation_dir, summary))
    return matches[0] if matches else None


def reconstruct(batch_dir: Path) -> dict[str, Any]:
    batch_dir = batch_dir.resolve()
    plan_path = batch_dir / "plan.json"
    plan = load_json(plan_path)
    episodes = int(plan["episodes_per_model"])
    evaluation_run_id = str(plan.get("run_id") or batch_dir.name)

    final_jobs: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []

    for job in plan["jobs"]:
        run_dir = recorded_path(job.get("run_dir") or job["model_run"])
        best_model = recorded_path(job.get("model") or job["best_model"])
        training_run_id = run_dir.name
        stem = (
            f"{job['scene']}_{job['method']}_seed_{job['seed']}"
            f"_port_{job['port']}"
        )
        stdout_log = batch_dir / f"{stem}.out.log"
        stderr_log = batch_dir / f"{stem}.err.log"
        found = find_evaluation(
            run_dir,
            episodes=episodes,
            best_model=best_model,
        )

        record: dict[str, Any] = {
            "run_id": training_run_id,
            "scene": job["scene"],
            "method": job["method"],
            "seed": int(job["seed"]),
            "port": int(job["port"]),
            "attempt": 1,
            "status": "completed" if found else "failed",
            "exit_code": 0 if found else None,
            "model_run": relative_text(run_dir),
            "best_model": relative_text(best_model),
            "evaluation_dir": relative_text(found[0]) if found else None,
            "stdout_log": relative_text(stdout_log),
            "stderr_log": relative_text(stderr_log),
        }
        final_jobs.append(record)

        if found:
            summary_row = {
                "scene": job["scene"],
                "method": job["method"],
                "seed": int(job["seed"]),
                "model_run": relative_text(run_dir),
                "evaluation_dir": relative_text(found[0]),
            }
            summary_row.update({field: found[1][field] for field in SUMMARY_FIELDS})
            summaries.append(summary_row)

    completed = sum(job["status"] == "completed" for job in final_jobs)
    failed = len(final_jobs) - completed
    results = {
        "evaluation_run_id": evaluation_run_id,
        "completed": completed,
        "skipped_existing": 0,
        "failed": failed,
        "final_jobs": final_jobs,
        "attempts": final_jobs,
        "reconstructed_at": datetime.now().isoformat(timespec="seconds"),
        "reconstructed_from": "plan.json and per-model evaluation artifacts",
    }

    (batch_dir / "results.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (batch_dir / "summary_all_models.json").write_text(
        json.dumps(summaries, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    with (batch_dir / "summary_all_models.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(summaries)

    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Rebuild results.json and summary_all_models.{json,csv} without "
            "rerunning AirSim evaluation."
        )
    )
    parser.add_argument(
        "batch_dir",
        type=Path,
        help="Existing Logs/EvaData/FormalBatches/<run_id> directory",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    results = reconstruct(args.batch_dir)
    print(
        f"Rebuilt {results['evaluation_run_id']}: "
        f"completed={results['completed']} failed={results['failed']}"
    )
    return 0 if results["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
