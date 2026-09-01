#!/usr/bin/env python3
"""Summarise TensorBoard training scalars for one scene.

Usage:
    python -m scripts.analysis.analyze_training_metrics --scene tunnel
"""

from __future__ import annotations

import argparse
from pathlib import Path

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from scripts.training.train_parallel import DEFAULT_METHODS, SCENES
from tools.experiment_paths import TRAIN_RUNS_ROOT

METRICS = (
    ("best/success_rate", "Best Success Rate", "{:.2%}"),
    ("core/success_rate_recent", "Recent Success Rate", "{:.2%}"),
    ("episode/success_rate_ma20", "Success Rate (MA20)", "{:.2%}"),
    ("episode/return_ma20", "Episode Return MA20", "{:.2f}"),
    ("episode/return_raw", "Latest Episode Return", "{:.2f}"),
    ("rollout/success_ep_len_mean", "Avg Success Episode Length", "{:.1f} steps"),
    ("uav/distance_to_goal", "Current Distance to Goal", "{:.2f} m"),
    ("time/episodes", "Completed Episodes", "{:.0f}"),
    ("train/actor_loss", "Actor Loss", "{:.4f}"),
    ("train/critic_loss", "Critic Loss", "{:.4f}"),
    ("train/ent_coef", "Entropy Coefficient", "{:.6f}"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", choices=SCENES, default="tunnel")
    parser.add_argument("--methods", nargs="+", default=list(DEFAULT_METHODS))
    parser.add_argument(
        "--training-root",
        type=Path,
        default=TRAIN_RUNS_ROOT,
        help="Root holding scene/method/seed/run_id training runs.",
    )
    parser.add_argument(
        "--target-steps",
        type=int,
        default=None,
        help="Completion target used for the progress line (default: scene budget).",
    )
    return parser.parse_args()


def load_scalars(event_file: Path) -> tuple[dict[str, float], dict[str, float]]:
    """Return the last and the peak value of every scalar tag in one event file."""
    accumulator = EventAccumulator(str(event_file))
    accumulator.Reload()
    latest: dict[str, float] = {}
    peak: dict[str, float] = {}
    for tag in accumulator.Tags().get("scalars", []):
        events = accumulator.Scalars(tag)
        if events:
            latest[tag] = events[-1].value
            peak[tag] = max(event.value for event in events)
    return latest, peak


def find_event_files(method_dir: Path) -> list[Path]:
    return sorted(method_dir.rglob("events.out.tfevents*"))


def report_method(method: str, method_dir: Path, target_steps: int) -> dict[str, float]:
    print(f"\n{method.upper():^70}")
    print("-" * 70)

    event_files = find_event_files(method_dir)
    if not event_files:
        print(f"  No TensorBoard event files under {method_dir}")
        return {}

    summary: dict[str, float] = {}
    for event_file in event_files:
        latest, peak = load_scalars(event_file)
        if not latest:
            continue
        print(f"  {event_file.parent.relative_to(method_dir)}")
        for tag, label, fmt in METRICS:
            if tag in latest:
                print(f"    {label:35s}: {fmt.format(latest[tag])}")

        steps = int(latest.get("time/total_timesteps", 0))
        if steps:
            share = steps / target_steps * 100
            state = "COMPLETED" if steps >= target_steps else f"TRAINING - {share:.1f}%"
            print(f"    {'Training Steps':35s}: {steps:,} / {target_steps:,} [{state}]")

        # Keep the furthest-trained run of this method for the comparison table.
        if steps >= summary.get("steps", -1):
            summary = {
                "best_sr": peak.get("best/success_rate", 0.0),
                "recent_sr": latest.get("core/success_rate_recent", 0.0),
                "return": latest.get("episode/return_ma20", 0.0),
                "steps": steps,
            }
    return summary


def main() -> int:
    args = parse_args()
    target_steps = args.target_steps or SCENES[args.scene]["timesteps"]
    scene_dir = args.training_root / args.scene

    print("=" * 80)
    print(f"TRAINING ANALYSIS REPORT - scene: {args.scene} ({scene_dir})")
    print("=" * 80)

    comparison = {
        method: report_method(method, scene_dir / method, target_steps)
        for method in args.methods
    }

    print("\n" + "=" * 80)
    print("COMPARATIVE ANALYSIS")
    print("=" * 80)
    print("\nMethod       | Best SR   | Recent SR | Return MA20 | Steps")
    print("------------ | --------- | --------- | ----------- | --------")
    for method in args.methods:
        row = comparison.get(method) or {}
        print(
            f"{method:12s} | {row.get('best_sr', 0):8.2%} | {row.get('recent_sr', 0):8.2%} | "
            f"{row.get('return', 0):11.2f} | {row.get('steps', 0):,}"
        )
    print(
        "\nValues are the last point of each scalar, so they describe the run's "
        "current state rather than a verified experiment result."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
