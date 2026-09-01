#!/usr/bin/env python3
"""Guarded ENV4 CoordAtt-only checkpoint rescreening preset.

The preset scans every saved 10k periodic checkpoint for CoordAtt seeds 0/1/2
in ``tunnel_mixed``.  It runs validation screening only; it never launches the
formal-test stage.  An all-zero method/seed group is recorded as
``NO_VALID_CHECKPOINT`` instead of being forced into a rank-1 selection.

Without ``--execute`` this entrypoint writes a frozen dry-run plan only.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.selection import select_checkpoints
from tools.experiment_paths import ENV4_SELECTION_ROOT, TRAIN_RUNS_ROOT


TARGET = "tunnel_mixed:coordatt"
ACCEPTED_TRUNCATED = "tunnel_mixed:coordatt:0:150000"
DEFAULT_SCREEN_SEEDS = [33_001, 33_002]
RESERVED_TEST_SEEDS = [43_001, 43_002]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Dry-run or execute ENV4 CoordAtt-only exhaustive checkpoint screening."
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Allow UE/AirSim validation screening to start.",
    )
    parser.add_argument(
        "--resume",
        nargs="?",
        const="latest",
        help="Resume latest, a run ID, a run directory, or a plan.json path.",
    )
    parser.add_argument(
        "--training-root",
        type=Path,
        default=TRAIN_RUNS_ROOT,
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ENV4_SELECTION_ROOT,
    )
    parser.add_argument("--screen-episodes", type=int, default=10)
    parser.add_argument(
        "--screen-seeds", nargs="+", type=int, default=list(DEFAULT_SCREEN_SEEDS)
    )
    parser.add_argument("--max-parallel", type=int, default=3)
    parser.add_argument("--ports", nargs="+", type=int, default=[41_451, 41_461, 41_471])
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
    parser.add_argument("--startup-wait", type=int, default=60)
    parser.add_argument("--no-launch-airsim", action="store_true")
    return parser.parse_args(argv)


def build_selector_argv(args: argparse.Namespace) -> list[str]:
    selector_argv = [
        "--training-root",
        str(args.training_root),
        "--output-root",
        str(args.output_root),
        "--targets",
        TARGET,
        "--seeds",
        "0",
        "1",
        "2",
        "--profile",
        "dynamic",
        "--candidate-mode",
        "full",
        "--candidate-interval",
        "10000",
        "--accept-truncated",
        ACCEPTED_TRUNCATED,
        "--screen-episodes",
        str(args.screen_episodes),
        "--screen-seeds",
        *(str(seed) for seed in args.screen_seeds),
        "--test-episodes",
        "50",
        "--test-seeds",
        *(str(seed) for seed in RESERVED_TEST_SEEDS),
        "--reject-all-zero-candidates",
        "--selection-only",
        "--max-parallel",
        str(args.max_parallel),
        "--ports",
        *(str(port) for port in args.ports),
        "--ue-exe",
        str(args.ue_exe),
        "--settings-template",
        str(args.settings_template),
        "--startup-wait",
        str(args.startup_wait),
    ]
    if args.resume:
        selector_argv.extend(["--resume", args.resume])
    if args.no_launch_airsim:
        selector_argv.append("--no-launch-airsim")

    if args.execute:
        selector_argv.extend(["--stage", "screen"])
    else:
        selector_argv.extend(["--stage", "plan", "--dry-run"])
    return selector_argv


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.execute:
        print(
            "Safety guard active: frozen plan/dry-run only. "
            "No UE, AirSim, or screening worker can start without --execute."
        )
    return select_checkpoints.main(build_selector_argv(args))


if __name__ == "__main__":
    raise SystemExit(main())
