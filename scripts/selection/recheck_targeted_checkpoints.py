#!/usr/bin/env python3
"""Safe launcher for the checkpoint rechecks agreed for the paper.

The target matrix is deliberately exact rather than a scene/method Cartesian
product:

* tunnel: SAC, CoordAtt, LSTM
* tunnel_obs: SAC, LSTM

By default this command only creates and prints a frozen plan.  Passing
``--execute`` is required before it can start evaluation processes or UE.
Checkpoint selection uses only the short screen; the 100-episode test remains
blind and cannot change the selected checkpoint.
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
from tools.experiment_paths import CROSS_SCENE_SELECTION_ROOT, TRAIN_RUNS_ROOT


TARGETS = ("tunnel:sac,coordatt,lstm", "tunnel_obs:sac,lstm")
DEFAULT_SCREEN_EPISODES = 10
DEFAULT_TEST_EPISODES = 100


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plan or execute the targeted Tunnel/Tunnel_obs checkpoint recheck. "
            "Without --execute, this is always a no-UE dry run."
        )
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Allow evaluation/UE processes to start. Omit this to inspect the plan only.",
    )
    parser.add_argument(
        "--stage",
        choices=("all", "screen", "test"),
        default="all",
        help="Execution stage; ignored unless --execute is supplied.",
    )
    parser.add_argument(
        "--resume",
        nargs="?",
        const="latest",
        help="Inspect or resume a frozen selector run (latest, run ID, directory, or plan.json).",
    )
    parser.add_argument("--training-root", type=Path, default=TRAIN_RUNS_ROOT)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=CROSS_SCENE_SELECTION_ROOT,
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--candidate-mode", choices=("full", "verify"), default="full")
    parser.add_argument("--candidate-interval", type=int, default=20_000)
    parser.add_argument("--screen-episodes", type=int, default=DEFAULT_SCREEN_EPISODES)
    parser.add_argument("--test-episodes", type=int, default=DEFAULT_TEST_EPISODES)
    parser.add_argument("--screen-seed", type=int, default=31_001)
    parser.add_argument("--test-seed", type=int, default=2_026)
    parser.add_argument("--max-parallel", type=int, default=3)
    parser.add_argument("--ports", nargs="+", type=int)
    parser.add_argument("--ue-exe", type=Path)
    parser.add_argument("--settings-template", type=Path)
    parser.add_argument("--startup-wait", type=int)
    parser.add_argument(
        "--no-launch-airsim",
        action="store_true",
        help="Use already-running AirSim instances; meaningful only with --execute.",
    )
    return parser.parse_args(argv)


def build_selector_argv(args: argparse.Namespace) -> list[str]:
    """Translate the guarded paper preset into selector arguments."""
    selector_argv = [
        "--training-root",
        str(args.training_root),
        "--output-root",
        str(args.output_root),
        "--targets",
        *TARGETS,
        "--seeds",
        *(str(seed) for seed in args.seeds),
        "--candidate-mode",
        args.candidate_mode,
        "--candidate-interval",
        str(args.candidate_interval),
        "--screen-episodes",
        str(args.screen_episodes),
        "--test-episodes",
        str(args.test_episodes),
        "--screen-seeds",
        str(args.screen_seed),
        "--test-seeds",
        str(args.test_seed),
        "--max-parallel",
        str(args.max_parallel),
    ]
    if args.ports:
        selector_argv.extend(["--ports", *(str(port) for port in args.ports)])
    if args.ue_exe is not None:
        selector_argv.extend(["--ue-exe", str(args.ue_exe)])
    if args.settings_template is not None:
        selector_argv.extend(["--settings-template", str(args.settings_template)])
    if args.startup_wait is not None:
        selector_argv.extend(["--startup-wait", str(args.startup_wait)])
    if args.resume:
        selector_argv.extend(["--resume", args.resume])
    if args.no_launch_airsim:
        selector_argv.append("--no-launch-airsim")

    if args.execute:
        selector_argv.extend(["--stage", args.stage])
    else:
        # This guard is intentionally unconditional: changing --stage cannot
        # accidentally bypass the explicit execution acknowledgement.
        selector_argv.extend(["--stage", "plan", "--dry-run"])
    return selector_argv


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.execute:
        print(
            "Safety guard active: plan/dry-run only. "
            "No UE or evaluation process can be started without --execute."
        )
    return select_checkpoints.main(build_selector_argv(args))


if __name__ == "__main__":
    raise SystemExit(main())
