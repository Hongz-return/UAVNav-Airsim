#!/usr/bin/env python3
"""Guarded one-command checkpoint selection and evaluation for Tunnel_AO.

The preset is deliberately limited to the models trained on this computer:
SAC and CoordAtt-SAC, seeds 0/1/2, in ``tunnel_mixed``.  It runs a dynamic
two-stage protocol:

1. build a compact candidate set from training-only TensorBoard metrics;
2. screen every frozen candidate on two independent 10-episode batches;
3. select and freeze one winner per method/seed using screen results only;
4. evaluate each frozen winner on two independent 50-episode test batches.

The two formal-test batches therefore contain 100 episodes in total.  Their
results can never change the selected checkpoint.  By default this entrypoint
only writes and prints a frozen dry-run plan; ``--execute`` is required to
launch Unreal Engine, AirSim, or evaluation workers.

CoordAtt seed 0 is explicitly recorded as an accepted truncated run whose
latest periodic checkpoint is 150,000 steps.  It is never relabelled as a
200,000-step checkpoint.
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


TARGET = "tunnel_mixed:sac,coordatt"
ACCEPTED_TRUNCATED_COORDINATE = "tunnel_mixed:coordatt:0"
DEFAULT_ACCEPTED_STEP = 150_000
DEFAULT_SCREEN_SEEDS = [32_001, 32_002]
DEFAULT_TEST_SEEDS = [2_026, 2_027]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plan or execute checkpoint selection plus blind 100-episode "
            "evaluation for local tunnel_mixed SAC/CoordAtt runs."
        )
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Allow UE/AirSim/evaluation processes to start; omit for dry-run planning.",
    )
    parser.add_argument(
        "--stage",
        choices=("all", "screen", "test"),
        default="all",
        help="Stage to execute. Test requires --resume of a frozen plan.",
    )
    parser.add_argument(
        "--resume",
        nargs="?",
        const="latest",
        help="Resume latest, a run ID, a run directory, or a plan.json path.",
    )
    parser.add_argument("--training-root", type=Path, default=TRAIN_RUNS_ROOT)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ENV4_SELECTION_ROOT,
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument(
        "--candidate-mode",
        choices=("full", "verify", "training_peaks"),
        default="training_peaks",
        help=(
            "training_peaks keeps best/final, two >=30k-separated return_ma20 peaks, "
            "and one success_rate_ma50 peak per method/seed."
        ),
    )
    parser.add_argument("--candidate-interval", type=int, default=20_000)
    parser.add_argument("--screen-episodes", type=int, default=10)
    parser.add_argument("--test-episodes", type=int, default=50)
    parser.add_argument(
        "--screen-seeds", nargs="+", type=int, default=list(DEFAULT_SCREEN_SEEDS)
    )
    parser.add_argument("--test-seeds", nargs="+", type=int, default=list(DEFAULT_TEST_SEEDS))
    parser.add_argument(
        "--coordatt-seed0-accepted-step",
        type=int,
        default=DEFAULT_ACCEPTED_STEP,
        help="Explicitly accepted latest periodic checkpoint for truncated CoordAtt seed 0.",
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
    parser.add_argument(
        "--no-launch-airsim",
        action="store_true",
        help="Use already-running AirSim instances; meaningful only with --execute.",
    )
    return parser.parse_args(argv)


def build_selector_argv(args: argparse.Namespace) -> list[str]:
    if args.coordatt_seed0_accepted_step < 1:
        raise ValueError("--coordatt-seed0-accepted-step must be positive")
    if args.stage == "test" and not args.resume:
        raise ValueError("--stage test requires --resume of a frozen plan")

    selector_argv = [
        "--training-root",
        str(args.training_root),
        "--output-root",
        str(args.output_root),
        "--targets",
        TARGET,
        "--seeds",
        *(str(seed) for seed in args.seeds),
        "--profile",
        "dynamic",
        "--candidate-mode",
        args.candidate_mode,
        "--candidate-interval",
        str(args.candidate_interval),
        "--accept-truncated",
        f"{ACCEPTED_TRUNCATED_COORDINATE}:{args.coordatt_seed0_accepted_step}",
        "--screen-episodes",
        str(args.screen_episodes),
        "--test-episodes",
        str(args.test_episodes),
        "--screen-seeds",
        *(str(seed) for seed in args.screen_seeds),
        "--test-seeds",
        *(str(seed) for seed in args.test_seeds),
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
        selector_argv.extend(["--stage", args.stage])
    else:
        selector_argv.extend(["--stage", "plan", "--dry-run"])
    return selector_argv


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.execute:
        print(
            "Safety guard active: frozen plan/dry-run only. "
            "No UE, AirSim, or evaluation process can start without --execute."
        )
    return select_checkpoints.main(build_selector_argv(args))


if __name__ == "__main__":
    raise SystemExit(main())
