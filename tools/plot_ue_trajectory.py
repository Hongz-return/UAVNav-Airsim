"""Plot a recorded episode trajectory into a live AirSim/UE scene.

Use this to capture local UE screenshots (occlusion expected).
Pair with paper/experiments/trajectory-style-compare-2026-08-02 candidates.

Example:
  # 1) Launch Tunnel map windowed (ViewMode != NoDisplay)
  # 2) Run:
  pixi run python tools/plot_ue_trajectory.py \
    --csv <episode_success.csv> \
    --airsim-port 41451 \
    --duration 120
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

try:
    import cosysairsim as airsim
except ImportError:  # pragma: no cover - fallback for stock AirSim installs
    import airsim
import numpy as np


def load_xyz(csv_path: Path) -> list[airsim.Vector3r]:
    with csv_path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    pts = []
    for r in rows:
        # AirSim NED
        pts.append(airsim.Vector3r(float(r["x"]), float(r["y"]), float(r["z"])))
    return pts


def subsample(points: list[airsim.Vector3r], max_points: int) -> list[airsim.Vector3r]:
    if len(points) <= max_points:
        return points
    idx = np.linspace(0, len(points) - 1, max_points).astype(int)
    return [points[i] for i in idx]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--csv",
        type=Path,
        required=True,
        help="episode_*.csv from scripts.evaluation.evaluate_single",
    )
    parser.add_argument("--airsim-ip", default="127.0.0.1")
    parser.add_argument("--airsim-port", type=int, default=41451)
    parser.add_argument("--max-points", type=int, default=400, help="downsample for simPlot")
    parser.add_argument("--thickness", type=float, default=12.0)
    parser.add_argument("--duration", type=float, default=180.0, help="seconds to keep overlays")
    parser.add_argument("--color", default="orange", choices=["orange", "red", "cyan", "green"])
    args = parser.parse_args()

    if not args.csv.is_file():
        raise FileNotFoundError(args.csv)

    colors = {
        "orange": [1.0, 0.45, 0.0, 1.0],
        "red": [1.0, 0.1, 0.1, 1.0],
        "cyan": [0.0, 0.85, 0.95, 1.0],
        "green": [0.1, 0.8, 0.3, 1.0],
    }

    points = subsample(load_xyz(args.csv), args.max_points)
    print(f"Loaded {args.csv} -> {len(points)} plot points")

    client = airsim.MultirotorClient(ip=args.airsim_ip, port=args.airsim_port)
    client.confirmConnection()

    # Persistent debug overlays (duration_sec)
    client.simPlotLineList(
        points,
        color_rgba=colors[args.color],
        thickness=args.thickness,
        duration=args.duration,
        is_persistent=False,
    )
    client.simPlotPoints(
        [points[0]],
        color_rgba=[0.0, 0.8, 0.3, 1.0],
        size=20,
        duration=args.duration,
        is_persistent=False,
    )
    client.simPlotPoints(
        [points[-1]],
        color_rgba=[0.8, 0.2, 0.8, 1.0],
        size=20,
        duration=args.duration,
        is_persistent=False,
    )

    # Mark closest-obstacle step if column exists
    with args.csv.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if rows and "obstacle_dist_m" in rows[0]:
        dists = []
        for r in rows:
            raw = (r.get("obstacle_dist_m") or "").strip()
            if raw:
                dists.append(float(raw))
            else:
                dists.append(float("inf"))
        if any(np.isfinite(dists)):
            i = int(np.argmin(dists))
            scale = (len(points) - 1) / max(1, len(rows) - 1)
            pi = int(round(i * scale))
            client.simPlotPoints(
                [points[pi]],
                color_rgba=[1.0, 1.0, 0.0, 1.0],
                size=28,
                duration=args.duration,
                is_persistent=False,
            )
            print(f"Closest obstacle marker near step {i}, d={dists[i]:.3f} m")

    print(
        f"Trajectory plotted for {args.duration:.0f}s. "
        "Orbit to a local viewpoint and capture screenshots."
    )
    # Keep process alive briefly so users know it finished
    time.sleep(1.0)


if __name__ == "__main__":
    main()
