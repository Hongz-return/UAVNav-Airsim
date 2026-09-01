#!/usr/bin/env python3
"""Plot failure modes from the formal ObsTunnel N=100 evaluation matrix.

The primary experiment matrix contains three independently trained seeds per
method and 100 frozen-model evaluation episodes per seed.  The dynamic
MA-SAC repeat batch is deliberately excluded from the main matrix.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
BATCH_A = (
    ROOT
    / "Logs"
    / "EvaData"
    / "FormalBatches"
    / "ENV2_TunnelObs_and_ENV3_TunnelActivate"
    / "summary_all_models.csv"
)
BATCH_B = (
    ROOT
    / "Logs"
    / "EvaData"
    / "FormalBatches"
    / "ENV1_Tunnel_and_ENV3_TunnelActivate"
    / "summary_all_models.csv"
)
DEFAULT_LAYOUT_CSV = (
    ROOT
    / "Logs"
    / "EvaData"
    / "Results"
    / "tunnel_activate"
    / "transformer"
    / "seed_0"
    / "2026_07_28_07_49_00_328963_Multirotor_cnn_SAC"
    / "eval_1_best_2026_07_30_14_21_31_237881"
    / "scene_object_tracks.csv"
)

METHOD_ORDER = ["sac", "lstm", "transformer", "mamba", "coordatt", "ma"]
METHOD_LABELS = {
    "sac": "SAC",
    "lstm": "LSTM-SAC",
    "transformer": "Transformer-SAC",
    "mamba": "Mamba-SAC",
    "coordatt": "CoordAtt-SAC",
    "ma": "MA-SAC",
}
OUTCOME_FIELDS = {
    "success": "success_rate_pct",
    "collision": "collision_rate_pct",
    "timeout": "timeout_rate_pct",
    "out_of_bounds": "out_of_bounds_rate_pct",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--paper-root",
        type=Path,
        default=ROOT.parent / "paper",
        help="Paper directory containing images/ and review/.",
    )
    parser.add_argument(
        "--layout-csv",
        type=Path,
        default=DEFAULT_LAYOUT_CSV,
        help="ObsTunnel t=0 scene-object snapshot used only as a layout backdrop.",
    )
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def select_primary_dynamic_rows() -> list[dict[str, str]]:
    rows_a = [
        row for row in read_csv(BATCH_A) if row["scene"] == "tunnel_activate"
    ]
    rows_b = [
        row
        for row in read_csv(BATCH_B)
        if row["scene"] == "tunnel_activate" and row["method"] != "ma"
    ]
    selected = rows_a + rows_b
    key_counts: dict[tuple[str, int], int] = defaultdict(int)
    for row in selected:
        key_counts[(row["method"], int(row["seed"]))] += 1
    expected = {(method, seed) for method in METHOD_ORDER for seed in range(3)}
    if set(key_counts) != expected or any(count != 1 for count in key_counts.values()):
        raise RuntimeError("Primary dynamic matrix is not exactly 6 methods x 3 seeds.")
    return selected


def sample_sd(values: list[float]) -> float:
    return float(np.std(values, ddof=1)) if len(values) > 1 else 0.0


def summarize_methods(rows: list[dict[str, str]]) -> dict[str, dict]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row["method"]].append(row)

    summary: dict[str, dict] = {}
    for method in METHOD_ORDER:
        method_rows = sorted(grouped[method], key=lambda row: int(row["seed"]))
        episodes = sum(int(row["episodes"]) for row in method_rows)
        metrics: dict[str, dict] = {}
        for outcome, field in OUTCOME_FIELDS.items():
            seed_rates = [float(row[field]) for row in method_rows]
            count = sum(
                round(float(row[field]) * int(row["episodes"]) / 100.0)
                for row in method_rows
            )
            metrics[outcome] = {
                "count": int(count),
                "rate_pct": float(100.0 * count / episodes),
                "seed_mean_pct": float(np.mean(seed_rates)),
                "seed_sample_sd_pct": sample_sd(seed_rates),
            }
        summary[method] = {
            "label": METHOD_LABELS[method],
            "seeds": [int(row["seed"]) for row in method_rows],
            "episodes": episodes,
            "outcomes": metrics,
        }
    return summary


def resolve_eval_dir(row: dict[str, str]) -> Path:
    path = Path(row["evaluation_dir"])
    return path if path.is_absolute() else ROOT / path


def read_ma_endpoints(rows: list[dict[str, str]]) -> list[dict]:
    endpoints: list[dict] = []
    for row in rows:
        if row["method"] != "ma":
            continue
        eval_dir = resolve_eval_dir(row)
        trajectory_dir = eval_dir / "trajectories_csv"
        files = sorted(trajectory_dir.glob("episode_*.csv"))
        if len(files) != int(row["episodes"]):
            raise RuntimeError(f"Unexpected trajectory count: {trajectory_dir}")
        for path in files:
            trajectory = read_csv(path)
            terminal = trajectory[-1]
            endpoints.append(
                {
                    "seed": int(row["seed"]),
                    "episode": int(terminal["episode"]),
                    "status": terminal["terminal_type"],
                    "x": float(terminal["x"]),
                    "y": float(terminal["y"]),
                    "z": float(terminal["z"]),
                    "step": int(terminal["step_index"]),
                    "obstacle_dist_m": (
                        float(terminal["obstacle_dist_m"])
                        if terminal["obstacle_dist_m"]
                        else None
                    ),
                    "max_x": max(float(point["x"]) for point in trajectory),
                }
            )
    if len(endpoints) != 300:
        raise RuntimeError(f"Expected 300 primary MA-SAC episodes, found {len(endpoints)}")
    return endpoints


def read_layout(path: Path) -> tuple[np.ndarray, np.ndarray]:
    rows = read_csv(path)
    tunnel, obstacles = [], []
    for row in rows:
        name = row["object_name"]
        point = (float(row["x"]), float(row["y"]))
        if "BP_Worker" in name:
            continue
        if "SM_TunCol1_Tunnel_base" in name:
            tunnel.append(point)
        else:
            obstacles.append(point)
    return np.asarray(tunnel), np.asarray(obstacles)


def quantiles(values: list[float]) -> dict[str, float]:
    q25, median, q75 = np.quantile(np.asarray(values, dtype=float), [0.25, 0.5, 0.75])
    return {"q25": float(q25), "median": float(median), "q75": float(q75)}


def summarize_ma_failures(endpoints: list[dict]) -> dict:
    collisions = [row for row in endpoints if row["status"] == "collision"]
    timeouts = [row for row in endpoints if row["status"] == "timeout"]
    out_of_bounds = [row for row in endpoints if row["status"] == "out_of_bounds"]
    segments = [(-50, 50), (50, 100), (100, 150), (150, 225)]
    hazards = []
    for lower, upper in segments:
        at_risk = sum(row["max_x"] >= lower for row in endpoints)
        failures = sum(lower <= row["x"] < upper for row in collisions)
        hazards.append(
            {
                "x_segment_m": [lower, upper],
                "at_risk": at_risk,
                "collisions": failures,
                "conditional_collision_pct": float(100.0 * failures / at_risk),
            }
        )
    terminal_distances = [
        row["obstacle_dist_m"]
        for row in collisions
        if row["obstacle_dist_m"] is not None
    ]
    return {
        "episodes": len(endpoints),
        "successes": sum(row["status"] == "success" for row in endpoints),
        "collisions": len(collisions),
        "timeouts": len(timeouts),
        "out_of_bounds": len(out_of_bounds),
        "collision_terminal_x_m": quantiles([row["x"] for row in collisions]),
        "collision_terminal_step": quantiles([row["step"] for row in collisions]),
        "collision_terminal_obstacle_distance_m": quantiles(terminal_distances),
        "segment_conditioned_collision": hazards,
        "interpretation_limit": (
            "Formal N=100 evaluations did not record per-step scene-object poses. "
            "Collision terminal locations can be mapped to the same ObsTunnel layout, "
            "but individual collisions cannot be assigned to a BP_Worker or a static object."
        ),
    }


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.titleweight": "bold",
            "axes.labelsize": 9,
            "legend.fontsize": 7.5,
            "legend.frameon": False,
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.15,
            "grid.linestyle": "-",
        }
    )


def plot(
    method_summary: dict[str, dict],
    endpoints: list[dict],
    tunnel: np.ndarray,
    static_obstacles: np.ndarray,
    output_dir: Path,
) -> None:
    configure_style()
    fig, (ax_outcomes, ax_map) = plt.subplots(
        1, 2, figsize=(7.2, 3.55), gridspec_kw={"width_ratios": [1.0, 1.22]}
    )

    y = np.arange(len(METHOD_ORDER))
    left = np.zeros(len(METHOD_ORDER))
    colors = {
        "success": "#0072B2",
        "collision": "#D55E00",
        "timeout": "#999999",
        "out_of_bounds": "#CC79A7",
    }
    labels = {
        "success": "Success",
        "collision": "Collision",
        "timeout": "Timeout",
        "out_of_bounds": "Out of bounds",
    }
    for outcome in OUTCOME_FIELDS:
        values = np.asarray(
            [method_summary[method]["outcomes"][outcome]["rate_pct"] for method in METHOD_ORDER]
        )
        if not np.any(values):
            continue
        ax_outcomes.barh(
            y,
            values,
            left=left,
            height=0.63,
            color=colors[outcome],
            edgecolor="white",
            linewidth=0.5,
            label=labels[outcome],
        )
        if outcome in {"success", "collision"}:
            for index, value in enumerate(values):
                if value >= 9:
                    ax_outcomes.text(
                        left[index] + value / 2,
                        index,
                        f"{value:.1f}",
                        ha="center",
                        va="center",
                        fontsize=7,
                        color="white",
                        fontweight="bold",
                    )
        left += values
    ax_outcomes.set_yticks(y)
    ax_outcomes.set_yticklabels([METHOD_LABELS[method] for method in METHOD_ORDER])
    ax_outcomes.get_yticklabels()[-1].set_fontweight("bold")
    ax_outcomes.set_xlim(0, 100)
    ax_outcomes.set_xlabel("Outcome rate over 300 episodes (%)")
    ax_outcomes.set_title("(a) Formal evaluation outcomes")
    ax_outcomes.invert_yaxis()
    ax_outcomes.legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=3)

    if len(tunnel):
        ax_map.scatter(
            tunnel[:, 0],
            tunnel[:, 1],
            s=10,
            marker="s",
            color="#B0BEC5",
            alpha=0.28,
            linewidths=0,
            label="Tunnel modules",
            zorder=1,
        )
    if len(static_obstacles):
        ax_map.scatter(
            static_obstacles[:, 0],
            static_obstacles[:, 1],
            s=18,
            marker="^",
            color="#E69F00",
            alpha=0.60,
            linewidths=0,
            label="Static layout",
            zorder=2,
        )
    collision_points = [row for row in endpoints if row["status"] == "collision"]
    ax_map.scatter(
        [row["x"] for row in collision_points],
        [row["y"] for row in collision_points],
        s=17,
        marker="x",
        color="#D55E00",
        alpha=0.58,
        linewidths=0.9,
        label="Collision terminal",
        zorder=4,
    )
    ax_map.scatter(
        [0.0],
        [0.0],
        s=45,
        marker="o",
        color="#111111",
        edgecolor="white",
        linewidth=0.6,
        label="Start",
        zorder=5,
    )
    ax_map.scatter(
        [201.3],
        [53.9],
        s=75,
        marker="*",
        color="#CC79A7",
        edgecolor="white",
        linewidth=0.5,
        label="Goal",
        zorder=5,
    )
    ax_map.set_xlabel("X (m)")
    ax_map.set_ylabel("Y (m)")
    ax_map.set_title("(b) MA-SAC collision terminals (145/300)")
    ax_map.set_aspect("equal", adjustable="box")
    ax_map.legend(loc="upper center", bbox_to_anchor=(0.5, -0.25), ncol=3)

    fig.subplots_adjust(bottom=0.36, wspace=0.30)
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / "dynamic_failure_analysis.pdf")
    fig.savefig(output_dir / "dynamic_failure_analysis.png", dpi=300)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    rows = select_primary_dynamic_rows()
    method_summary = summarize_methods(rows)
    endpoints = read_ma_endpoints(rows)
    tunnel, static_obstacles = read_layout(args.layout_csv)
    ma_failures = summarize_ma_failures(endpoints)

    image_dir = args.paper_root / "images" / "ObsTunnel"
    review_dir = args.paper_root / "review"
    plot(method_summary, endpoints, tunnel, static_obstacles, image_dir)

    payload = {
        "scope": {
            "scene": "ObsTunnel",
            "primary_models": 18,
            "episodes_per_model": 100,
            "total_episodes_per_method": 300,
            "summary_batches": [str(BATCH_A), str(BATCH_B)],
            "excluded": (
                "Dynamic MA-SAC training batch 2026_07_28_15_55_32_685674 "
                "(three independent supplementary repeats)."
            ),
            "layout_csv": str(args.layout_csv),
        },
        "methods": method_summary,
        "ma_sac_failure_analysis": ma_failures,
    }
    review_dir.mkdir(parents=True, exist_ok=True)
    output_json = review_dir / "dynamic_failure_analysis_data.json"
    output_json.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Wrote {image_dir / 'dynamic_failure_analysis.pdf'}")
    print(f"Wrote {image_dir / 'dynamic_failure_analysis.png'}")
    print(f"Wrote {output_json}")


if __name__ == "__main__":
    main()
