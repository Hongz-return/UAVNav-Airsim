"""
Export an AirSim voxel map and overlay evaluated UAV trajectories.

Examples:
    python tools/map_generation/map_generation.py --mode export
    python tools/map_generation/map_generation.py --scene Tunnel --episodes 20 --eval-tag Obs
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


DEFAULT_MODEL_ORDER = [
    "SAC_FrameStack4",
    "LSTM",
    "Mamba",
    "Mamba_CoordAtt",
]

MODEL_COLORS = {
    "SAC_FrameStack4": "#4C78A8",
    "LSTM": "#F4A261",
    "Mamba": "#4CC9F0",
    "Mamba_CoordAtt": "#2A9D8F",
}

MODEL_LABELS = {
    "SAC_FrameStack4": "SAC",
    "Mamba_CoordAtt": "MA-Mamba",
}

STATUS_COLORS = {
    "success": "#2A9D8F",
    "collision": "#D62828",
    "timeout": "#E9C46A",
    "out_of_bounds": "#6C757D",
}

STATUS_MARKERS = {
    "success": "o",
    "collision": "X",
    "timeout": "s",
    "out_of_bounds": "^",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export voxel maps and overlay UAV trajectories")
    parser.add_argument("--mode", choices=["plot", "export", "both"], default="plot")
    parser.add_argument("--binvox", type=str, default="tools/map_generation/map.binvox")
    parser.add_argument("--scene", type=str, default="Tunnel", help="Scene log directory under logs/")
    parser.add_argument("--episodes", type=int, default=20, help="Prefer eval_<episodes> if available")
    parser.add_argument("--eval-tag", type=str, default="Obs", help="Optional eval suffix, e.g. Obs")
    parser.add_argument("--models", nargs="+", default=None, help="Optional model subdirectories to plot")
    parser.add_argument("--outdir", type=str, default=None, help="Output directory for rendered figures")
    parser.add_argument("--max-traj", type=int, default=5, help="Max trajectories per model to overlay")
    parser.add_argument("--max-voxels", type=int, default=18000, help="Max occupied voxels to draw in 3D")
    parser.add_argument(
        "--terminal-types",
        nargs="+",
        default=["success", "collision", "timeout", "out_of_bounds"],
        help="Terminal markers to display",
    )
    parser.add_argument("--resolution", type=float, default=0.5, help="Voxel resolution used during export")
    parser.add_argument("--center", nargs=3, type=float, default=[0.0, 0.0, 0.0], help="Voxel export center")
    parser.add_argument("--voxel-size", nargs=3, type=float, default=[200.0, 200.0, 100.0], help="Voxel grid size")
    return parser.parse_args()


def export_voxel_grid(binvox_path: Path, center: List[float], voxel_size: List[float], resolution: float) -> None:
    try:
        import cosysairsim as airsim
    except ImportError as exc:
        raise ImportError("cosysairsim is required for --mode export/both") from exc

    client = airsim.VehicleClient()
    client.confirmConnection()
    binvox_path.parent.mkdir(parents=True, exist_ok=True)
    center_vec = airsim.Vector3r(center[0], center[1], center[2])
    client.simCreateVoxelGrid(
        center_vec,
        voxel_size[0],
        voxel_size[1],
        voxel_size[2],
        resolution,
        str(binvox_path.resolve()),
    )
    print(f"[DONE] Voxel grid exported to {binvox_path}")


def read_binvox(path: Path) -> Dict[str, object]:
    with path.open("rb") as handle:
        header = handle.readline().strip()
        if not header.startswith(b"#binvox"):
            raise ValueError(f"Invalid binvox header in {path}")

        dims = None
        translate = None
        scale = None
        while True:
            line = handle.readline()
            if not line:
                raise ValueError("Unexpected EOF while reading binvox header")
            line = line.strip()
            if line == b"data":
                break
            if line.startswith(b"dim"):
                dims = np.array([int(part) for part in line.split()[1:4]], dtype=int)
            elif line.startswith(b"translate"):
                translate = np.array([float(part) for part in line.split()[1:4]], dtype=float)
            elif line.startswith(b"scale"):
                scale = float(line.split()[1])

        if dims is None or translate is None or scale is None:
            raise ValueError(f"Incomplete binvox header in {path}")

        raw = np.frombuffer(handle.read(), dtype=np.uint8)
        values = raw[0::2]
        counts = raw[1::2]
        flat = np.repeat(values, counts).astype(bool)
        if flat.size != int(np.prod(dims)):
            raise ValueError(f"Decoded voxel count mismatch in {path}")

        # AirSim writes dims in x-z-y order; transpose to x-y-z for plotting.
        occupancy = flat.reshape((dims[0], dims[1], dims[2])).transpose((0, 2, 1))
        return {
            "occupancy": occupancy,
            "header_dims": dims,
            "dims_xyz": np.array(occupancy.shape, dtype=int),
            "translate": translate,
            "scale": scale,
        }


def display_model_name(model_name: str) -> str:
    return MODEL_LABELS.get(model_name, model_name)


def extract_eval_count(eval_dir: Path) -> int:
    try:
        return int(eval_dir.name.split("_", 1)[1].split("_", 1)[0])
    except (IndexError, ValueError):
        return -1


def order_models(model_names: List[str]) -> List[str]:
    preferred_index = {name: idx for idx, name in enumerate(DEFAULT_MODEL_ORDER)}
    return sorted(model_names, key=lambda name: (preferred_index.get(name, 999), name))


def discover_models(scene_root: Path) -> List[str]:
    discovered = []
    for name in DEFAULT_MODEL_ORDER:
        if (scene_root / name).is_dir():
            discovered.append(name)
    return discovered


def find_eval_dir(model_root: Path, episodes: int, eval_tag: Optional[str]) -> Optional[Path]:
    if not model_root.exists():
        return None

    expected = f"eval_{episodes}"
    if eval_tag:
        expected = f"{expected}_{eval_tag}"

    candidates = []
    for candidate in model_root.rglob("eval_*"):
        if candidate.is_dir() and (candidate / "trajectories.npy").exists():
            if candidate.name == expected:
                candidates.append(candidate)

    if not candidates:
        return None

    candidates.sort(key=lambda path: (extract_eval_count(path), path.stat().st_mtime), reverse=True)
    return candidates[0]


def infer_terminal_events(results: Dict, trajectories: np.ndarray) -> List[Dict[str, float]]:
    terminal_positions = results.get("terminal_positions", [])
    terminal_statuses = results.get("terminal_statuses", [])
    successes = results.get("successes", [])
    crashes = results.get("crashes", [])
    out_of_bounds = results.get("out_of_bounds", [])

    rows = []
    for idx, trajectory in enumerate(trajectories):
        if idx < len(terminal_positions):
            terminal_position = np.asarray(terminal_positions[idx], dtype=float)
        else:
            traj_np = np.asarray(trajectory, dtype=float)
            terminal_position = traj_np[-1] if traj_np.ndim == 2 and traj_np.shape[0] > 0 else np.full(3, np.nan)

        if idx < len(terminal_statuses):
            terminal_type = str(terminal_statuses[idx])
        else:
            is_success = idx < len(successes) and float(successes[idx]) == 1.0
            is_crash = idx < len(crashes) and float(crashes[idx]) == 1.0
            is_oob = idx < len(out_of_bounds) and float(out_of_bounds[idx]) == 1.0
            terminal_type = "success" if is_success else ("collision" if is_crash else ("out_of_bounds" if is_oob else "timeout"))

        rows.append(
            {
                "episode": idx + 1,
                "terminal_type": terminal_type,
                "terminal_x": float(terminal_position[0]),
                "terminal_y": float(terminal_position[1]),
                "terminal_z": float(terminal_position[2]),
            }
        )
    return rows


def load_eval_bundle(eval_dir: Path) -> Dict[str, object]:
    results = np.load(eval_dir / "results.npy", allow_pickle=True).item()
    trajectories = np.load(eval_dir / "trajectories.npy", allow_pickle=True)

    terminal_csv = eval_dir / "terminal_events.csv"
    if terminal_csv.exists():
        terminal_events = []
        with terminal_csv.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                terminal_events.append(
                    {
                        "episode": int(row["episode"]),
                        "terminal_type": row["terminal_type"],
                        "terminal_x": float(row["terminal_x"]),
                        "terminal_y": float(row["terminal_y"]),
                        "terminal_z": float(row["terminal_z"]),
                        "start_x": float(row["start_x"]),
                        "start_y": float(row["start_y"]),
                        "start_z": float(row["start_z"]),
                        "goal_x": float(row["goal_x"]),
                        "goal_y": float(row["goal_y"]),
                        "goal_z": float(row["goal_z"]),
                    }
                )
    else:
        terminal_events = infer_terminal_events(results, trajectories)

    if terminal_events:
        start = np.array(
            [
                terminal_events[0].get("start_x", np.nan),
                terminal_events[0].get("start_y", np.nan),
                terminal_events[0].get("start_z", np.nan),
            ],
            dtype=float,
        )
        goal = np.array(
            [
                terminal_events[0].get("goal_x", np.nan),
                terminal_events[0].get("goal_y", np.nan),
                terminal_events[0].get("goal_z", np.nan),
            ],
            dtype=float,
        )
    else:
        start_positions = results.get("start_positions", [])
        goal_positions = results.get("goal_positions", [])
        start = np.asarray(start_positions[0], dtype=float) if start_positions else np.full(3, np.nan)
        goal = np.asarray(goal_positions[0], dtype=float) if goal_positions else np.full(3, np.nan)

    return {
        "results": results,
        "trajectories": trajectories,
        "terminal_events": terminal_events,
        "start": start,
        "goal": goal,
    }


def load_scene_bundles(scene: str, episodes: int, eval_tag: Optional[str], models: Optional[List[str]]) -> Dict[str, Dict[str, object]]:
    scene_root = Path("logs") / scene
    model_names = order_models(models) if models else discover_models(scene_root)
    bundles: Dict[str, Dict[str, object]] = {}
    for model_name in model_names:
        eval_dir = find_eval_dir(scene_root / model_name, episodes, eval_tag)
        if eval_dir is None:
            print(f"[SKIP] {model_name}: no matching eval directory found")
            continue
        bundles[model_name] = load_eval_bundle(eval_dir)
        print(f"[OK] {model_name}: {eval_dir}")
    if not bundles:
        raise FileNotFoundError(f"No evaluation bundles found under logs/{scene}")
    return bundles


def select_representative_indices(bundle: Dict[str, object], max_traj: int) -> List[int]:
    terminal_events = bundle["terminal_events"]
    path_lengths = bundle["results"].get("path_lengths", [])

    status_to_indices: Dict[str, List[int]] = {}
    for idx, event in enumerate(terminal_events):
        status_to_indices.setdefault(event["terminal_type"], []).append(idx)

    chosen: List[int] = []
    for status in ("success", "collision", "timeout", "out_of_bounds"):
        indices = status_to_indices.get(status, [])
        if not indices:
            continue
        ranked = sorted(indices, key=lambda i: path_lengths[i] if i < len(path_lengths) else float("inf"))
        chosen.append(ranked[len(ranked) // 2])
        if len(chosen) >= max_traj:
            return chosen[:max_traj]

    success_indices = status_to_indices.get("success", [])
    if success_indices:
        ranked = sorted(success_indices, key=lambda i: path_lengths[i] if i < len(path_lengths) else float("inf"))
        for idx in ranked:
            if idx not in chosen:
                chosen.append(idx)
            if len(chosen) >= max_traj:
                break

    if len(chosen) < max_traj:
        for idx in range(len(bundle["trajectories"])):
            if idx not in chosen:
                chosen.append(idx)
            if len(chosen) >= max_traj:
                break

    return chosen[:max_traj]


def voxel_extent(translate: np.ndarray, dims_xyz: np.ndarray, resolution: float) -> List[float]:
    return [
        float(translate[0]),
        float(translate[0] + dims_xyz[0] * resolution),
        float(translate[1]),
        float(translate[1] + dims_xyz[1] * resolution),
        float(translate[2]),
        float(translate[2] + dims_xyz[2] * resolution),
    ]


def sample_occupied_points(occupancy: np.ndarray, translate: np.ndarray, resolution: float, max_voxels: int) -> np.ndarray:
    occupied = np.argwhere(occupancy)
    if occupied.size == 0:
        return np.zeros((0, 3), dtype=float)
    if occupied.shape[0] > max_voxels:
        indices = np.linspace(0, occupied.shape[0] - 1, max_voxels, dtype=int)
        occupied = occupied[indices]
    world = translate + (occupied.astype(float) + 0.5) * resolution
    return world


def plot_topdown_map(
    binvox: Dict[str, object],
    bundles: Dict[str, Dict[str, object]],
    resolution: float,
    output_path: Path,
    max_traj: int,
    terminal_types: List[str],
) -> None:
    occupancy = binvox["occupancy"]
    translate = np.asarray(binvox["translate"], dtype=float)
    dims_xyz = np.asarray(binvox["dims_xyz"], dtype=int)
    extent = voxel_extent(translate, dims_xyz, resolution)

    topdown = occupancy.any(axis=2).T.astype(float)

    fig, ax = plt.subplots(figsize=(10.5, 6.2), dpi=260)
    ax.imshow(
        topdown,
        origin="lower",
        extent=[extent[0], extent[1], extent[2], extent[3]],
        cmap="Greys",
        alpha=0.42,
        interpolation="nearest",
    )

    for model_name, bundle in bundles.items():
        color = MODEL_COLORS.get(model_name, "#444444")
        rep_indices = select_representative_indices(bundle, max_traj)
        for traj_idx in rep_indices:
            trajectory = bundle["trajectories"][traj_idx]
            traj_np = np.asarray(trajectory, dtype=float)
            if traj_np.ndim != 2 or traj_np.shape[0] < 2 or traj_np.shape[1] < 2:
                continue
            terminal_type = bundle["terminal_events"][traj_idx]["terminal_type"] if traj_idx < len(bundle["terminal_events"]) else "unknown"
            is_success = terminal_type == "success"
            ax.plot(
                traj_np[:, 0],
                traj_np[:, 1],
                color=color,
                linewidth=2.8 if is_success else 1.8,
                alpha=0.98 if is_success else 0.72,
                linestyle="-" if is_success else "--",
            )

        for event in bundle["terminal_events"]:
            terminal_type = event["terminal_type"]
            if terminal_type not in terminal_types:
                continue
            ax.scatter(
                event["terminal_x"],
                event["terminal_y"],
                marker=STATUS_MARKERS.get(terminal_type, "o"),
                s=42,
                color=STATUS_COLORS.get(terminal_type, "#333333"),
                edgecolors="white",
                linewidths=0.45,
                zorder=5,
            )

        start = bundle["start"]
        goal = bundle["goal"]
        if np.all(np.isfinite(start[:2])):
            ax.scatter(start[0], start[1], marker="*", s=180, color="#111111", edgecolors="white", linewidths=0.6, zorder=6)
        if np.all(np.isfinite(goal[:2])):
            ax.scatter(goal[0], goal[1], marker="P", s=130, color="#6A4C93", edgecolors="white", linewidths=0.6, zorder=6)

    model_handles = [
        Line2D([], [], color=MODEL_COLORS.get(model_name, "#444444"), linewidth=2.2, label=display_model_name(model_name))
        for model_name in bundles
    ]
    status_handles = [
        Line2D([], [], color=STATUS_COLORS[name], marker=STATUS_MARKERS[name], linestyle="None", markersize=8, label=name.replace("_", " ").title())
        for name in ("success", "collision", "timeout", "out_of_bounds")
        if name in terminal_types
    ]
    special_handles = [
        Line2D([], [], color="#111111", marker="*", linestyle="None", markersize=10, label="Start"),
        Line2D([], [], color="#6A4C93", marker="P", linestyle="None", markersize=8, label="Goal"),
    ]

    ax.legend(
        handles=model_handles + special_handles + status_handles,
        loc="upper right",
        frameon=True,
        fontsize=8.5,
        ncol=2,
        handlelength=1.8,
        columnspacing=0.9,
    )
    ax.set_title("Top-Down Obstacle Map with UAV Trajectories")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_aspect("equal")
    ax.grid(alpha=0.22, linestyle=":")

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_3d_map(
    binvox: Dict[str, object],
    bundles: Dict[str, Dict[str, object]],
    resolution: float,
    output_path: Path,
    max_traj: int,
    max_voxels: int,
    terminal_types: List[str],
) -> None:
    occupancy = binvox["occupancy"]
    translate = np.asarray(binvox["translate"], dtype=float)
    dims_xyz = np.asarray(binvox["dims_xyz"], dtype=int)
    extent = voxel_extent(translate, dims_xyz, resolution)
    occupied_world = sample_occupied_points(occupancy, translate, resolution, max_voxels)

    fig = plt.figure(figsize=(11.5, 7.0), dpi=240)
    ax = fig.add_subplot(111, projection="3d")

    if occupied_world.size > 0:
        ax.scatter(
            occupied_world[:, 0],
            occupied_world[:, 1],
            occupied_world[:, 2],
            s=1.2,
            color="#6C757D",
            alpha=0.045,
            depthshade=False,
        )

    for model_name, bundle in bundles.items():
        color = MODEL_COLORS.get(model_name, "#444444")
        rep_indices = select_representative_indices(bundle, max_traj)
        for traj_idx in rep_indices:
            trajectory = bundle["trajectories"][traj_idx]
            traj_np = np.asarray(trajectory, dtype=float)
            if traj_np.ndim != 2 or traj_np.shape[0] < 2 or traj_np.shape[1] < 3:
                continue
            terminal_type = bundle["terminal_events"][traj_idx]["terminal_type"] if traj_idx < len(bundle["terminal_events"]) else "unknown"
            is_success = terminal_type == "success"
            ax.plot(
                traj_np[:, 0],
                traj_np[:, 1],
                traj_np[:, 2],
                color=color,
                linewidth=2.8 if is_success else 1.9,
                alpha=0.98 if is_success else 0.70,
                linestyle="-" if is_success else "--",
            )

        for event in bundle["terminal_events"]:
            terminal_type = event["terminal_type"]
            if terminal_type not in terminal_types:
                continue
            ax.scatter(
                event["terminal_x"],
                event["terminal_y"],
                event["terminal_z"],
                marker=STATUS_MARKERS.get(terminal_type, "o"),
                s=38,
                color=STATUS_COLORS.get(terminal_type, "#333333"),
                depthshade=False,
            )

        start = bundle["start"]
        goal = bundle["goal"]
        if np.all(np.isfinite(start[:3])):
            ax.scatter(start[0], start[1], start[2], marker="*", s=200, color="#111111", depthshade=False)
        if np.all(np.isfinite(goal[:3])):
            ax.scatter(goal[0], goal[1], goal[2], marker="P", s=140, color="#6A4C93", depthshade=False)

    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.set_zlim(extent[4], extent[5])
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title("3D Obstacle Map with UAV Trajectories")
    ax.view_init(elev=24, azim=-58)

    handles = [Line2D([], [], color=MODEL_COLORS.get(model_name, "#444444"), linewidth=2.4, label=display_model_name(model_name)) for model_name in bundles]
    fig.legend(handles=handles, loc="upper center", ncol=max(1, len(handles)), frameon=False, bbox_to_anchor=(0.5, 0.98), columnspacing=1.2)

    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    binvox_path = Path(args.binvox)

    if args.mode in {"export", "both"}:
        export_voxel_grid(binvox_path, args.center, args.voxel_size, args.resolution)

    if args.mode in {"plot", "both"}:
        if not binvox_path.exists():
            raise FileNotFoundError(f"Binvox file not found: {binvox_path}")

        bundles = load_scene_bundles(args.scene, args.episodes, args.eval_tag, args.models)
        binvox = read_binvox(binvox_path)

        if args.outdir:
            output_dir = Path(args.outdir)
        else:
            suffix = args.eval_tag if args.eval_tag else str(args.episodes)
            output_dir = Path("logs") / args.scene / f"map_overlay_{suffix}"
        output_dir.mkdir(parents=True, exist_ok=True)

        topdown_path = output_dir / "obstacle_map_2d.png"
        map3d_path = output_dir / "obstacle_map_3d.png"
        plot_topdown_map(binvox, bundles, args.resolution, topdown_path, args.max_traj, args.terminal_types)
        plot_3d_map(binvox, bundles, args.resolution, map3d_path, args.max_traj, args.max_voxels, args.terminal_types)

        print(f"[DONE] 2D map overlay: {topdown_path}")
        print(f"[DONE] 3D map overlay: {map3d_path}")


if __name__ == "__main__":
    main()
