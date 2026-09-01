"""Capture CoordAtt and Mamba responses during real ObsTunnel rollouts.

This is a read-only diagnostic: it loads a frozen MA-SAC checkpoint, runs
deterministic AirSim episodes, and records forward-hook activations without
changing model parameters.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from configparser import ConfigParser
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


UAV_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(UAV_ROOT))
os.chdir(UAV_ROOT)

from gym_env.networks.mamba_policy import CoordAttention2d
from scripts.evaluation.evaluate_single import (
    apply_evaluation_overrides,
    build_evaluation_scenarios,
    get_base_env,
)
from tools.runtime_support import build_runtime_env, load_model


DEFAULT_MODEL_DIR = UAV_ROOT / (
    "Logs/TrainData/Runs/tunnel_activate/ma/seed_0/"
    "2026_07_28_16_54_21_375604_Multirotor_cnn_SAC"
)
DEFAULT_EXPERIMENT_DIR = Path(
    r"D:\code\ZipFIle\paper\experiments\real-obstunnel-intermediate-features"
)
DEFAULT_FIGURE_DIR = Path(r"D:\code\ZipFIle\paper\images\ObsTunnel")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize MA-SAC intermediate features on real ObsTunnel rollouts."
    )
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--scenario-seed", type=int, default=2026)
    parser.add_argument("--yaw-range-deg", type=float, default=10.0)
    parser.add_argument("--airsim-ip", default="127.0.0.1")
    parser.add_argument("--airsim-port", type=int, default=41451)
    parser.add_argument("--experiment-dir", type=Path, default=DEFAULT_EXPERIMENT_DIR)
    parser.add_argument("--figure-dir", type=Path, default=DEFAULT_FIGURE_DIR)
    parser.add_argument(
        "--env-label",
        default="ObsTunnel",
        help="Scene label written into summary/figure titles (e.g. Tunnel-Obs / ENV2).",
    )
    parser.add_argument(
        "--replot-only",
        action="store_true",
        help="Rebuild the figure from saved rollout data without connecting to AirSim.",
    )
    return parser.parse_args()


def rankdata(values: np.ndarray) -> np.ndarray:
    """Return average ranks, including deterministic handling of ties."""
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        stop = start + 1
        while stop < len(values) and sorted_values[stop] == sorted_values[start]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * (start + stop - 1) + 1.0
        start = stop
    return ranks


def spearman_correlation(x: np.ndarray, y: np.ndarray) -> float:
    """Compute Spearman correlation without an optional SciPy dependency."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    valid = np.isfinite(x) & np.isfinite(y)
    if valid.sum() < 3:
        return float("nan")
    x_rank = rankdata(x[valid])
    y_rank = rankdata(y[valid])
    if np.std(x_rank) == 0 or np.std(y_rank) == 0:
        return float("nan")
    return float(np.corrcoef(x_rank, y_rank)[0, 1])


class CoordAttentionHook:
    """Reconstruct the CoordAtt height-width gate for the latest real frame."""

    def __init__(self) -> None:
        self.latest: np.ndarray | None = None
        self.handle = None

    def __call__(self, module, inputs, _output) -> None:
        with torch.no_grad():
            features = inputs[0]
            _, _, height, width = features.shape
            pooled_h = features.mean(dim=3, keepdim=True)
            pooled_w = features.mean(dim=2, keepdim=True).permute(0, 1, 3, 2)
            mixed = module.shared(torch.cat([pooled_h, pooled_w], dim=2))
            part_h, part_w = torch.split(mixed, [height, width], dim=2)
            part_w = part_w.permute(0, 1, 3, 2)
            gate_h = torch.sigmoid(module.attn_h(part_h))
            gate_w = torch.sigmoid(module.attn_w(part_w))
            maps = (gate_h * gate_w).mean(dim=1)
            self.latest = maps[-1].detach().cpu().numpy().astype(np.float32)

    def register(self, module) -> None:
        self.handle = module.register_forward_hook(self)

    def remove(self) -> None:
        if self.handle is not None:
            self.handle.remove()


class MambaContextHook:
    """Capture the latest input token and output context from the actor backend."""

    def __init__(self) -> None:
        self.latest_input: np.ndarray | None = None
        self.latest_output: np.ndarray | None = None
        self.handle = None

    def __call__(self, _module, inputs, output) -> None:
        input_tensor = inputs[0]
        output_tensor = output[0] if isinstance(output, tuple) else output
        self.latest_input = (
            input_tensor[:, -1, :].detach().cpu().numpy()[0].astype(np.float32)
        )
        if output_tensor.ndim == 3:
            output_tensor = output_tensor[:, -1, :]
        self.latest_output = output_tensor.detach().cpu().numpy()[0].astype(np.float32)

    def register(self, module) -> None:
        self.handle = module.register_forward_hook(self)

    def remove(self) -> None:
        if self.handle is not None:
            self.handle.remove()


def find_coordatt(policy) -> CoordAttention2d:
    for module in policy.shared_cnn.modules():
        if isinstance(module, CoordAttention2d):
            return module
    raise RuntimeError("The loaded policy does not contain a CoordAttention2d module.")


def depth_frame_from_obs(obs: dict[str, np.ndarray]) -> np.ndarray:
    image = np.asarray(obs["image"][0])
    return np.squeeze(image).astype(np.uint8)


def robust_depth_metres(depth_uint8: np.ndarray, max_depth: float) -> float:
    depth_m = depth_uint8.astype(np.float32) / 255.0 * max_depth
    valid = depth_m[(depth_m > 0.1) & (depth_m < max_depth)]
    return float(np.percentile(valid, 1)) if valid.size else float(max_depth)


def terminal_status(info: dict) -> str:
    if info.get("is_success", False):
        return "success"
    if info.get("is_crash", False):
        return "collision"
    if info.get("is_not_in_workspace", False):
        return "out_of_bounds"
    return "timeout"


def normalize_map(attention: np.ndarray) -> np.ndarray:
    low = float(np.min(attention))
    high = float(np.max(attention))
    if high - low <= 1e-12:
        return np.zeros_like(attention, dtype=np.float32)
    return ((attention - low) / (high - low)).astype(np.float32)


def run_rollouts(args: argparse.Namespace):
    config_path = args.model_dir / "config" / "config.ini"
    model_path = args.model_dir / "models" / "best_model.zip"
    if not config_path.exists():
        raise FileNotFoundError(config_path)
    if not model_path.exists():
        raise FileNotFoundError(model_path)
    if args.episodes < 1:
        raise ValueError("--episodes must be positive")

    cfg = ConfigParser()
    cfg.read(config_path)
    apply_evaluation_overrides(
        cfg,
        env_name=None,
        airsim_ip=args.airsim_ip,
        airsim_port=args.airsim_port,
    )
    max_depth = cfg.getfloat("environment", "max_depth_meters")
    scenarios, _ = build_evaluation_scenarios(
        cfg,
        args.episodes,
        args.scenario_seed,
        args.yaw_range_deg,
    )

    env = build_runtime_env(cfg)
    np.random.seed(args.scenario_seed)
    torch.manual_seed(args.scenario_seed)
    env.seed(args.scenario_seed)
    base_env = get_base_env(env)
    base_env.set_evaluation_scenarios(scenarios + [scenarios[-1]])
    model = load_model(cfg.get("options", "algo"), str(model_path), env, cfg=cfg)
    model.policy.set_training_mode(False)

    attention_hook = CoordAttentionHook()
    context_hook = MambaContextHook()
    attention_hook.register(find_coordatt(model.policy))
    context_hook.register(model.policy.actor.mamba)

    rows: list[dict] = []
    depth_frames: list[np.ndarray] = []
    attention_maps: list[np.ndarray] = []
    input_vectors: list[np.ndarray] = []
    context_vectors: list[np.ndarray] = []
    episode_summaries: list[dict] = []

    try:
        obs = env.reset()
        global_step = 0
        for episode in range(1, args.episodes + 1):
            scenario = dict(getattr(base_env, "current_scenario", None) or {})
            previous_depth = None
            previous_input = None
            previous_context = None
            episode_start = np.array([True])
            step = 0

            while True:
                current_depth = depth_frame_from_obs(obs)
                action, _ = model.predict(
                    obs,
                    deterministic=True,
                    episode_start=episode_start,
                )
                episode_start = np.array([False])
                step += 1
                global_step += 1

                if (
                    attention_hook.latest is None
                    or context_hook.latest_input is None
                    or context_hook.latest_output is None
                ):
                    raise RuntimeError("A forward hook did not capture the expected tensor.")

                attention = attention_hook.latest.copy()
                input_vector = context_hook.latest_input.copy()
                context_vector = context_hook.latest_output.copy()

                frame_change = (
                    float(np.mean(np.abs(current_depth.astype(np.float32) - previous_depth)))
                    / 255.0
                    if previous_depth is not None
                    else float("nan")
                )
                input_change = (
                    float(np.linalg.norm(input_vector - previous_input))
                    if previous_input is not None
                    else float("nan")
                )
                context_change = (
                    float(np.linalg.norm(context_vector - previous_context))
                    if previous_context is not None
                    else float("nan")
                )

                next_obs, rewards, dones, infos = env.step(action)
                info = infos[0] if infos else {}
                position = info.get("terminal_position", info.get("position"))
                if position is None:
                    position = base_env.dynamic_model.get_position()
                position = np.asarray(position, dtype=np.float32)

                rows.append(
                    {
                        "global_step": global_step,
                        "episode": episode,
                        "step": step,
                        "scenario_id": scenario.get("scenario_id"),
                        "scenario_yaw_deg": scenario.get("yaw_deg"),
                        "reward": float(rewards[0]),
                        "action_x": float(action[0][0]),
                        "action_y": float(action[0][1]),
                        "action_yaw": float(action[0][2]),
                        "x": float(position[0]),
                        "y": float(position[1]),
                        "z": float(position[2]),
                        "robust_depth_m": robust_depth_metres(current_depth, max_depth),
                        "frame_change": frame_change,
                        "input_feature_change": input_change,
                        "mamba_context_change": context_change,
                        "attention_raw_mean": float(np.mean(attention)),
                        "attention_raw_std": float(np.std(attention)),
                        "terminal": bool(dones[0]),
                        "outcome": terminal_status(info) if dones[0] else "",
                    }
                )
                depth_frames.append(current_depth)
                attention_maps.append(attention)
                input_vectors.append(input_vector)
                context_vectors.append(context_vector)

                previous_depth = current_depth.astype(np.float32)
                previous_input = input_vector
                previous_context = context_vector
                obs = next_obs

                if dones[0]:
                    outcome = terminal_status(info)
                    episode_summaries.append(
                        {
                            "episode": episode,
                            "scenario_id": scenario.get("scenario_id"),
                            "scenario_yaw_deg": scenario.get("yaw_deg"),
                            "steps": step,
                            "outcome": outcome,
                        }
                    )
                    print(
                        f"Episode {episode}/{args.episodes}: {outcome}, "
                        f"steps={step}, scenario={scenario.get('scenario_id')}"
                    )
                    break
    finally:
        attention_hook.remove()
        context_hook.remove()
        env.close()

    return (
        rows,
        np.stack(depth_frames),
        np.stack(attention_maps),
        np.stack(input_vectors),
        np.stack(context_vectors),
        episode_summaries,
        config_path,
        model_path,
    )


def exploratory_edge_alignment(
    depth_frames: np.ndarray,
    attention_maps: np.ndarray,
) -> dict:
    """Post-registered audit of CoordAtt alignment with depth discontinuities."""
    correlations = []
    contrasts = []
    for depth_uint8, attention in zip(depth_frames, attention_maps):
        depth_m = depth_uint8.astype(np.float32) / 255.0 * 25.0
        grad_y, grad_x = np.gradient(depth_m)
        edge = np.sqrt(grad_x**2 + grad_y**2)
        attn_h, attn_w = attention.shape
        if edge.shape[0] % attn_h or edge.shape[1] % attn_w:
            raise ValueError(
                f"Depth shape {edge.shape} is not divisible by attention shape "
                f"{attention.shape}."
            )
        block_h = edge.shape[0] // attn_h
        block_w = edge.shape[1] // attn_w
        edge_grid = edge.reshape(
            attn_h,
            block_h,
            attn_w,
            block_w,
        ).mean(axis=(1, 3))
        attention_norm = normalize_map(attention)
        correlations.append(
            spearman_correlation(attention_norm.ravel(), edge_grid.ravel())
        )
        high = edge_grid >= np.percentile(edge_grid, 75)
        low = edge_grid <= np.percentile(edge_grid, 25)
        contrasts.append(
            float(np.mean(attention_norm[high]) - np.mean(attention_norm[low]))
        )

    correlations_array = np.asarray(correlations, dtype=float)
    contrasts_array = np.asarray(contrasts, dtype=float)
    return {
        "status": "exploratory_not_preregistered",
        "per_frame_attention_edge_spearman_median": float(
            np.nanmedian(correlations_array)
        ),
        "per_frame_attention_edge_spearman_mean": float(
            np.nanmean(correlations_array)
        ),
        "positive_frame_fraction": float(
            np.mean(correlations_array[np.isfinite(correlations_array)] > 0)
        ),
        "top_vs_bottom_edge_quartile_attention_contrast_median": float(
            np.nanmedian(contrasts_array)
        ),
        "n_frames": int(len(attention_maps)),
    }


def analyze(
    rows: list[dict],
    depth_frames: np.ndarray,
    attention_maps: np.ndarray,
) -> dict:
    input_change = np.array([row["input_feature_change"] for row in rows], dtype=float)
    context_change = np.array([row["mamba_context_change"] for row in rows], dtype=float)
    frame_change = np.array([row["frame_change"] for row in rows], dtype=float)
    depth = np.array([row["robust_depth_m"] for row in rows], dtype=float)

    normalized_attention = np.stack([normalize_map(item) for item in attention_maps])
    attention_std = float(np.std(normalized_attention))
    rho = spearman_correlation(input_change, context_change)

    valid_context = np.flatnonzero(np.isfinite(context_change))
    top_indices = valid_context[
        np.argsort(context_change[valid_context])[-min(3, len(valid_context)) :]
    ][::-1]
    valid_frame = frame_change[np.isfinite(frame_change)]
    frame_q75 = float(np.percentile(valid_frame, 75)) if valid_frame.size else float("nan")
    top_condition = bool(
        any(
            depth[index] < 5.0
            or (
                np.isfinite(frame_change[index])
                and np.isfinite(frame_q75)
                and frame_change[index] >= frame_q75
            )
            for index in top_indices
        )
    )

    endpoint_1 = attention_std > 0.01
    endpoint_2 = bool(np.isfinite(rho) and rho > 0.0)
    endpoint_3 = top_condition
    supported = endpoint_1 and endpoint_2 and endpoint_3

    return {
        "confirmatory_status": "supported" if supported else "mixed_or_unsupported",
        "endpoints": {
            "attention_pooled_normalized_std_gt_0_01": endpoint_1,
            "input_mamba_spearman_positive": endpoint_2,
            "top3_response_near_or_high_frame_change": endpoint_3,
        },
        "attention_pooled_normalized_std": attention_std,
        "input_mamba_spearman_rho": rho,
        "frame_change_q75": frame_q75,
        "top3_response_rows_zero_based": top_indices.tolist(),
        "top3_response_steps": [
            {
                "episode": rows[index]["episode"],
                "step": rows[index]["step"],
                "robust_depth_m": rows[index]["robust_depth_m"],
                "frame_change": rows[index]["frame_change"],
                "mamba_context_change": rows[index]["mamba_context_change"],
            }
            for index in top_indices
        ],
        "n_steps": len(rows),
        "exploratory_edge_alignment": exploratory_edge_alignment(
            depth_frames,
            attention_maps,
        ),
    }


def save_data(
    args: argparse.Namespace,
    rows: list[dict],
    depth_frames: np.ndarray,
    attention_maps: np.ndarray,
    input_vectors: np.ndarray,
    context_vectors: np.ndarray,
    episodes: list[dict],
    summary: dict,
    config_path: Path,
    model_path: Path,
) -> None:
    args.experiment_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.experiment_dir / "step_metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    np.savez_compressed(
        args.experiment_dir / "rollout_features.npz",
        depth_uint8=depth_frames,
        attention_maps=attention_maps,
        mamba_input_vectors=input_vectors,
        mamba_context_vectors=context_vectors,
    )

    payload = {
        **summary,
        "episodes": episodes,
        "model_file": str(model_path.resolve()),
        "config_file": str(config_path.resolve()),
        "environment": args.env_label,
        "scenario_seed": args.scenario_seed,
        "yaw_range_deg": args.yaw_range_deg,
        "airsim_rpc": f"{args.airsim_ip}:{args.airsim_port}",
        "protocol_sha256": "681B46F52F743922B15BFC7756427D318DEE87887A961768C137BDD1808400D5",
    }
    with (args.experiment_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")


def plot_figure(
    args: argparse.Namespace,
    rows: list[dict],
    depth_frames: np.ndarray,
    attention_maps: np.ndarray,
    summary: dict,
) -> None:
    args.figure_dir.mkdir(parents=True, exist_ok=True)
    episode_ids = sorted({int(row["episode"]) for row in rows})
    selected = []
    for episode in episode_ids:
        candidates = [
            index
            for index, row in enumerate(rows)
            if (
                row["episode"] == episode
                and row["step"] > 8
                and np.isfinite(row["mamba_context_change"])
            )
        ]
        if not candidates:
            candidates = [
                index
                for index, row in enumerate(rows)
                if row["episode"] == episode
                and np.isfinite(row["mamba_context_change"])
            ]
        if not candidates:
            candidates = [
                index for index, row in enumerate(rows) if row["episode"] == episode
            ]
        if not candidates:
            continue
        selected.append(
            max(
                candidates,
                key=lambda index: (
                    rows[index]["mamba_context_change"]
                    if np.isfinite(rows[index]["mamba_context_change"])
                    else -1.0
                ),
            )
        )
    if not selected:
        # Fall back to the strongest pooled Mamba responses.
        ranked = sorted(
            range(len(rows)),
            key=lambda index: (
                rows[index]["mamba_context_change"]
                if np.isfinite(rows[index]["mamba_context_change"])
                else -1.0
            ),
            reverse=True,
        )
        selected = ranked[: min(3, len(ranked))]
    if not selected:
        raise RuntimeError("No rollout frames available to plot.")


    fig = plt.figure(figsize=(13.4, 7.6), dpi=200)
    grid = fig.add_gridspec(2, 6, height_ratios=[1.0, 1.05], hspace=0.35, wspace=0.65)

    for panel, index in enumerate(selected[:3]):
        ax = fig.add_subplot(grid[0, panel * 2 : panel * 2 + 2])
        ax.imshow(depth_frames[index], cmap="gray", vmin=0, vmax=255)
        ax.imshow(
            normalize_map(attention_maps[index]),
            cmap="inferno",
            alpha=0.58,
            vmin=0,
            vmax=1,
            extent=(0, depth_frames.shape[2], depth_frames.shape[1], 0),
        )
        row = rows[index]
        ax.set_title(
            f"Episode {row['episode']}, step {row['step']}\n"
            f"depth$_{{1\\%}}$={row['robust_depth_m']:.2f} m, "
            f"$\\Delta h$={row['mamba_context_change']:.2f}",
            fontsize=9.2,
        )
        ax.set_xlabel("Image width")
        if panel == 0:
            ax.set_ylabel("Image height")
        else:
            ax.set_yticklabels([])

    ax_series = fig.add_subplot(grid[1, :4])
    input_change = np.array([row["input_feature_change"] for row in rows], dtype=float)
    context_change = np.array([row["mamba_context_change"] for row in rows], dtype=float)
    x = np.arange(1, len(rows) + 1)

    def robust_normalize(values):
        result = np.full_like(values, np.nan, dtype=float)
        valid = np.isfinite(values)
        if valid.any():
            scale = float(np.percentile(values[valid], 95))
            if scale > 0:
                result[valid] = np.clip(values[valid] / scale, 0, 1.25)
        return result

    ax_series.plot(
        x,
        robust_normalize(input_change),
        color="#D95F02",
        linewidth=1.0,
        label="Input-feature change",
    )
    ax_series.plot(
        x,
        robust_normalize(context_change),
        color="#1B70A6",
        linewidth=1.1,
        label="Mamba-context change",
    )
    for index in range(1, len(rows)):
        if rows[index]["episode"] != rows[index - 1]["episode"]:
            ax_series.axvline(index + 0.5, color="0.65", linestyle="--", linewidth=0.8)
    ax_series.set_xlabel("Pooled real-rollout control step")
    ax_series.set_ylabel("Change / pooled 95th percentile")
    ax_series.set_ylim(0, 1.28)
    ax_series.grid(alpha=0.22)
    ax_series.legend(frameon=False, ncol=2, loc="upper right")

    ax_scatter = fig.add_subplot(grid[1, 4:])
    valid = np.isfinite(input_change) & np.isfinite(context_change)
    colors = np.array([row["robust_depth_m"] for row in rows], dtype=float)[valid]
    scatter = ax_scatter.scatter(
        input_change[valid],
        context_change[valid],
        c=colors,
        cmap="viridis",
        s=8,
        alpha=0.55,
        linewidths=0,
    )
    ax_scatter.set_xlabel("Input-feature change")
    ax_scatter.set_ylabel("Mamba-context change")
    ax_scatter.set_title(
        f"Spearman $\\rho$={summary['input_mamba_spearman_rho']:.3f}",
        fontsize=10,
    )
    ax_scatter.grid(alpha=0.2)
    colorbar = fig.colorbar(scatter, ax=ax_scatter, fraction=0.046, pad=0.04)
    colorbar.set_label("Depth 1st percentile (m)", fontsize=8.5)

    status = summary["confirmatory_status"].replace("_", " ")
    fig.suptitle(
        f"Real {args.env_label} intermediate responses of MA-SAC "
        f"(confirmatory diagnostic: {status})",
        fontsize=12.5,
        fontweight="bold",
    )
    edge_audit = summary.get("exploratory_edge_alignment", {})
    if edge_audit:
        fig.text(
            0.5,
            0.895,
            "Exploratory CoordAtt--depth-edge alignment: "
            f"median per-frame $\\rho$="
            f"{edge_audit['per_frame_attention_edge_spearman_median']:.3f}",
            ha="center",
            va="center",
            fontsize=9.3,
        )
    fig.subplots_adjust(top=0.85, bottom=0.09, left=0.07, right=0.97)
    for suffix in ("pdf", "png"):
        fig.savefig(
            args.figure_dir / f"real_intermediate_features.{suffix}",
            bbox_inches="tight",
            dpi=300,
        )
    plt.close(fig)


def load_saved_rollout(experiment_dir: Path):
    numeric_int = {"global_step", "episode", "step"}
    numeric_float = {
        "scenario_yaw_deg",
        "reward",
        "action_x",
        "action_y",
        "action_yaw",
        "x",
        "y",
        "z",
        "robust_depth_m",
        "frame_change",
        "input_feature_change",
        "mamba_context_change",
        "attention_raw_mean",
        "attention_raw_std",
    }
    rows = []
    with (experiment_dir / "step_metrics.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        for raw in csv.DictReader(handle):
            row = dict(raw)
            for key in numeric_int:
                row[key] = int(row[key])
            for key in numeric_float:
                row[key] = float(row[key]) if row[key] else float("nan")
            row["terminal"] = row["terminal"].lower() == "true"
            rows.append(row)

    with np.load(experiment_dir / "rollout_features.npz") as data:
        depth_frames = data["depth_uint8"]
        attention_maps = data["attention_maps"]
    with (experiment_dir / "summary.json").open(encoding="utf-8") as handle:
        summary = json.load(handle)
    return rows, depth_frames, attention_maps, summary


def main() -> None:
    args = parse_args()
    if args.replot_only:
        rows, depth_frames, attention_maps, summary = load_saved_rollout(
            args.experiment_dir
        )
        summary["exploratory_edge_alignment"] = exploratory_edge_alignment(
            depth_frames,
            attention_maps,
        )
        with (args.experiment_dir / "summary.json").open(
            "w", encoding="utf-8"
        ) as handle:
            json.dump(
                summary,
                handle,
                indent=2,
                ensure_ascii=False,
                allow_nan=False,
            )
            handle.write("\n")
        plot_figure(args, rows, depth_frames, attention_maps, summary)
        print(f"Figure: {args.figure_dir / 'real_intermediate_features.pdf'}")
        return

    (
        rows,
        depth_frames,
        attention_maps,
        input_vectors,
        context_vectors,
        episodes,
        config_path,
        model_path,
    ) = run_rollouts(args)
    summary = analyze(rows, depth_frames, attention_maps)
    save_data(
        args,
        rows,
        depth_frames,
        attention_maps,
        input_vectors,
        context_vectors,
        episodes,
        summary,
        config_path,
        model_path,
    )
    plot_figure(args, rows, depth_frames, attention_maps, summary)

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Data: {args.experiment_dir}")
    print(f"Figure: {args.figure_dir / 'real_intermediate_features.pdf'}")


if __name__ == "__main__":
    main()
