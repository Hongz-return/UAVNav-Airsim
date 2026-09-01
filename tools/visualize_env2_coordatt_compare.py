"""Paper figure data: depth | CNN (pre-CoordAtt) | after CoordAtt.

Captures real MA-SAC forward features on a live AirSim scene.  The historical
defaults still reproduce the ENV2 diagnostic, while explicit model and scene
arguments support the same auditable capture protocol for additional scenes.
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
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
from matplotlib import font_manager

UAV_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(UAV_ROOT))
os.chdir(UAV_ROOT)

from gym_env.networks.mamba_policy import CoordAttention2d
from scripts.evaluation.evaluate_single import (
    apply_evaluation_overrides,
    build_evaluation_scenarios,
    get_base_env,
)
from tools.airsim_instances import launch_instances, terminate_processes
from tools.runtime_support import build_runtime_env, load_model

DEFAULT_MODEL_DIR = UAV_ROOT / (
    "Logs/TrainData/Runs/tunnel_obs/ma/seed_0/"
    "2026_07_27_16_05_31_659067_Multirotor_cnn_SAC"
)
# Pending review: write here until user confirms move into images/<scene>/.
DEFAULT_OUT = Path(r"D:\code\ZipFIle\paper\images\analysis")
DEFAULT_EXP = Path(r"D:\code\ZipFIle\paper\experiments\env2-coordatt-compare")


def setup_chinese_font() -> None:
    for path in (
        r"C:\Windows\Fonts\simhei.ttf",
        r"C:\Windows\Fonts\msyh.ttc",
        r"C:\Windows\Fonts\simsun.ttc",
    ):
        if Path(path).exists():
            font_manager.fontManager.addfont(path)
            prop = font_manager.FontProperties(fname=path)
            plt.rcParams["font.family"] = prop.get_name()
            plt.rcParams["axes.unicode_minus"] = False
            return


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    p.add_argument(
        "--model-path",
        type=Path,
        help="Explicit frozen model/checkpoint. Defaults to <model-dir>/models/best_model.zip.",
    )
    p.add_argument("--episodes", type=int, default=3)
    p.add_argument(
        "--stop-after-first-success",
        action="store_true",
        help="Stop after the first completed successful episode.",
    )
    p.add_argument("--scenario-seed", type=int, default=2026)
    p.add_argument("--yaw-range-deg", type=float, default=10.0)
    p.add_argument("--airsim-ip", default="127.0.0.1")
    p.add_argument("--airsim-port", type=int, default=41481)
    p.add_argument("--experiment-dir", type=Path, default=DEFAULT_EXP)
    p.add_argument("--figure-dir", type=Path, default=DEFAULT_OUT)
    p.add_argument("--env-key", default="ENV2")
    p.add_argument("--scene-label", default="static-obstacle tunnel")
    p.add_argument("--launch-ue", action="store_true")
    p.add_argument(
        "--ue-exe",
        type=Path,
        default=Path(r"D:\UEBuilds\TunnelEnv\Windows\Tunnel.exe"),
    )
    p.add_argument(
        "--map-path",
        default="/Game/Tunnel_Col_Vol1/Maps/Tunnel",
    )
    p.add_argument(
        "--settings-template",
        type=Path,
        default=Path(r"C:\Users\31732\Documents\AirSim\settings.json"),
    )
    p.add_argument("--startup-wait", type=int, default=60)
    p.add_argument("--clock-speed", type=float, default=1.0)
    p.add_argument("--res-x", type=int, default=640)
    p.add_argument("--res-y", type=int, default=480)
    p.add_argument("--replot-only", action="store_true")
    p.add_argument(
        "--coordatt-only",
        action="store_true",
        help="Only plot CoordAtt three-panel figure (skip depth strip).",
    )
    return p.parse_args()


def normalize(arr: np.ndarray) -> np.ndarray:
    low, high = float(np.min(arr)), float(np.max(arr))
    if high - low < 1e-12:
        return np.zeros_like(arr, dtype=np.float32)
    return ((arr - low) / (high - low)).astype(np.float32)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_model_path(args: argparse.Namespace) -> Path:
    path = args.model_path or (args.model_dir / "models" / "best_model.zip")
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Frozen model not found: {path}")
    return path


def read_start_position(config_path: Path) -> list[float]:
    cfg = ConfigParser()
    cfg.read(config_path, encoding="utf-8")
    value = ast.literal_eval(cfg.get("environment", "start_position"))
    if not isinstance(value, list) or len(value) != 3:
        raise ValueError(f"Invalid start_position in {config_path}: {value!r}")
    return [float(item) for item in value]


def feature_strength(feat: np.ndarray) -> np.ndarray:
    """Channel-mean absolute activation -> HxW map."""
    if feat.ndim == 3:
        return np.mean(np.abs(feat), axis=0).astype(np.float32)
    return np.abs(feat).astype(np.float32)


def upsample(map_hw: np.ndarray, shape_hw: tuple[int, int]) -> np.ndarray:
    h, w = shape_hw
    ah, aw = map_hw.shape
    yy = np.linspace(0, ah - 1, h, dtype=np.float32)
    xx = np.linspace(0, aw - 1, w, dtype=np.float32)
    y0 = np.floor(yy).astype(int)
    x0 = np.floor(xx).astype(int)
    y1 = np.clip(y0 + 1, 0, ah - 1)
    x1 = np.clip(x0 + 1, 0, aw - 1)
    wy = yy - y0
    wx = xx - x0
    top = map_hw[y0][:, x0] * (1 - wx)[None, :] + map_hw[y0][:, x1] * wx[None, :]
    bot = map_hw[y1][:, x0] * (1 - wx)[None, :] + map_hw[y1][:, x1] * wx[None, :]
    return top * (1 - wy)[:, None] + bot * wy[:, None]


class PrePostCoordAttHook:
    """Capture pre/post features and CoordAtt gates a_h, a_w, a_h*a_w."""

    def __init__(self) -> None:
        self.pre: np.ndarray | None = None
        self.post: np.ndarray | None = None
        self.attn: np.ndarray | None = None
        self.a_h: np.ndarray | None = None  # (H,) channel-mean
        self.a_w: np.ndarray | None = None  # (W,) channel-mean
        self.handle = None

    def __call__(self, module: CoordAttention2d, inputs, output) -> None:
        x = inputs[0]
        identity = x
        _, _, height, width = x.shape
        x_h = x.mean(dim=3, keepdim=True)
        x_w = x.mean(dim=2, keepdim=True).permute(0, 1, 3, 2)
        mixed = torch.cat([x_h, x_w], dim=2)
        mixed = module.shared(mixed)
        x_h, x_w = torch.split(mixed, [height, width], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)
        a_h = torch.sigmoid(module.attn_h(x_h))
        a_w = torch.sigmoid(module.attn_w(x_w))
        gate = (a_h * a_w)[-1].detach().cpu().numpy().astype(np.float32)
        a_h_np = a_h[-1].detach().cpu().numpy().astype(np.float32)
        a_w_np = a_w[-1].detach().cpu().numpy().astype(np.float32)
        self.pre = identity[-1].detach().cpu().numpy().astype(np.float32)
        self.post = output[-1].detach().cpu().numpy().astype(np.float32)
        self.attn = gate.mean(axis=0)
        self.a_h = a_h_np.mean(axis=0).reshape(-1)  # H
        self.a_w = a_w_np.mean(axis=0).reshape(-1)  # W

    def register(self, module: CoordAttention2d) -> None:
        self.handle = module.register_forward_hook(self)

    def remove(self) -> None:
        if self.handle is not None:
            self.handle.remove()


def find_coordatt(policy) -> CoordAttention2d:
    for module in policy.shared_cnn.modules():
        if isinstance(module, CoordAttention2d):
            return module
    raise RuntimeError("CoordAttention2d not found")


def depth_from_obs(obs) -> np.ndarray:
    return np.squeeze(np.asarray(obs["image"][0])).astype(np.uint8)


def robust_depth_m(depth_u8: np.ndarray, max_depth: float = 25.0) -> float:
    depth = depth_u8.astype(np.float32) / 255.0 * max_depth
    valid = depth[(depth > 0.1) & (depth < max_depth)]
    return float(np.percentile(valid, 1)) if valid.size else float(max_depth)


def attention_from_pre_post(pre: np.ndarray, post: np.ndarray) -> np.ndarray:
    """Recover a_h*a_w ≈ post/pre when live gates were not saved."""
    denom = np.abs(pre).astype(np.float64)
    ratio = np.divide(
        post.astype(np.float64),
        denom,
        out=np.full_like(denom, np.nan),
        where=denom > 1e-5,
    )
    ratio = np.clip(ratio, 0.0, 1.0)
    return np.nanmean(ratio, axis=0).astype(np.float32)


def depth_edge_map(depth_u8: np.ndarray) -> np.ndarray:
    d = depth_u8.astype(np.float64)
    gy, gx = np.gradient(d)
    edge = np.hypot(gx, gy)
    return normalize(edge.astype(np.float32))


def corr2d(a: np.ndarray, b: np.ndarray) -> float:
    aa = a.ravel().astype(np.float64)
    bb = b.ravel().astype(np.float64)
    if aa.std() < 1e-12 or bb.std() < 1e-12:
        return 0.0
    return float(np.corrcoef(aa, bb)[0, 1])


def run_rollouts(args: argparse.Namespace):
    config_path = args.model_dir / "config" / "config.ini"
    model_path = resolve_model_path(args)
    cfg = ConfigParser()
    cfg.read(config_path)
    apply_evaluation_overrides(
        cfg,
        env_name=None,
        airsim_ip=args.airsim_ip,
        airsim_port=args.airsim_port,
    )
    scenarios, _ = build_evaluation_scenarios(
        cfg, args.episodes, args.scenario_seed, args.yaw_range_deg
    )
    env = build_runtime_env(cfg)
    np.random.seed(args.scenario_seed)
    torch.manual_seed(args.scenario_seed)
    env.seed(args.scenario_seed)
    base_env = get_base_env(env)
    base_env.set_evaluation_scenarios(scenarios + [scenarios[-1]])
    model = load_model(cfg.get("options", "algo"), str(model_path), env, cfg=cfg)
    model.policy.set_training_mode(False)

    hook = PrePostCoordAttHook()
    hook.register(find_coordatt(model.policy))

    rows = []
    episode_outcomes = []
    depths, pres, posts, attns = [], [], [], []
    a_hs, a_ws = [], []
    try:
        obs = env.reset()
        for episode in range(1, args.episodes + 1):
            episode_start = np.array([True])
            episode_row_start = len(rows)
            step = 0
            while True:
                depth = depth_from_obs(obs)
                action, _ = model.predict(
                    obs, deterministic=True, episode_start=episode_start
                )
                episode_start = np.array([False])
                step += 1
                if (
                    hook.pre is None
                    or hook.post is None
                    or hook.attn is None
                    or hook.a_h is None
                    or hook.a_w is None
                ):
                    raise RuntimeError("CoordAtt hook missed a forward pass")
                pre = hook.pre.copy()
                post = hook.post.copy()
                attn = hook.attn.copy()
                a_h = hook.a_h.copy()
                a_w = hook.a_w.copy()
                attn_u = upsample(attn, depth.shape)
                edge = depth_edge_map(depth)
                near = (depth.astype(np.float32) / 255.0 * 25.0) < 4.0
                near_mass = float(attn_u[near].mean()) if near.any() else 0.0
                far_mass = float(attn_u[~near].mean()) if (~near).any() else 0.0
                rows.append(
                    {
                        "episode": episode,
                        "step": step,
                        "robust_depth_m": robust_depth_m(depth),
                        "pre_std": float(np.std(feature_strength(pre))),
                        "post_std": float(np.std(feature_strength(post))),
                        "focus_gain": float(
                            np.std(feature_strength(post))
                            - np.std(feature_strength(pre))
                        ),
                        "attn_std": float(np.std(attn)),
                        "attn_range": float(attn.max() - attn.min()),
                        "attn_edge_corr": corr2d(attn_u, edge),
                        "attn_near_far": near_mass - far_mass,
                        "a_h_std": float(np.std(a_h)),
                        "a_w_std": float(np.std(a_w)),
                    }
                )
                depths.append(depth)
                pres.append(pre)
                posts.append(post)
                attns.append(attn)
                a_hs.append(a_h)
                a_ws.append(a_w)
                obs, _reward, done, info = env.step(action)
                if isinstance(done, (list, tuple, np.ndarray)):
                    done = bool(done[0])
                if isinstance(info, (list, tuple)):
                    info = info[0]
                if done:
                    crash = bool(info.get("is_crash"))
                    success = bool(info.get("is_success"))
                    outcome = "success" if success else ("collision" if crash else "other")
                    for row in rows[episode_row_start:]:
                        row["episode_success"] = success
                        row["episode_crash"] = crash
                        row["episode_outcome"] = outcome
                    episode_outcomes.append(
                        {
                            "episode": episode,
                            "steps": step,
                            "crash": crash,
                            "success": success,
                            "outcome": outcome,
                        }
                    )
                    # DummyVecEnv already auto-resets on done; obs is next episode.
                    print(
                        f"Episode {episode}/{args.episodes}: "
                        f"steps={step}, crash={crash}, success={success}"
                    )
                    break
            if args.stop_after_first_success and episode_outcomes[-1]["success"]:
                print(f"Stopping after successful episode {episode}.")
                break
    finally:
        hook.remove()
        env.close()

    return (
        rows,
        np.stack(depths),
        np.stack(pres),
        np.stack(posts),
        np.stack(attns),
        np.stack(a_hs),
        np.stack(a_ws),
        episode_outcomes,
    )


def pick_frame(rows, depths, attns, a_hs, a_ws) -> int:
    """Prefer frames with non-trivial directional gates near obstacles."""
    best_i, best = 0, -1e9
    for i, row in enumerate(rows):
        if row["step"] < 12:
            continue
        if row["robust_depth_m"] > 5.0:
            continue
        # Require some real gate variation (avoid flat ~constant maps).
        if row.get("attn_range", 0.0) < 0.02 and row.get("a_h_std", 0.0) < 0.01:
            continue
        attn_u = upsample(attns[i], depths[i].shape)  # absolute, not renormalized
        edge = depth_edge_map(depths[i])
        edge_corr = corr2d(attn_u, edge)
        near = (depths[i].astype(np.float32) / 255.0 * 25.0) < 4.0
        near_far = (
            float(attn_u[near].mean() - attn_u[~near].mean()) if near.any() else 0.0
        )
        score = (
            0.9 * edge_corr
            + 0.7 * near_far
            + 1.5 * float(row.get("attn_range", np.ptp(attns[i])))
            + 0.8 * float(np.std(a_hs[i]))
            + 0.8 * float(np.std(a_ws[i]))
        )
        if row["robust_depth_m"] < 3.5:
            score += 0.1
        if score > best:
            best, best_i = score, i
    if best < -1e8 / 2:
        # Fallback: largest attention range among near frames.
        for i, row in enumerate(rows):
            if row["step"] < 12 or row["robust_depth_m"] > 5.0:
                continue
            score = float(np.ptp(attns[i])) + 0.1 * float(np.std(a_hs[i]))
            if score > best:
                best, best_i = score, i
    return best_i


def plot_coordatt(
    rows, depths, attns, a_hs, a_ws, out_dir: Path, env_key: str
) -> Path:
    """Depth | vertical gate | horizontal gate — CoordAtt's actual factors."""
    setup_chinese_font()
    idx = pick_frame(rows, depths, attns, a_hs, a_ws)
    row = rows[idx]
    depth = depths[idx]
    a_h = a_hs[idx]
    a_w = a_ws[idx]
    # Expand 1D directional gates to feature HxW, then upsample to depth size.
    h_map = np.repeat(a_h[:, None], a_w.shape[0], axis=1)
    w_map = np.repeat(a_w[None, :], a_h.shape[0], axis=0)
    h_map = upsample(h_map, depth.shape)
    w_map = upsample(w_map, depth.shape)
    prod = h_map * w_map

    fig, axes = plt.subplots(1, 3, figsize=(9.8, 3.2), dpi=200)
    axes[0].imshow(depth, cmap="gray", vmin=0, vmax=255)
    axes[0].set_title("原始深度图", fontsize=12)

    im1 = axes[1].imshow(h_map, cmap="magma", vmin=0.0, vmax=1.0)
    axes[1].set_title(r"垂直门控 $\mathbf{g}^{h}$", fontsize=12)
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    im2 = axes[2].imshow(w_map, cmap="magma", vmin=0.0, vmax=1.0)
    axes[2].set_title(r"水平门控 $\mathbf{g}^{w}$", fontsize=12)
    fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(
        f"{env_key} 中 CoordAtt 方向门控（绝对值，探索性示意）",
        fontsize=13,
        y=1.04,
    )
    fig.text(
        0.5,
        -0.05,
        f"MA-SAC 冻结模型 · 第{row['episode']}回合第{row['step']}步 · "
        f"g^h范围=[{a_h.min():.2f},{a_h.max():.2f}]  "
        f"g^w范围=[{a_w.min():.2f},{a_w.max():.2f}]  "
        f"乘积范围=[{prod.min():.2f},{prod.max():.2f}] · "
        "展示模块门控，不作有效性机理证明",
        ha="center",
        fontsize=8.2,
        color="#444444",
    )
    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = out_dir / "coordatt_attention"
    fig.savefig(f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(f"{stem}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # Companion: depth + product overlay with absolute scale annotation.
    fig2, axes2 = plt.subplots(1, 2, figsize=(7.2, 3.2), dpi=200)
    axes2[0].imshow(depth, cmap="gray", vmin=0, vmax=255)
    axes2[0].set_title("原始深度图", fontsize=12)
    axes2[1].imshow(depth, cmap="gray", vmin=0, vmax=255)
    axes2[1].imshow(prod, cmap="magma", alpha=0.55, vmin=0.0, vmax=1.0)
    axes2[1].set_title(r"深度 $\times$ $\mathbf{g}^{h}\mathbf{g}^{w}$", fontsize=12)
    for ax in axes2:
        ax.set_xticks([])
        ax.set_yticks([])
    fig2.suptitle("门控乘积叠加（同帧）", fontsize=12, y=1.02)
    fig2.tight_layout()
    stem2 = out_dir / "coordatt_attention_overlay"
    fig2.savefig(f"{stem2}.pdf", bbox_inches="tight")
    fig2.savefig(f"{stem2}.png", dpi=300, bbox_inches="tight")
    plt.close(fig2)

    print(
        f"wrote {stem}.pdf / .png and overlay "
        f"(ep{row['episode']} step{row['step']}, "
        f"h=[{a_h.min():.3f},{a_h.max():.3f}], "
        f"w=[{a_w.min():.3f},{a_w.max():.3f}])"
    )
    return Path(f"{stem}.pdf")


def plot_mamba_frames(rows, depths, out_dir: Path, env_key: str) -> Path | None:
    """Simple continuous-frame strip for temporal context (secondary panel)."""
    # Pick episode with most steps
    ep_counts: dict[int, int] = {}
    for r in rows:
        ep_counts[r["episode"]] = ep_counts.get(r["episode"], 0) + 1
    if not ep_counts:
        return None
    ep = max(ep_counts, key=ep_counts.get)
    idxs = [i for i, r in enumerate(rows) if r["episode"] == ep]
    if len(idxs) < 8:
        return None
    # 5 frames evenly from mid segment
    mid = idxs[len(idxs) // 4 : (3 * len(idxs)) // 4] or idxs
    chosen = [mid[int(k)] for k in np.linspace(0, len(mid) - 1, 5)]

    setup_chinese_font()
    fig, axes = plt.subplots(1, 5, figsize=(10.5, 2.35), dpi=200)
    for ax, i in zip(axes, chosen):
        ax.imshow(depths[i], cmap="gray", vmin=0, vmax=255)
        ax.set_title(f"步 {rows[i]['step']}", fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(
        f"{env_key} 连续深度观测示意（第{ep}回合，供对照时序建模输入）",
        fontsize=12,
        y=1.05,
    )
    fig.text(
        0.5,
        -0.08,
        "探索性可视化：展示时序主干所见的连续深度变化，不作选择性记忆的机理证明",
        ha="center",
        fontsize=8.5,
        color="#444444",
    )
    fig.tight_layout()
    stem = out_dir / "mamba_depth_sequence"
    fig.savefig(f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(f"{stem}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {stem}.pdf / .png")
    return Path(f"{stem}.pdf")


def save_data(args, rows, depths, pres, posts, attns, a_hs, a_ws, episode_outcomes):
    args.experiment_dir.mkdir(parents=True, exist_ok=True)
    with (args.experiment_dir / "frames.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    np.savez_compressed(
        args.experiment_dir / "coordatt_features.npz",
        depth_uint8=depths,
        pre_features=pres,
        post_features=posts,
        attn_maps=attns,
        a_h=a_hs,
        a_w=a_ws,
    )
    with (args.experiment_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "n_frames": len(rows),
                "model_dir": str(resolve_model_path(args).parent.parent),
                "model_path": str(resolve_model_path(args)),
                "model_sha256": sha256(resolve_model_path(args)),
                "airsim": f"{args.airsim_ip}:{args.airsim_port}",
                "scenario_seed": args.scenario_seed,
                "viz": "coordatt_directional_gates",
                "env_key": args.env_key,
                "scene_label": args.scene_label,
                "map_path": args.map_path,
                "episodes_requested": args.episodes,
                "episodes_collected": len(episode_outcomes),
                "stop_after_first_success": args.stop_after_first_success,
                "episode_outcomes": episode_outcomes,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
        f.write("\n")


def load_saved(exp: Path):
    rows = list(csv.DictReader((exp / "frames.csv").open(encoding="utf-8")))
    for r in rows:
        r["episode"] = int(r["episode"])
        r["step"] = int(r["step"])
        for key in ("episode_success", "episode_crash"):
            if key in r:
                r[key] = str(r[key]).lower() == "true"
        for k in (
            "robust_depth_m",
            "pre_std",
            "post_std",
            "focus_gain",
            "attn_std",
            "attn_range",
            "attn_edge_corr",
            "attn_near_far",
            "a_h_std",
            "a_w_std",
        ):
            if k in r and r[k] != "":
                r[k] = float(r[k])
    with np.load(exp / "coordatt_features.npz") as data:
        depths = data["depth_uint8"]
        pres = data["pre_features"]
        posts = data["post_features"]
        if "a_h" in data.files and "a_w" in data.files:
            a_hs = data["a_h"]
            a_ws = data["a_w"]
            if "attn_maps" in data.files:
                attns = data["attn_maps"]
            else:
                attns = np.stack([np.outer(h, w) for h, w in zip(a_hs, a_ws)])
        else:
            attns = np.stack(
                [attention_from_pre_post(pre, post) for pre, post in zip(pres, posts)]
            )
            # Approximate directional profiles from recovered product (weak).
            a_hs = attns.mean(axis=2)
            a_ws = attns.mean(axis=1)
            for i, r in enumerate(rows):
                r["attn_std"] = float(np.std(attns[i]))
                r["attn_range"] = float(np.ptp(attns[i]))
                r["a_h_std"] = float(np.std(a_hs[i]))
                r["a_w_std"] = float(np.std(a_ws[i]))
        return rows, depths, pres, posts, attns, a_hs, a_ws


def main() -> None:
    args = parse_args()
    ue_processes = []
    try:
        if args.launch_ue and not args.replot_only:
            config_path = args.model_dir / "config" / "config.ini"
            ue_processes = launch_instances(
                ue_exe=args.ue_exe,
                map_path=args.map_path,
                settings_template=args.settings_template,
                output_dir=args.experiment_dir / "airsim_settings",
                ports=[args.airsim_port],
                start_position=read_start_position(config_path),
                clock_speed=args.clock_speed,
                res_x=args.res_x,
                res_y=args.res_y,
                wait_seconds=args.startup_wait,
            )
        if args.replot_only:
            rows, depths, _pres, _posts, attns, a_hs, a_ws = load_saved(
                args.experiment_dir
            )
        else:
            (
                rows,
                depths,
                pres,
                posts,
                attns,
                a_hs,
                a_ws,
                episode_outcomes,
            ) = run_rollouts(args)
            save_data(
                args,
                rows,
                depths,
                pres,
                posts,
                attns,
                a_hs,
                a_ws,
                episode_outcomes,
            )
        plot_coordatt(
            rows, depths, attns, a_hs, a_ws, args.figure_dir, args.env_key
        )
        if not args.coordatt_only:
            plot_mamba_frames(rows, depths, args.figure_dir, args.env_key)
    finally:
        if ue_processes:
            terminate_processes(ue_processes)


if __name__ == "__main__":
    main()
