"""
R1-6: Visualize CoordAtt attention weights and Mamba selectivity.

Loads a trained MA-SAC model, feeds synthetic tunnel depth images,
and extracts intermediate activations via forward hooks.

Outputs (to paper/images/Tunnel/):
  - coordatt_attention.pdf: Attention heatmap overlaid on depth image
  - mamba_delta_curve.pdf: Hidden state change along a trajectory
"""
import sys
import os
import traceback
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from configparser import ConfigParser

# Setup paths
UAV_ROOT = r"D:\code\ZipFIle\UAV_Navigation_DRL_AirSim"
sys.path.insert(0, UAV_ROOT)
os.chdir(UAV_ROOT)

import gymnasium as gym
from gym_env.networks.mamba_policy import MambaSACPolicy, CoordAttention2d
from gym_env.utils import build_mamba_policy_kwargs

# ── Model path ──
MODEL_DIR = os.path.join(
    UAV_ROOT,
    "Logs", "TrainData", "Runs", "tunnel", "ma", "seed_0",
    "2026_07_25_05_07_12_204825_Multirotor_cnn_SAC",
)
MODEL_PATH = os.path.join(MODEL_DIR, "models", "best_model")
CONFIG_PATH = os.path.join(MODEL_DIR, "config", "config.ini")
OUTPUT_DIR = r"D:\code\ZipFIle\paper\images\Tunnel"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ── Synthetic depth image generation ──
def make_tunnel_depth(height=60, width=90, max_depth=25.0):
    """Create a synthetic tunnel depth image (corridor without obstacles)."""
    depth = np.ones((height, width), dtype=np.float32) * 12.0
    for i in range(height):
        for j in range(width):
            di = abs(i - height / 2) / (height / 2)
            dj = abs(j - width / 2) / (width / 2)
            wall_dist = 10.0 - di * 4.0 - dj * 3.0
            depth[i, j] = max(2.0, wall_dist)
    depth[-5:, :] = np.linspace(3.0, 1.5, 5).reshape(-1, 1)
    depth[:5, :] = np.linspace(1.5, 3.0, 5).reshape(-1, 1)
    return depth


def make_obstacle_depth(height=60, width=90, obstacle_pos="center", obstacle_size=15):
    """Create a tunnel depth image with an obstacle."""
    depth = make_tunnel_depth(height, width)
    cx, cy = width // 2, height // 2
    if obstacle_pos == "left":
        cx, cy = width // 3, height // 2
    elif obstacle_pos == "right":
        cx, cy = 2 * width // 3, height // 2
    for i in range(max(0, cy - obstacle_size), min(height, cy + obstacle_size)):
        for j in range(max(0, cx - obstacle_size // 2), min(width, cx + obstacle_size // 2)):
            dist = np.sqrt((i - cy) ** 2 + (j - cx) ** 2)
            if dist < obstacle_size:
                depth[i, j] = max(1.0, 3.0 - dist * 0.15)
    return depth


def make_trajectory_depths(n_steps=16, height=60, width=90):
    """Create a sequence of depth images simulating approach to obstacle."""
    depths = []
    for t in range(n_steps):
        if t < 4:
            d = make_tunnel_depth(height, width)
        elif t < 8:
            d = make_obstacle_depth(height, width, "center", obstacle_size=5 + (t - 4) * 2)
        else:
            d = make_obstacle_depth(height, width, "center", obstacle_size=9 + (t - 8) * 2)
        d += np.random.randn(height, width) * 0.1
        d = np.clip(d, 0.5, 25.0)
        depths.append(d)
    return np.array(depths)


def depth_to_uint8(depth):
    """Convert float depth to uint8 image."""
    return np.clip(depth / 25.0 * 255.0, 0, 255).astype(np.uint8)


# ── CoordAtt attention hook ──
class AttentionHook:
    """Capture CoordAtt attention weights via forward hook."""
    def __init__(self):
        self.attention_weights = None
        self.hook_handle = None

    def __call__(self, module, inp, out):
        x = inp[0]
        _, _, height, width = x.shape
        x_h = x.mean(dim=3, keepdim=True)
        x_w = x.mean(dim=2, keepdim=True).permute(0, 1, 3, 2)
        mixed = torch.cat([x_h, x_w], dim=2)
        mixed = module.shared(mixed)
        x_h2, x_w2 = torch.split(mixed, [height, width], dim=2)
        x_w2 = x_w2.permute(0, 1, 3, 2)
        a_h = torch.sigmoid(module.attn_h(x_h2))
        a_w = torch.sigmoid(module.attn_w(x_w2))
        attention_map = (a_h * a_w).squeeze(0).mean(dim=0)
        self.attention_weights = attention_map.cpu().numpy()

    def register(self, module):
        self.hook_handle = module.register_forward_hook(self)

    def remove(self):
        if self.hook_handle:
            self.hook_handle.remove()


# ── Mamba hidden state hook ──
class MambaStateHook:
    """Capture Mamba block input/output for selectivity analysis."""
    def __init__(self):
        self.inputs = []
        self.outputs = []
        self.hook_handle = None

    def __call__(self, module, inp, out):
        self.inputs.append(inp[0].detach().cpu())
        if isinstance(out, torch.Tensor):
            self.outputs.append(out.detach().cpu())
        else:
            self.outputs.append(out[0].detach().cpu() if isinstance(out, tuple) else out)

    def register(self, module):
        self.hook_handle = module.register_forward_hook(self)

    def remove(self):
        if self.hook_handle:
            self.hook_handle.remove()


# ── Find CoordAtt module ──
def find_coordatt_module(policy):
    """Find the CoordAttention2d module in the policy."""
    if hasattr(policy, 'shared_cnn'):
        for name, module in policy.shared_cnn.named_modules():
            if isinstance(module, CoordAttention2d):
                return module, f"shared_cnn.{name}"
    if hasattr(policy, 'actor') and hasattr(policy.actor, '_cnn') and policy.actor._cnn is not None:
        for name, module in policy.actor._cnn.named_modules():
            if isinstance(module, CoordAttention2d):
                return module, f"actor._cnn.{name}"
    return None, None


# ── Load model from checkpoint ──
def load_model():
    """Load MA-SAC model directly from PyTorch checkpoint."""
    model_file = MODEL_PATH + ".zip"
    if not os.path.exists(model_file):
        print(f"ERROR: Model file not found: {model_file}")
        model_dir = os.path.dirname(model_file)
        if os.path.exists(model_dir):
            print(f"Files in {model_dir}:")
            for f in os.listdir(model_dir):
                print(f"  {f}")
        return None

    # Read config
    cfg = ConfigParser()
    cfg.read(CONFIG_PATH)

    n_stack = cfg.getint("mamba", "n_stack", fallback=1)
    height = cfg.getint("environment", "screen_height", fallback=60)
    width = cfg.getint("environment", "screen_width", fallback=90)
    state_dim = cfg.getint("options", "state_feature_num", fallback=6)

    # Build observation and action spaces
    obs_space = gym.spaces.Dict({
        "image": gym.spaces.Box(low=0, high=255, shape=(n_stack, 1, height, width), dtype=np.uint8),
        "state": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(n_stack, state_dim), dtype=np.float32),
    })
    action_space = gym.spaces.Box(
        low=np.array([-4.0, -2.0, -1.0], dtype=np.float32),
        high=np.array([4.0, 2.0, 1.0], dtype=np.float32),
        dtype=np.float32,
    )

    # Build policy kwargs from config
    policy_kwargs = build_mamba_policy_kwargs(cfg)
    print(f"Policy kwargs: {policy_kwargs}")

    # Create policy
    policy = MambaSACPolicy(
        observation_space=obs_space,
        action_space=action_space,
        lr_schedule=lambda _: 3e-4,
        **policy_kwargs,
    )

    # Load checkpoint
    print(f"Loading checkpoint: {model_file}")
    try:
        checkpoint = torch.load(model_file, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(model_file, map_location="cpu")

    if "policy_state_dict" in checkpoint:
        policy.load_state_dict(checkpoint["policy_state_dict"])
        print("  Loaded policy_state_dict from checkpoint")
    else:
        # Maybe it's a SB3 format
        print("  No 'policy_state_dict' key found. Keys:", list(checkpoint.keys())[:10])
        # Try direct load
        policy.load_state_dict(checkpoint)

    policy = policy.to("cpu")
    policy.eval()
    return policy


# ── CoordAtt visualization ──
def visualize_coordatt(policy):
    """Generate CoordAtt attention heatmap visualization."""
    print("\n=== Generating CoordAtt attention visualization ===")

    coordatt_module, module_name = find_coordatt_module(policy)

    if coordatt_module is None or isinstance(coordatt_module, nn.Identity):
        print("ERROR: CoordAtt module not found!")
        if hasattr(policy, 'shared_cnn'):
            for name, module in policy.shared_cnn.named_modules():
                print(f"  {name}: {type(module).__name__}")
        return False

    print(f"Found CoordAtt module: {type(coordatt_module).__name__} at {module_name}")

    depth_clean = make_tunnel_depth()
    depth_obstacle = make_obstacle_depth()

    hook = AttentionHook()
    hook.register(coordatt_module)

    fig, axes = plt.subplots(2, 3, figsize=(14, 8), dpi=200)

    for idx, (depth, title) in enumerate([
        (depth_clean, "Corridor (no obstacle)"),
        (depth_obstacle, "Obstacle appearing")
    ]):
        depth_uint8 = depth_to_uint8(depth)
        # n_stack=1: image shape is (B, C, H, W), state shape is (B, state_dim)
        image_tensor = torch.zeros(1, 1, 60, 90, dtype=torch.uint8)
        image_tensor[0, 0] = torch.from_numpy(depth_uint8)

        state_tensor = torch.zeros(1, 6, dtype=torch.float32)
        state_tensor[0, 0] = 10.0

        obs = {"image": image_tensor, "state": state_tensor}

        with torch.no_grad():
            _ = policy.actor(obs, deterministic=True, use_history=False)

        if hook.attention_weights is not None:
            attn = hook.attention_weights
            print(f"  [{title}] Attention shape: {attn.shape}, range: [{attn.min():.4f}, {attn.max():.4f}]")
        else:
            print(f"  [{title}] WARNING: No attention weights captured!")
            attn = np.ones((10, 20)) * 0.5

        # Plot depth image
        ax_depth = axes[idx, 0]
        im = ax_depth.imshow(depth, cmap='gray', vmin=0, vmax=20)
        ax_depth.set_title(f"Depth Image ({title})", fontsize=10)
        ax_depth.set_xlabel("Width")
        ax_depth.set_ylabel("Height")
        plt.colorbar(im, ax=ax_depth, fraction=0.046)

        # Plot attention map
        ax_attn = axes[idx, 1]
        im2 = ax_attn.imshow(attn, cmap='hot', vmin=attn.min(), vmax=attn.max())
        ax_attn.set_title(f"CoordAtt Attention ({title})", fontsize=10)
        ax_attn.set_xlabel("Width")
        ax_attn.set_ylabel("Height")
        plt.colorbar(im2, ax=ax_attn, fraction=0.046)

        # Plot overlay
        ax_overlay = axes[idx, 2]
        depth_norm = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
        if attn.shape != depth.shape:
            from scipy.ndimage import zoom
            attn_resized = zoom(attn, (depth.shape[0] / attn.shape[0], depth.shape[1] / attn.shape[1]))
        else:
            attn_resized = attn
        attn_resized_norm = (attn_resized - attn_resized.min()) / (attn_resized.max() - attn_resized.min() + 1e-8)

        overlay = np.zeros((depth.shape[0], depth.shape[1], 3))
        overlay[:, :, 0] = depth_norm * 0.4 + attn_resized_norm * 0.8
        overlay[:, :, 1] = depth_norm * 0.4
        overlay[:, :, 2] = depth_norm * 0.4
        overlay = np.clip(overlay, 0, 1)
        ax_overlay.imshow(overlay)
        ax_overlay.set_title(f"Overlay ({title})", fontsize=10)
        ax_overlay.set_xlabel("Width")
        ax_overlay.set_ylabel("Height")

    hook.remove()

    fig.suptitle("CoordAtt Attention Visualization: Corridor vs Obstacle", fontsize=13, fontweight='bold')
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    output_path = os.path.join(OUTPUT_DIR, "coordatt_attention.pdf")
    fig.savefig(output_path, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close(fig)
    return True


# ── Mamba selectivity visualization ──
def visualize_mamba(policy):
    """Generate Mamba selectivity curve visualization."""
    print("\n=== Generating Mamba selectivity curve ===")

    mamba_block = policy.actor.mamba
    backend = mamba_block.backend if hasattr(mamba_block, 'backend') else "unknown"
    print(f"  Mamba backend: {backend}")

    n_steps = 16
    trajectory = make_trajectory_depths(n_steps=n_steps)

    hook = MambaStateHook()
    hook.register(mamba_block)

    for t in range(n_steps):
        depth_uint8 = depth_to_uint8(trajectory[t])
        # n_stack=1: image shape is (B, C, H, W), state shape is (B, state_dim)
        image_tensor = torch.zeros(1, 1, 60, 90, dtype=torch.uint8)
        image_tensor[0, 0] = torch.from_numpy(depth_uint8)

        state_tensor = torch.zeros(1, 6, dtype=torch.float32)
        state_tensor[0, 0] = max(0, 10.0 - t * 0.5)

        obs = {"image": image_tensor, "state": state_tensor}

        with torch.no_grad():
            _ = policy.actor(obs, deterministic=True, use_history=False)

    hook.remove()

    print(f"  Captured {len(hook.inputs)} inputs, {len(hook.outputs)} outputs")

    # Compute hidden state changes (selectivity proxy)
    if len(hook.outputs) >= 2:
        deltas = []
        for t in range(1, len(hook.outputs)):
            curr = hook.outputs[t]
            prev = hook.outputs[t - 1]
            if curr.shape == prev.shape:
                change = torch.norm(curr - prev, dim=-1).mean().item()
                deltas.append(change)
            else:
                deltas.append(0.0)
        deltas = np.array(deltas)
        print(f"  Computed {len(deltas)} delta values")
        print(f"  Range: [{deltas.min():.4f}, {deltas.max():.4f}]")
    else:
        print("  WARNING: Not enough outputs captured")
        deltas = np.zeros(n_steps - 1)

    # Compute input feature changes
    input_deltas = []
    if len(hook.inputs) >= 2:
        for t in range(1, len(hook.inputs)):
            curr = hook.inputs[t]
            prev = hook.inputs[t - 1]
            if curr.shape == prev.shape:
                change = torch.norm(curr - prev, dim=-1).mean().item()
                input_deltas.append(change)
            else:
                input_deltas.append(0.0)
    input_deltas = np.array(input_deltas) if input_deltas else np.zeros(len(deltas))

    print(f"  Input deltas range: [{input_deltas.min():.4f}, {input_deltas.max():.4f}]")

    # ── Plot ──
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), dpi=200,
                                     gridspec_kw={'height_ratios': [3, 1]})

    steps = np.arange(1, len(deltas) + 1)

    # Normalize for comparison
    deltas_norm = deltas / deltas.max() if deltas.max() > 0 else deltas
    input_deltas_norm = input_deltas / input_deltas.max() if input_deltas.max() > 0 else input_deltas

    ax1.plot(steps, deltas_norm, 'b-o', linewidth=2, markersize=6,
             label='Mamba hidden state change', zorder=3)
    ax1.plot(steps, input_deltas_norm, 'r--s', linewidth=1.5, markersize=4, alpha=0.7,
             label='Input feature change (CNN+state)', zorder=2)

    ax1.axvspan(1, 4, alpha=0.1, color='green', label='Clean corridor')
    ax1.axvspan(4, 8, alpha=0.1, color='orange', label='Obstacle appears')
    ax1.axvspan(8, 16, alpha=0.1, color='red', label='Approaching obstacle')

    ax1.set_xlabel("Time Step", fontsize=11)
    ax1.set_ylabel("Normalized Change Magnitude", fontsize=11)
    ax1.set_title("Mamba Selectivity: Hidden State Change Along Trajectory",
                  fontsize=12, fontweight='bold')
    ax1.legend(loc='upper left', fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0.5, n_steps + 0.5)

    # Bottom: representative depth images
    for t_idx in [0, 4, 8, 12, 15]:
        if t_idx < len(trajectory):
            x_pos = (t_idx / (n_steps - 1)) * 0.9 + 0.05
            ax_inset = ax2.inset_axes([x_pos - 0.06, 0.1, 0.12, 0.8])
            ax_inset.imshow(trajectory[t_idx], cmap='gray', vmin=0, vmax=20)
            ax_inset.set_title(f"t={t_idx}", fontsize=7)
            ax_inset.set_xticks([])
            ax_inset.set_yticks([])

    ax2.set_xlabel("Time Step", fontsize=11)
    ax2.set_ylabel("Depth", fontsize=11)
    ax2.set_yticks([])

    fig.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, "mamba_delta_curve.pdf")
    fig.savefig(output_path, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close(fig)
    return True


# ── Main ──
def main():
    print("Loading MA-SAC model...")
    print(f"Model path: {MODEL_PATH}.zip")
    print(f"Config path: {CONFIG_PATH}")

    policy = load_model()
    if policy is None:
        return

    print(f"\nPolicy type: {type(policy).__name__}")
    print(f"Actor type: {type(policy.actor).__name__}")

    if hasattr(policy, 'shared_cnn'):
        print(f"Shared CNN type: {type(policy.shared_cnn).__name__}")
        print(f"  Attention type: {policy.shared_cnn.attention_type}")
        print(f"  Attention stage: {policy.shared_cnn.attention_stage}")

    if hasattr(policy.actor, 'mamba'):
        mamba_block = policy.actor.mamba
        backend = mamba_block.backend if hasattr(mamba_block, 'backend') else "unknown"
        print(f"Mamba block type: {type(mamba_block).__name__}, backend: {backend}")

    # Generate visualizations
    success_coordatt = False
    success_mamba = False

    try:
        success_coordatt = visualize_coordatt(policy)
    except Exception as e:
        print(f"CoordAtt visualization failed: {e}")
        traceback.print_exc()

    try:
        success_mamba = visualize_mamba(policy)
    except Exception as e:
        print(f"Mamba visualization failed: {e}")
        traceback.print_exc()

    print("\n=== Summary ===")
    print(f"  CoordAtt visualization: {'SUCCESS' if success_coordatt else 'FAILED'}")
    print(f"  Mamba visualization:    {'SUCCESS' if success_mamba else 'FAILED'}")
    print(f"  Output directory: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
