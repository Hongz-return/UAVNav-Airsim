"""Verify the full rail gates and render paired 3D flight evidence."""

import argparse
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from .ch4_trainenv_3d_course import COURSE_FILE, obstacles, read_json, write_json
from .screen_ch4_stage3_v2 import RAIL, screen_rail
from .summarize_ch4_stage3 import FAMILIES, HELDOUT, read_csv, records


COLORS = {"success": "#007c76", "collision": "#b94a32", "out_of_bounds": "#7654a5",
          "timeout": "#606b77", "route_not_completed": "#b0751e"}


def heldout_traces(root, family, lane):
    traces = []
    for episode in records(root / family / "episodes.jsonl"):
        points = np.asarray([row["position_xyz_agl_m"] for row in records(root / episode["trajectory"])])
        points[:, 1] -= lane
        traces.append((episode["outcome"], points))
    return traces


def rail_traces(directory):
    ends = read_csv(directory / "terminal_events.csv")
    traces = []
    for trajectory, result in zip(sorted((directory / "trajectories_csv").glob("episode_*.csv")), ends, strict=True):
        points = np.asarray([[float(row[axis]) for axis in "xyz"] for row in read_csv(trajectory)])
        traces.append((result["terminal_type"], points))
    return traces


def add_box(ax, item, lane):
    x0, x1 = item["x_m"]
    y0, y1 = (y - lane for y in item["y_m"])
    z0, z1 = item["z_agl_m"]
    vertices = [(x, y, z) for z in (z0, z1) for x, y in ((x0, y0), (x0, y1), (x1, y1), (x1, y0))]
    quads = ((0, 1, 2, 3), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7))
    ax.add_collection3d(Poly3DCollection([[vertices[i] for i in face] for face in quads],
                                         facecolor="#a5b3c2", edgecolor="#667b8e", linewidth=.3, alpha=.22))


def axis_bounds(values, pad):
    return float(min(values) - pad), float(max(values) + pad)


def add_tracks(ax, cases):
    for outcome, pts in cases:
        color = COLORS[outcome]
        ax.plot(*pts.T, color=color, lw=1.05, alpha=.60)
        ax.scatter(*pts[-1], color=color, s=10, depthshade=False)
    ax.set_box_aspect((3.6, 1.2, 1.0))
    ax.view_init(elev=20, azim=-69)
    ax.tick_params(axis="both", labelsize=7, pad=0)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=3))
    ax.zaxis.set_major_locator(MaxNLocator(nbins=4))


def draw(root, heldout, rails, stage, baseline):
    geometry = read_json(COURSE_FILE)
    rail_geometry = read_json(RAIL.parent / "benchmark_manifest.json")
    fig = plt.figure(figsize=(13.4, 20.4))
    fig.suptitle("Stage 2 vs. Stage 3 | paired flights on the 70 locked cases", fontsize=18, y=.982)
    for index, case in enumerate((*FAMILIES, "t1", "t2", "t3")):
        rail = case in rails
        if rail:
            versions = [rail_traces(Path(path)) for path in (baseline[case]["artifacts"], rails[case]["artifacts"])]
            line = np.asarray(rail_geometry[case]["centerline_ned_m"], dtype=float)
            points = np.concatenate([pts for version in versions for _, pts in version])
            limits = (axis_bounds(points[:, 0], 3), axis_bounds(points[:, 1], 2), axis_bounds(points[:, 2], .2))
        else:
            lane = geometry["families"][case]["lane_y_m"]
            versions = [heldout_traces(folder, case, lane) for folder in (HELDOUT, heldout)]
            limits = ((0, 66), (-6.5, 6.5), (0, 6.4))
        for column, (model, cases) in enumerate(zip(("Stage 2 best@495k", f"Stage 3 @{stage:,}"), versions, strict=True)):
            ax = fig.add_subplot(7, 2, 2 * index + column + 1, projection="3d")
            if rail:
                ax.plot(line[:, 0], line[:, 1], np.zeros(len(line)), "--", color="#7d8790", lw=.75, alpha=.7)
            else:
                for box in obstacles(case, geometry):
                    add_box(ax, box, lane)
            add_tracks(ax, cases)
            successes = sum(outcome == "success" for outcome, _ in cases)
            ax.set(title=f"{case.upper()}   {model}   {successes}/10 reached goal",
                   xlabel="X (m)", ylabel="Y (m)", zlabel="Z up (m)" if rail else "AGL (m)",
                   xlim=limits[0], ylim=limits[1], zlim=limits[2])
    handles = [Line2D([0], [0], color=color, lw=2, label=name.replace("_", " ").title())
               for name, color in COLORS.items()]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), bbox_to_anchor=(.5, .005), frameon=False)
    fig.subplots_adjust(top=.940, bottom=.025, left=.035, right=.955, hspace=.27, wspace=.12)
    fig.savefig(root / "flight_evidence.png", dpi=164, facecolor="white")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--heldout", type=Path)
    parser.add_argument("--rail", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rail-gate-sha256", help="Verify full T1/T2 results before continuing to the remaining cases")
    parser.add_argument("--rail-gate-scenes", nargs="+", choices=("t1", "t2"), default=("t1", "t2"))
    args = parser.parse_args()
    baseline = read_json(RAIL / "overall_summary.json")
    if args.rail_gate_sha256:
        rail_gates = {scene: screen_rail(args.rail / scene, scene, baseline[scene], args.rail_gate_sha256)
                      for scene in args.rail_gate_scenes}
        if any(result["episodes"] != 10 for result in rail_gates.values()):
            raise RuntimeError("Full railway acceptance requires all ten locked episodes in T1 and T2")
        passed = all(result["passed"] for result in rail_gates.values())
        write_json(args.output / "final_rail_gate.json",
                   {"passed": passed, "checkpoint_sha256": args.rail_gate_sha256, "rail": rail_gates})
        print(f"[Full rail audit] {'/'.join(rail_gates).upper()} gate={passed}", flush=True)
        return
    if args.heldout is None:
        parser.error("Full trajectory evidence requires the held-out results")
    comparison = read_json(args.output / "benchmark_comparison.json")
    if comparison["reset_depth_checks_passed"] != 40:
        raise RuntimeError("The selected checkpoint must have 40 consistent held-out depth resets")
    results = {scene: read_json(args.rail / scene / "result.json") for scene in ("t1", "t2", "t3")}
    rail_gate = read_json(args.output / "final_rail_gate.json")
    if (not rail_gate["passed"] or set(rail_gate["rail"]) != {"t1", "t2"}
            or rail_gate["checkpoint_sha256"] != comparison["stage3_checkpoint_sha256"]):
        raise RuntimeError("Full evidence requires a checkpoint with verified T1/T2 results")
    draw(args.output, args.heldout, results, comparison["stage3_checkpoint_step"], baseline)
    print("[Final 70 cases] depth resets 40/40, full T1/T2 gate passed", flush=True)


if __name__ == "__main__":
    main()
