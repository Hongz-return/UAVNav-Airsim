"""Reject Stage 3 candidates that forget either of the two locked rail tasks."""

import argparse
import ast
from collections import Counter
from pathlib import Path

from .ch4_trainenv_3d_course import ROOT, read_json, sha256, write_json
from .ch4_trainenv_3d_runtime import read_ini
from .evaluate_ch4_stage3_rail import BASELINE as RAIL, SCENE_CONFIGS
from .summarize_ch4_stage3 import FAMILIES, HELDOUT, read_csv, records


PROTOCOL = ROOT / "reports/ch4_trainenv_3d_20261125/eval_protocol.json"
CASE_COUNT = 3


def count_heldout(root, count):
    rows = {family: records(root / family / "episodes.jsonl")[:count] for family in FAMILIES}
    if any(len(items) != count for items in rows.values()):
        raise RuntimeError(f"Incomplete TrainEnv_3D screening evidence: {root}")
    return rows


def screen_rail(root, scene, baseline, checkpoint_sha):
    identity = read_json(root / "protocol_identity.json")
    result = read_json(root / "result.json")
    if identity["seed"] != 2026 or read_json(root / "settings.json")["ClockSpeed"] != 1.0:
        raise RuntimeError(f"{scene} seed or clock speed differs from the locked run")
    if result["map_sha256"] != baseline["map_sha256"] or result["checkpoint_sha256"] != checkpoint_sha:
        raise RuntimeError(f"{scene} map or checkpoint identity does not match the screen")
    live = Path(result["artifacts"])
    original = Path(baseline["artifacts"])
    candidates = read_csv(live / "terminal_events.csv")
    count = len(candidates)
    if not CASE_COUNT <= count <= 10 or count != identity.get("episodes", result["summary"]["episodes"]):
        raise RuntimeError(f"Incomplete screened episode set: {scene}")
    manifest = read_json(live / "scenario_manifest.json")["scenarios"]
    if manifest != read_json(original / "scenario_manifest.json")["scenarios"][:count]:
        raise RuntimeError(f"{scene} evaluated a different episode prefix")
    report = read_json(live / "evaluation_manifest.json")
    if not report["use_history"] or report["history_mode"] != "full":
        raise RuntimeError(f"{scene} disabled the checkpoint's full policy history")
    config = read_ini(ROOT / "configs" / SCENE_CONFIGS[scene])
    low, high = ast.literal_eval(config.get("environment", "work_space_z"))
    heights = []
    for path in sorted((live / "trajectories_csv").glob("episode_*.csv")):
        zs = [float(row["z"]) for row in read_csv(path)]
        heights.append({"episode": path.stem, "minimum_m": min(zs), "maximum_m": max(zs),
                        "outside_vertical_workspace": min(zs) < low or max(zs) > high})
    if len(heights) != count:
        raise RuntimeError(f"Missing {scene} trajectory data needed to audit the height envelope")
    current = Counter(row["terminal_type"] for row in candidates)
    reference = Counter(row["terminal_type"] for row in read_csv(original / "terminal_events.csv")[:count])
    violations = sum(row["outside_vertical_workspace"] for row in heights)
    passed = (violations == 0 and current["success"] >= reference["success"]
              and current["collision"] <= reference["collision"])
    return {"episodes": count, "passed": passed, "stage2": dict(reference), "stage3": dict(current),
            "altitude_violations": violations, "workspace_z_m": [low, high], "heights": heights}


def screen(root, run, step):
    checkpoint = run / "models" / f"checkpoint_{step}_steps.zip"
    digest = sha256(checkpoint)
    summary = read_json(root / "heldout/summary.json")
    if summary["protocol_sha256"] != sha256(PROTOCOL) or summary["checkpoint_sha256"] != digest:
        raise RuntimeError("Held-out checkpoint or locked protocol changed")
    candidate = count_heldout(root / "heldout", CASE_COUNT)
    stage2 = count_heldout(HELDOUT, CASE_COUNT)
    families = {}
    for family, episodes in candidate.items():
        reference = stage2[family]
        if [row["scenario_id"] for row in episodes] != [row["scenario_id"] for row in reference]:
            raise RuntimeError(f"The locked {family} scenario prefix changed")
        if not all(row["reset_depth_consistent"] for row in episodes):
            raise RuntimeError(f"Missing valid reset-depth evidence in {family}")
        families[family] = {
            "stage2": dict(Counter(row["outcome"] for row in reference)),
            "stage3": dict(Counter(row["outcome"] for row in episodes)),
            "stage2_passage": sum(row["route_completed"] for row in reference),
            "stage3_passage": sum(row["route_completed"] for row in episodes),
        }
    rail_baseline = read_json(RAIL / "overall_summary.json")
    rails = {scene: screen_rail(root / scene, scene, rail_baseline[scene], digest) for scene in ("t1", "t2")}
    valid_rails = all(row["passed"] for row in rails.values())
    successes = sum(row["stage3"].get("success", 0) for row in families.values())
    baseline_successes = sum(row["stage2"].get("success", 0) for row in families.values())
    under_successes = families["under"]["stage3"].get("success", 0)
    under_passages = families["under"]["stage3_passage"]
    return {
        "checkpoint": str(checkpoint), "checkpoint_step": step, "checkpoint_sha256": digest,
        "training_run": str(run), "eligible": valid_rails and successes >= baseline_successes,
        "rail_gate_passed": valid_rails, "under_successes": under_successes, "under_passages": under_passages,
        "heldout_successes": successes, "baseline_heldout_successes": baseline_successes,
        "heldout_passages": sum(row["stage3_passage"] for row in families.values()),
        "heldout": families, "rail": rails,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--screen", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--step", type=int, required=True)
    args = parser.parse_args()
    root = args.screen.resolve()
    audit = screen(root, args.run.resolve(), args.step)
    write_json(root / "screening.json", audit)
    print(f"[{args.step}] rail_gate={audit['rail_gate_passed']} 3D={audit['heldout_successes']}/12 "
          f"under={audit['under_successes']}/3 ({audit['under_passages']} passages)", flush=True)


if __name__ == "__main__":
    main()
