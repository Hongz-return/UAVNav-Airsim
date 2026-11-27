"""Compare Stage 3 with the frozen 40-case and 30-case Stage 2 baselines."""

import argparse
from collections import Counter
import csv
from pathlib import Path

import numpy as np

from tools.ue_benchmark.ch4_trainenv_3d_course import ROOT, read_json, write_json


HELDOUT = ROOT / "reports/ch4_trainenv_3d_20261125/eval_stage2_495k_depth_fixed"
RAIL = ROOT / "reports/ch4_benchmark_rebuild_20261125"
FAMILIES = ("over", "under", "side", "over_under")
RAIL_EXIT_STATIONS_M = {"t1": 143.0, "t2": 140.0, "t3": 100.0}


def records(path):
    import json
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream]


def metrics(outcomes, passes, altitude_errors):
    counts = Counter(outcomes)
    n = sum(counts.values())
    return {
        "episodes": n, "success": counts["success"], "collision": counts["collision"],
        "out_of_bounds": counts["out_of_bounds"], "timeout": counts["timeout"],
        "route_not_completed": counts["route_not_completed"], "passage": int(sum(passes)),
        "mean_terminal_altitude_error_m": float(np.mean(altitude_errors)),
    }


def heldout_case(directory, family, *, audited=False):
    rows = records(directory / family / "episodes.jsonl")
    if len(rows) != 10 or (audited and not all(row["reset_depth_consistent"] for row in rows)):
        raise ValueError(f"Incomplete locked flight evidence for {directory}/{family}")
    return metrics([row["outcome"] for row in rows], [row["route_completed"] for row in rows],
                   [row["terminal_altitude_error_m"] for row in rows]), rows


def projected_max_station(points, centerline):
    starts = centerline[:-1]
    vectors = np.diff(centerline, axis=0)
    lengths = np.linalg.norm(vectors, axis=1)
    valid = lengths > 0
    stations = np.r_[0, np.cumsum(lengths)][:-1][valid]
    starts, vectors, lengths = starts[valid], vectors[valid], lengths[valid]
    shifts = points[:, None, :] - starts[None, :, :]
    fraction = np.clip(np.sum(shifts * vectors, axis=-1) / lengths ** 2, 0.0, 1.0)
    nearest = np.argmin(np.sum((shifts - fraction[..., None] * vectors) ** 2, axis=-1), axis=1)
    return float(np.max(stations[nearest] + lengths[nearest] * fraction[np.arange(len(points)), nearest]))


def read_csv(path):
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def rail_case(directory, scene, geometry):
    end = read_csv(directory / "terminal_events.csv")
    csvs = sorted((directory / "trajectories_csv").glob("episode_*.csv"))
    if len(end) != 10 or len(csvs) != 10:
        raise ValueError(f"Expected ten completed railway episodes: {directory}")
    line = np.array(geometry["centerline_ned_m"], dtype=float)[:, :2]
    crossings = []
    for path in csvs:
        trajectory = read_csv(path)
        points = np.array([[float(row["x"]), float(row["y"])] for row in trajectory])
        crossings.append(projected_max_station(points, line) >= RAIL_EXIT_STATIONS_M[scene])
    return metrics(
        [row["terminal_type"] for row in end], crossings,
        [abs(float(row["terminal_z"]) - float(row["goal_z"])) for row in end],
    )


def combine(rows):
    counts = {key: sum(row[key] for row in rows) for key in rows[0] if key != "mean_terminal_altitude_error_m"}
    return {
        **counts,
        "mean_terminal_altitude_error_m": sum(
            row["episodes"] * row["mean_terminal_altitude_error_m"] for row in rows
        ) / counts["episodes"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--heldout", type=Path, required=True)
    parser.add_argument("--rail", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rail_baseline = read_json(RAIL / "eval_stage2_495k/overall_summary.json")
    heldout_baseline = read_json(HELDOUT / "summary.json")
    candidate = read_json(args.heldout / "summary.json")
    for key in ("protocol_sha256", "geometry_sha256"):
        if candidate[key] != heldout_baseline[key]:
            raise RuntimeError(f"Stage 3 changed the locked held-out {key}")
    if not candidate["complete_protocol"]:
        raise RuntimeError("The Stage 3 candidate does not have 40 completed held-out cases")
    geometry = read_json(RAIL / "benchmark_manifest.json")
    rows = []
    heldout_pairs = []
    for family in FAMILIES:
        stage2, base_cases = heldout_case(HELDOUT, family)
        stage3, new_cases = heldout_case(args.heldout, family, audited=True)
        for a, b in zip(base_cases, new_cases, strict=True):
            if a["scenario_id"] != b["scenario_id"]:
                raise RuntimeError("Stage 3 used a different case ordering")
            heldout_pairs.append((a["outcome"] == "success", b["outcome"] == "success"))
        rows.append({"case": family, "stage2": stage2, "stage3": stage3})
    tracks_3d = list(rows)
    rows.append({"case": "TrainEnv_3D", **{
        stage: combine([row[stage] for row in tracks_3d]) for stage in ("stage2", "stage3")
    }})
    tracks_rail = []
    for scene, original in sorted(rail_baseline.items()):
        result = read_json(args.rail / scene / "result.json")
        if result["map_sha256"] != original["map_sha256"] or result["checkpoint_sha256"] != candidate["checkpoint_sha256"]:
            raise RuntimeError(f"Mismatched railway map or model identity in {scene}")
        outputs = Path(result["artifacts"])
        baseline_outputs = Path(original["artifacts"])
        if read_json(outputs / "scenario_manifest.json") != read_json(baseline_outputs / "scenario_manifest.json"):
            raise RuntimeError(f"Stage 3 evaluated different railway cases in {scene}")
        tracks_rail.append({
            "case": scene.upper(), "stage2": rail_case(baseline_outputs, scene, geometry[scene]),
            "stage3": rail_case(outputs, scene, geometry[scene]),
        })
    rows.extend(tracks_rail)
    rows.append({"case": "Railway", **{
        stage: combine([row[stage] for row in tracks_rail]) for stage in ("stage2", "stage3")
    }})
    args.output.mkdir(parents=True, exist_ok=True)
    write_json(args.output / "benchmark_comparison.json", {
        "stage2_checkpoint_sha256": heldout_baseline["checkpoint_sha256"],
        "stage3_checkpoint_step": candidate["checkpoint_step"],
        "stage3_checkpoint_sha256": candidate["checkpoint_sha256"],
        "protocol_sha256": candidate["protocol_sha256"],
        "geometry_sha256": candidate["geometry_sha256"],
        "reset_depth_checks_passed": sum(
            result["reset_depth_checks_passed"] for result in candidate["families"].values()
        ),
        "heldout_new_successes": sum(not a and b for a, b in heldout_pairs),
        "heldout_lost_successes": sum(a and not b for a, b in heldout_pairs),
        "rail_passage_exit_station_m": RAIL_EXIT_STATIONS_M,
        "rows": rows,
    })
    with (args.output / "benchmark_comparison.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=["case", "stage", *rows[0]["stage2"]])
        writer.writeheader()
        for row in rows:
            for stage in ("stage2", "stage3"):
                writer.writerow({"case": row["case"], "stage": stage, **row[stage]})


if __name__ == "__main__":
    main()
