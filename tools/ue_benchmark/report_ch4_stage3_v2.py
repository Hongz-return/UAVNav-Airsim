"""Rank gated checkpoints and summarize their training telemetry."""

import argparse
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from .ch4_trainenv_3d_course import read_json, sha256, write_json
from .summarize_ch4_stage3 import records


def metrics(candidate):
    return (candidate["heldout_successes"], candidate["heldout_passages"], candidate["under_successes"],
            candidate["under_passages"], candidate["rail"]["t1"]["stage3"].get("success", 0),
            candidate["rail"]["t2"]["stage3"].get("success", 0))


def dominates(better, other):
    comparisons = list(zip(metrics(better), metrics(other), strict=True))
    return all(a >= b for a, b in comparisons) and any(a > b for a, b in comparisons)


def selection(report):
    candidates = []
    for document in sorted((report / "screens").glob("*/screening.json")):
        candidate = read_json(document)
        if sha256(Path(candidate["checkpoint"])) != candidate["checkpoint_sha256"]:
            raise RuntimeError(f"Candidate weights changed after screening: {candidate['checkpoint']}")
        tag = document.parent.name
        final_path = report / f"final_{tag}/final_rail_gate.json"
        final_passed = read_json(final_path)["passed"] if final_path.is_file() else None
        candidates.append({**candidate, "tag": tag, "screen": str(document), "full_rail_gate": final_passed})
    eligible = [item for item in candidates if item["eligible"] and item["full_rail_gate"] is not False]
    frontier = [item for item in eligible if not any(dominates(other, item) for other in eligible)]
    ordered = sorted(frontier, key=lambda item: (
        item["under_successes"] > 0, item["heldout_successes"], item["under_successes"],
        item["under_passages"], item["heldout_passages"], sum(metrics(item)[-2:]), -item["checkpoint_step"],
    ), reverse=True)
    return {"screens": candidates, "pareto_front": [item["tag"] for item in ordered],
            "selected": ordered[0] if ordered else None}


def series_metrics(path):
    by_family = defaultdict(list)
    episodes = records(path / "course_episodes.jsonl")
    for row in episodes:
        by_family[row["family"]].append(row)
    families = {}
    for name, rows in sorted(by_family.items()):
        families[name] = {
            "episodes": len(rows), "successes": sum(row["success"] for row in rows),
            "collisions": sum(row["collision"] for row in rows),
            "out_of_bounds": sum(row["out_of_bounds"] for row in rows),
            "timeouts": sum(row["timeout"] for row in rows),
            "completed_routes_with_gates": sum(row["route_complete"] and row["gates_total"] > 0 for row in rows),
            "passes_by_gate_count": dict(Counter(row["gates_passed"] for row in rows if row["gates_total"] > 0)),
            "median_terminal_altitude_error_m": median(row["terminal_altitude_error_m"] for row in rows),
        }
    traces = defaultdict(list)
    for file in (path / "tb_logs").rglob("events.out.tfevents*"):
        accumulator = EventAccumulator(str(file), size_guidance={"scalars": 0}).Reload()
        for tag in ("q/mean", "q/td_error", "train/critic_loss", "train/ent_coef", "train/demo_action_loss"):
            if tag in accumulator.Tags()["scalars"]:
                traces[tag].extend(accumulator.Scalars(tag))
    telemetry = {}
    for tag, readings in traces.items():
        readings.sort(key=lambda row: row.step)
        telemetry[tag] = {"scalars": len(readings), "first_ten_median": median(row.value for row in readings[:10]),
                          "last_ten_median": median(row.value for row in readings[-10:])}
    manifest = read_json(path / "run.json")
    return {**read_json(path / "training_summary.json"), "run": str(path), "families": families,
            "demo_action_anchor": manifest.get("demo_action_anchor"),
            "demo_terminal_altitude_weight": manifest.get("demo_terminal_altitude_weight", 0.0),
            "demo_replay_fraction": manifest.get("demo_replay_fraction", 0.0),
            "tensorboard": telemetry}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--series", type=Path, nargs="+", required=True)
    args = parser.parse_args()
    report = args.report.resolve()
    ranking = selection(report)
    slices = [series_metrics(file.parent) for series in args.series
              for file in sorted(series.resolve().glob("slice_*/training_summary.json"))]
    write_json(report / "candidate_selection.json", ranking)
    write_json(report / "training_stability.json", {
        "total_new_steps": sum(row["added_steps"] for row in slices),
        "total_gradient_updates": sum(row["final"]["updates"] - row["initial"]["updates"] for row in slices),
        "slices": slices,
    })
    chosen = ranking["selected"]
    print(f"Completed training transitions: {sum(row['added_steps'] for row in slices):,}")
    print(f"Pareto candidates: {', '.join(ranking['pareto_front']) or 'none'}")
    if chosen is not None:
        print(f"Screened checkpoint {chosen['tag']}: {chosen['heldout_successes']}/12 held-out successes, "
              f"{chosen['under_successes']}/3 under successes")


if __name__ == "__main__":
    main()
