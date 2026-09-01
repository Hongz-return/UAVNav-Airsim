"""Build an auditable inventory of canonical training and evaluation data.

The stable data grain is:

* training: scene / method / seed / run_id
* evaluation: scene / method / seed / run_id / evaluation_id

Historical repeats live below ``_repeat`` and are deliberately excluded from
the 4-scene x 6-method x 3-seed core-coverage calculation.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from tools.experiment_paths import (
    CHECKPOINT_SELECTION_ROOT,
    EVALUATION_RESULTS_ROOT,
    LOGS_ROOT,
    TRAIN_RUNS_ROOT,
)


CORE_SCENES = ("tunnel", "tunnel_obs", "tunnel_activate", "tunnel_mixed")
CORE_METHODS = ("sac", "lstm", "transformer", "mamba", "ma", "coordatt")
CORE_SEEDS = ("seed_0", "seed_1", "seed_2")


def _identity(relative: Path) -> tuple[str, str, str, str]:
    """Extract scene, method, seed and run id from a canonical relative path."""
    parts = relative.parts
    if parts and parts[0] == "_repeat":
        if len(parts) < 5:
            raise ValueError(f"Invalid repeat path: {relative}")
        return f"_repeat/{parts[1]}", parts[2], parts[3], parts[4]
    if len(parts) < 4:
        raise ValueError(f"Invalid experiment path: {relative}")
    return parts[0], parts[1], parts[2], parts[3]


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _new_group() -> dict[str, Any]:
    return {
        "run_manifest_count": 0,
        "run_without_manifest_count": 0,
        "seeds": set(),
        "model_zip_count": 0,
        "periodic_checkpoint_count": 0,
        "best_model_count": 0,
        "final_model_count": 0,
        "evaluation_manifest_count": 0,
        "evaluation_100_episode_count": 0,
        "evaluation_episode_counts": Counter(),
    }


def build_inventory() -> dict[str, Any]:
    groups: defaultdict[tuple[str, str], dict[str, Any]] = defaultdict(_new_group)
    parse_failures: list[dict[str, str]] = []
    run_dirs: set[Path] = set()
    run_coverage: set[tuple[str, str, str]] = set()
    evaluation_coverage: set[tuple[str, str, str]] = set()
    formal_combined_tests: defaultdict[
        tuple[str, str, str], dict[str, set[Any]]
    ] = defaultdict(lambda: {"episodes": set(), "sources": set()})
    incomplete_runs: list[dict[str, Any]] = []

    for manifest in sorted(TRAIN_RUNS_ROOT.rglob("run.json")):
        run_dir = manifest.parent
        relative = run_dir.relative_to(TRAIN_RUNS_ROOT)
        try:
            scene, method, seed, run_id = _identity(relative)
            _load_json(manifest)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            parse_failures.append({"file": str(manifest), "error": str(exc)})
            continue

        run_dirs.add(run_dir)
        group = groups[(scene, method)]
        group["run_manifest_count"] += 1
        group["seeds"].add(seed)
        if scene in CORE_SCENES and method in CORE_METHODS and seed in CORE_SEEDS:
            run_coverage.add((scene, method, seed))

        model_dir = run_dir / "models"
        has_best = (model_dir / "best_model.zip").is_file()
        has_final = (model_dir / "model_sb3.zip").is_file()
        if not has_best or not has_final:
            incomplete_runs.append(
                {
                    "relative_run": relative.as_posix(),
                    "scene": scene,
                    "method": method,
                    "seed": seed,
                    "run_id": run_id,
                    "has_best_model": has_best,
                    "has_final_model": has_final,
                    "periodic_checkpoint_count": sum(
                        1
                        for candidate in model_dir.glob("*.zip")
                        if candidate.name not in {"best_model.zip", "model_sb3.zip"}
                    ),
                }
            )

    model_run_dirs: set[Path] = set()
    for model_file in sorted(TRAIN_RUNS_ROOT.rglob("*.zip")):
        if model_file.parent.name != "models":
            continue
        run_dir = model_file.parent.parent
        relative = run_dir.relative_to(TRAIN_RUNS_ROOT)
        try:
            scene, method, seed, _run_id = _identity(relative)
        except ValueError as exc:
            parse_failures.append({"file": str(model_file), "error": str(exc)})
            continue

        model_run_dirs.add(run_dir)
        group = groups[(scene, method)]
        group["seeds"].add(seed)
        group["model_zip_count"] += 1
        if model_file.name == "best_model.zip":
            group["best_model_count"] += 1
        elif model_file.name == "model_sb3.zip":
            group["final_model_count"] += 1
        else:
            group["periodic_checkpoint_count"] += 1

    for run_dir in sorted(model_run_dirs - run_dirs):
        relative = run_dir.relative_to(TRAIN_RUNS_ROOT)
        scene, method, _seed, _run_id = _identity(relative)
        groups[(scene, method)]["run_without_manifest_count"] += 1

    for manifest in sorted(EVALUATION_RESULTS_ROOT.rglob("evaluation_manifest.json")):
        evaluation_dir = manifest.parent
        relative = evaluation_dir.relative_to(EVALUATION_RESULTS_ROOT)
        try:
            scene, method, seed, _run_id = _identity(relative)
            payload = _load_json(manifest)
            episodes = int(payload["episodes"])
        except (KeyError, TypeError, OSError, ValueError, json.JSONDecodeError) as exc:
            parse_failures.append({"file": str(manifest), "error": str(exc)})
            continue

        group = groups[(scene, method)]
        group["evaluation_manifest_count"] += 1
        group["evaluation_episode_counts"][episodes] += 1
        if episodes == 100:
            group["evaluation_100_episode_count"] += 1
        if scene in CORE_SCENES and method in CORE_METHODS and seed in CORE_SEEDS:
            evaluation_coverage.add((scene, method, seed))

    for result_csv in sorted(CHECKPOINT_SELECTION_ROOT.rglob("test_results.csv")):
        try:
            with result_csv.open("r", encoding="utf-8-sig", newline="") as handle:
                for row in csv.DictReader(handle):
                    if row.get("record_type") != "combined" or row.get("stage") != "test":
                        continue
                    scene = row.get("scene", "")
                    method = row.get("method", "")
                    seed = f"seed_{int(row['seed'])}"
                    episodes = int(row["episodes"])
                    key = (scene, method, seed)
                    if (
                        scene == "tunnel_mixed"
                        and method in CORE_METHODS
                        and seed in CORE_SEEDS
                    ):
                        formal_combined_tests[key]["episodes"].add(episodes)
                        formal_combined_tests[key]["sources"].add(
                            result_csv.relative_to(LOGS_ROOT).as_posix()
                        )
        except (KeyError, TypeError, OSError, ValueError, csv.Error) as exc:
            parse_failures.append({"file": str(result_csv), "error": str(exc)})

    expected = {
        (scene, method, seed)
        for scene in CORE_SCENES
        for method in CORE_METHODS
        for seed in CORE_SEEDS
    }

    group_rows: list[dict[str, Any]] = []
    for (scene, method), values in sorted(groups.items()):
        group_rows.append(
            {
                "scene": scene,
                "method": method,
                "run_manifest_count": values["run_manifest_count"],
                "run_without_manifest_count": values["run_without_manifest_count"],
                "seeds": sorted(values["seeds"]),
                "model_zip_count": values["model_zip_count"],
                "periodic_checkpoint_count": values["periodic_checkpoint_count"],
                "best_model_count": values["best_model_count"],
                "final_model_count": values["final_model_count"],
                "evaluation_manifest_count": values["evaluation_manifest_count"],
                "evaluation_100_episode_count": values["evaluation_100_episode_count"],
                "evaluation_episode_counts": {
                    str(key): count
                    for key, count in sorted(values["evaluation_episode_counts"].items())
                },
            }
        )

    core_run_missing = sorted(expected - run_coverage)
    core_evaluation_missing = sorted(expected - evaluation_coverage)
    expected_env4 = {
        ("tunnel_mixed", method, seed)
        for method in CORE_METHODS
        for seed in CORE_SEEDS
    }
    formal_combined_missing = sorted(expected_env4 - set(formal_combined_tests))
    formal_protocol_rows = []
    for (scene, method, seed), values in sorted(formal_combined_tests.items()):
        formal_protocol_rows.append(
            {
                "scene": scene,
                "method": method,
                "seed": seed,
                "episode_options": sorted(values["episodes"]),
                "sources": sorted(values["sources"]),
            }
        )
    model_totals = Counter()
    for row in group_rows:
        model_totals["model_zip_count"] += row["model_zip_count"]
        model_totals["periodic_checkpoint_count"] += row["periodic_checkpoint_count"]
        model_totals["best_model_count"] += row["best_model_count"]
        model_totals["final_model_count"] += row["final_model_count"]

    return {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "data_grain": {
            "training": "scene/method/seed/run_id",
            "evaluation": "scene/method/seed/run_id/evaluation_id",
            "repeat_policy": "_repeat is preserved separately and excluded from core coverage",
        },
        "roots": {
            "training": str(TRAIN_RUNS_ROOT),
            "evaluation": str(EVALUATION_RESULTS_ROOT),
        },
        "totals": {
            "training_run_manifests": len(run_dirs),
            "training_runs_without_manifest": len(model_run_dirs - run_dirs),
            **dict(model_totals),
            "evaluation_manifests": sum(
                row["evaluation_manifest_count"] for row in group_rows
            ),
            "parse_failures": len(parse_failures),
        },
        "core_coverage": {
            "definition": {
                "scenes": list(CORE_SCENES),
                "methods": list(CORE_METHODS),
                "seeds": list(CORE_SEEDS),
                "expected_combinations": len(expected),
            },
            "training_combinations_found": len(run_coverage),
            "training_combinations_missing": [list(item) for item in core_run_missing],
            "evaluation_combinations_found": len(evaluation_coverage),
            "evaluation_combinations_missing": [
                list(item) for item in core_evaluation_missing
            ],
            "status": (
                "pass"
                if not core_run_missing and not core_evaluation_missing
                else "fail"
            ),
        },
        "env4_formal_combined_test_coverage": {
            "definition": (
                "tunnel_mixed combined/test rows in checkpoint-selection "
                "test_results.csv"
            ),
            "expected_combinations": len(expected_env4),
            "combinations_found": len(formal_combined_tests),
            "combinations_missing": [list(item) for item in formal_combined_missing],
            "status": "pass" if not formal_combined_missing else "fail",
            "protocols": formal_protocol_rows,
        },
        "known_exceptions": {
            "incomplete_or_superseded_run_count": len(incomplete_runs),
            "incomplete_or_superseded_runs": incomplete_runs,
            "artifact_runs_without_manifest": [
                run_dir.relative_to(TRAIN_RUNS_ROOT).as_posix()
                for run_dir in sorted(model_run_dirs - run_dirs)
            ],
        },
        "groups": group_rows,
        "parse_failures": parse_failures,
    }


def write_inventory(inventory: dict[str, Any], json_path: Path, csv_path: Path) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(
        json.dumps(inventory, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    columns = (
        "scene",
        "method",
        "run_manifest_count",
        "run_without_manifest_count",
        "seeds",
        "model_zip_count",
        "periodic_checkpoint_count",
        "best_model_count",
        "final_model_count",
        "evaluation_manifest_count",
        "evaluation_100_episode_count",
        "evaluation_episode_counts",
    )
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in inventory["groups"]:
            output = dict(row)
            output["seeds"] = ";".join(row["seeds"])
            output["evaluation_episode_counts"] = json.dumps(
                row["evaluation_episode_counts"], sort_keys=True
            )
            writer.writerow(output)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--json-output",
        type=Path,
        default=LOGS_ROOT / "experiment_inventory.json",
    )
    parser.add_argument(
        "--csv-output",
        type=Path,
        default=LOGS_ROOT / "experiment_inventory.csv",
    )
    args = parser.parse_args()

    inventory = build_inventory()
    write_inventory(inventory, args.json_output, args.csv_output)
    print(json.dumps(inventory["totals"], ensure_ascii=False, indent=2))
    print(json.dumps(inventory["core_coverage"], ensure_ascii=False, indent=2))
    print(
        json.dumps(
            inventory["env4_formal_combined_test_coverage"],
            ensure_ascii=False,
            indent=2,
        )
    )
    failed = bool(inventory["parse_failures"])
    failed = failed or inventory["core_coverage"]["status"] != "pass"
    failed = (
        failed
        or inventory["env4_formal_combined_test_coverage"]["status"] != "pass"
    )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
