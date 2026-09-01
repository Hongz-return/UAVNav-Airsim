"""Validate canonical experiment JSON, CSV structure, and referenced paths."""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
TRAIN_RUNS = ROOT / "Logs" / "TrainData" / "Runs"
EVA_RESULTS = ROOT / "Logs" / "EvaData" / "Results"
SELECTION = ROOT / "Logs" / "EvaData" / "CheckpointSelection"
FORMAL = ROOT / "Logs" / "EvaData" / "FormalBatches"
VALIDATION_REPORT = ROOT / "reports" / "data_validation" / "metadata_validation.json"
PATH_KEYS = {
    "model_absolute_path",
    "training_run_absolute_path",
    "model_file",
    "model_run",
    "result_dir",
    "run_dir",
    "best_model",
    "model",
}
STALE_TRAINING = re.compile(r"logs_revision[\\/]training", re.IGNORECASE)


def resolve_recorded(raw: str) -> Path:
    path = Path(raw)
    return (path if path.is_absolute() else ROOT / path).resolve()


def path_values(value: Any, location: str = "$"):
    if isinstance(value, dict):
        for key, child in value.items():
            child_location = f"{location}.{key}"
            if key in PATH_KEYS and isinstance(child, str) and child.strip():
                yield child_location, child
            yield from path_values(child, child_location)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from path_values(child, f"{location}[{index}]")


def main() -> int:
    active_roots = (TRAIN_RUNS, EVA_RESULTS, SELECTION, FORMAL)
    json_paths = sorted(
        path
        for active_root in active_roots
        for path in active_root.rglob("*.json")
    )
    parse_failures: list[dict[str, str]] = []
    missing_references: list[dict[str, str]] = []
    stale_path_files: list[str] = []
    parsed: dict[Path, Any] = {}
    for path in json_paths:
        text = path.read_text(encoding="utf-8-sig")
        if STALE_TRAINING.search(text):
            stale_path_files.append(path.relative_to(ROOT).as_posix())
        try:
            value = json.loads(text)
        except json.JSONDecodeError as exc:
            parse_failures.append(
                {"path": path.relative_to(ROOT).as_posix(), "error": str(exc)}
            )
            continue
        parsed[path] = value
        for location, raw in path_values(value):
            referenced = resolve_recorded(raw)
            if not referenced.exists():
                missing_references.append(
                    {
                        "source": path.relative_to(ROOT).as_posix(),
                        "location": location,
                        "recorded_path": raw,
                    }
                )

    csv_failures: list[dict[str, object]] = []
    selection_csv_count = 0
    for path in sorted(SELECTION.rglob("*.csv")):
        if path.name not in {
            "screen_results.csv",
            "confirm_results.csv",
            "test_results.csv",
            "selected_checkpoints.csv",
        }:
            continue
        selection_csv_count += 1
        with path.open("r", encoding="utf-8-sig", newline="") as stream:
            rows = list(csv.DictReader(stream))
        malformed = sum(1 for row in rows if row.get(None))
        if malformed:
            csv_failures.append(
                {
                    "path": path.relative_to(ROOT).as_posix(),
                    "malformed_row_count": malformed,
                }
            )
        if STALE_TRAINING.search(path.read_text(encoding="utf-8-sig")):
            stale_path_files.append(path.relative_to(ROOT).as_posix())

    evaluation_manifests = list(EVA_RESULTS.rglob("evaluation_manifest.json"))
    run_manifests = list(TRAIN_RUNS.rglob("run.json"))
    result = {
        "status": (
            "pass"
            if not parse_failures
            and not missing_references
            and not csv_failures
            and not stale_path_files
            else "fail"
        ),
        "json_file_count": len(json_paths),
        "json_parse_failure_count": len(parse_failures),
        "referenced_path_failure_count": len(missing_references),
        "selection_csv_count": selection_csv_count,
        "selection_csv_failure_count": len(csv_failures),
        "training_run_manifest_count": len(run_manifests),
        "evaluation_manifest_count": len(evaluation_manifests),
        "stale_training_path_file_count": len(set(stale_path_files)),
        "parse_failures": parse_failures[:100],
        "missing_references": missing_references[:100],
        "csv_failures": csv_failures[:100],
        "stale_training_path_files": sorted(set(stale_path_files))[:100],
    }
    VALIDATION_REPORT.parent.mkdir(parents=True, exist_ok=True)
    VALIDATION_REPORT.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
