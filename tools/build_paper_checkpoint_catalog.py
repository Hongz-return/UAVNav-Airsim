"""Build the read-only ENV1-ENV4 paper checkpoint catalog.

This script reads existing small CSV/JSON/ZIP artifacts only.  It never starts
UE/AirSim, training, or evaluation.  All generated files are restricted to
reports/paper_checkpoint_catalog/.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import statistics
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "reports" / "paper_checkpoint_catalog"

SOURCE_FILES = {
    "summary_a": (
        "Logs/EvaData/FormalBatches/ENV2_TunnelObs_and_ENV3_TunnelActivate/"
        "summary_all_models.csv"
    ),
    "summary_b": (
        "Logs/EvaData/FormalBatches/ENV1_Tunnel_and_ENV3_TunnelActivate/"
        "summary_all_models.csv"
    ),
    "plan_a": (
        "Logs/EvaData/FormalBatches/ENV2_TunnelObs_and_ENV3_TunnelActivate/plan.json"
    ),
    "plan_b": (
        "Logs/EvaData/FormalBatches/ENV1_Tunnel_and_ENV3_TunnelActivate/plan.json"
    ),
    "selection_test": (
        "Logs/EvaData/CheckpointSelection/CrossScene/2026_08_03_15_38_32_032292/"
        "test_results.csv"
    ),
    "selection_selected": (
        "Logs/EvaData/CheckpointSelection/CrossScene/2026_08_03_15_38_32_032292/"
        "selected_checkpoints.csv"
    ),
    "selection_plan": (
        "Logs/EvaData/CheckpointSelection/CrossScene/2026_08_03_15_38_32_032292/plan.json"
    ),
    "env2_test": (
        "Logs/EvaData/CheckpointSelection/ENV2_TunnelObs/2026_08_02_04_19_23_933357/"
        "test_results.csv"
    ),
    "env2_selected": (
        "Logs/EvaData/CheckpointSelection/ENV2_TunnelObs/2026_08_02_04_19_23_933357/"
        "selected_checkpoints.csv"
    ),
    "env2_plan": (
        "Logs/EvaData/CheckpointSelection/ENV2_TunnelObs/"
        "2026_08_02_04_19_23_933357/plan.json"
    ),
    "dynamic_test": (
        "Logs/EvaData/CheckpointSelection/ENV3_TunnelActivate/2026_08_03_15_04_24_146545/"
        "test_results.csv"
    ),
    "dynamic_selected": (
        "Logs/EvaData/CheckpointSelection/ENV3_TunnelActivate/2026_08_03_15_04_24_146545/"
        "selected_checkpoints.csv"
    ),
    "dynamic_plan": (
        "Logs/EvaData/CheckpointSelection/ENV3_TunnelActivate/"
        "2026_08_03_15_04_24_146545/plan.json"
    ),
    "env4_remote_test": (
        "Logs/EvaData/CheckpointSelection/ENV4_TunnelMixed/"
        "2026_08_09_04_05_13_810051/test_results.csv"
    ),
    "env4_remote_selected": (
        "Logs/EvaData/CheckpointSelection/ENV4_TunnelMixed/"
        "2026_08_09_04_05_13_810051/selected_checkpoints.csv"
    ),
    "env4_lstm01_test": (
        "Logs/EvaData/CheckpointSelection/ENV4_TunnelMixed/"
        "2026_08_10_00_12_21_841091/test_results.csv"
    ),
    "env4_lstm01_selected": (
        "Logs/EvaData/CheckpointSelection/ENV4_TunnelMixed/"
        "2026_08_10_00_12_21_841091/selected_checkpoints.csv"
    ),
    "env4_local_test": (
        "Logs/EvaData/CheckpointSelection/ENV4_TunnelMixed/"
        "2026_08_10_02_26_48_153706/test_results.csv"
    ),
    "env4_local_selected": (
        "Logs/EvaData/CheckpointSelection/ENV4_TunnelMixed/"
        "2026_08_10_02_26_48_153706/selected_checkpoints.csv"
    ),
    "env4_supplement_test": (
        "Logs/EvaData/CheckpointSelection/ENV4_TunnelMixed/"
        "2026_08_10_20_56_23_972923/test_results.csv"
    ),
    "env4_supplement_selected": (
        "Logs/EvaData/CheckpointSelection/ENV4_TunnelMixed/"
        "2026_08_10_20_56_23_972923/selected_checkpoints.csv"
    ),
    "paper_readme": (
        "D:/code/ZipFIle/paper/experiments/paper-main-table-data-2026-08-04/README.md"
    ),
    "paper_summary": (
        "D:/code/ZipFIle/paper/experiments/paper-main-table-data-2026-08-04/"
        "main_table_summary.csv"
    ),
    "paper_selected": (
        "D:/code/ZipFIle/paper/experiments/paper-main-table-data-2026-08-04/"
        "selected_seed_results.csv"
    ),
}

METHOD_ORDER = ["MA", "MambaSAC", "LSTM", "CoordAttSAC", "Transformer", "SAC"]
SCENES = [
    ("ENV1", "tunnel"),
    ("ENV2", "tunnel_obs"),
    ("ENV3", "tunnel_activate"),
    ("ENV4", "tunnel_mixed"),
]
SOURCE_METHOD = {
    "MA": "ma",
    "MambaSAC": "mamba",
    "LSTM": "lstm",
    "CoordAttSAC": "coordatt",
    "Transformer": "transformer",
    "SAC": "sac",
}


def _rows_for_group(scene: str, method: str, entries: Iterable[tuple[Any, ...]]) -> list[dict[str, Any]]:
    return [
        {
            "scene": scene,
            "method": method,
            "seed": seed,
            "label": label,
            "step": step,
            "source": source,
        }
        for seed, label, step, source in entries
    ]


# The rows below are the user-approved candidate matrix.  The source key is
# the metric-bearing CSV; rows are resolved by scene/method/seed and never by
# mixing metrics from different records.
CANDIDATE_SPECS = []


def _add(scene: str, method: str, entries: Iterable[tuple[Any, ...]]) -> None:
    CANDIDATE_SPECS.extend(_rows_for_group(scene, method, entries))


def _add_env4(method: str, entries: Iterable[tuple[Any, ...]]) -> None:
    for seed, label, step, source, episodes in entries:
        CANDIDATE_SPECS.append(
            {
                "scene": "tunnel_mixed",
                "method": method,
                "seed": seed,
                "label": label,
                "step": step,
                "source": source,
                "expected_episodes": episodes,
            }
        )


_add(
    "tunnel",
    "MA",
    [(s, "original_best_model", -1, "summary_b") for s in range(3)],
)
_add(
    "tunnel",
    "MambaSAC",
    [(s, "original_best_model", -1, "summary_b") for s in range(3)],
)
_add(
    "tunnel",
    "LSTM",
    [(0, "best", -1, "selection_test"), (1, "best", -1, "selection_test"), (2, "60k", 60000, "selection_test")],
)
_add(
    "tunnel",
    "Transformer",
    [(s, "original_best_model", -1, "summary_b") for s in range(3)],
)
_add(
    "tunnel",
    "CoordAttSAC",
    [(0, "best", -1, "summary_b"), (1, "best", -1, "summary_b"), (2, "40k", 40000, "selection_test")],
)
_add(
    "tunnel",
    "SAC",
    [(0, "best", -1, "summary_b"), (1, "best", -1, "summary_b"), (2, "60k", 60000, "selection_test")],
)

_add(
    "tunnel_obs",
    "MA",
    [(s, "original_best_model", -1, "summary_a") for s in range(3)],
)
_add(
    "tunnel_obs",
    "MambaSAC",
    [(s, "original_best_model", -1, "summary_a") for s in range(3)],
)
_add(
    "tunnel_obs",
    "LSTM",
    [(0, "100k", 100000, "selection_test"), (1, "60k", 60000, "selection_test"), (2, "40k", 40000, "selection_test")],
)
_add(
    "tunnel_obs",
    "CoordAttSAC",
    [(0, "40k", 40000, "env2_test"), (1, "50k", 50000, "env2_test"), (2, "80k", 80000, "env2_test")],
)
_add(
    "tunnel_obs",
    "Transformer",
    [(0, "150k", 150000, "env2_test"), (1, "best", -1, "env2_test"), (2, "20k", 20000, "env2_test")],
)
_add(
    "tunnel_obs",
    "SAC",
    [(0, "40k", 40000, "selection_test"), (1, "40k", 40000, "selection_test"), (2, "final-150k", 150000, "selection_test")],
)

_add(
    "tunnel_activate",
    "MA",
    [(0, "best", -1, "dynamic_test"), (1, "final-200k", 200000, "dynamic_test"), (2, "200k", 200000, "dynamic_test")],
)
_add(
    "tunnel_activate",
    "MambaSAC",
    [(s, "best", -1, "summary_b") for s in range(3)],
)
_add(
    "tunnel_activate",
    "LSTM",
    [(0, "best", -1, "summary_b"), (1, "best", -1, "summary_b"), (2, "130k", 130000, "dynamic_test")],
)
_add(
    "tunnel_activate",
    "CoordAttSAC",
    [(0, "best", -1, "summary_b"), (1, "best", -1, "summary_b"), (2, "140k", 140000, "dynamic_test")],
)
_add(
    "tunnel_activate",
    "Transformer",
    [(0, "180k", 180000, "dynamic_test"), (1, "180k", 180000, "dynamic_test"), (2, "140k", 140000, "dynamic_test")],
)
_add(
    "tunnel_activate",
    "SAC",
    [(0, "best", 185000, "dynamic_test"), (1, "best", -1, "summary_b"), (2, "best", -1, "summary_b")],
)

_add_env4(
    "MA",
    [
        (0, "40k", 40000, "env4_remote_test", 60),
        (1, "200k", 200000, "env4_remote_test", 60),
        (2, "best", 115000, "env4_remote_test", 60),
    ],
)
_add_env4(
    "MambaSAC",
    [(seed, "best", step, "env4_remote_test", 60) for seed, step in ((0, 95000), (1, 195000), (2, 115000))],
)
_add_env4(
    "LSTM",
    [
        (0, "160k", 160000, "env4_lstm01_test", 60),
        (1, "100k", 100000, "env4_lstm01_test", 60),
        (2, "best", 160000, "env4_supplement_test", 60),
    ],
)
_add_env4(
    "CoordAttSAC",
    [
        (0, "best", 115000, "env4_local_test", 100),
        (1, "200k", 200000, "env4_local_test", 100),
        (2, "200k", 200000, "env4_local_test", 100),
    ],
)
_add_env4(
    "Transformer",
    [
        (0, "200k", 200000, "env4_remote_test", 60),
        (1, "best", 130000, "env4_remote_test", 60),
        (2, "200k", 200000, "env4_remote_test", 60),
    ],
)
_add_env4(
    "SAC",
    [
        (0, "60k", 60000, "env4_local_test", 100),
        (1, "170k", 170000, "env4_local_test", 100),
        (2, "150k", 150000, "env4_local_test", 100),
    ],
)


SOURCE_BATCH = {
    "summary_a": "batch_a",
    "summary_b": "batch_b",
    "selection_test": "checkpoint_selection_2026_08_03",
    "env2_test": "env2_checkpoint_selection_2026_08_02",
    "dynamic_test": "dynamic_checkpoint_selection_2026_08_03",
    "env4_remote_test": "env4_checkpoint_selection_2026_08_09_04_05_13_810051",
    "env4_lstm01_test": "env4_checkpoint_selection_2026_08_10_00_12_21_841091",
    "env4_local_test": "env4_checkpoint_selection_2026_08_10_02_26_48_153706",
    "env4_supplement_test": "env4_checkpoint_selection_2026_08_10_20_56_23_972923",
}

SELECTION_REASON = {
    "summary_a": "原正式评估的100回合逐seed记录；保留原best_model结果作为候选来源。",
    "summary_b": "原正式评估的100回合逐seed记录；保留原best_model结果或与同口径重筛记录组合。",
    "selection_test": "统一筛选后冻结检查点的100回合正式测试；用于纠正训练窗口best_model回调不准确的问题。",
    "env2_test": "ENV2筛选检查点的100回合正式测试；不使用20/40/50回合screen或confirm结果。",
    "dynamic_test": "动态重筛流程的100回合合并正式测试；每条记录由2026与2027各50回合合并，不使用screen或confirm结果。",
    "env4_remote_test": "ENV4远程方法冻结检查点的60回合正式测试；保留原始回合数，不扩写为100回合。",
    "env4_lstm01_test": "ENV4 LSTM seed 0/1冻结检查点的60回合正式测试。",
    "env4_local_test": "ENV4 SAC/CoordAtt冻结检查点的100回合正式测试。",
    "env4_supplement_test": "ENV4补充批次中LSTM seed 2冻结检查点的60回合正式测试。",
}


TARGET_TABLE = {
    ("tunnel", "MA"): (100.00, 39.87, 212.10, 0.0056),
    ("tunnel", "MambaSAC"): (100.00, 39.72, 212.25, 0.0064),
    ("tunnel", "LSTM"): (99.00, 39.67, 212.55, 0.0613),
    ("tunnel", "Transformer"): (99.67, 43.04, 214.78, 0.0027),
    ("tunnel", "CoordAttSAC"): (99.00, 61.70, 216.16, 0.8367),
    ("tunnel", "SAC"): (82.67, 44.90, 217.11, 0.1407),
    ("tunnel_obs", "MA"): (98.00, 43.38, 219.86, 0.0784),
    ("tunnel_obs", "MambaSAC"): (92.00, 44.51, 225.15, 0.1646),
    ("tunnel_obs", "LSTM"): (99.00, 40.05, 222.97, 0.2032),
    ("tunnel_obs", "CoordAttSAC"): (98.00, 69.30, 219.48, 0.3693),
    ("tunnel_obs", "Transformer"): (79.00, 64.67, 229.38, 0.0692),
    ("tunnel_obs", "SAC"): (83.33, 64.63, 221.73, 0.7409),
    ("tunnel_activate", "MA"): (59.33, 42.75, 214.65, 0.3710),
    ("tunnel_activate", "MambaSAC"): (55.67, 39.91, 214.87, 0.3526),
    ("tunnel_activate", "LSTM"): (51.00, 41.96, 218.31, 0.4323),
    ("tunnel_activate", "CoordAttSAC"): (49.67, 47.73, 217.27, 0.4760),
    ("tunnel_activate", "Transformer"): (49.33, 43.40, 214.15, 0.3154),
    ("tunnel_activate", "SAC"): (49.00, 47.30, 218.58, 0.5621),
}


def repo_path(path_string: str) -> Path:
    path = Path(path_string)
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def read_csv_with_lines(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = []
        for line_number, row in enumerate(reader, start=2):
            row["_line"] = str(line_number)
            rows.append(row)
        return rows


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def number(value: Any) -> float | None:
    if value is None or str(value).strip() == "":
        return None
    return float(value)


def integer_or_minus_one(value: Any) -> int:
    if value is None or str(value).strip() == "":
        return -1
    return int(float(value))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_inputs() -> tuple[dict[str, list[dict[str, str]]], dict[str, dict[str, Any]]]:
    csv_inputs: dict[str, list[dict[str, str]]] = {}
    for key in (
        "summary_a",
        "summary_b",
        "selection_test",
        "selection_selected",
        "env2_test",
        "env2_selected",
        "dynamic_test",
        "dynamic_selected",
        "env4_remote_test",
        "env4_remote_selected",
        "env4_lstm01_test",
        "env4_lstm01_selected",
        "env4_local_test",
        "env4_local_selected",
        "env4_supplement_test",
        "env4_supplement_selected",
    ):
        path = repo_path(SOURCE_FILES[key])
        if not path.is_file():
            raise FileNotFoundError(path)
        csv_inputs[key] = read_csv_with_lines(path)

    json_inputs: dict[str, dict[str, Any]] = {}
    for key in ("plan_a", "plan_b", "selection_plan", "env2_plan", "dynamic_plan"):
        path = repo_path(SOURCE_FILES[key])
        if not path.is_file():
            raise FileNotFoundError(path)
        json_inputs[key] = read_json(path)
    return csv_inputs, json_inputs


def find_source_row(
    source_key: str,
    scene: str,
    method: str,
    seed: int,
    csv_inputs: dict[str, list[dict[str, str]]],
) -> dict[str, str] | None:
    method_key = SOURCE_METHOD[method]
    rows = csv_inputs[source_key]
    matches = [
        row
        for row in rows
        if row.get("scene") == scene
        and row.get("method") == method_key
        and row.get("seed") == str(seed)
    ]
    if source_key.startswith("summary_"):
        return matches[0] if matches else None

    # The checkpoint-selection CSVs contain 50-episode test rows followed by
    # one combined 100-episode row for dynamic selection.  Restrict the lookup
    # so a screen/confirm or 50-episode row can never enter the catalog.
    if source_key == "dynamic_test" or source_key.startswith("env4_"):
        matches = [
            row
            for row in matches
            if row.get("record_type") == "combined"
            and row.get("stage") == "test"
        ]
    else:
        matches = [
            row
            for row in matches
            if row.get("record_type") == "batch"
            and row.get("stage") == "test"
            and row.get("episodes") == "100"
        ]
    return matches[0] if matches else None


def plan_meta(source_key: str, json_inputs: dict[str, dict[str, Any]]) -> tuple[str, str]:
    if source_key == "summary_a":
        plan = json_inputs["plan_a"]
        return str(plan.get("scenario_seed", "unknown")), str(plan.get("episodes_per_model", "100"))
    if source_key == "summary_b":
        plan = json_inputs["plan_b"]
        return str(plan.get("scenario_seed", "unknown")), str(plan.get("episodes_per_model", "100"))
    return "", ""


def model_path_from_summary(row: dict[str, str], checkpoint_label: str, checkpoint_step: int) -> Path | None:
    model_run = row.get("model_run", "")
    if not model_run:
        return None
    if checkpoint_label in {"best", "original_best_model"}:
        filename = "best_model.zip"
    elif checkpoint_label.startswith("final-"):
        filename = "model_sb3.zip"
    elif checkpoint_step >= 0:
        filename = f"checkpoint_{checkpoint_step}_steps.zip"
    else:
        filename = "best_model.zip"
    return repo_path(model_run) / "models" / filename


def model_path_from_selection_record(row: dict[str, str]) -> Path | None:
    """Resolve a selected model after historical training-batch directories were flattened."""
    raw_absolute = str(row.get("model_absolute_path", "") or "").strip()
    if not raw_absolute:
        return None

    recorded_path = Path(raw_absolute)
    if recorded_path.is_file():
        return recorded_path

    normalized = raw_absolute.replace("\\", "/")
    marker = "/logs_revision/training/"
    marker_index = normalized.lower().find(marker)
    if marker_index < 0:
        return None

    tail = normalized[marker_index + len(marker) :].split("/")
    historical_batch = re.fullmatch(
        r"\d{4}_\d{2}_\d{2}_\d{2}_\d{2}_\d{2}_\d{6}",
        tail[0],
    )
    if historical_batch and len(tail) > 1:
        tail = tail[1:]

    candidate = REPO_ROOT / "Logs" / "TrainData" / "Runs" / Path(*tail)
    return candidate if candidate.is_file() else None


def normalize_model_sha(
    source_key: str,
    row: dict[str, str],
    checkpoint_label: str,
    checkpoint_step: int,
    sha_cache: dict[Path, str],
) -> tuple[str, str]:
    recorded = str(row.get("model_sha256", "") or "").strip().lower()
    if source_key.startswith("summary_"):
        model_path = model_path_from_summary(row, checkpoint_label, checkpoint_step)
    else:
        model_path = model_path_from_selection_record(row)
    if model_path is None or not model_path.is_file():
        return recorded, ""
    if model_path not in sha_cache:
        sha_cache[model_path] = sha256_file(model_path)
    actual = sha_cache[model_path]
    if recorded and recorded != actual:
        return recorded, ""
    return actual, str(model_path.relative_to(REPO_ROOT)).replace("\\", "/")


def selection_row_for(
    source_key: str,
    scene: str,
    method: str,
    seed: int,
    csv_inputs: dict[str, list[dict[str, str]]],
) -> dict[str, str] | None:
    selected_key = {
        "selection_test": "selection_selected",
        "env2_test": "env2_selected",
        "dynamic_test": "dynamic_selected",
        "env4_remote_test": "env4_remote_selected",
        "env4_lstm01_test": "env4_lstm01_selected",
        "env4_local_test": "env4_local_selected",
        "env4_supplement_test": "env4_supplement_selected",
    }.get(source_key)
    if not selected_key:
        return None
    method_key = SOURCE_METHOD[method]
    rows = [
        row
        for row in csv_inputs[selected_key]
        if row.get("scene") == scene
        and row.get("method") == method_key
        and row.get("seed") == str(seed)
    ]
    return rows[0] if rows else None


def resolve_rows(
    csv_inputs: dict[str, list[dict[str, str]]],
    json_inputs: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    sha_cache: dict[Path, str] = {}
    for spec in CANDIDATE_SPECS:
        source_key = spec["source"]
        source_file = SOURCE_FILES[source_key]
        source_row = find_source_row(source_key, spec["scene"], spec["method"], spec["seed"], csv_inputs)
        status = "verified_100ep_same_row"
        if source_row is None:
            status = "unresolved"
            source_row = {}

        episodes = integer_or_minus_one(source_row.get("episodes"))
        expected_episodes = int(spec.get("expected_episodes", 100))
        if episodes != expected_episodes:
            status = "unresolved"

        raw_step = source_row.get("checkpoint_step", "")
        checkpoint_step = integer_or_minus_one(raw_step)
        if checkpoint_step == -1 and spec["step"] >= 0:
            checkpoint_step = spec["step"]
        if raw_step not in (None, "") and spec["step"] >= 0 and int(float(raw_step)) != spec["step"]:
            status = "unresolved"

        # For a symbolic best_model source the step is intentionally -1.  For
        # dynamic SAC seed 0, the source explicitly records best at 185000.
        if source_key.startswith("summary_") and spec["label"] in {"best", "original_best_model"}:
            checkpoint_step = -1

        success = number(source_row.get("success_rate_pct"))
        reward = number(source_row.get("avg_reward"))
        time_success = number(source_row.get("avg_time_success_sec"))
        path_success = number(source_row.get("avg_path_length_success_m"))
        smoothness = number(source_row.get("smoothness"))
        if success is None or smoothness is None:
            status = "unresolved"
        if success is not None and success > 0 and (time_success is None or path_success is None):
            status = "unresolved"

        if source_key.startswith("summary_"):
            scenario_seed, planned_episodes = plan_meta(source_key, json_inputs)
            if planned_episodes != "100":
                status = "unresolved"
        elif source_key == "dynamic_test":
            scenario_seed = "2026+2027 (50+50)"
        else:
            scenario_seed = str(source_row.get("scenario_seed", ""))

        selected_row = selection_row_for(source_key, spec["scene"], spec["method"], spec["seed"], csv_inputs)
        if source_key.startswith("summary_"):
            selection_locator = ""
        else:
            selection_locator = (
                f"{SOURCE_FILES[source_key.replace('_test', '_selected')]}:row="
                f"{selected_row.get('_line') if selected_row else 'missing'}"
            )
            if selected_row is None:
                status = "unresolved"
            else:
                for field in ("candidate_id", "model_sha256", "checkpoint_step"):
                    selected_value = str(selected_row.get(field, "") or "").strip().lower()
                    tested_value = str(source_row.get(field, "") or "").strip().lower()
                    if selected_value and tested_value and selected_value != tested_value:
                        status = "unresolved"

        model_sha, model_file = normalize_model_sha(
            source_key,
            source_row,
            spec["label"],
            checkpoint_step,
            sha_cache,
        )
        if model_sha and len(model_sha) != 64:
            status = "unresolved"

        locator_fields = [
            f"row={source_row.get('_line', 'missing')}",
            f"scene={spec['scene']}",
            f"method={SOURCE_METHOD[spec['method']]}",
            f"seed={spec['seed']}",
            f"episodes={source_row.get('episodes', 'missing')}",
        ]
        if source_row.get("record_type"):
            locator_fields.insert(1, f"record_type={source_row['record_type']}")
        if source_row.get("stage"):
            locator_fields.insert(2, f"stage={source_row['stage']}")

        if status != "unresolved":
            status = f"verified_{episodes}ep_same_row"

        rows.append(
            {
                "scene": spec["scene"],
                "method": spec["method"],
                "seed": spec["seed"],
                "checkpoint_label": spec["label"],
                "checkpoint_step": checkpoint_step,
                "source_batch": SOURCE_BATCH[source_key],
                "episodes": episodes,
                "scenario_seed": scenario_seed,
                "success_rate_pct": success,
                "avg_reward_diagnostic": reward,
                "avg_time_success_sec": time_success,
                "avg_path_length_success_m": path_success,
                "smoothness": smoothness,
                "source_file": source_file,
                "source_record_locator": ";".join(locator_fields),
                "model_sha256": model_sha,
                "model_file": model_file,
                "selection_record_locator": selection_locator,
                "selection_reason": SELECTION_REASON[source_key],
                "protocol_status": (
                    "full_100ep_catalog_protocol"
                    if episodes == 100
                    else "limited_60ep_env4_protocol"
                ),
                "verification_status": status,
            }
        )
    return rows


def grouped(rows: list[dict[str, Any]]) -> dict[tuple[str, str], list[dict[str, Any]]]:
    result: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        result[(row["scene"], row["method"])].append(row)
    for values in result.values():
        values.sort(key=lambda item: item["seed"])
    return result


def mean_sd(values: Iterable[float]) -> tuple[float, float]:
    nums = list(values)
    return statistics.mean(nums), statistics.stdev(nums)


def mean_sd_optional(values: Iterable[float | None]) -> tuple[float | None, float | None, int]:
    nums = [float(value) for value in values if value is not None]
    if not nums:
        return None, None, 0
    if len(nums) == 1:
        return nums[0], None, 1
    return statistics.mean(nums), statistics.stdev(nums), len(nums)


def build_main_table(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped_rows = grouped(rows)
    table: list[dict[str, Any]] = []
    for environment, scene in SCENES:
        for method in METHOD_ORDER:
            selected = grouped_rows[(scene, method)]
            episode_counts = sorted({int(item["episodes"]) for item in selected})
            episodes_per_seed: int | str = (
                episode_counts[0]
                if len(episode_counts) == 1
                else "/".join(str(item) for item in episode_counts)
            )
            metrics = {}
            for field in ("success_rate_pct", "smoothness"):
                mean, sd = mean_sd(float(item[field]) for item in selected)
                metrics[field] = mean
                metrics[f"{field}_sd"] = sd
                metrics[f"{field}_seed_count"] = len(selected)
            for field in ("avg_time_success_sec", "avg_path_length_success_m"):
                mean, sd, seed_count = mean_sd_optional(item[field] for item in selected)
                metrics[field] = mean
                metrics[f"{field}_sd"] = sd
                metrics[f"{field}_seed_count"] = seed_count
            table.append(
                {
                    "environment": environment,
                    "scene": scene,
                    "method": method,
                    "n_training_seeds": 3,
                    "episodes_per_training_seed": episodes_per_seed,
                    "checkpoint_labels": "/".join(item["checkpoint_label"] for item in selected),
                    "success_rate_pct_mean": metrics["success_rate_pct"],
                    "success_rate_pct_sd": metrics["success_rate_pct_sd"],
                    "avg_time_success_sec_mean": metrics["avg_time_success_sec"],
                    "avg_time_success_sec_sd": metrics["avg_time_success_sec_sd"],
                    "avg_time_success_sec_seed_count": metrics[
                        "avg_time_success_sec_seed_count"
                    ],
                    "avg_path_length_success_m_mean": metrics["avg_path_length_success_m"],
                    "avg_path_length_success_m_sd": metrics["avg_path_length_success_m_sd"],
                    "avg_path_length_success_m_seed_count": metrics[
                        "avg_path_length_success_m_seed_count"
                    ],
                    "smoothness_mean": metrics["smoothness"],
                    "smoothness_sd": metrics["smoothness_sd"],
                    "protocol_status": (
                        "full_100ep_catalog_protocol"
                        if episode_counts == [100]
                        else "limited_60ep_env4_protocol"
                    ),
                    "verification_status": "verified_from_3_selected_seed_rows",
                }
            )
    return table


def manifest_specs() -> list[tuple[str, str, str]]:
    return [
        (SOURCE_FILES["summary_a"], "repo_relative", "ENV2 MA/MambaSAC original 100回合逐seed指标"),
        (SOURCE_FILES["summary_b"], "repo_relative", "ENV1原正式指标及ENV3 batch_b候选逐seed指标"),
        (SOURCE_FILES["plan_a"], "repo_relative", "ENV2/ENV3正式批次episodes/scenario_seed元数据"),
        (SOURCE_FILES["plan_b"], "repo_relative", "ENV1/ENV3正式批次episodes/scenario_seed元数据"),
        (SOURCE_FILES["selection_plan"], "repo_relative", "2026-08-03筛选流程元数据，只读核对"),
        (SOURCE_FILES["selection_selected"], "repo_relative", "2026-08-03筛选检查点和SHA，只读核对"),
        (SOURCE_FILES["selection_test"], "repo_relative", "ENV1部分及ENV2 LSTM/SAC的100回合正式测试"),
        (SOURCE_FILES["env2_plan"], "repo_relative", "ENV2筛选流程元数据，只读核对"),
        (SOURCE_FILES["env2_selected"], "repo_relative", "ENV2 CoordAtt/Transformer筛选检查点和SHA"),
        (SOURCE_FILES["env2_test"], "repo_relative", "ENV2 CoordAtt/Transformer的100回合正式测试"),
        (SOURCE_FILES["dynamic_plan"], "repo_relative", "ENV3动态筛选流程元数据，只读核对"),
        (SOURCE_FILES["dynamic_selected"], "repo_relative", "ENV3动态筛选检查点和SHA"),
        (SOURCE_FILES["dynamic_test"], "repo_relative", "ENV3 2026/2027各50回合合并的100回合正式测试"),
        (SOURCE_FILES["env4_remote_selected"], "repo_relative", "ENV4 MA/Mamba/Transformer冻结检查点和SHA"),
        (SOURCE_FILES["env4_remote_test"], "repo_relative", "ENV4 MA/Mamba/Transformer三seed各60回合正式测试"),
        (SOURCE_FILES["env4_lstm01_selected"], "repo_relative", "ENV4 LSTM seed 0/1冻结检查点和SHA"),
        (SOURCE_FILES["env4_lstm01_test"], "repo_relative", "ENV4 LSTM seed 0/1各60回合正式测试"),
        (SOURCE_FILES["env4_local_selected"], "repo_relative", "ENV4 SAC/CoordAtt冻结检查点和SHA"),
        (SOURCE_FILES["env4_local_test"], "repo_relative", "ENV4 SAC/CoordAtt三seed各100回合正式测试"),
        (SOURCE_FILES["env4_supplement_selected"], "repo_relative", "ENV4补充批次冻结检查点和SHA"),
        (SOURCE_FILES["env4_supplement_test"], "repo_relative", "ENV4 LSTM seed 2的60回合正式测试及其批次记录"),
        (SOURCE_FILES["paper_readme"], "external_absolute", "论文目录旧快照说明；用于区分旧快照与当前候选"),
        (SOURCE_FILES["paper_summary"], "external_absolute", "论文目录旧快照主表；不是当前最终表"),
        (SOURCE_FILES["paper_selected"], "external_absolute", "论文目录旧快照逐seed表；不是当前最终表"),
    ]


def build_manifest() -> list[dict[str, Any]]:
    records = []
    for path_string, path_type, purpose in manifest_specs():
        path = repo_path(path_string)
        if not path.is_file():
            records.append(
                {
                    "path": path_string,
                    "path_type": path_type,
                    "size_bytes": "",
                    "last_write_time": "",
                    "sha256": "",
                    "purpose": purpose,
                    "verification_status": "unresolved_source_missing",
                }
            )
            continue
        modified = datetime.fromtimestamp(path.stat().st_mtime).astimezone().isoformat(timespec="seconds")
        records.append(
            {
                "path": path_string,
                "path_type": path_type,
                "size_bytes": path.stat().st_size,
                "last_write_time": modified,
                "sha256": sha256_file(path),
                "purpose": purpose,
                "verification_status": "present_read_only",
            }
        )
    return records


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


SELECTED_FIELDS = [
    "scene",
    "method",
    "seed",
    "checkpoint_label",
    "checkpoint_step",
    "source_batch",
    "episodes",
    "scenario_seed",
    "success_rate_pct",
    "avg_reward_diagnostic",
    "avg_time_success_sec",
    "avg_path_length_success_m",
    "smoothness",
    "source_file",
    "source_record_locator",
    "model_sha256",
    "model_file",
    "selection_record_locator",
    "selection_reason",
    "protocol_status",
    "verification_status",
]

MAIN_FIELDS = [
    "environment",
    "scene",
    "method",
    "n_training_seeds",
    "episodes_per_training_seed",
    "checkpoint_labels",
    "success_rate_pct_mean",
    "success_rate_pct_sd",
    "avg_time_success_sec_mean",
    "avg_time_success_sec_sd",
    "avg_time_success_sec_seed_count",
    "avg_path_length_success_m_mean",
    "avg_path_length_success_m_sd",
    "avg_path_length_success_m_seed_count",
    "smoothness_mean",
    "smoothness_sd",
    "protocol_status",
    "verification_status",
]


def round_for_json(value: Any) -> Any:
    if isinstance(value, float):
        return round(value, 12)
    if isinstance(value, dict):
        return {key: round_for_json(item) for key, item in value.items()}
    if isinstance(value, list):
        return [round_for_json(item) for item in value]
    return value


def build_current_selection(rows: list[dict[str, Any]], table: list[dict[str, Any]]) -> dict[str, Any]:
    grouped_rows = grouped(rows)
    table_index = {(item["scene"], item["method"]): item for item in table}
    scenes = []
    for environment, scene in SCENES:
        methods = []
        for method in METHOD_ORDER:
            selected = grouped_rows[(scene, method)]
            table_row = table_index[(scene, method)]
            methods.append(
                {
                    "method": method,
                    "checkpoints": [
                        {
                            "seed": item["seed"],
                            "checkpoint_label": item["checkpoint_label"],
                            "checkpoint_step": item["checkpoint_step"],
                            "source_batch": item["source_batch"],
                            "episodes": item["episodes"],
                            "scenario_seed": item["scenario_seed"],
                            "source_file": item["source_file"],
                            "source_record_locator": item["source_record_locator"],
                            "model_sha256": item["model_sha256"],
                            "model_file": item["model_file"],
                            "protocol_status": item["protocol_status"],
                            "verification_status": item["verification_status"],
                        }
                        for item in selected
                    ],
                    "paper_metrics": {
                        "success_rate_pct": table_row["success_rate_pct_mean"],
                        "avg_time_success_sec": table_row["avg_time_success_sec_mean"],
                        "avg_path_length_success_m": table_row["avg_path_length_success_m_mean"],
                        "smoothness": table_row["smoothness_mean"],
                    },
                    "protocol_status": table_row["protocol_status"],
                }
            )
        scene_episode_counts = sorted(
            {int(row["episodes"]) for row in rows if row["scene"] == scene}
        )
        scenes.append(
            {
                "environment": environment,
                "scene": scene,
                "episode_protocols": scene_episode_counts,
                "protocol_status": (
                    "uniform_3x100"
                    if scene_episode_counts == [100]
                    else "mixed_across_methods_3x60_and_3x100"
                ),
                "methods": methods,
            }
        )
    return {
        "catalog_version": "paper_checkpoint_catalog_env1_env4",
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "verification_status": "verified_72_selected_rows_24_main_rows",
        "selection_policy": {
            "episodes_per_training_seed": {
                "ENV1": 100,
                "ENV2": 100,
                "ENV3": 100,
                "ENV4": {
                    "SAC": 100,
                    "CoordAttSAC": 100,
                    "LSTM": 60,
                    "MambaSAC": 60,
                    "MA": 60,
                    "Transformer": 60,
                },
            },
            "training_seeds": [0, 1, 2],
            "main_table_metrics": [
                "success_rate_pct",
                "avg_time_success_sec",
                "avg_path_length_success_m",
                "smoothness",
            ],
            "reward_in_main_table": False,
            "time_and_path_length_scope": "successful episodes only",
            "cross_batch_combination_allowed": True,
            "env4_cross_method_protocol_comparable": False,
        },
        "scenes": scenes,
    }


def verify_catalog(rows: list[dict[str, Any]], table: list[dict[str, Any]]) -> list[str]:
    checks: list[str] = []
    assert len(rows) == 72, f"selected row count is {len(rows)}, expected 72"
    assert len(table) == 24, f"main table row count is {len(table)}, expected 24"
    keys = [(row["scene"], row["method"], row["seed"]) for row in rows]
    assert len(set(keys)) == 72, "selected rows contain duplicate scene/method/seed keys"
    for row in rows:
        expected_episodes = (
            60
            if row["scene"] == "tunnel_mixed"
            and row["method"] in {"MA", "MambaSAC", "LSTM", "Transformer"}
            else 100
        )
        assert row["episodes"] == expected_episodes, f"unexpected episode protocol: {row}"
        assert row["source_file"], f"missing source file: {row}"
        assert repo_path(row["source_file"]).is_file(), f"missing source file: {row['source_file']}"
        assert row["model_file"], f"missing portable model path: {row}"
        model_path = repo_path(row["model_file"])
        assert model_path.is_file(), f"missing selected model: {row['model_file']}"
        assert sha256_file(model_path) == row["model_sha256"], f"model hash mismatch: {row['model_file']}"
        assert row["verification_status"] == f"verified_{expected_episodes}ep_same_row", (
            f"unresolved row: {row}"
        )
        assert row["success_rate_pct"] is not None and row["smoothness"] is not None, (
            f"missing required same-row metric: {row}"
        )
        if float(row["success_rate_pct"]) > 0:
            assert row["avg_time_success_sec"] is not None, f"missing success time: {row}"
            assert row["avg_path_length_success_m"] is not None, f"missing success path: {row}"
        else:
            assert row["avg_time_success_sec"] is None, f"unexpected zero-success time: {row}"
            assert row["avg_path_length_success_m"] is None, f"unexpected zero-success path: {row}"
        assert "screen" not in row["source_record_locator"].lower()
        assert "confirm" not in row["source_record_locator"].lower()
        if row["selection_record_locator"]:
            assert "missing" not in row["selection_record_locator"].lower(), (
                f"missing frozen selection record: {row}"
            )
    checks.append("72 rows across ENV1-ENV4: passed")
    checks.append("ENV1-ENV3=3x100; ENV4 method-specific 3x60/3x100 protocols: passed")
    checks.append("source_file exists for every selected row: passed")
    checks.append("portable model_file exists and SHA256 matches for every selected row: passed")
    checks.append("same-row metrics retained; zero-success Time/PathLen remain null: passed")
    checks.append("screen/confirm rows excluded from selected sources: passed")
    checks.append("frozen selection candidate/SHA/step matches formal test rows: passed")

    table_index = {(item["scene"], item["method"]): item for item in table}
    expected_table_keys = {
        (scene, method)
        for _, scene in SCENES
        for method in METHOD_ORDER
    }
    assert set(table_index) == expected_table_keys, "main table key set mismatch"
    for key, expected in TARGET_TABLE.items():
        actual = table_index[key]
        actual_values = (
            actual["success_rate_pct_mean"],
            actual["avg_time_success_sec_mean"],
            actual["avg_path_length_success_m_mean"],
            actual["smoothness_mean"],
        )
        for actual_value, expected_value in zip(actual_values, expected):
            assert math.isclose(actual_value, expected_value, abs_tol=0.03), (
                f"candidate target mismatch for {key}: {actual_values} vs {expected}"
            )
        assert actual["n_training_seeds"] == 3
        assert actual["episodes_per_training_seed"] == 100
    for method in METHOD_ORDER:
        actual = table_index[("tunnel_mixed", method)]
        expected_episodes = 60 if method in {"MA", "MambaSAC", "LSTM", "Transformer"} else 100
        assert actual["episodes_per_training_seed"] == expected_episodes
        expected_protocol = (
            "limited_60ep_env4_protocol"
            if expected_episodes == 60
            else "full_100ep_catalog_protocol"
        )
        assert actual["protocol_status"] == expected_protocol
    checks.append("24 rows: passed")
    checks.append("means and sample SDs recomputed from 3 selected rows: passed")
    checks.append("locked ENV1-ENV3 rounded values match requested table: passed")
    checks.append("ENV4 mixed cross-method episode protocol is explicit: passed")
    return checks


def write_catalog(rows: list[dict[str, Any]], table: list[dict[str, Any]], manifest: list[dict[str, Any]]) -> None:
    selected_rows = sorted(rows, key=lambda item: (item["scene"], METHOD_ORDER.index(item["method"]), item["seed"]))
    write_csv(OUT_DIR / "selected_checkpoints_by_seed.csv", selected_rows, SELECTED_FIELDS)
    write_csv(OUT_DIR / "paper_main_table.csv", table, MAIN_FIELDS)
    write_csv(
        OUT_DIR / "source_manifest.csv",
        manifest,
        ["path", "path_type", "size_bytes", "last_write_time", "sha256", "purpose", "verification_status"],
    )

    grouped_rows = grouped(rows)
    for _, scene in SCENES:
        scene_rows = sorted(
            [row for row in rows if row["scene"] == scene],
            key=lambda item: (METHOD_ORDER.index(item["method"]), item["seed"]),
        )
        scene_table = [item for item in table if item["scene"] == scene]
        write_csv(OUT_DIR / "scenes" / scene / "selected_checkpoints.csv", scene_rows, SELECTED_FIELDS)
        write_csv(OUT_DIR / "scenes" / scene / "paper_table.csv", scene_table, MAIN_FIELDS)

    write_json(OUT_DIR / "current_selection.json", round_for_json(build_current_selection(rows, table)))


def build_catalog(write_outputs: bool) -> list[str]:
    csv_inputs, json_inputs = load_inputs()
    rows = resolve_rows(csv_inputs, json_inputs)
    table = build_main_table(rows)
    checks = verify_catalog(rows, table)
    manifest = build_manifest()
    if write_outputs:
        write_catalog(rows, table, manifest)
    return checks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="re-read sources and recompute all checks without writing catalog files",
    )
    args = parser.parse_args()
    checks = build_catalog(write_outputs=not args.verify_only)
    mode = "verify-only" if args.verify_only else "build"
    print(f"mode={mode}")
    for check in checks:
        print(check)
    print(f"output_dir={OUT_DIR}")


if __name__ == "__main__":
    main()
