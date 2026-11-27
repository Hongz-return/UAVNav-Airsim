#!/usr/bin/env python3
"""Auditable, resumable two-stage checkpoint selection for tunnel scenes.

The script deliberately delegates every model evaluation to
``scripts.evaluation.evaluate_single``. It owns only inventory discovery,
immutable candidate/manifest planning, parallel AirSim orchestration, result
validation, aggregation, and ranking.

The protocol is intentionally limited to a short validation screen followed
by a blind formal test. Formal-test results are never accepted by the
checkpoint-selection function.
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import math
import os
import re
import statistics
import subprocess
import sys
import zipfile
from bisect import bisect_right
from collections import defaultdict
from configparser import ConfigParser
from dataclasses import asdict, dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.training.train_parallel import METHOD_CONFIGS, SCENES, default_ports
from tools.airsim_instances import auto_pin_p_core, launch_instances, terminate_processes
from tools.experiment_paths import (
    CROSS_SCENE_SELECTION_ROOT,
    EVALUATION_RESULTS_ROOT,
    FORMAL_EVALUATION_BATCHES_ROOT,
    PARALLEL_RUNS_ROOT,
    TRAIN_RUNS_ROOT,
    evaluation_parent_for_run,
)


SCHEMA_VERSION = 2
DEFAULT_OUTPUT_ROOT = CROSS_SCENE_SELECTION_ROOT
DEFAULT_METHODS = ["sac", "coordatt", "lstm", "transformer", "mamba", "ma"]
DEFAULT_SEEDS = [0, 1, 2]
STAGES = ("screen", "test")
PROFILE_BY_SCENE = {
    "tunnel": "deterministic",
    "tunnel_obs": "deterministic",
    "tunnel_activate": "dynamic",
    "tunnel_mixed": "dynamic",
}
DEFAULT_PROTOCOLS = {
    "deterministic": {
        "screen": {"episodes": 10, "seeds": [31001]},
        "test": {"episodes": 100, "seeds": [2026]},
    },
    "dynamic": {
        "screen": {"episodes": 10, "seeds": [32001, 32002]},
        "test": {"episodes": 50, "seeds": [2026, 2027]},
    },
}
CHECKPOINT_RE = re.compile(r"^checkpoint_(\d+)_steps\.zip$", re.IGNORECASE)
TRAINING_REWARD_TAG = "episode/return_ma20"
TRAINING_SUCCESS_TAG = "episode/success_rate_ma50"
TRAINING_REWARD_PEAK_MIN_GAP = 30_000
RANK_METRICS = (
    "success_rate_pct",
    "collision_rate_pct",
    "out_of_bounds_rate_pct",
    "timeout_rate_pct",
    "spl",
    "avg_reward",
)
SUCCESS_WEIGHTED_METRICS = {
    "avg_path_length_success_m",
    "avg_steps_success",
    "avg_time_success_sec",
    "avg_speed_success_mps",
}
IDENTITY_FIELDS = [
    "record_type",
    "stage",
    "scene",
    "method",
    "seed",
    "candidate_id",
    "checkpoint_type",
    "checkpoint_step",
    "training_step",
    "model_relative_path",
    "model_absolute_path",
    "model_sha256",
    "validation_batch",
    "scenario_seed",
    "episodes",
    "result_dir",
    "batch_success_rate_std_pct",
    "batch_success_rate_range_pct",
]


@dataclass(frozen=True)
class TrainingRun:
    scene: str
    method: str
    seed: int
    run_dir: Path
    run_relative_path: str
    target_steps: int
    created_at: str
    batch_id: str | None = None
    declared_target_steps: int | None = None
    accepted_truncated: bool = False


@dataclass(frozen=True)
class TrainingMetricSnapshot:
    checkpoint_step: int
    checkpoint_path: Path
    reward_event_step: int
    reward: float
    success_event_step: int
    success_rate: float


def utcish_now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def policy_sha256(path: Path) -> str:
    """Hash only the serialized policy weights for strategy-level deduplication."""

    try:
        with zipfile.ZipFile(path) as archive:
            return hashlib.sha256(archive.read("policy.pth")).hexdigest()
    except (OSError, KeyError, zipfile.BadZipFile) as exc:
        raise ValueError(f"Cannot hash policy.pth in model archive: {path}") from exc


def normalized_path(value: str | Path) -> str:
    """Normalize Windows paths case-insensitively, even in non-Windows tests."""
    text = str(value).strip().replace("\\", "/")
    while "//" in text:
        text = text.replace("//", "/")
    if text.startswith("//?/"):
        text = text[4:]
    return text.rstrip("/").casefold()


def paths_equal(left: str | Path | None, right: str | Path | None) -> bool:
    if left is None or right is None:
        return left is right
    return normalized_path(left) == normalized_path(right)


def relative_or_absolute(path: Path, root: Path = ROOT) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path.resolve())


def atomic_json_write(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _load_parallel_plan(training_root: Path, batch_id: str | None) -> dict[str, Any] | None:
    if not batch_id:
        return None
    path = PARALLEL_RUNS_ROOT / batch_id / "plan.json"
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _method_evidence(metadata: dict[str, Any], parallel_plan: dict[str, Any] | None, method: str, seed: int) -> list[str]:
    evidence: list[str] = []
    if metadata.get("method") is not None:
        evidence.append(str(metadata["method"]))
    config_source = Path(str(metadata.get("config_source", ""))).name.casefold()
    if re.match(rf"^{re.escape(method.casefold())}_seed_{seed}(?:_|\.)", config_source):
        evidence.append(method)
    if parallel_plan:
        for job in parallel_plan.get("jobs", []):
            try:
                if str(job.get("method")) == method and int(job.get("seed")) == seed:
                    evidence.append(method)
                    break
            except (TypeError, ValueError):
                continue
    return evidence


def _scene_evidence(metadata: dict[str, Any], parallel_plan: dict[str, Any] | None) -> list[str]:
    evidence: list[str] = []
    if metadata.get("scene") is not None:
        evidence.append(str(metadata["scene"]))
    if parallel_plan and parallel_plan.get("scene") is not None:
        evidence.append(str(parallel_plan["scene"]))
    # The cleaned direct-layout archive no longer has the original batch
    # directory, but immutable metadata paths still contain the scene name.
    for field in ("config_snapshot", "tensorboard_dir"):
        value = normalized_path(str(metadata.get(field, "")))
        evidence.extend(scene for scene in SCENES if f"/{scene}/" in f"/{value}/")
    # Tunnel is unambiguous. Obs is shared by static and dynamic maps, so it
    # cannot independently verify either directory name.
    if str(metadata.get("environment", "")).casefold() == "tunnel":
        evidence.append("tunnel")
    return evidence


def _parse_created_at(value: str) -> datetime:
    try:
        return datetime.fromisoformat(value)
    except (TypeError, ValueError):
        return datetime.min


def parse_targets(values: Sequence[str] | None) -> set[tuple[str, str]] | None:
    """Parse ``scene:method,method`` tokens into an exact target matrix."""
    if not values:
        return None
    targets: set[tuple[str, str]] = set()
    for token in values:
        scene, separator, method_text = token.partition(":")
        if not separator or scene not in SCENES:
            raise ValueError(
                f"Invalid --targets token {token!r}; expected scene:method,method"
            )
        methods = [item.strip() for item in method_text.split(",") if item.strip()]
        if not methods:
            raise ValueError(f"No methods specified in --targets token {token!r}")
        unknown = sorted(set(methods) - set(METHOD_CONFIGS))
        if unknown:
            raise ValueError(f"Unknown method(s) in --targets token {token!r}: {unknown}")
        for method in methods:
            pair = (scene, method)
            if pair in targets:
                raise ValueError(f"Duplicate --targets coordinate: {scene}:{method}")
            targets.add(pair)
    return targets


def parse_accepted_truncated(
    values: Sequence[str] | None,
) -> dict[tuple[str, str, int], int]:
    """Parse explicit ``scene:method:seed:step`` truncation approvals.

    This is intentionally opt-in and coordinate-specific.  It allows a run
    that stopped before its declared target to participate without renaming a
    checkpoint or pretending that the declared target was reached.
    """
    accepted: dict[tuple[str, str, int], int] = {}
    for token in values or ():
        parts = token.split(":")
        if len(parts) != 4:
            raise ValueError(
                f"Invalid --accept-truncated token {token!r}; "
                "expected scene:method:seed:step"
            )
        scene, method, seed_text, step_text = parts
        if scene not in SCENES:
            raise ValueError(f"Unknown scene in --accept-truncated token {token!r}")
        if method not in METHOD_CONFIGS:
            raise ValueError(f"Unknown method in --accept-truncated token {token!r}")
        try:
            seed = int(seed_text)
            step = int(step_text)
        except ValueError as exc:
            raise ValueError(
                f"Invalid seed/step in --accept-truncated token {token!r}"
            ) from exc
        if seed < 0 or step < 1:
            raise ValueError(
                f"Seed must be non-negative and step positive in {token!r}"
            )
        coordinate = (scene, method, seed)
        if coordinate in accepted:
            raise ValueError(f"Duplicate --accept-truncated coordinate: {coordinate}")
        accepted[coordinate] = step
    return accepted


def discover_training_inventory(
    training_root: Path,
    *,
    scenes: Sequence[str] | None = None,
    methods: Sequence[str] | None = None,
    seeds: Sequence[int] | None = None,
    accepted_truncated: dict[tuple[str, str, int], int] | None = None,
) -> tuple[list[TrainingRun], list[dict[str, Any]]]:
    """Discover artifact-complete training runs, never evaluation CSV rows.

    Directory identity is cross-checked against run metadata and, for the
    repository's batch-prefixed layout, the frozen parallel-training plan.
    Duplicate scene/method/seed coordinates are resolved by run.json's
    ``created_at`` (never filesystem modification time) and fully audited.
    """
    root = training_root.resolve()
    wanted_scenes = set(scenes or SCENES)
    wanted_methods = set(methods or DEFAULT_METHODS)
    wanted_seeds = set(DEFAULT_SEEDS if seeds is None else seeds)
    accepted_truncated = dict(accepted_truncated or {})
    issues: list[dict[str, Any]] = []
    discovered: list[TrainingRun] = []
    if not root.is_dir():
        return [], [{"reason": "training_root_missing", "path": str(root)}]

    for metadata_path in sorted(root.rglob("run.json"), key=lambda p: normalized_path(p)):
        run_dir = metadata_path.parent
        try:
            rel = run_dir.relative_to(root)
            parts = rel.parts
            if len(parts) < 4:
                raise ValueError("expected .../<scene>/<method>/seed_<n>/<run_directory>")
            scene, method, seed_name = parts[-4], parts[-3], parts[-2]
            if not seed_name.casefold().startswith("seed_"):
                raise ValueError(f"invalid seed directory: {seed_name}")
            seed = int(seed_name[5:])
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            declared_target_steps = int(metadata["total_timesteps"])
            metadata_seed = int(metadata["seed"])
            if metadata_seed != seed:
                raise ValueError(f"seed mismatch: directory={seed}, run.json={metadata_seed}")

            batch_id = parts[0] if len(parts) > 4 else None
            parallel_plan = _load_parallel_plan(root, batch_id)
            scene_evidence = _scene_evidence(metadata, parallel_plan)
            method_evidence = _method_evidence(metadata, parallel_plan, method, seed)
            if not scene_evidence or any(item != scene for item in scene_evidence):
                raise ValueError(
                    f"scene is not corroborated: directory={scene}, evidence={scene_evidence}"
                )
            if not method_evidence or any(item != method for item in method_evidence):
                raise ValueError(
                    f"method is not corroborated: directory={method}, evidence={method_evidence}"
                )

            coordinate = (scene, method, seed)
            accepted_step = accepted_truncated.get(coordinate)
            accepted = accepted_step is not None
            if accepted:
                if accepted_step > declared_target_steps:
                    raise ValueError(
                        "accepted truncated step exceeds declared target: "
                        f"accepted={accepted_step}, declared={declared_target_steps}"
                    )
                target_steps = accepted_step
            else:
                target_steps = declared_target_steps

            required = [
                run_dir / "config" / "config.ini",
                run_dir / "models" / "best_model.zip",
                run_dir / "models" / f"checkpoint_{target_steps}_steps.zip",
            ]
            if not accepted:
                required.append(run_dir / "models" / "model_sb3.zip")
            missing = [str(path) for path in required if not path.is_file()]
            if missing:
                raise ValueError(f"missing required artifacts: {missing}")
            if accepted:
                periodic_steps = sorted(
                    int(match.group(1))
                    for path in (run_dir / "models").iterdir()
                    if path.is_file() and (match := CHECKPOINT_RE.match(path.name))
                )
                if not periodic_steps or periodic_steps[-1] != target_steps:
                    raise ValueError(
                        "accepted truncated step must be the latest periodic checkpoint: "
                        f"accepted={target_steps}, available={periodic_steps}"
                    )
                issues.append(
                    {
                        "reason": "accepted_truncated_training_run",
                        "scene": scene,
                        "method": method,
                        "seed": seed,
                        "path": relative_or_absolute(run_dir, root),
                        "declared_target_steps": declared_target_steps,
                        "accepted_step": target_steps,
                    }
                )
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            issues.append(
                {
                    "reason": "invalid_training_run",
                    "path": relative_or_absolute(run_dir, root),
                    "detail": str(exc),
                }
            )
            continue

        if scene not in wanted_scenes or method not in wanted_methods or seed not in wanted_seeds:
            continue
        discovered.append(
            TrainingRun(
                scene=scene,
                method=method,
                seed=seed,
                run_dir=run_dir.resolve(),
                run_relative_path=str(rel),
                target_steps=target_steps,
                created_at=str(metadata.get("created_at", "")),
                batch_id=batch_id,
                declared_target_steps=declared_target_steps,
                accepted_truncated=accepted,
            )
        )

    grouped: dict[tuple[str, str, int], list[TrainingRun]] = defaultdict(list)
    for run in discovered:
        grouped[(run.scene, run.method, run.seed)].append(run)

    selected: list[TrainingRun] = []
    for key, group in sorted(grouped.items()):
        ordered = sorted(
            group,
            key=lambda run: (_parse_created_at(run.created_at), normalized_path(run.run_dir)),
            reverse=True,
        )
        if len(ordered) > 1:
            newest = _parse_created_at(ordered[0].created_at)
            if newest == datetime.min or _parse_created_at(ordered[1].created_at) == newest:
                raise ValueError(
                    f"Ambiguous duplicate training runs for {key}; use uniquely dated run.json metadata"
                )
            for duplicate in ordered[1:]:
                issues.append(
                    {
                        "reason": "duplicate_coordinate_older_run_json_created_at",
                        "scene": duplicate.scene,
                        "method": duplicate.method,
                        "seed": duplicate.seed,
                        "excluded_run": duplicate.run_relative_path,
                        "selected_run": ordered[0].run_relative_path,
                    }
                )
        selected.append(ordered[0])

    return sorted(selected, key=lambda run: (run.scene, run.method, run.seed)), issues


def read_model_training_step(path: Path, *, fallback: int | None = None) -> int | None:
    match = CHECKPOINT_RE.match(path.name)
    if match:
        return int(match.group(1))
    try:
        with zipfile.ZipFile(path) as archive:
            payload = json.loads(archive.read("data"))
        value = payload.get("num_timesteps")
        if isinstance(value, (int, float)) and int(value) >= 0:
            return int(value)
    except (OSError, KeyError, TypeError, ValueError, zipfile.BadZipFile, json.JSONDecodeError):
        pass
    return fallback


def _candidate_alias(path: Path, run: TrainingRun) -> dict[str, Any]:
    match = CHECKPOINT_RE.match(path.name)
    if match:
        kind = "checkpoint"
        step = int(match.group(1))
    elif path.name.casefold() == "best_model.zip":
        kind = "best"
        step = read_model_training_step(path)
    elif path.name.casefold() == "model_sb3.zip":
        kind = "final"
        step = read_model_training_step(path, fallback=run.target_steps)
    else:
        raise ValueError(f"Unsupported model artifact: {path}")
    return {"checkpoint_type": kind, "checkpoint_step": step, "path": path}


def _training_metric_snapshots(
    run: TrainingRun, periodic: dict[int, Path]
) -> list[TrainingMetricSnapshot]:
    """Map TensorBoard training metrics causally onto saved checkpoints.

    A checkpoint at step S receives the most recent scalar event at or before
    S.  This prevents later training observations from being attributed to an
    earlier policy snapshot.  Evaluation results are never read here.
    """

    try:
        from tensorboard.backend.event_processing.event_accumulator import (
            EventAccumulator,
        )
    except ImportError as exc:  # pragma: no cover - dependency is locked by pixi
        raise RuntimeError(
            "candidate-mode=training_peaks requires TensorBoard in the Pixi environment"
        ) from exc

    event_files = sorted(run.run_dir.glob("tb_logs/**/events.out.tfevents.*"))
    if len(event_files) != 1:
        raise ValueError(
            f"Expected exactly one TensorBoard event file for {run.run_relative_path}, "
            f"found {len(event_files)}"
        )
    event_path = event_files[0]
    accumulator = EventAccumulator(str(event_path), size_guidance={"scalars": 0})
    accumulator.Reload()
    scalar_tags = set(accumulator.Tags().get("scalars", []))
    required_tags = {TRAINING_REWARD_TAG, TRAINING_SUCCESS_TAG}
    missing_tags = sorted(required_tags - scalar_tags)
    if missing_tags:
        raise ValueError(
            f"TensorBoard tags missing for {run.run_relative_path}: {missing_tags}"
        )

    def series(tag: str) -> tuple[list[int], list[float]]:
        events = accumulator.Scalars(tag)
        if not events:
            raise ValueError(
                f"TensorBoard tag {tag!r} has no events for {run.run_relative_path}"
            )
        return [int(event.step) for event in events], [float(event.value) for event in events]

    reward_steps, reward_values = series(TRAINING_REWARD_TAG)
    success_steps, success_values = series(TRAINING_SUCCESS_TAG)

    def at_or_before(
        steps: Sequence[int], values: Sequence[float], checkpoint_step: int
    ) -> tuple[int, float]:
        index = bisect_right(steps, checkpoint_step) - 1
        if index < 0:
            raise ValueError(
                f"No TensorBoard event exists at or before checkpoint {checkpoint_step} "
                f"for {run.run_relative_path}"
            )
        return steps[index], values[index]

    snapshots: list[TrainingMetricSnapshot] = []
    for checkpoint_step, checkpoint_path in sorted(periodic.items()):
        reward_event_step, reward = at_or_before(
            reward_steps, reward_values, checkpoint_step
        )
        success_event_step, success_rate = at_or_before(
            success_steps, success_values, checkpoint_step
        )
        snapshots.append(
            TrainingMetricSnapshot(
                checkpoint_step=checkpoint_step,
                checkpoint_path=checkpoint_path,
                reward_event_step=reward_event_step,
                reward=reward,
                success_event_step=success_event_step,
                success_rate=success_rate,
            )
        )
    return snapshots


def _training_peak_rows(
    snapshots: Sequence[TrainingMetricSnapshot],
) -> tuple[TrainingMetricSnapshot, TrainingMetricSnapshot, TrainingMetricSnapshot]:
    """Return two reward peaks and one success-rate peak deterministically."""

    if not snapshots:
        raise ValueError("No checkpoint-level TensorBoard snapshots available")
    reward_ranked = sorted(
        snapshots,
        key=lambda row: (row.reward, row.success_rate, row.checkpoint_step),
        reverse=True,
    )
    reward_first = reward_ranked[0]
    reward_second = next(
        (
            row
            for row in reward_ranked[1:]
            if abs(row.checkpoint_step - reward_first.checkpoint_step)
            >= TRAINING_REWARD_PEAK_MIN_GAP
        ),
        None,
    )
    if reward_second is None:
        raise ValueError(
            "Cannot select two reward peaks separated by at least "
            f"{TRAINING_REWARD_PEAK_MIN_GAP} steps"
        )
    success_peak = max(
        snapshots,
        key=lambda row: (row.success_rate, row.reward, row.checkpoint_step),
    )
    return reward_first, reward_second, success_peak


def discover_candidates(
    run: TrainingRun,
    *,
    mode: str = "full",
    interval: int = 20_000,
) -> list[dict[str, Any]]:
    if mode not in {"full", "verify", "training_peaks"}:
        raise ValueError(f"Unsupported candidate mode: {mode}")
    if interval < 1:
        raise ValueError("--candidate-interval must be at least 1")

    models = run.run_dir / "models"
    periodic: dict[int, Path] = {}
    for path in models.iterdir():
        match = CHECKPOINT_RE.match(path.name)
        if match and path.is_file() and int(match.group(1)) <= run.target_steps:
            periodic[int(match.group(1))] = path.resolve()
    if run.target_steps not in periodic:
        raise ValueError(f"Final periodic checkpoint missing for {run.run_relative_path}")

    best = (models / "best_model.zip").resolve()
    final = (models / "model_sb3.zip").resolve()
    best_step = read_model_training_step(best)
    if run.accepted_truncated and best_step is not None and best_step > run.target_steps:
        raise ValueError(
            f"Best model exceeds accepted truncated step for {run.run_relative_path}: "
            f"best={best_step}, accepted={run.target_steps}"
        )
    chosen: set[Path] = set()
    reasons_by_path: dict[Path, list[dict[str, Any]]] = defaultdict(list)

    def choose(path: Path, reason: str, **evidence: Any) -> None:
        chosen.add(path)
        reasons_by_path[path].append({"reason": reason, **evidence})

    choose(
        best,
        "best_model",
        model_step=best_step,
        detail="SB3 num_timesteps metadata; callback prioritizes success then reward",
    )
    choose(
        periodic[run.target_steps],
        "accepted_truncated_final" if run.accepted_truncated else "final_checkpoint",
        model_step=run.target_steps,
        detail=(
            "Explicitly accepted truncated final checkpoint; never relabelled"
            if run.accepted_truncated
            else "Declared final periodic checkpoint"
        ),
    )
    if not run.accepted_truncated and mode != "training_peaks":
        choose(
            final,
            "final_checkpoint_alias",
            model_step=run.target_steps,
            detail="SB3 final-model alias retained for artifact completeness",
        )
    if mode == "full":
        for step, path in periodic.items():
            if step % interval == 0 or step == run.target_steps:
                choose(path, "periodic_interval", model_step=step, interval=interval)
    elif mode == "training_peaks":
        reward_first, reward_second, success_peak = _training_peak_rows(
            _training_metric_snapshots(run, periodic)
        )
        choose(
            reward_first.checkpoint_path,
            "reward_peak_1",
            model_step=reward_first.checkpoint_step,
            metric_tag=TRAINING_REWARD_TAG,
            metric_event_step=reward_first.reward_event_step,
            metric_value=reward_first.reward,
            minimum_pair_gap=TRAINING_REWARD_PEAK_MIN_GAP,
        )
        choose(
            reward_second.checkpoint_path,
            "reward_peak_2",
            model_step=reward_second.checkpoint_step,
            metric_tag=TRAINING_REWARD_TAG,
            metric_event_step=reward_second.reward_event_step,
            metric_value=reward_second.reward,
            minimum_pair_gap=TRAINING_REWARD_PEAK_MIN_GAP,
        )
        choose(
            success_peak.checkpoint_path,
            "success_rate_peak",
            model_step=success_peak.checkpoint_step,
            metric_tag=TRAINING_SUCCESS_TAG,
            metric_event_step=success_peak.success_event_step,
            metric_value=success_peak.success_rate,
            tie_break=f"{TRAINING_REWARD_TAG}, then later checkpoint",
        )

    # Reliably locate the closest saved periodic checkpoints around the best
    # step. The step comes from SB3's serialized num_timesteps, never mtime.
    if mode in {"full", "verify"} and best_step is not None:
        lower = [step for step in periodic if step <= best_step]
        upper = [step for step in periodic if step >= best_step]
        if lower:
            step = max(lower)
            choose(periodic[step], "closest_periodic_at_or_below_best", model_step=step)
        if upper:
            step = min(upper)
            choose(periodic[step], "closest_periodic_at_or_above_best", model_step=step)

    aliases_by_hash: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for path in sorted(chosen, key=lambda item: normalized_path(item)):
        alias = _candidate_alias(path, run)
        alias["model_relative_path"] = str(path.relative_to(run.run_dir.resolve()))
        alias["model_absolute_path"] = str(path)
        alias["selection_reasons"] = reasons_by_path[path]
        alias["model_sha256"] = sha256(path)
        if mode == "training_peaks":
            alias["policy_sha256"] = policy_sha256(path)
            grouping_digest = alias["policy_sha256"]
        else:
            grouping_digest = alias["model_sha256"]
        aliases_by_hash[grouping_digest].append(alias)

    priority = {"checkpoint": 0, "best": 1, "final": 2}
    candidates: list[dict[str, Any]] = []
    for grouping_digest, aliases in aliases_by_hash.items():
        aliases.sort(
            key=lambda item: (
                priority[item["checkpoint_type"]],
                -(item["checkpoint_step"] if item["checkpoint_step"] is not None else -1),
                normalized_path(item["model_absolute_path"]),
            )
        )
        canonical = aliases[0]
        training_steps = [
            int(item["checkpoint_step"])
            for item in aliases
            if item["checkpoint_step"] is not None
        ]
        training_step = max(training_steps) if training_steps else -1
        selection_reasons: list[dict[str, Any]] = []
        seen_reasons: set[str] = set()
        for alias in aliases:
            for reason in alias["selection_reasons"]:
                serialized = json.dumps(reason, ensure_ascii=False, sort_keys=True)
                if serialized not in seen_reasons:
                    seen_reasons.add(serialized)
                    selection_reasons.append(reason)
        candidates.append(
            {
                "candidate_id": (
                    f"{run.scene}_{run.method}_s{run.seed}_{grouping_digest[:12]}"
                ),
                "scene": run.scene,
                "method": run.method,
                "seed": run.seed,
                "checkpoint_type": canonical["checkpoint_type"],
                "checkpoint_step": canonical["checkpoint_step"],
                "training_step": training_step,
                "model_relative_path": canonical["model_relative_path"],
                "model_absolute_path": canonical["model_absolute_path"],
                "model_sha256": canonical["model_sha256"],
                "policy_sha256": (
                    grouping_digest if mode == "training_peaks" else None
                ),
                "selection_reasons": selection_reasons,
                "aliases": [
                    {
                        key: value
                        for key, value in alias.items()
                        if key != "path"
                    }
                    for alias in aliases
                ],
            }
        )
    return sorted(
        candidates,
        key=lambda item: (
            item["training_step"],
            priority[item["checkpoint_type"]],
            item["candidate_id"],
        ),
    )


def _protocol_for_scene(scene: str, args: argparse.Namespace) -> dict[str, Any]:
    profile = PROFILE_BY_SCENE[scene]
    if args.profile is not None and args.profile != profile:
        raise ValueError(f"Scene {scene} requires profile={profile}, got {args.profile}")
    defaults = DEFAULT_PROTOCOLS[profile]
    protocol: dict[str, Any] = {"profile": profile}
    for stage in STAGES:
        episodes_override = getattr(args, f"{stage}_episodes")
        seeds_override = getattr(args, f"{stage}_seeds")
        protocol[stage] = {
            "episodes": (
                defaults[stage]["episodes"]
                if episodes_override is None
                else episodes_override
            ),
            "seeds": list(
                defaults[stage]["seeds"]
                if seeds_override is None
                else seeds_override
            ),
        }
        if protocol[stage]["episodes"] < 1:
            raise ValueError(f"--{stage}-episodes must be at least 1")
    expected_batches = 1 if profile == "deterministic" else 2
    for stage in STAGES:
        if len(protocol[stage]["seeds"]) != expected_batches:
            raise ValueError(
                f"{scene}/{stage} requires {expected_batches} scenario seed(s), "
                f"got {protocol[stage]['seeds']}"
            )
    return protocol


def _read_positions(run_dir: Path) -> tuple[list[float], list[float]]:
    cfg = ConfigParser()
    path = run_dir / "config" / "config.ini"
    if not cfg.read(path, encoding="utf-8"):
        raise FileNotFoundError(path)
    start = [float(value) for value in ast.literal_eval(cfg.get("environment", "start_position"))]
    goal = [float(value) for value in ast.literal_eval(cfg.get("environment", "goal_position"))]
    if len(start) != 3 or len(goal) != 3:
        raise ValueError(f"Invalid start/goal position in {path}")
    return start, goal


def _manifest_name(scene: str, stage: str, batch_count: int, batch_index: int) -> str:
    suffix = f"_batch_{batch_index}" if batch_count > 1 else ""
    return f"{scene}_{stage}{suffix}.json"


def materialize_scenario_manifests(
    run_dir: Path,
    runs: Sequence[TrainingRun],
    protocols: dict[str, dict[str, Any]],
    *,
    yaw_range_deg: float,
    source_root: Path | None = None,
) -> dict[str, dict[str, list[dict[str, Any]]]]:
    output = run_dir / "scenario_manifests"
    output.mkdir(parents=True, exist_ok=True)
    by_scene: dict[str, list[TrainingRun]] = defaultdict(list)
    for run in runs:
        by_scene[run.scene].append(run)

    result: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for scene, scene_runs in sorted(by_scene.items()):
        all_positions = [_read_positions(item.run_dir) for item in scene_runs]
        first_start, first_goal = all_positions[0]
        if any(start != first_start or goal != first_goal for start, goal in all_positions[1:]):
            raise ValueError(f"All {scene} training runs must share start/goal positions")
        result[scene] = {}
        for stage in STAGES:
            spec = protocols[scene][stage]
            entries: list[dict[str, Any]] = []
            for batch_index, scenario_seed in enumerate(spec["seeds"], start=1):
                name = _manifest_name(scene, stage, len(spec["seeds"]), batch_index)
                path = output / name
                if source_root is not None:
                    source = source_root / name
                    if not source.is_file():
                        raise FileNotFoundError(f"Scenario manifest not found: {source}")
                    payload = json.loads(source.read_text(encoding="utf-8"))
                    scenarios = payload.get("scenarios", payload) if isinstance(payload, dict) else payload
                    if len(scenarios) < spec["episodes"]:
                        raise ValueError(f"{source} has {len(scenarios)} scenarios; need {spec['episodes']}")
                    scenarios = scenarios[: spec["episodes"]]
                    source_path = str(source.resolve())
                else:
                    rng = np.random.default_rng(scenario_seed)
                    yaws = rng.uniform(-yaw_range_deg, yaw_range_deg, size=spec["episodes"])
                    scenarios = [
                        {
                            "scenario_id": f"seed_{scenario_seed}_episode_{index:03d}",
                            "start_position": first_start,
                            "goal_position": first_goal,
                            "yaw_deg": float(yaw),
                            "yaw_rad": float(np.deg2rad(yaw)),
                        }
                        for index, yaw in enumerate(yaws)
                    ]
                    source_path = None
                payload = {
                    "scene": scene,
                    "stage": stage,
                    "validation_batch": batch_index,
                    "scenario_seed": scenario_seed,
                    "episodes": spec["episodes"],
                    "yaw_range_deg": yaw_range_deg,
                    "source_manifest": source_path,
                    "scenarios": scenarios,
                }
                atomic_json_write(path, payload)
                entries.append(
                    {
                        "validation_batch": batch_index,
                        "episodes": spec["episodes"],
                        "scenario_seed": scenario_seed,
                        "path": str(path.resolve()),
                        "relative_path": relative_or_absolute(path, run_dir),
                        "sha256": sha256(path),
                    }
                )
            result[scene][stage] = entries
    return result


def _selector_args(candidate: dict[str, Any]) -> list[str]:
    kind = candidate["checkpoint_type"]
    if kind == "checkpoint":
        return ["--checkpoint-step", str(candidate["checkpoint_step"])]
    if kind == "best":
        return ["--best"]
    if kind == "final":
        return ["--final"]
    raise ValueError(f"Unsupported checkpoint type: {kind}")


def build_evaluation_command(
    *,
    run_absolute_path: str | Path,
    candidate: dict[str, Any],
    episodes: int,
    scenario_seed: int,
    scenario_manifest: str | Path,
    yaw_range_deg: float,
    airsim_ip: str,
    port: int,
    save_tag: str,
) -> list[str]:
    return [
        sys.executable,
        "-m",
        "scripts.evaluation.evaluate_single",
        "-m",
        str(run_absolute_path),
        "-n",
        str(episodes),
        *_selector_args(candidate),
        "--save",
        "--output-root",
        str(EVALUATION_RESULTS_ROOT),
        "--save-tag",
        save_tag,
        "--airsim-ip",
        airsim_ip,
        "--airsim-port",
        str(port),
        "--scenario-seed",
        str(scenario_seed),
        "--scenario-manifest",
        str(scenario_manifest),
        "--yaw-range-deg",
        str(yaw_range_deg),
    ]


def printable(command: Sequence[str]) -> str:
    return subprocess.list2cmdline(list(command))


def make_tasks(
    plan: dict[str, Any],
    *,
    stage: str,
    candidate_ids: dict[tuple[str, str, int], list[str]],
) -> list[dict[str, Any]]:
    candidates = {item["candidate_id"]: item for item in plan["candidates"]}
    runs = {
        (item["scene"], item["method"], int(item["seed"])): item
        for item in plan["training_runs"]
    }
    ports = plan["runtime"]["ports"]
    tasks: list[dict[str, Any]] = []
    index = 0
    for key in sorted(candidate_ids):
        run = runs[key]
        for candidate_id in candidate_ids[key]:
            candidate = candidates[candidate_id]
            manifest_entries = plan["scenario_manifests"][key[0]][stage]
            for manifest in manifest_entries:
                port = ports[index % len(ports)]
                batch = int(manifest["validation_batch"])
                tag = (
                    f"ckptsel_{plan['run_id']}_{stage}_b{batch}_"
                    f"{candidate_id.rsplit('_', 1)[-1]}"
                )
                command = build_evaluation_command(
                    run_absolute_path=run["run_absolute_path"],
                    candidate=candidate,
                    episodes=int(manifest["episodes"]),
                    scenario_seed=int(manifest["scenario_seed"]),
                    scenario_manifest=manifest["path"],
                    yaw_range_deg=float(plan["runtime"]["yaw_range_deg"]),
                    airsim_ip=plan["runtime"]["airsim_ip"],
                    port=port,
                    save_tag=tag,
                )
                tasks.append(
                    {
                        "task_id": f"{key[0]}__{key[1]}__s{key[2]}__{candidate_id}__{stage}__b{batch}",
                        "scene": key[0],
                        "method": key[1],
                        "seed": key[2],
                        "training_run_relative_path": run["run_relative_path"],
                        "training_run_absolute_path": run["run_absolute_path"],
                        "candidate_id": candidate_id,
                        "checkpoint_type": candidate["checkpoint_type"],
                        "checkpoint_step": candidate["checkpoint_step"],
                        "training_step": candidate["training_step"],
                        "model_relative_path": candidate["model_relative_path"],
                        "model_absolute_path": candidate["model_absolute_path"],
                        "model_sha256": candidate["model_sha256"],
                        "stage": stage,
                        "validation_batch": batch,
                        "episodes": int(manifest["episodes"]),
                        "scenario_seed": int(manifest["scenario_seed"]),
                        "scenario_manifest": manifest["path"],
                        "scenario_manifest_sha256": manifest["sha256"],
                        "yaw_range_deg": float(plan["runtime"]["yaw_range_deg"]),
                        "port": port,
                        "save_tag": tag,
                        "command": command,
                        "planned_at": utcish_now(),
                        "status": "pending",
                        "result_dir": None,
                    }
                )
                index += 1
    return tasks


def create_plan(args: argparse.Namespace) -> tuple[Path, dict[str, Any]]:
    training_root = args.training_root.resolve()
    output_root = args.output_root.resolve()
    legacy_root = FORMAL_EVALUATION_BATCHES_ROOT.resolve()
    if output_root == legacy_root or legacy_root in output_root.parents:
        raise ValueError("--output-root must not be inside Logs/EvaData/FormalBatches")
    if len(set(args.scenes)) != len(args.scenes):
        raise ValueError("--scenes values must be unique")
    if len(set(args.methods)) != len(args.methods):
        raise ValueError("--methods values must be unique")
    if len(set(args.seeds)) != len(args.seeds):
        raise ValueError("--seeds values must be unique")
    target_pairs = parse_targets(args.targets)
    accepted_truncated = parse_accepted_truncated(args.accept_truncated)
    if target_pairs is None:
        target_pairs = {
            (scene, method) for scene in args.scenes for method in args.methods
        }
    requested_scenes = sorted({scene for scene, _ in target_pairs})
    requested_methods = sorted({method for _, method in target_pairs})
    requested_coordinates = {
        (scene, method, seed)
        for scene, method in target_pairs
        for seed in args.seeds
    }
    outside_targets = sorted(set(accepted_truncated) - requested_coordinates)
    if outside_targets:
        raise ValueError(
            "--accept-truncated coordinates must be inside the requested matrix: "
            f"{outside_targets}"
        )
    if args.max_parallel < 1:
        raise ValueError("--max-parallel must be at least 1")
    ports = list(args.ports or default_ports(args.max_parallel))
    if len(ports) < args.max_parallel:
        raise ValueError("--ports must provide at least --max-parallel values")
    ports = ports[: args.max_parallel]
    if len(set(ports)) != len(ports):
        raise ValueError("--ports must be unique")

    protocols = {scene: _protocol_for_scene(scene, args) for scene in requested_scenes}
    runs, issues = discover_training_inventory(
        training_root,
        scenes=requested_scenes,
        methods=requested_methods,
        seeds=args.seeds,
        accepted_truncated=accepted_truncated,
    )
    runs = [run for run in runs if (run.scene, run.method) in target_pairs]
    expected = {
        (scene, method, seed)
        for scene, method in target_pairs
        for seed in args.seeds
    }
    actual = {(run.scene, run.method, run.seed) for run in runs}
    missing = sorted(expected - actual)
    if missing:
        raise RuntimeError(f"Missing artifact-complete training runs: {missing}")

    run_id = datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%f")
    output_dir = output_root / run_id
    output_dir.mkdir(parents=True, exist_ok=False)
    manifests = materialize_scenario_manifests(
        output_dir,
        runs,
        protocols,
        yaw_range_deg=args.yaw_range_deg,
        source_root=args.scenario_manifest_root.resolve() if args.scenario_manifest_root else None,
    )

    training_rows = [
        {
            **asdict(run),
            "run_dir": None,
            "run_relative_path": run.run_relative_path,
            "run_absolute_path": str(run.run_dir),
        }
        for run in runs
    ]
    for row in training_rows:
        row.pop("run_dir", None)

    candidates: list[dict[str, Any]] = []
    for run in runs:
        candidates.extend(
            discover_candidates(
                run,
                mode=args.candidate_mode,
                interval=args.candidate_interval,
            )
        )

    plan: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "created_at": utcish_now(),
        "training_root": str(training_root),
        "output_dir": str(output_dir.resolve()),
        "requested_stage": args.stage,
        "targets": [
            {"scene": scene, "method": method}
            for scene, method in sorted(target_pairs)
        ],
        "candidate_mode": args.candidate_mode,
        "candidate_interval": args.candidate_interval,
        "reject_all_zero_candidates": args.reject_all_zero_candidates,
        "selection_only": args.selection_only,
        "accepted_truncated_runs": [
            {
                "scene": scene,
                "method": method,
                "seed": seed,
                "accepted_step": step,
            }
            for (scene, method, seed), step in sorted(accepted_truncated.items())
        ],
        "verify_escalation_threshold_pct": args.verify_escalation_threshold_pct,
        "protocols": protocols,
        "runtime": {
            "max_parallel": args.max_parallel,
            "ports": ports,
            "ue_exe": str(args.ue_exe.resolve()),
            "settings_template": str(args.settings_template.resolve()),
            "airsim_ip": args.airsim_ip,
            "startup_wait": args.startup_wait,
            "clock_speed": args.clock_speed,
            "res_x": args.res_x,
            "res_y": args.res_y,
            "yaw_range_deg": args.yaw_range_deg,
        },
        "discovery": {
            "eligible_run_count_before_filters": len(list(training_root.rglob("run.json"))),
            "selected_run_count": len(runs),
            "issues_and_exclusions": issues,
            "source": "run.json_and_training_artifacts_only",
            "evaluation_csv_used": False,
        },
        "training_runs": training_rows,
        "candidates": candidates,
        "scenario_manifests": manifests,
        "tasks": [],
        "screen_winners": {},
        "selected_checkpoints": [],
        "no_valid_checkpoints": [],
        "verify_escalations": [],
        "plan_revisions": [
            {"at": utcish_now(), "action": "freeze_training_runs_candidates_and_manifests"}
        ],
    }
    all_candidate_ids: dict[tuple[str, str, int], list[str]] = defaultdict(list)
    for candidate in candidates:
        all_candidate_ids[(candidate["scene"], candidate["method"], candidate["seed"])].append(
            candidate["candidate_id"]
        )
    plan["tasks"] = make_tasks(plan, stage="screen", candidate_ids=all_candidate_ids)

    screen_count = len(plan["tasks"])
    test_count = (
        0
        if args.selection_only
        else sum(len(manifests[key[0]]["test"]) for key in all_candidate_ids)
    )
    plan["estimated_task_counts"] = {
        "screen": screen_count,
        "test": test_count,
        "total": screen_count + test_count,
    }
    atomic_json_write(output_dir / "plan.json", plan)
    return output_dir, plan


def resolve_resume_plan(value: str, output_root: Path) -> Path:
    if value != "latest":
        candidate = Path(value)
        if not candidate.is_absolute():
            by_id = output_root / candidate
            candidate = by_id if by_id.exists() else candidate
        if candidate.is_dir():
            candidate = candidate / "plan.json"
        if not candidate.is_file():
            raise FileNotFoundError(f"Resume plan not found: {candidate}")
        return candidate.resolve()
    plans = sorted(output_root.glob("*/plan.json"), key=lambda path: path.parent.name, reverse=True)
    if not plans:
        raise FileNotFoundError(f"No checkpoint-selection plans under {output_root}")
    return plans[0].resolve()


def load_plan(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"Unsupported plan schema: {payload.get('schema_version')}")
    return payload


def validate_frozen_inputs(plan: dict[str, Any]) -> None:
    for candidate in plan["candidates"]:
        path = Path(candidate["model_absolute_path"])
        if not path.is_file():
            raise FileNotFoundError(f"Frozen model is missing: {path}")
        actual = sha256(path)
        if actual != candidate["model_sha256"]:
            raise ValueError(f"Frozen model SHA256 changed: {path}")
    for scene_stages in plan["scenario_manifests"].values():
        for manifests in scene_stages.values():
            for manifest in manifests:
                path = Path(manifest["path"])
                if not path.is_file() or sha256(path) != manifest["sha256"]:
                    raise ValueError(f"Frozen scenario manifest is missing or changed: {path}")


def read_summary_csv(path: Path) -> dict[str, float | int | None]:
    metrics: dict[str, float | int | None] = {}
    with path.open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows or set(rows[0]) != {"metric", "value"}:
        raise ValueError(f"Invalid summary_metrics.csv: {path}")
    for row in rows:
        key = row["metric"]
        value = row["value"]
        if value == "":
            metrics[key] = None
        elif key == "episodes":
            metrics[key] = int(float(value))
        else:
            metrics[key] = float(value)
    for key in ("episodes", *RANK_METRICS):
        if key not in metrics:
            raise ValueError(f"Missing metric {key} in {path}")
    return metrics


def expected_model_type(task: dict[str, Any]) -> str:
    if task["checkpoint_type"] == "checkpoint":
        return f"checkpoint_{task['checkpoint_step']}"
    return task["checkpoint_type"]


def validate_result_dir(task: dict[str, Any], result_dir: Path) -> dict[str, Any]:
    if not result_dir.is_dir() or task["save_tag"] not in result_dir.name:
        raise ValueError(f"Result directory does not match save tag: {result_dir}")
    manifest_path = result_dir / "evaluation_manifest.json"
    summary_path = result_dir / "summary_metrics.csv"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    metrics = read_summary_csv(summary_path)
    checks = [
        (int(manifest.get("episodes", -1)) == int(task["episodes"]), "episodes"),
        (str(manifest.get("model_type")) == expected_model_type(task), "model_type"),
        (paths_equal(manifest.get("model_file"), task["model_absolute_path"]), "model_file"),
        (int(manifest.get("scenario_seed", -1)) == int(task["scenario_seed"]), "scenario_seed"),
        (
            paths_equal(manifest.get("scenario_manifest_source"), task["scenario_manifest"]),
            "scenario_manifest_source",
        ),
        (math.isclose(float(manifest.get("yaw_range_deg", -1)), float(task["yaw_range_deg"])), "yaw_range_deg"),
        (int(metrics["episodes"] or -1) == int(task["episodes"]), "summary episodes"),
    ]
    if manifest.get("save_tag") is not None:
        checks.append((str(manifest["save_tag"]) == task["save_tag"], "save_tag"))
    if manifest.get("model_sha256") is not None:
        checks.append((str(manifest["model_sha256"]) == task["model_sha256"], "manifest model_sha256"))
    failed = [label for ok, label in checks if not ok]
    if failed:
        raise ValueError(f"Result manifest mismatch ({', '.join(failed)}): {result_dir}")
    model_path = Path(task["model_absolute_path"])
    if not model_path.is_file() or sha256(model_path) != task["model_sha256"]:
        raise ValueError(f"Result model SHA256 mismatch: {model_path}")
    return {"manifest": manifest, "metrics": metrics}


def _candidate_result_dirs(task: dict[str, Any]) -> list[Path]:
    run_dir = Path(task["training_run_absolute_path"])
    try:
        result_parent = evaluation_parent_for_run(run_dir)
    except ValueError:
        # Frozen plans created for a temporary or externally mounted training
        # root may predate the canonical Logs layout. Their existing results
        # remain discoverable beside the run without weakening canonical runs.
        result_parent = run_dir
    if not result_parent.is_dir():
        return []
    return sorted(
        [path for path in result_parent.iterdir() if path.is_dir() and task["save_tag"] in path.name],
        key=lambda path: normalized_path(path),
    )


def find_valid_result(task: dict[str, Any]) -> tuple[Path, dict[str, Any]] | None:
    candidates: list[Path] = []
    if task.get("result_dir"):
        candidates.append(Path(task["result_dir"]))
    candidates.extend(path for path in _candidate_result_dirs(task) if path not in candidates)
    valid: list[tuple[Path, dict[str, Any]]] = []
    for path in candidates:
        try:
            valid.append((path, validate_result_dir(task, path)))
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            continue
    if len(valid) > 1:
        raise ValueError(f"Multiple valid results for frozen task {task['task_id']}: {[str(p) for p, _ in valid]}")
    return valid[0] if valid else None


def _result_record(task: dict[str, Any], result_dir: Path, validated: dict[str, Any]) -> dict[str, Any]:
    return {
        "record_type": "batch",
        "stage": task["stage"],
        "scene": task["scene"],
        "method": task["method"],
        "seed": task["seed"],
        "candidate_id": task["candidate_id"],
        "checkpoint_type": task["checkpoint_type"],
        "checkpoint_step": task["checkpoint_step"],
        "training_step": task["training_step"],
        "model_relative_path": task["model_relative_path"],
        "model_absolute_path": task["model_absolute_path"],
        "model_sha256": task["model_sha256"],
        "validation_batch": task["validation_batch"],
        "scenario_seed": task["scenario_seed"],
        "episodes": task["episodes"],
        "result_dir": str(result_dir.resolve()),
        **validated["metrics"],
    }


def _metric_weight(metric: str, record: dict[str, Any]) -> float:
    if metric in SUCCESS_WEIGHTED_METRICS:
        return float(record["episodes"]) * float(record["success_rate_pct"]) / 100.0
    return float(record["episodes"])


def merge_batch_metrics(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        raise ValueError("Cannot merge an empty result list")
    total_episodes = sum(int(record["episodes"]) for record in records)
    metric_names = sorted(
        set.intersection(
            *(set(record) - set(IDENTITY_FIELDS) for record in records)
        )
    )
    merged: dict[str, Any] = {"episodes": total_episodes}
    for metric in metric_names:
        if metric == "episodes" or metric == "reward_std":
            continue
        values_and_weights = [
            (float(record[metric]), _metric_weight(metric, record))
            for record in records
            if record.get(metric) is not None and _metric_weight(metric, record) > 0
        ]
        merged[metric] = (
            sum(value * weight for value, weight in values_and_weights)
            / sum(weight for _, weight in values_and_weights)
            if values_and_weights
            else None
        )
    reward_rows = [record for record in records if record.get("avg_reward") is not None]
    if reward_rows and all(record.get("reward_std") is not None for record in reward_rows):
        mean = float(merged["avg_reward"])
        second_moment = sum(
            int(record["episodes"])
            * (float(record["reward_std"]) ** 2 + float(record["avg_reward"]) ** 2)
            for record in reward_rows
        ) / sum(int(record["episodes"]) for record in reward_rows)
        merged["reward_std"] = math.sqrt(max(0.0, second_moment - mean**2))
    rates = [float(record["success_rate_pct"]) for record in records]
    merged["batch_success_rate_std_pct"] = statistics.pstdev(rates) if len(rates) > 1 else 0.0
    merged["batch_success_rate_range_pct"] = max(rates) - min(rates) if rates else 0.0
    return merged


def combine_candidate_results(records: Sequence[dict[str, Any]], *, stage: str) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, int, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[(record["scene"], record["method"], int(record["seed"]), record["candidate_id"])].append(record)
    combined: list[dict[str, Any]] = []
    for key, batch_rows in sorted(grouped.items()):
        first = batch_rows[0]
        combined.append(
            {
                "record_type": "combined",
                "stage": stage,
                "scene": key[0],
                "method": key[1],
                "seed": key[2],
                "candidate_id": key[3],
                "checkpoint_type": first["checkpoint_type"],
                "checkpoint_step": first["checkpoint_step"],
                "training_step": first["training_step"],
                "model_relative_path": first["model_relative_path"],
                "model_absolute_path": first["model_absolute_path"],
                "model_sha256": first["model_sha256"],
                "validation_batch": "combined",
                "scenario_seed": "multiple" if len(batch_rows) > 1 else batch_rows[0]["scenario_seed"],
                "result_dir": "",
                **merge_batch_metrics(batch_rows),
            }
        )
    return combined


def ranking_key(record: dict[str, Any]) -> tuple[float, ...]:
    """Lower tuple is better; exact implementation of the fixed lexicographic rule."""
    return (
        -float(record["success_rate_pct"]),
        float(record["collision_rate_pct"]),
        float(record["out_of_bounds_rate_pct"]),
        float(record["timeout_rate_pct"]),
        -float(record["spl"]),
        -float(record["avg_reward"]),
        -float(record.get("training_step", -1) or -1),
        str(record["candidate_id"]),
    )


def rank_by_training_run(records: Sequence[dict[str, Any]]) -> dict[tuple[str, str, int], list[dict[str, Any]]]:
    grouped: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[(record["scene"], record["method"], int(record["seed"]))].append(record)
    return {key: sorted(rows, key=ranking_key) for key, rows in sorted(grouped.items())}


def select_screen_outcomes(
    screen_batch_results: Sequence[dict[str, Any]],
    *,
    reject_all_zero_candidates: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return eligible winners and explicit no-valid-checkpoint records.

    The optional gate prevents an all-zero candidate group from being presented
    as a successful selection.  It is deliberately opt-in so existing frozen
    plans retain their original behavior when resumed.
    """
    if any(record.get("stage") != "screen" for record in screen_batch_results):
        raise ValueError("Checkpoint selection accepts screen-stage results only")
    combined = combine_candidate_results(screen_batch_results, stage="screen")
    winners = []
    no_valid = []
    for key, ranked in rank_by_training_run(combined).items():
        best_success = max(float(row["success_rate_pct"]) for row in ranked)
        if reject_all_zero_candidates and best_success <= 0.0:
            no_valid.append(
                {
                    "status": "NO_VALID_CHECKPOINT",
                    "scene": key[0],
                    "method": key[1],
                    "seed": key[2],
                    "candidate_count": len(ranked),
                    "maximum_success_rate_pct": best_success,
                    "reason": "all_screen_candidates_zero_success",
                }
            )
            continue
        winner = dict(ranked[0])
        winner["selection_rank"] = 1
        winner["selection_reason"] = (
            "lexicographic: success desc, collision asc, out_of_bounds asc, "
            "timeout asc, SPL desc, average reward desc, training step desc"
        )
        winners.append(winner)
    return winners, no_valid


def _write_results(output_dir: Path, stage: str, batch_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    combined = combine_candidate_results(batch_rows, stage=stage)
    payload = {"stage": stage, "batch_results": batch_rows, "combined_results": combined}
    atomic_json_write(output_dir / f"{stage}_results.json", payload)
    rows = [*batch_rows, *combined]
    metric_fields = sorted(set().union(*(row.keys() for row in rows)) - set(IDENTITY_FIELDS)) if rows else []
    fieldnames = [*IDENTITY_FIELDS, *metric_fields]
    with (output_dir / f"{stage}_results.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return combined


def collect_stage_results(plan: dict[str, Any], stage: str) -> list[dict[str, Any]]:
    records = []
    for task in plan["tasks"]:
        if task["stage"] != stage:
            continue
        found = find_valid_result(task)
        if found is None:
            raise RuntimeError(f"Missing or invalid result for task {task['task_id']}")
        records.append(_result_record(task, found[0], found[1]))
    return records


def _candidate(plan: dict[str, Any], candidate_id: str) -> dict[str, Any]:
    return next(item for item in plan["candidates"] if item["candidate_id"] == candidate_id)


def _verify_escalations(
    plan: dict[str, Any],
    ranked: dict[tuple[str, str, int], list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    if plan["candidate_mode"] != "verify":
        return []
    escalations = []
    for key, rows in ranked.items():
        success_values = [float(row["success_rate_pct"]) for row in rows]
        best_alias = next(
            (
                row
                for row in rows
                if any(
                    alias["checkpoint_type"] == "best"
                    for alias in _candidate(plan, row["candidate_id"])["aliases"]
                )
            ),
            None,
        )
        winner = rows[0]
        spread = max(success_values) - min(success_values)
        best_gap = (
            float(winner["success_rate_pct"]) - float(best_alias["success_rate_pct"])
            if best_alias is not None
            else spread
        )
        if max(spread, best_gap) >= float(plan["verify_escalation_threshold_pct"]):
            escalations.append(
                {
                    "scene": key[0],
                    "method": key[1],
                    "seed": key[2],
                    "success_rate_spread_pct": spread,
                    "winner_vs_best_gap_pct": best_gap,
                    "required_action": "rerun_with_candidate_mode_full",
                }
            )
    return escalations


def _write_selected(output_dir: Path, winners: list[dict[str, Any]]) -> None:
    atomic_json_write(output_dir / "selected_checkpoints.json", {"selected": winners})
    fields = [*IDENTITY_FIELDS, *RANK_METRICS, "selection_rank", "selection_reason"]
    with (output_dir / "selected_checkpoints.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(winners)


def _write_no_valid(output_dir: Path, rows: list[dict[str, Any]]) -> None:
    atomic_json_write(output_dir / "no_valid_checkpoints.json", {"no_valid": rows})
    fields = [
        "status",
        "scene",
        "method",
        "seed",
        "candidate_count",
        "maximum_success_rate_pct",
        "reason",
    ]
    with (output_dir / "no_valid_checkpoints.csv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _prepare_test_tasks(plan: dict[str, Any], output_dir: Path) -> None:
    if any(task["stage"] == "test" for task in plan["tasks"]):
        return
    screen_rows = collect_stage_results(plan, "screen")
    screen_combined = _write_results(output_dir, "screen", screen_rows)
    ranked = rank_by_training_run(screen_combined)
    plan["verify_escalations"] = _verify_escalations(plan, ranked)
    if plan["verify_escalations"]:
        atomic_json_write(output_dir / "plan.json", plan)
        return
    winners, no_valid = select_screen_outcomes(
        screen_rows,
        reject_all_zero_candidates=bool(
            plan.get("reject_all_zero_candidates", False)
        ),
    )
    plan["screen_winners"] = {
        f"{row['scene']}/{row['method']}/{int(row['seed'])}": row["candidate_id"]
        for row in winners
    }
    plan["selected_checkpoints"] = winners
    plan["no_valid_checkpoints"] = no_valid
    _write_selected(output_dir, winners)
    _write_no_valid(output_dir, no_valid)
    for row in no_valid:
        print(
            "NO_VALID_CHECKPOINT: "
            f"{row['scene']}/{row['method']}/seed_{row['seed']} "
            f"({row['candidate_count']} screened candidates, all 0% success)"
        )
    selected_ids = {
        (row["scene"], row["method"], int(row["seed"])): [row["candidate_id"]]
        for row in winners
    }
    if not bool(plan.get("selection_only", False)):
        plan["tasks"].extend(make_tasks(plan, stage="test", candidate_ids=selected_ids))
    plan["plan_revisions"].append(
        {
            "at": utcish_now(),
            "action": (
                "freeze_screen_winners_selection_only"
                if bool(plan.get("selection_only", False))
                else "freeze_screen_winners_and_test_tasks"
            ),
        }
    )
    atomic_json_write(output_dir / "plan.json", plan)


def _read_start_position(run_dir: Path) -> list[float]:
    return _read_positions(run_dir)[0]


def _task_waves(tasks: Sequence[dict[str, Any]], width: int) -> list[list[dict[str, Any]]]:
    """Build waves without assigning the same frozen AirSim port twice."""
    remaining = list(tasks)
    waves: list[list[dict[str, Any]]] = []
    while remaining:
        wave: list[dict[str, Any]] = []
        deferred: list[dict[str, Any]] = []
        used_ports: set[int] = set()
        for task in remaining:
            port = int(task["port"])
            if len(wave) < width and port not in used_ports:
                wave.append(task)
                used_ports.add(port)
            else:
                deferred.append(task)
        if not wave:
            raise RuntimeError("Could not construct a unique-port evaluation wave")
        waves.append(wave)
        remaining = deferred
    return waves


def execute_stage(
    plan: dict[str, Any],
    output_dir: Path,
    stage: str,
    *,
    no_launch_airsim: bool,
) -> list[str]:
    plan_path = output_dir / "plan.json"
    pending: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for task in plan["tasks"]:
        if task["stage"] != stage:
            continue
        found = find_valid_result(task)
        if found is not None:
            task["status"] = "complete"
            task["result_dir"] = str(found[0].resolve())
            print(f"[SKIP valid] {task['task_id']}")
        else:
            task["status"] = "pending"
            task["result_dir"] = None
            pending[task["scene"]].append(task)
    atomic_json_write(plan_path, plan)

    failures: list[str] = []
    runtime = plan["runtime"]
    for scene, scene_tasks in sorted(pending.items()):
        processes = []
        try:
            active_ports = list(dict.fromkeys(task["port"] for task in scene_tasks))
            if not no_launch_airsim:
                processes = launch_instances(
                    ue_exe=Path(runtime["ue_exe"]),
                    map_path=SCENES[scene]["map_path"],
                    settings_template=Path(runtime["settings_template"]),
                    output_dir=output_dir / "airsim_settings" / f"{scene}_{stage}",
                    ports=active_ports,
                    start_position=_read_start_position(Path(scene_tasks[0]["training_run_absolute_path"])),
                    clock_speed=float(runtime["clock_speed"]),
                    res_x=int(runtime["res_x"]),
                    res_y=int(runtime["res_y"]),
                    wait_seconds=int(runtime["startup_wait"]),
                )
            for wave_index, wave in enumerate(
                _task_waves(scene_tasks, int(runtime["max_parallel"])), start=1
            ):
                running = []
                print(f"[{scene}/{stage}] wave {wave_index}: {len(wave)} task(s)")
                try:
                    for task in wave:
                        before = {normalized_path(path) for path in _candidate_result_dirs(task)}
                        log_stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", task["task_id"])
                        stdout_path = output_dir / "logs" / f"{log_stem}.out.log"
                        stderr_path = output_dir / "logs" / f"{log_stem}.err.log"
                        stdout_path.parent.mkdir(parents=True, exist_ok=True)
                        stdout = stdout_path.open("w", encoding="utf-8")
                        stderr = stderr_path.open("w", encoding="utf-8")
                        try:
                            process = subprocess.Popen(
                                task["command"],
                                cwd=ROOT,
                                stdout=stdout,
                                stderr=stderr,
                                env=os.environ.copy(),
                            )
                        except BaseException:
                            stdout.close()
                            stderr.close()
                            raise
                        auto_pin_p_core([process.pid])
                        running.append((task, process, stdout, stderr, before))
                        task["status"] = "running"
                        print(f"[START] {task['task_id']} pid={process.pid} port={task['port']}")
                    atomic_json_write(plan_path, plan)
                    for task, process, stdout, stderr, before in running:
                        code = process.wait()
                        stdout.close()
                        stderr.close()
                        if code:
                            task["status"] = "failed"
                            failures.append(f"{task['task_id']}: exit {code}")
                            continue
                        after_paths = _candidate_result_dirs(task)
                        new_paths = [path for path in after_paths if normalized_path(path) not in before]
                        valid_new = []
                        for path in new_paths:
                            try:
                                valid_new.append((path, validate_result_dir(task, path)))
                            except (OSError, TypeError, ValueError, json.JSONDecodeError):
                                continue
                        if len(valid_new) != 1:
                            task["status"] = "failed"
                            failures.append(
                                f"{task['task_id']}: expected one new valid result, got {len(valid_new)}"
                            )
                        else:
                            task["status"] = "complete"
                            task["result_dir"] = str(valid_new[0][0].resolve())
                            print(f"[DONE] {task['task_id']}")
                finally:
                    for _, process, stdout, stderr, _ in running:
                        if process.poll() is None:
                            process.terminate()
                            try:
                                process.wait(timeout=10)
                            except subprocess.TimeoutExpired:
                                process.kill()
                        if not stdout.closed:
                            stdout.close()
                        if not stderr.closed:
                            stderr.close()
                atomic_json_write(plan_path, plan)
                if failures:
                    break
        finally:
            if processes:
                terminate_processes(processes)
        if failures:
            break
    return failures


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Select UAV checkpoints with frozen validation manifests and resumable evaluation tasks."
    )
    parser.add_argument("--training-root", type=Path, default=TRAIN_RUNS_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--scenes", nargs="+", choices=SCENES, default=list(SCENES))
    parser.add_argument("--methods", nargs="+", choices=METHOD_CONFIGS, default=list(DEFAULT_METHODS))
    parser.add_argument(
        "--targets",
        nargs="+",
        help=(
            "Exact scene/method matrix, for example "
            "tunnel:sac,coordatt,lstm tunnel_obs:sac,lstm. "
            "When supplied, this takes precedence over --scenes/--methods."
        ),
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument("--stage", choices=("plan", "screen", "test", "all"), default="all")
    parser.add_argument("--profile", choices=("deterministic", "dynamic"))
    parser.add_argument(
        "--candidate-mode",
        choices=("full", "verify", "training_peaks"),
        default="full",
        help=(
            "full scans periodic checkpoints, verify checks a small structural set, "
            "and training_peaks uses best/final plus two return_ma20 peaks and one "
            "success_rate_ma50 peak without reading evaluation results."
        ),
    )
    parser.add_argument("--candidate-interval", type=int, default=20_000)
    parser.add_argument(
        "--accept-truncated",
        nargs="+",
        default=[],
        metavar="SCENE:METHOD:SEED:STEP",
        help=(
            "Explicitly accept a run that stopped early at its latest periodic "
            "checkpoint. The exception is frozen in plan.json and never relabels "
            "the checkpoint as the declared target."
        ),
    )
    parser.add_argument("--screen-episodes", type=int)
    parser.add_argument("--test-episodes", type=int)
    parser.add_argument("--screen-seeds", nargs="+", type=int)
    parser.add_argument("--test-seeds", nargs="+", type=int)
    parser.add_argument("--max-parallel", type=int, default=3)
    parser.add_argument("--ports", nargs="+", type=int)
    parser.add_argument("--ue-exe", type=Path, default=Path(r"D:\UEBuilds\TunnelEnv\Windows\Tunnel.exe"))
    parser.add_argument(
        "--settings-template",
        type=Path,
        default=Path(r"C:\Users\31732\Documents\AirSim\settings.json"),
    )
    parser.add_argument("--airsim-ip", default="127.0.0.1")
    parser.add_argument("--startup-wait", type=int, default=60)
    parser.add_argument("--clock-speed", type=float, default=1.0)
    parser.add_argument("--res-x", type=int, default=640)
    parser.add_argument("--res-y", type=int, default=480)
    parser.add_argument("--yaw-range-deg", type=float, default=10.0)
    parser.add_argument("--scenario-manifest-root", type=Path)
    parser.add_argument("--verify-escalation-threshold-pct", type=float, default=5.0)
    parser.add_argument(
        "--reject-all-zero-candidates",
        action="store_true",
        help=(
            "Write NO_VALID_CHECKPOINT and skip formal-test task creation when "
            "every screened candidate for a method/seed has 0%% success."
        ),
    )
    parser.add_argument(
        "--selection-only",
        action="store_true",
        help="Finalize screen selection without creating any formal-test tasks.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--resume",
        nargs="?",
        const="latest",
        help="Resume the latest plan, a run ID, a run directory, or a plan.json path.",
    )
    parser.add_argument("--no-launch-airsim", action="store_true")
    return parser.parse_args(argv)


def _print_plan_summary(plan: dict[str, Any], output_dir: Path, *, dry_run: bool) -> None:
    counts = plan["estimated_task_counts"]
    print(
        f"Discovered {len(plan['training_runs'])} canonical training runs and "
        f"{len(plan['candidates'])} SHA256-unique candidates."
    )
    print(
        f"Tasks: screen={counts['screen']}, test={counts['test']}, "
        f"total={counts['total']}"
    )
    excluded = [
        item for item in plan["discovery"]["issues_and_exclusions"]
        if item.get("reason") == "duplicate_coordinate_older_run_json_created_at"
    ]
    if excluded:
        print(f"Audited duplicate-coordinate exclusions: {len(excluded)}")
    print(f"Plan: {output_dir / 'plan.json'}")
    if dry_run:
        for task in plan["tasks"]:
            print(f"  {task['task_id']}: {printable(task['command'])}")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.yaw_range_deg < 0:
        raise ValueError("--yaw-range-deg must be non-negative")
    if args.resume:
        plan_path = resolve_resume_plan(args.resume, args.output_root.resolve())
        plan = load_plan(plan_path)
        output_dir = plan_path.parent
        validate_frozen_inputs(plan)
        print(f"Resuming frozen plan {plan['run_id']} (no candidate rediscovery).")
    else:
        if args.stage == "test":
            raise ValueError(f"--stage {args.stage} requires --resume of an earlier plan")
        output_dir, plan = create_plan(args)

    _print_plan_summary(plan, output_dir, dry_run=args.dry_run)
    if args.dry_run or args.stage == "plan":
        print("Dry run/plan only: no UE or evaluation process was started.")
        return 0

    stages = ["screen", "test"] if args.stage == "all" else [args.stage]
    for stage in stages:
        if stage == "test":
            if bool(plan.get("selection_only", False)):
                raise ValueError("This frozen plan is selection-only; formal test is disabled")
            _prepare_test_tasks(plan, output_dir)
            if plan["verify_escalations"]:
                raise RuntimeError("Cannot test a verify-mode plan that requires full screening")

        failures = execute_stage(
            plan,
            output_dir,
            stage,
            no_launch_airsim=args.no_launch_airsim,
        )
        if failures:
            print("Checkpoint-selection failures:", file=sys.stderr)
            for failure in failures:
                print(f"  {failure}", file=sys.stderr)
            return 1
        batch_rows = collect_stage_results(plan, stage)
        _write_results(output_dir, stage, batch_rows)
        if stage == "screen":
            _prepare_test_tasks(plan, output_dir)
            if plan["verify_escalations"]:
                print(
                    "Verify mode found material differences; full screening is required for: "
                    + ", ".join(
                        f"{row['scene']}/{row['method']}/seed_{row['seed']}"
                        for row in plan["verify_escalations"]
                    ),
                    file=sys.stderr,
                )
                return 2

    print(f"Completed requested stage(s). Results: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
