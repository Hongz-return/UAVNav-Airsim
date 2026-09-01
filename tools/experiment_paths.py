"""Canonical experiment-data paths shared by training and evaluation tools."""

from __future__ import annotations

import re
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
LOGS_ROOT = REPO_ROOT / "Logs"

TRAIN_DATA_ROOT = LOGS_ROOT / "TrainData"
TRAIN_RUNS_ROOT = TRAIN_DATA_ROOT / "Runs"
TRAIN_RUN_METADATA_ROOT = TRAIN_DATA_ROOT / "RunMetadata"
PARALLEL_RUNS_ROOT = TRAIN_RUN_METADATA_ROOT / "ParallelRuns"

EVA_DATA_ROOT = LOGS_ROOT / "EvaData"
EVALUATION_RESULTS_ROOT = EVA_DATA_ROOT / "Results"
FORMAL_EVALUATION_BATCHES_ROOT = EVA_DATA_ROOT / "FormalBatches"
CHECKPOINT_SELECTION_ROOT = EVA_DATA_ROOT / "CheckpointSelection"
ENV1_SELECTION_ROOT = CHECKPOINT_SELECTION_ROOT / "ENV1_Tunnel"
ENV2_SELECTION_ROOT = CHECKPOINT_SELECTION_ROOT / "ENV2_TunnelObs"
ENV3_DYNAMIC_SELECTION_ROOT = CHECKPOINT_SELECTION_ROOT / "ENV3_TunnelActivate"
ENV4_SELECTION_ROOT = CHECKPOINT_SELECTION_ROOT / "ENV4_TunnelMixed"
CROSS_SCENE_SELECTION_ROOT = CHECKPOINT_SELECTION_ROOT / "CrossScene"
DEMO_RECORDINGS_ROOT = EVA_DATA_ROOT / "DemoRecordings"
DIAGNOSTICS_ROOT = EVA_DATA_ROOT / "Diagnostics"
UE_LIVE_ROOT = DIAGNOSTICS_ROOT / "UELiveObsTunnel"
GENERATED_PACKAGES_ROOT = REPO_ROOT / "outputs" / "packages"

_BATCH_ID = re.compile(r"\d{4}_\d{2}_\d{2}_\d{2}_\d{2}_\d{2}_\d{6}")


def normalize_training_relative(relative: Path) -> Path:
    """Remove only the obsolete outer batch layer used by older ENV4 runs."""
    parts = relative.parts
    if parts and _BATCH_ID.fullmatch(parts[0]) and len(parts) > 1:
        parts = parts[1:]
    return Path(*parts)


def training_run_relative(run_dir: str | Path) -> Path:
    """Return the stable scene/method/seed/run path for a training run."""
    resolved = Path(run_dir).resolve()
    try:
        return normalize_training_relative(resolved.relative_to(TRAIN_RUNS_ROOT.resolve()))
    except ValueError as exc:
        raise ValueError(f"Training run is outside the canonical data root: {resolved}") from exc


def evaluation_parent_for_run(
    run_dir: str | Path,
    output_root: str | Path = EVALUATION_RESULTS_ROOT,
) -> Path:
    """Mirror a training run's stable identity below the evaluation-results root."""
    return Path(output_root) / training_run_relative(run_dir)
