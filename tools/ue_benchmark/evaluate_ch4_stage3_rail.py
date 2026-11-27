"""Evaluate a Stage 3 checkpoint with one frozen Chapter 4 railway protocol."""

import argparse
from pathlib import Path
import subprocess
from uuid import uuid4

from scripts.evaluation.evaluate_single import main as evaluate
from tools.airsim_instances import validate_ports_available
from tools.trainenv_instances import DEFAULT_EDITOR, HOST, _UnrealJob, _wait_ready
from tools.ue_benchmark.ch4_trainenv_3d_course import ROOT, read_json, sha256, write_json
from tools.ue_benchmark.ch4_trainenv_3d_runtime import assert_observation_contract, read_ini


BASELINE = ROOT / "reports/ch4_benchmark_rebuild_20261125/eval_stage2_495k"
PROJECT = Path("D:/software/Epic Games/UE/UE_5.4/Env/Tunnel/Tunnel.uproject")
SCENE_CONFIGS = {
    "t1": "ch4_eval_t1_double_static.ini",
    "t2": "ch4_eval_t2_double_dynamic.ini",
    "t3": "ch4_eval_t3_single_scurve.ini",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", choices=SCENE_CONFIGS, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--checkpoint-step", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--port", type=int, default=41982)
    parser.add_argument("--episodes", type=int, default=10, help="Use a locked prefix for candidate screening")
    args = parser.parse_args()
    if not 1 <= args.episodes <= 10:
        parser.error("--episodes must select a non-empty prefix of the ten locked cases")

    original = read_json(BASELINE / "overall_summary.json")[args.scene]
    map_path = original["map"]
    package = PROJECT.parent / "Content" / (map_path.removeprefix("/Game/") + ".umap")
    if sha256(package) != original["map_sha256"]:
        raise RuntimeError(f"The frozen {args.scene} map has changed")
    baseline_checkpoint = Path(original["checkpoint"])
    if sha256(baseline_checkpoint) != original["checkpoint_sha256"]:
        raise RuntimeError("The frozen railway checkpoint has changed")
    baseline_config = baseline_checkpoint.parent.parent / "config/config.ini"
    assert_observation_contract(read_ini(baseline_config))
    run = args.run.resolve(strict=True)
    assert_observation_contract(read_ini(run / "config/config.ini"))
    checkpoint = run / "models" / f"checkpoint_{args.checkpoint_step}_steps.zip"
    fingerprint = sha256(checkpoint)
    scene_config = ROOT / "configs" / SCENE_CONFIGS[args.scene]
    scenario_manifest = Path(original["artifacts"]) / "scenario_manifest.json"
    if len(read_json(scenario_manifest)["scenarios"]) != 10 or not original["use_history"]:
        raise RuntimeError("Expected the original ten-case full-history protocol")

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    launch_id = uuid4().hex
    settings = read_json(BASELINE / args.scene / "settings.json")
    if settings["ClockSpeed"] != original["clock_speed"]:
        raise RuntimeError("Railway clock speed differs from baseline")
    settings.update(ApiServerPort=args.port, TrainEnvLaunchId=launch_id)
    settings_path = output / "settings.json"
    write_json(settings_path, settings)
    validate_ports_available(HOST, [args.port])

    unreal_log = output / "unreal.log"
    command = [
        str(DEFAULT_EDITOR), str(PROJECT), map_path, "-game", "-windowed", "-ResX=320", "-ResY=240",
        "-NoSound", "-unattended", "-nosplash", "-NoVSync", f"-settings={settings_path}", f"-abslog={unreal_log}",
    ]
    write_json(output / "protocol_identity.json", {
        "map": map_path, "map_sha256": original["map_sha256"],
        "scenario_manifest": str(scenario_manifest), "scenario_sha256": sha256(scenario_manifest),
        "scene_config": str(scene_config), "scene_config_sha256": sha256(scene_config),
        "runtime_config": str(baseline_config), "runtime_config_sha256": sha256(baseline_config),
        "candidate": str(checkpoint), "candidate_sha256": fingerprint, "checkpoint_step": args.checkpoint_step,
        "command": command, "seed": original["seed"], "yaw_range_deg": original["yaw_range_deg"],
        "episodes": args.episodes,
    })
    process = None
    try:
        with _UnrealJob() as job:
            process = subprocess.Popen(command, cwd=DEFAULT_EDITOR.parent, creationflags=subprocess.CREATE_NO_WINDOW)
            try:
                job.add(process)
            except BaseException:
                process.kill()
                raise
            print(f"[Rail {args.scene}] Starting UE PID {process.pid} on {args.port}", flush=True)
            _wait_ready({args.port: process}, launch_id=launch_id, timeout=180)
            result = evaluate([
                "--model_path", str(run), "--checkpoint-step", str(args.checkpoint_step),
                "--scene-config", str(scene_config), "--scenario-manifest", str(scenario_manifest),
                "--scenario-seed", str(original["seed"]), "--yaw-range-deg", str(original["yaw_range_deg"]),
                "--airsim-ip", HOST, "--airsim-port", str(args.port), "--episodes", str(args.episodes),
                "--save", "--output-root", str(output / "results"), "--save-tag", args.scene,
            ], runtime_config=baseline_config)
    finally:
        if process is not None:
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=10)

    summary = read_json(result / "summary_metrics.json")
    if (sha256(checkpoint) != fingerprint or sha256(package) != original["map_sha256"]
            or summary["episodes"] != args.episodes):
        raise RuntimeError("Railway evaluation evidence is incomplete or has changed")
    write_json(output / "result.json", {
        "scene": args.scene, "map_sha256": original["map_sha256"], "checkpoint_sha256": fingerprint,
        "checkpoint_step": args.checkpoint_step, "artifacts": str(result), "summary": summary,
        "complete_protocol": args.episodes == 10,
    })


if __name__ == "__main__":
    main()
