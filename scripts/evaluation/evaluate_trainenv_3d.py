"""Fly a fixed checkpoint on the locked four-family TrainEnv_3D protocol."""

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import shutil
import traceback

import numpy as np
import torch

from scripts.evaluation.evaluate_single import get_base_env
from tools.runtime_support import build_runtime_env, load_model
from tools.trainenv_instances import DEFAULT_3D_EXE, DEFAULT_EDITOR, HOST, TRAINENV_3D_MAP, launch_trainenv
from tools.ue_benchmark.ch4_trainenv_3d_course import (
    COURSE_FILE, ROOT, build_protocol, obstacles, read_json, sha256, write_json,
)
from tools.ue_benchmark.ch4_trainenv_3d_runtime import (
    REPORT, SETTINGS_TEMPLATE, SPAWN, TRAIN_CONFIG, VEHICLE, DepthAudit,
    assert_observation_contract, assert_reset_depth_consistent, collision_details, json_line, progress_through_obstacles,
    read_ini, scenario_errors, start_sample, step_sample,
)


PRIOR_RESULTS = ROOT / "reports/ch4_benchmark_rebuild_20261125/eval_stage2_495k/overall_summary.json"


def frozen_checkpoint(protocol, output, checkpoint_override, resume):
    spec = protocol["settings"]
    prior = read_json(PRIOR_RESULTS)
    for family in ("t1", "t2", "t3"):
        if (prior[family]["checkpoint_sha256"] != spec["checkpoint_sha256"]
                or prior[family]["model_step"] != spec["checkpoint_step"]
                or not prior[family]["use_history"]):
            raise RuntimeError(f"{family} used a different checkpoint or history mode")
    original = checkpoint_override or Path(prior["t1"]["checkpoint"])
    original_config = original.parent.parent / "config/config.ini"
    if sha256(original) != spec["checkpoint_sha256"]:
        raise RuntimeError(f"Checkpoint identity mismatch: {original}")
    frozen = output / "best_495k.zip"
    config_copy = output / "original_training.ini"
    if not resume:
        shutil.copyfile(original, frozen)
        shutil.copyfile(original_config, config_copy)
    if sha256(frozen) != spec["checkpoint_sha256"] or sha256(config_copy) != sha256(original_config):
        raise RuntimeError("Frozen checkpoint or its training config has the wrong identity")
    return frozen, config_copy, original


def freeze_candidate(candidate, checkpoint_step, output, resume):
    candidate = candidate.resolve(strict=True)
    source_config = candidate.parent.parent / "config/config.ini"
    assert_observation_contract(read_ini(source_config))
    frozen = output / "candidate.zip"
    config_copy = output / "candidate_training.ini"
    manifest = output / "candidate_identity.json"
    expected = {
        "checkpoint_step": checkpoint_step,
        "checkpoint_sha256": sha256(candidate),
        "training_config_sha256": sha256(source_config),
    }
    if not resume:
        shutil.copyfile(candidate, frozen)
        shutil.copyfile(source_config, config_copy)
        write_json(manifest, expected)
    if (read_json(manifest) != expected or sha256(frozen) != expected["checkpoint_sha256"]
            or sha256(config_copy) != expected["training_config_sha256"]):
        raise RuntimeError("Candidate identity differs from the fixed evaluation copy")
    return frozen, candidate, expected


def evaluation_config(original_config, protocol):
    cfg = read_ini(original_config)
    scene = read_ini(TRAIN_CONFIG)
    spec = protocol["settings"]
    assert_observation_contract(cfg)
    for key in ("start_position", "goal_position", "work_space_x", "work_space_y", "work_space_z"):
        cfg["environment"][key] = scene["environment"][key]
    cfg["environment"].update({
        "accept_radius": str(spec["accept_radius_m"]),
        "accept_altitude_tolerance": str(spec["accept_altitude_tolerance_m"]),
        "max_episode_steps": str(spec["max_episode_steps"]),
        "start_random_angle": "0",
        "random_route_reverse": "False",
    })
    cfg.remove_option("environment", "training_routes")
    cfg["perception"]["virtual_ceiling_prob"] = "0"
    for key in ("altitude_deviation_weight", "altitude_boundary_margin", "altitude_boundary_weight"):
        cfg["reward"][key] = scene["reward"][key]
    cfg["airsim"].update(
        ip=HOST, vehicle_name=VEHICLE, depth_only="True", synchronous_steps="False", timeout_value="60",
    )
    return cfg


def policy_digest(policy):
    digest = hashlib.sha256()
    for name, tensor in policy.state_dict().items():
        digest.update(name.encode("utf-8"))
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def fly_episode(env, vector_env, model, course, scenario, spec, output):
    depth = DepthAudit(spec)
    scenario_id = scenario["scenario_id"]
    goal = scenario["goal_xyz_agl_m"]
    trajectory = output / f"{scenario_id}.jsonl"
    with depth.attach(env.client), trajectory.open("x", encoding="utf-8") as records:
        observation, reset_info = env.reset()
        if reset_info["scenario_id"] != scenario_id:
            raise RuntimeError(f"Scenario mismatch: {reset_info}")
        first = start_sample(env, course, depth.latest(0))
        assert_reset_depth_consistent(first, course)
        start_errors = scenario_errors(first["position_xyz_agl_m"], scenario["start_xyz_agl_m"])
        yaw_error = math.remainder(first["attitude_rad"][2] - scenario["yaw_rad"], math.tau)
        if max(start_errors) > 0.30 or abs(yaw_error) > 0.12:
            raise RuntimeError(f"{scenario_id} reset error: {start_errors} m, {yaw_error} rad")
        initial_depth = depth.frame.copy()
        samples = [first]
        json_line(records, first)
        total_reward = 0.0
        while True:
            previous_depth_ns = depth.previous_time_ns
            action, _ = model.predict(
                vector_env.transpose_observations(observation), deterministic=True,
                episode_start=np.array([len(samples) == 1]), use_history=True,
            )
            observation, reward, terminated, truncated, info = env.step(action)
            total_reward += float(reward)
            sample = step_sample(info, course, depth.latest(previous_depth_ns))
            samples.append(sample)
            json_line(records, sample)
            if terminated or truncated:
                break
    np.savez_compressed(output / f"{scenario_id}_depth.npz", reset=initial_depth, terminal=depth.frame)
    terminal = samples[-1]
    horizontal, altitude = scenario_errors(terminal["position_xyz_agl_m"], goal)
    family = scenario["family"]
    obstacle_records, order = progress_through_obstacles(samples, family, course)
    expected_order = [obstacle["name"] for obstacle in obstacles(family, course)]
    route_completed = order == expected_order
    if info["is_crash"]:
        outcome = "collision"
    elif info["is_not_in_workspace"]:
        outcome = "out_of_bounds"
    elif truncated:
        outcome = "timeout"
    elif info["is_success"]:
        outcome = "success" if route_completed else "route_not_completed"
    else:
        raise RuntimeError(f"Unclassified episode termination: {info}")
    if info["is_success"] and (horizontal >= spec["accept_radius_m"]
                                or altitude >= spec["accept_altitude_tolerance_m"]):
        raise RuntimeError("Environment success does not satisfy the locked goal tolerances")
    return {
        **scenario,
        "outcome": outcome,
        "goal_reached": info["is_success"],
        "collision": info["is_crash"],
        "out_of_bounds": info["is_not_in_workspace"],
        "timeout": truncated,
        "terminal_xyz_agl_m": terminal["position_xyz_agl_m"],
        "terminal_xy_error_m": horizontal,
        "terminal_altitude_error_m": altitude,
        "observed_start_xyz_agl_m": first["position_xyz_agl_m"],
        "observed_start_yaw_deg": math.degrees(first["attitude_rad"][2]),
        "route_completed": route_completed,
        "obstacle_order": order,
        "obstacle_records": obstacle_records,
        "collision_event": collision_details(env.client) if info["is_crash"] else None,
        "steps": len(samples) - 1,
        "reset_depth_consistent": True,
        "depth_frames_checked": len(samples),
        "min_finite_positive_depth_fraction": min(row["depth"]["positive_finite_fraction"] for row in samples),
        "min_observed_depth_m": min(row["depth"]["min_m"] for row in samples),
        "sim_seconds": (terminal["state_timestamp_ns"] - first["state_timestamp_ns"]) / 1e9,
        "total_reward": total_reward,
        "trajectory": trajectory.relative_to(output.parent).as_posix(),
    }


def completed_episodes(output, scenarios):
    path = output / "episodes.jsonl"
    if not path.is_file():
        return []
    with path.open(encoding="utf-8") as stream:
        results = [json.loads(line) for line in stream]
    expected = [row["scenario_id"] for row in scenarios[:len(results)]]
    if [row["scenario_id"] for row in results] != expected:
        raise RuntimeError(f"Recorded episodes in {output} do not match the locked scenario prefix")
    for row in results:
        trace = output.parent / row["trajectory"]
        with trace.open(encoding="utf-8") as stream:
            frames = sum(1 for _ in stream)
        if frames != row["depth_frames_checked"] or frames != row["steps"] + 1:
            raise RuntimeError(f"Incomplete trace for {row['scenario_id']}")
    return results


def family_summary(family, results, **verification):
    return {
        "family": family,
        "episodes": len(results),
        "outcome_counts": dict(Counter(result["outcome"] for result in results)),
        "route_completions": sum(row["route_completed"] for row in results),
        "mean_terminal_altitude_error_m": float(np.mean([r["terminal_altitude_error_m"] for r in results])),
        "depth_frames_checked": sum(row["depth_frames_checked"] for row in results),
        "reset_depth_checks_passed": sum(row.get("reset_depth_consistent", False) for row in results),
        **verification,
    }


def evaluate_family(family, scenarios, completed, port, cfg, checkpoint, checkpoint_step, course, spec, output, attempt):
    torch.set_num_threads(1)
    cfg["airsim"]["rpc_port"] = str(port)
    output.mkdir(exist_ok=True)
    if attempt is not None:
        cfg["airsim"]["timeout_value"] = "60"
        archive = output / "interrupted" / attempt
        previous = [output / "effective_config.ini"]
        for scenario in scenarios[len(completed):]:
            scenario_id = scenario["scenario_id"]
            previous.extend([output / f"{scenario_id}.jsonl", output / f"{scenario_id}_depth.npz"])
        for path in previous:
            if path.is_file():
                archive.mkdir(parents=True, exist_ok=True)
                path.rename(archive / path.name)
    with (output / "effective_config.ini").open("w", encoding="utf-8") as stream:
        cfg.write(stream)
    vector_env = build_runtime_env(cfg)
    env = get_base_env(vector_env)
    pending = scenarios[len(completed):]
    env.set_evaluation_scenarios(pending)
    try:
        model = load_model("SAC", str(checkpoint), vector_env, cfg=cfg)
        if model.num_timesteps != checkpoint_step or model.policy.actor.inference_seq_len != 8:
            raise RuntimeError("Loaded the wrong checkpoint step or actor history length")
        model.policy.set_training_mode(False)
        model.policy.requires_grad_(False)
        digest_before, updates_before = policy_digest(model.policy), model._n_updates
        results = list(completed)
        with torch.inference_mode(), (output / "episodes.jsonl").open("a", encoding="utf-8") as summary:
            for scenario in pending:
                result = fly_episode(env, vector_env, model, course, scenario, spec, output)
                results.append(result)
                json_line(summary, result)
                summary.flush()
                print(
                    f"{result['scenario_id']}: {result['outcome']}, "
                    f"{result['steps']} steps, final altitude error {result['terminal_altitude_error_m']:.2f} m",
                    flush=True,
                )
        digest_after = policy_digest(model.policy)
        if model._n_updates != updates_before or digest_after != digest_before:
            raise RuntimeError("The frozen policy changed during flight")
        result = family_summary(
            family, results, checkpoint_step=model.num_timesteps, device=str(model.device),
            history_length=model.policy.actor.inference_seq_len, policy_sha256_before=digest_before,
            policy_sha256_after=digest_after, training_updates_before=updates_before,
            training_updates_after=model._n_updates, previously_completed=len(completed),
        )
        write_json(output / "family_summary.json", result)
        return result
    finally:
        vector_env.close()
        env.client.client.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=REPORT / "eval_protocol.json")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--resume", action="store_true", help="Retain complete episodes and fly only missing cases")
    parser.add_argument("--ue-exe", type=Path)
    parser.add_argument("--ue-project", type=Path, help="Use standalone UE Editor with this source project")
    models = parser.add_mutually_exclusive_group()
    models.add_argument("--checkpoint", type=Path, help="Optional copy of the same baseline, with its run config")
    models.add_argument("--candidate-checkpoint", type=Path, help="Fine-tuned weights to fly on the original protocol")
    parser.add_argument("--candidate-step", type=int, help="Actual training step saved in a candidate checkpoint")
    parser.add_argument("--first-port", type=int, default=41970)
    parser.add_argument("--max-instances", type=int, default=4, help="Maximum simultaneously running simulators")
    parser.add_argument("--episodes-per-family", type=int, help="Fly a locked prefix for candidate screening")
    args = parser.parse_args()
    if (args.candidate_checkpoint is None) != (args.candidate_step is None):
        parser.error("--candidate-checkpoint and --candidate-step must be supplied together")
    if args.max_instances < 1:
        parser.error("--max-instances must be positive")

    course, protocol = read_json(COURSE_FILE), read_json(args.protocol)
    if protocol != build_protocol(course) or course["map"] != TRAINENV_3D_MAP:
        raise RuntimeError("The locked scenarios or protocol do not match the authored course")
    output = args.run_dir.resolve()
    launch_output, attempt = output, None
    if args.resume:
        if sha256(args.protocol) != sha256(output / "eval_protocol.json"):
            raise RuntimeError("The existing run used a different protocol file")
        attempt = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        launch_output = output / "resumes" / attempt
        launch_output.mkdir(parents=True)
    else:
        output.mkdir(parents=True, exist_ok=False)
        shutil.copyfile(args.protocol, output / "eval_protocol.json")
    spec = protocol["settings"]
    episode_limit = args.episodes_per_family
    if episode_limit is not None and not 1 <= episode_limit <= spec["episodes_per_family"]:
        parser.error("--episodes-per-family must select a non-empty prefix of the locked protocol")
    if args.candidate_checkpoint is not None:
        checkpoint, original_checkpoint, identity = freeze_candidate(
            args.candidate_checkpoint, args.candidate_step, output, args.resume,
        )
        baseline = Path(read_json(PRIOR_RESULTS)["t1"]["checkpoint"])
        if sha256(baseline) != spec["checkpoint_sha256"]:
            raise RuntimeError("The locked Stage 2 baseline checkpoint does not match the protocol")
        original_config = baseline.parent.parent / "config/config.ini"
    else:
        checkpoint, original_config, original_checkpoint = frozen_checkpoint(
            protocol, output, args.checkpoint, args.resume,
        )
        identity = {"checkpoint_step": spec["checkpoint_step"], "checkpoint_sha256": spec["checkpoint_sha256"]}
    cfg = evaluation_config(original_config, protocol)
    families = list(course["families"])
    by_family = {
        name: [row for row in protocol["scenarios"] if row["family"] == name][:episode_limit]
        for name in families
    }
    recorded = {family: completed_episodes(output / family, by_family[family]) for family in families}
    unfinished = [family for family in families if len(recorded[family]) < len(by_family[family])]
    results, errors = {}, {}
    for family in families:
        if family in unfinished:
            continue
        summary = output / family / "family_summary.json"
        results[family] = read_json(summary) if summary.is_file() else family_summary(
            family, recorded[family], previously_completed=len(recorded[family]),
        )
    executable = args.ue_exe or (DEFAULT_EDITOR if args.ue_project else DEFAULT_3D_EXE)
    for start in range(0, len(unfinished), args.max_instances):
        batch = unfinished[start:start + args.max_instances]
        ports = list(range(args.first_port, args.first_port + len(batch)))
        family_ports = dict(zip(batch, ports, strict=True))
        batch_output = launch_output
        if len(unfinished) > args.max_instances:
            batch_output = launch_output / f"batch_{start // args.max_instances + 1:02d}"
            batch_output.mkdir()
        with launch_trainenv(
            executable=executable.resolve(), template=SETTINGS_TEMPLATE, ports=ports, start_position=SPAWN,
            clock_speed=spec["clock_speed"], run_dir=batch_output, timeout=180, map_path=TRAINENV_3D_MAP,
            editor_project=args.ue_project.resolve() if args.ue_project else None,
        ) as job, ProcessPoolExecutor(max_workers=len(batch)) as pool:
            workers = {
                pool.submit(
                    evaluate_family, family, by_family[family], recorded[family], port, deepcopy(cfg),
                    checkpoint, identity["checkpoint_step"], course, spec, output / family, attempt,
                ): family
                for family, port in family_ports.items()
            }
            # Spawned Python workers share the simulator lifetime on Windows.
            # Otherwise a forced parent exit can leave their inherited pipes open.
            children = list(pool._processes.values())
            try:
                for process in children:
                    job.add(process)
                for worker in as_completed(workers):
                    family = workers[worker]
                    try:
                        results[family] = worker.result()
                    except Exception:
                        errors[family] = traceback.format_exc()
                        print(errors[family], flush=True)
                    finally:
                        job.stop_simulator(family_ports[family])
            except BaseException:
                for process in children:
                    if process.is_alive():
                        process.terminate()
                raise
    if unfinished:
        write_json(launch_output / "family_results.json", results)
    if errors:
        write_json(launch_output / "evaluation_errors.json", errors)
        raise RuntimeError(f"Incomplete evaluation, failed families: {', '.join(errors)}")
    fingerprint = identity["checkpoint_sha256"]
    if sha256(checkpoint) != fingerprint or sha256(original_checkpoint) != fingerprint:
        raise RuntimeError("Checkpoint file changed during evaluation")
    write_json(output / "summary.json", {
        "protocol_sha256": sha256(args.protocol),
        "geometry_sha256": protocol["geometry_sha256"],
        **identity,
        "source_training_config_sha256": sha256(original_config),
        "map": TRAINENV_3D_MAP,
        "deterministic": True,
        "gradient_updates": 0,
        "complete_protocol": all(len(by_family[name]) == spec["episodes_per_family"] for name in families),
        "normalization": {
            "distance_m": cfg.getfloat("environment", "distance_norm_scale"),
            "altitude_m": cfg.getfloat("environment", "max_vertical_difference"),
            "depth_m": cfg.getfloat("environment", "max_depth_meters"),
        },
        "families": {family: results[family] for family in families},
    })


if __name__ == "__main__":
    main()
