"""Commission and run the frozen MA-SAC paired comparison on packaged TrainEnv."""

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime, timezone
from hashlib import sha256 as tensor_hash
import math
from pathlib import Path
import time
import traceback

import numpy as np
import torch
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage

from tools.runtime_support import _build_monitored_env, load_model
from tools.trainenv_instances import DEFAULT_EXE, TRAINENV_MAP, launch_trainenv
from tools.ue_benchmark.ch4_stage3_crossing import (
    CROSSING_KEY, CROSSING_SIZE_M, HALF_TRAVEL_M, SPEED_MPS, CrossingCase, CrossingPath, enable_crossing_clock,
    ready_reset_observation,
)
from tools.ue_benchmark.ch4_stage3_v2_course import SPAWN_NED, TransientBoxes, tunnel_roof
from tools.ue_benchmark.ch4_trainenv_3d_course import ROOT, read_json, sha256, write_json
from tools.ue_benchmark.ch4_trainenv_3d_runtime import assert_observation_contract, json_line, read_ini
from tools.ue_benchmark.ch4_trainenv_crossing_audit import (
    MAX_UNBLOCKED_LAG_M, MAX_UPDATE_GAP_S, NEAR_SURFACE_M, SURVEY, assert_surveyed_section, capture_depth,
    check_scene, current_obstacle, depth_geometry, distance_to_box, episode_clearance, surveyed_boxes,
)


REPORT = ROOT / "reports/ch4_trainenv_crossing_20261126"
TEMPLATE = ROOT / "configs/ch4_stage3_trainenv_airsim.json"
IDENTITY = ROOT / "reports/ch4_stage3_v2_20261125/final_500000_scratch500k/rail/t2/protocol_identity.json"
CHECKPOINT_SHA256 = "ae24021e62f35f2d134fd4c563db2460d98ea1f42150c666ab10052454300324"
BOUNDS = ([.5, -.8, -math.pi / 3], [4., .8, math.pi / 3])
CLOCK_SPEED = 1.0
SEED = 200
PHASES = {"early": -1.45, "near": 0.0, "late": 2.2}
REFERENCES = (
    ("A", (0, -15, 1), (162, -15, 1), 10.5, -2., 1.75, -1),
    ("B", (0, 0, 1), (162, 15, 1), 12.5, 3., 2.05, 1),
    ("C", (162, -15, 1), (0, 0, 1), 151.5, -3., 2.15, -1),
    ("D", (162, 0, 1), (0, 0, 1), 151., 2., 1.95, 1),
)


def checkpoint():
    identity = read_json(IDENTITY)
    path = Path(identity["candidate"])
    if identity["candidate_sha256"] != CHECKPOINT_SHA256 or sha256(path) != CHECKPOINT_SHA256:
        raise RuntimeError("Frozen checkpoint differs from the T2 500k protocol identity")
    return path


def config(path, port, *, crossing=False):
    cfg = read_ini(path.parents[1] / "config/config.ini")
    assert_observation_contract(cfg)
    cfg["airsim"].update(rpc_port=str(port), synchronous_steps="True")
    if crossing:
        cfg["stage3_v2"]["crossing_mode"] = "moving"
        enable_crossing_clock(cfg)
    return cfg


def launch(port, output):
    return launch_trainenv(executable=DEFAULT_EXE, template=TEMPLATE, ports=[port], start_position=SPAWN_NED,
                           clock_speed=CLOCK_SPEED, run_dir=output, timeout=120, clock_type="SteppableClock")


def weights_digest(policy):
    digest = tensor_hash()
    for name, tensor in policy.state_dict().items():
        digest.update(name.encode("utf-8"))
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


@contextmanager
def frozen_env(cfg, path, *, crossing):
    env = _build_monitored_env(cfg)
    base = env.unwrapped
    vec = VecTransposeImage(DummyVecEnv([lambda: env]))
    try:
        model = load_model("SAC", str(path), vec, cfg=cfg)
        if model.policy.actor.inference_seq_len != 8 or model.num_timesteps != 500000:
            raise RuntimeError("Wrong frozen history length or checkpoint training step")
        if not (np.allclose(env.action_space.low, BOUNDS[0]) and np.allclose(env.action_space.high, BOUNDS[1])):
            raise RuntimeError("Wrong checkpoint action limits")
        model.policy.set_training_mode(False)
        model.policy.requires_grad_(False)
        identity = {
            "policy_sha256_before": weights_digest(model.policy),
            "model_updates_before": model._n_updates, "checkpoint_step": model.num_timesteps,
            "device": str(model.device), "history": model.policy.actor.inference_seq_len,
        }
        with torch.inference_mode():
            yield env, model, identity
        identity["policy_sha256_after"] = weights_digest(model.policy)
        identity["gradient_update_delta"] = model._n_updates - identity["model_updates_before"]
        if (identity["gradient_update_delta"] or model.num_timesteps != identity["checkpoint_step"]
                or identity["policy_sha256_before"] != identity["policy_sha256_after"]
                or any(p.grad is not None or p.requires_grad for p in model.policy.parameters())):
            raise RuntimeError("The evaluation changed the frozen policy")
    finally:
        vec.close()
        if not crossing:
            base.client.client.close()


def predict(model, obs):
    action, _ = model.predict({"image": VecTransposeImage.transpose_image(obs["image"]), "state": obs["state"]},
                              deterministic=True, use_history=True)
    if not np.isfinite(action).all() or not np.all(action >= np.asarray(BOUNDS[0]) - 1e-6) or not np.all(
        action <= np.asarray(BOUNDS[1]) + 1e-6
    ):
        raise RuntimeError(f"Invalid frozen action: {action}")
    return action


def heading(start, goal, offset_deg):
    return math.atan2(goal[1] - start[1], goal[0] - start[0]) + math.radians(offset_deg)


def reference_worker(path, port, output, definitions):
    output.mkdir()
    with launch(port, output), frozen_env(config(path, port), path, crossing=False) as (env, model, identity):
        base = env.unwrapped
        dynamics = base.dynamic_model
        scene = TransientBoxes(base.client)
        base.set_evaluation_scenarios([
            {"start_position": start, "goal_position": goal, "yaw_rad": heading(start, goal, yaw)}
            for _, start, goal, _, yaw, _, _ in definitions
        ])
        try:
            for label, start, goal, x, yaw, roof, _ in definitions:
                scene.replace((tunnel_roof(roof),), verify=True)
                dynamics.scene_step_callback = None
                obs, _ = env.reset(seed=SEED)
                obs, recaptures = ready_reset_observation(base, obs)
                model.policy.actor.reset_history()
                t0 = base.client.getMultirotorState(vehicle_name=dynamics.vehicle_name).timestamp
                # Match the subdivided control timing used in both formal arms.
                dynamics.scene_step_callback = lambda timestamp: None
                samples = [[0.0, *dynamics.get_position()]]
                direction = math.copysign(1.0, goal[0] - start[0])
                reference = None
                for _ in range(85):
                    obs, _, terminal, truncated, info = env.step(predict(model, obs))
                    current = [(info["state_timestamp"] - t0) / 1e9, *info["position"]]
                    prev = samples[-1]
                    samples.append(current)
                    if direction * (prev[1] - x) < 0 <= direction * (current[1] - x):
                        fraction = (x - prev[1]) / (current[1] - prev[1])
                        reference = [a + fraction * (b - a) for a, b in zip(prev, current, strict=True)]
                    if terminal or truncated or direction * (current[1] - x) >= 2.0:
                        break
                write_json(output.parent / f"{label}.json", {
                    "id": label, "start": start, "goal": goal, "cross_x": x, "yaw_offset_deg": yaw, "roof": roof,
                    "reference": reference, "end": samples[-1], "steps": len(samples) - 1,
                    "crash": bool(info["is_crash"]), "trace": samples, "reset_depth_recaptures": recaptures,
                })
                print(f"[reference] {label} {reference}", flush=True)
        finally:
            dynamics.scene_step_callback = None
            scene.clear()
    write_json(output / "frozen_policy.json", identity)


def source_identities(path):
    files = (
        path, path.parents[1] / "config/config.ini", TEMPLATE, IDENTITY, SURVEY,
        ROOT / "gym_env/envs/dynamics/multirotor_airsim.py",
        ROOT / "gym_env/envs/tasks/base_task.py",
        ROOT / "tools/ue_benchmark/ch4_stage3_crossing.py",
        ROOT / "tools/ue_benchmark/ch4_stage3_v2_course.py",
        ROOT / "tools/ue_benchmark/ch4_stage3_v2_training.py",
        ROOT / "tools/ue_benchmark/ch4_trainenv_crossing_audit.py",
        ROOT / "tools/ue_benchmark/verify_ch4_trainenv_crossing.py",
        ROOT / "tools/runtime_support.py",
        ROOT / "tools/trainenv_instances.py",
        Path(__file__).resolve(),
    )
    return {str(f): sha256(f) for f in files}


def lock_protocol(output, references, path):
    target = output / "protocol.json"
    if target.exists():
        raise FileExistsError(f"Protocol is already locked: {target}")
    native = surveyed_boxes()
    prepared = []
    for label, start, goal, x, yaw, roof, direction in REFERENCES:
        file = references / f"{label}.json"
        raw = read_json(file)
        if (raw["reference"] is None or raw["crash"] or raw["start"] != list(start)
                or raw["goal"] != list(goal) or raw["cross_x"] != x
                or raw["yaw_offset_deg"] != yaw or raw["roof"] != roof):
            raise RuntimeError(f"Missing valid prescoring reference for {label}")
        time_s, _, y, z = raw["reference"]
        prepared.append((label, start, goal, x, yaw, roof, direction, time_s, y, z, file))
    pairs = []
    for phase_index, (phase, offset) in enumerate(PHASES.items()):
        for i, (label, start, goal, x, yaw, roof, sign, time_s, y, z, file) in enumerate(prepared):
            center_time_s = round(max(HALF_TRAVEL_M / SPEED_MPS + .001, time_s + offset), 3)
            plan = CrossingPath(x, round(y, 2), -1., sign, center_time_s)
            section = assert_surveyed_section(plan, start, native)
            nominal_clearance = distance_to_box((x, y, z), plan.box_at(time_s, "moving"))
            modes = ["stationary", "moving"] if (i + phase_index) % 2 == 0 else ["moving", "stationary"]
            case = CrossingCase(start, goal, heading(start, goal, yaw), roof, plan, phase)
            pairs.append({
                "id": f"{phase}_{label}", "reference_id": label, "order": modes, "seed": SEED,
                "preset": asdict(case), "moving_start_s": plan.start_time_s, "moving_end_s": plan.end_time_s,
                "same_initial_pose_xyz_up_m": plan.center_at(0., "stationary"),
                "unobstructed_reference": {"file": str(file), "sha256": sha256(file), "t_s": time_s,
                                           "cross_x_m": x, "y_m": y, "z_m": z},
                "planned_obstacle_minus_reference_arrival_s": plan.center_time_s - time_s,
                "planned_reference_clearance_at_crossing_m": nominal_clearance,
                "expected_contact_band_at_reference_crossing": nominal_clearance <= NEAR_SURFACE_M,
                "layout_audit": section,
            })
    protocol = {
        "locked_at_utc": datetime.now(timezone.utc).isoformat(), "protocol_version": 2,
        "map": TRAINENV_MAP, "package": str(DEFAULT_EXE), "source_sha256": source_identities(path),
        "clock": {"type": "SteppableClock", "speed": CLOCK_SPEED, "time_zero": "post-reset first policy observation",
                  "control_dt_s": .1, "requested_physics_window_s": .05, "fine_windows_until": "planned traverse end",
                  "stalled_short_window_retry_s": .010,
                  "max_measured_update_gap_s": MAX_UPDATE_GAP_S, "unblocked_pose_lag_limit_m": MAX_UNBLOCKED_LAG_M},
        "camera_gate": {"ordinary_reset": "at most five captures while paused until lower rays see near geometry",
                        "scored_reset": "verify the actual input frame against the cube and fixed floor",
                        "flight": "treat a saturated nonterminal observation as infrastructure failure",
                        "maximum_concurrent_worlds": 2},
        "obstacle": {"asset": "Cube", "size_xyz_m": CROSSING_SIZE_M, "direction_axis": "Y", "speed_mps": SPEED_MPS,
                     "travel_m": 2 * HALF_TRAVEL_M, "motion": "one swept linear traverse then stop",
                     "stationary_mode": "hold the same initial pose"},
        "policy": {"checkpoint_sha256": CHECKPOINT_SHA256, "training_step": 500000,
                   "deterministic": True, "history": 8, "action_bounds": BOUNDS, "gradient_updates": 0,
                   "state_scaling": "unchanged 6-component checkpoint configuration",
                   "image_scaling": "DepthPlanar in meters, clip 25 m, resize 60x90, uint8 and existing policy scaling"},
        "termination": {"xy_goal_threshold_m_exclusive": 2., "altitude_threshold_m_exclusive": .5,
                        "max_steps": 1000, "terminate_on_collision": True, "terminate_on_workspace_exit": True,
                        "workspace_z_up_m": [-2.2, .2], "depth_only_collision_threshold_m": 0.},
        "metrics": {"within_contact_band_m": NEAR_SURFACE_M, "distance": "drone center to observed cube surface",
                    "arrival": "first forward X-plane crossing from the logged flight; missing if not reached",
                    "contact": "task's single AirSim collision RPC; choose named cube or other geometry"},
        "retry": {"max_attempts_per_pair": 2,
                  "rule": "Restart the same preset and arm order on a fresh owned simulator after an RPC, "
                          "reset/depth/clock validation, or execution failure, using a fresh port; "
                          "keep both arms of only one complete "
                          "attempt. All completed success, collision, workspace exit and timeout results count. "
                          "Preserve unscored attempts with their reason."},
        "pairs": pairs,
    }
    write_json(target, protocol)
    print(f"Locked {len(pairs)} pairs at {target} sha256={sha256(target)}", flush=True)


def decode_case(entry):
    data = dict(entry["preset"])
    data["path"] = CrossingPath(**data["path"])
    return CrossingCase(**data)


def reset_flight(env, case, mode, seed, output):
    base = env.unwrapped
    with capture_depth(base.client) as images:
        obs, reset = env.reset(seed=seed, options={
            "crossing_case": case, "crossing_arm": mode, "record_crossing_updates": True,
        })
    if not images or "stage3_crossing" not in reset:
        raise RuntimeError("Reset did not return a real DepthPlanar observation with the crossing actor")
    scene = base.stage3_course.crossing
    fov = base.client.simGetCameraInfo(base.perception.camera_id, base.dynamic_model.vehicle_name).fov
    depth = depth_geometry(images[-1], current_obstacle(scene.center), (*surveyed_boxes(), tunnel_roof(case.roof_clearance_m)),
                           fov, output / "reset_depth.npz")
    depth["recaptures"] = reset["stage3_crossing"]["reset_depth_recaptures"]
    initial = dict(base.dynamic_model.last_reset_state)
    body_up = np.asarray(initial["render_position_ned_m"]) * [1, 1, -1]
    # The packaged front-camera optical origin is 0.46 m ahead of the pawn.
    yaw = initial["attitude_rad"][2]
    optical = body_up + .46 * np.array([math.cos(yaw), math.sin(yaw), 0.])
    depth["optical_origin_error_m"] = math.dist(optical, depth["camera_xyz_up_m"])
    depth["obstacle_center_reset_error_m"] = math.dist(scene.center, case.path.center_at(0, mode))
    depth["passed"] = (depth["passed"] and depth["optical_origin_error_m"] <= .05
                       and depth["obstacle_center_reset_error_m"] <= .008)
    write_json(output / "reset.json", {"vehicle": initial, "first_depth": depth, "scene": reset["stage3_crossing"]})
    np.save(output / "first_policy_image.npy", obs["image"], allow_pickle=False)
    if not depth["passed"]:
        raise RuntimeError(f"Reset camera/obstacle geometry failed: {depth}")
    return obs, scene, depth, initial


def flight_sample(scene, position, step, *, action=None, info=None):
    sample = {
        "step": step, "t_s": scene.time_s, "drone_xyz_up_m": position,
        "box_xyz_up_m": scene.center, "center_to_box_m": distance_to_box(position, current_obstacle(scene.center)),
    }
    if info is not None:
        sample.update(action=action.tolist(), velocity_xyz_up_mps=info["velocity_world"],
                      simulation_timestamp_ns=info["state_timestamp"])
    return sample


def fly(env, case, mode, seed, output, decide):
    output.mkdir()
    base = env.unwrapped
    obs, scene, depth, initial = reset_flight(env, case, mode, seed, output)
    samples = [flight_sample(scene, base.dynamic_model.get_position(), 0)]
    total_reward = 0.0
    start_wall = time.perf_counter()
    with (output / "flight.jsonl").open("w", encoding="utf-8", buffering=1) as trace:
        json_line(trace, samples[0])
        while True:
            action = decide(obs, samples[-1])
            obs, reward, terminal, truncated, info = env.step(action)
            row = flight_sample(scene, info["position"], info["step_num"], action=action, info=info)
            if row["t_s"] <= samples[-1]["t_s"] or abs(
                row["t_s"] - (info["state_timestamp"] - scene.start_timestamp_ns) / 1e9
            ) > .002:
                raise RuntimeError("Flight and obstacle timestamps disagree")
            samples.append(row)
            total_reward += float(reward)
            json_line(trace, row)
            if terminal or truncated:
                break
            if not np.any(obs["image"] < 255):
                raise RuntimeError("The simulator produced a saturated nonterminal DepthPlanar frame")
    with (output / "obstacle_updates.jsonl").open("w", encoding="utf-8") as stream:
        for update in scene.updates:
            json_line(stream, update)
    hit = info["stage3_crossing"].get("collision")
    own_hit = hit is not None and hit["object_name"] == scene.actor
    clock = check_scene(scene, own_hit=own_hit)
    final = samples[-1]["drone_xyz_up_m"]
    gx, gy, gz = case.goal_position
    outcome = ("collision" if info["is_crash"] else "success" if info["is_success"]
               else "out_of_bounds" if info["is_not_in_workspace"] else "timeout")
    result = {
        "mode": mode, "phase": case.phase, "outcome": outcome, "steps": info["step_num"],
        "sim_start_ns": scene.start_timestamp_ns, "sim_duration_s": samples[-1]["t_s"],
        "wall_flight_s": time.perf_counter() - start_wall, "reward": total_reward,
        "initial_pose": initial, "first_depth": depth, "obstacle_name": scene.actor, "collision": hit,
        "trajectory_length_m": sum(math.dist(a["drone_xyz_up_m"], b["drone_xyz_up_m"])
                                   for a, b in zip(samples, samples[1:])),
        "terminal_xyz_up_m": final, "goal_horizontal_error_m": math.hypot(final[0] - gx, final[1] - gy),
        "goal_altitude_error_m": abs(final[2] + gz), "clock": clock,
        **episode_clearance(samples, scene, hit),
    }
    write_json(output / "result.json", result)
    print(f"[{case.phase}/{mode}] {outcome} steps={result['steps']} sim={result['sim_duration_s']:.2f}s "
          f"contact={result['contact']} gap={result['minimum_center_to_box_surface_m']:.2f}m", flush=True)
    return result


def score_pair(path, port, directory, entry, protocol_sha):
    directory.mkdir(exist_ok=True)
    accepted = directory / "accepted.json"
    if accepted.exists():
        record = read_json(accepted)
        if record["protocol_sha256"] != protocol_sha:
            raise RuntimeError(f"Existing results use a different protocol: {directory}")
        return record
    case = decode_case(entry)
    for index in (1, 2):
        attempt = directory / f"attempt_{index:02}"
        if attempt.exists():
            continue
        attempt.mkdir()
        try:
            arms = {}
            attempt_port = port + index - 1
            with launch(attempt_port, attempt), frozen_env(config(path, attempt_port, crossing=True), path, crossing=True) as (
                env, model, identity,
            ):
                for mode in entry["order"]:
                    model.policy.actor.reset_history()
                    arms[mode] = fly(env, case, mode, entry["seed"], attempt / mode,
                                     lambda obs, previous: predict(model, obs))
                a, b = (arms[mode]["initial_pose"] for mode in ("stationary", "moving"))
                pose_delta = math.dist(a["position_ned_m"], b["position_ned_m"])
                yaw_delta = abs(math.remainder(a["attitude_rad"][2] - b["attitude_rad"][2], math.tau))
                if pose_delta > .05 or yaw_delta > .01:
                    raise RuntimeError(f"Pair reset mismatch: {pose_delta:.3f} m or {yaw_delta:.3f} rad")
            result = {
                "id": entry["id"], "phase": case.phase, "protocol_sha256": protocol_sha, "arm_order": entry["order"],
                "attempt": index, "reset_position_difference_m": pose_delta, "reset_yaw_difference_rad": yaw_delta,
                "freeze": identity, "arms": arms,
            }
            write_json(accepted, result)
            print(f"[paired] {entry['id']} completed attempt={index}", flush=True)
            return result
        except Exception as error:
            write_json(attempt / "failure.json", {"error": str(error), "trace": traceback.format_exc()})
            print(f"[paired] {entry['id']} failed attempt={index}: {error}", flush=True)
    return None


def score_worker(path, port, output, pairs, protocol_sha):
    return [score_pair(path, port + 2 * index, output / entry["id"], entry, protocol_sha) for index, entry in pairs]


def summarize(output, protocol):
    rows = [read_json(output / "score" / pair["id"] / "accepted.json") for pair in protocol["pairs"]]
    cell = {}
    for phase in ("overall", *PHASES):
        chosen = [r for r in rows if phase == "overall" or r["phase"] == phase]
        cell[phase] = {"pairs": len(chosen), "paired_outcomes": dict(Counter(
            r["arms"]["stationary"]["outcome"] + " -> " + r["arms"]["moving"]["outcome"] for r in chosen
        ))}
        for mode in ("stationary", "moving"):
            arms = [r["arms"][mode] for r in chosen]
            cell[phase][mode] = {
                "outcomes": dict(Counter(r["outcome"] for r in arms)),
                "contact": dict(Counter(r["contact"] or "none" for r in arms)),
                "within_contact_band": sum(r["within_contact_band"] for r in arms),
                "x_plane_reached": sum(r["drone_at_plane"] is not None for r in arms),
                "min_center_to_surface_m": min(r["minimum_center_to_box_surface_m"] for r in arms),
            }
        cell[phase]["success_delta_moving_minus_stationary"] = (
            cell[phase]["moving"]["outcomes"].get("success", 0) - cell[phase]["stationary"]["outcomes"].get("success", 0)
        )
    write_json(output / "paired_summary.json", {"summary": cell, "pairs": rows})
    print(cell["overall"], flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("reference", "lock", "verify", "score", "summarize"))
    parser.add_argument("--output", type=Path, default=REPORT)
    parser.add_argument("--references", type=Path)
    parser.add_argument("--first-port", type=int, default=42340)
    parser.add_argument("--workers", type=int, choices=(1, 2), default=2)
    args = parser.parse_args(argv)
    output = args.output.resolve()
    output.mkdir(exist_ok=True, parents=True)
    path = checkpoint()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    if args.operation == "reference":
        workers = min(args.workers, len(REFERENCES))
        reference_root = (args.references or output / "reference").resolve()
        reference_root.mkdir()
        with ThreadPoolExecutor(max_workers=workers) as pool:
            jobs = [pool.submit(reference_worker, path, args.first_port + i, reference_root / f"worker_{i}",
                                REFERENCES[i::workers]) for i in range(workers)]
            for job in jobs:
                job.result()
        return
    if args.operation == "lock":
        lock_protocol(output, (args.references or output / "reference").resolve(), path)
        return
    protocol_path = output / "protocol.json"
    protocol = read_json(protocol_path)
    if protocol["source_sha256"] != source_identities(path):
        raise RuntimeError("A protocol-bound code, checkpoint, configuration or survey file changed")
    if args.operation == "verify":
        from tools.ue_benchmark.verify_ch4_trainenv_crossing import verify
        verify(config(path, args.first_port, crossing=True), args.first_port, output)
    elif args.operation == "score":
        if not read_json(output / "physical/verification.json")["passed"]:
            raise RuntimeError("The packaged cross-traffic physical verification has not passed")
        score = output / "score"
        score.mkdir(exist_ok=True)
        protocol_sha = sha256(protocol_path)
        pairs = list(enumerate(protocol["pairs"]))
        workers = min(args.workers, len(pairs))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            jobs = [pool.submit(score_worker, path, args.first_port, score, pairs[i::workers], protocol_sha)
                    for i in range(workers)]
            results = [record for job in jobs for record in job.result()]
        if sha256(path) != CHECKPOINT_SHA256 or sha256(protocol_path) != protocol_sha:
            raise RuntimeError("Checkpoint or locked protocol changed during the scoring runs")
        if any(r is None for r in results):
            raise RuntimeError(f"Only {sum(r is not None for r in results)}/{len(pairs)} complete pairs could be scored")
        summarize(output, protocol)
    else:
        summarize(output, protocol)


if __name__ == "__main__":
    main()
