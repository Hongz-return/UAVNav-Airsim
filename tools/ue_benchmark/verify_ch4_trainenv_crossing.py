"""Exercise the installed Cube, clock, collision and front DepthPlanar together."""

from copy import deepcopy
import math
import time

import numpy as np

from scripts.evaluation.evaluate_trainenv_crossing import SEED, fly, launch, reset_flight
from tools.runtime_support import _build_monitored_env
from tools.ue_benchmark.ch4_stage3_crossing import CROSSING_KEY, CrossingCase, CrossingPath
from tools.ue_benchmark.ch4_stage3_v2_course import tunnel_roof
from tools.ue_benchmark.ch4_trainenv_3d_course import write_json
from tools.ue_benchmark.ch4_trainenv_3d_runtime import json_line
from tools.ue_benchmark.ch4_trainenv_crossing_audit import (
    assert_surveyed_section, capture_depth, check_scene, current_obstacle, depth_geometry, surveyed_boxes,
)


def optical_run(env, case, output):
    output.mkdir()
    base = env.unwrapped
    _, scene, first_depth, _ = reset_flight(env, case, "moving", SEED, output)
    start_pose = scene.read_center()
    time.sleep(.2)
    paused_stamp = base.client.getMultirotorState(vehicle_name=base.dynamic_model.vehicle_name).timestamp
    if paused_stamp != scene.start_timestamp_ns or scene.read_center() != start_pose:
        raise RuntimeError("Clock or box moved while the simulator was paused")
    timeline = [{"name": "reset", "t_s": 0.0, **first_depth}]
    target_times = iter((case.path.start_time_s + .65, case.path.center_time_s, case.path.end_time_s + .15))
    target = next(target_times, None)
    config = (*surveyed_boxes(), tunnel_roof(case.roof_clearance_m))
    fov = base.client.simGetCameraInfo(base.perception.camera_id, base.dynamic_model.vehicle_name).fov
    while target is not None:
        with capture_depth(base.client) as images:
            _, _, terminal, truncated, _ = env.step(np.array([.50, 0.0, 0.0], dtype=np.float32))
        if terminal or truncated or len(images) != 1:
            raise RuntimeError("The optical witness failed before observing a full crossing")
        if scene.time_s >= target:
            name = f"motion_{len(timeline)}"
            optical = depth_geometry(images[0], current_obstacle(scene.center), config, fov, output / f"{name}.npz")
            timeline.append({"name": name, "t_s": scene.time_s, **optical})
            target = next(target_times, None)
    with (output / "obstacle_updates.jsonl").open("w", encoding="utf-8") as stream:
        for sample in scene.updates:
            json_line(stream, sample)
    deltas = [b["xyz_up_m"][1] - a["xyz_up_m"][1] for a, b in zip(scene.updates, scene.updates[1:])]
    result = {
        "samples": timeline, "paused_s": .2, "clock_changed_while_paused": False,
        "readback_travel_m": math.dist(start_pose, scene.read_center()),
        "maximum_absolute_pose_increment_m": max(abs(d) for d in deltas),
        "negative_progress_samples": sum(d * case.path.direction < -1e-5 for d in deltas),
        "clock": check_scene(scene),
    }
    result["passed"] = (
        all(frame["passed"] for frame in timeline)
        and abs(timeline[-1]["cube_median_col"] - timeline[0]["cube_median_col"]) >= 75
        and result["negative_progress_samples"] == 0 and abs(result["readback_travel_m"] - 6.) < .01
    )
    write_json(output / "capture_summary.json", result)
    return result


def bypass_controller(env):
    dynamics = env.unwrapped.dynamic_model
    xs = (0., 4.6, 10.5, 14.)
    ys = (0., -4.9, -4.9, 0.)

    def decide(obs, previous):
        x, y, z = previous["drone_xyz_up_m"]
        tx = min(x + 1.65, xs[-1])
        ty = float(np.interp(tx, xs, ys))
        target_yaw = math.atan2(ty - y, max(.18, tx - x))
        yaw_rate = 2.8 * math.remainder(target_yaw - dynamics.get_attitude()[2], math.tau)
        vertical = 2.0 * (-1.0 - z)
        forward = min(2.1, .75 + .45 * max(0.0, xs[-1] - x))
        return np.clip([forward, vertical, yaw_rate], env.action_space.low, env.action_space.high).astype(np.float32)

    return decide


def training_route_smoke(env):
    """Reset the selected training mode on a real roofed route and then a 3-D gate."""
    base = env.unwrapped
    course = base.stage3_course
    observations = {}
    for family in ("original_roofed", "over"):
        course.forced_family = family
        obs, reset = env.reset(seed=SEED)
        _, _, terminated, truncated, step = env.step(np.array([1., 0., 0.], dtype=np.float32))
        observations[family] = {
            "start": course.scenario["start_position"], "goal": course.scenario["goal_position"],
            "box_keys": [b.key for b in course.scenario["boxes"]], "crossing_reset": reset.get("stage3_crossing"),
            "crossing_after_step": step.get("stage3_crossing"),
            "first_image_shape": list(obs["image"].shape), "terminal_after_first_step": terminated or truncated,
        }
        present = base.client.simListSceneObjects("Ch4Stage3V2_.*_cross_traffic.*")
        if family == "original_roofed" and (len(present) != 1 or "stage3_crossing" not in step):
            raise RuntimeError("Ordinary training did not select the cross-traffic course")
        if family == "over" and (present or "stage3_crossing" in step or not course.scenario["gates"]):
            raise RuntimeError("The 3-D corridor retained a cross-traffic actor from the previous route")
    course.forced_family = None
    return observations


def verify(cfg, port, report):
    output = report / "physical"
    output.mkdir(exist_ok=False)
    audit = {"passed": False}
    case = CrossingCase((0., 0., 1.), (14., 0., 1.), 0.0, 2.05, CrossingPath(8., 0., -1., 1, 3.55), "near")
    audit["witness"] = {
        "start_ned_m": case.start_position, "goal_ned_m": case.goal_position, "path_x_m": case.path.x,
        "center_time_s": case.path.center_time_s, "direction": case.path.direction,
        **assert_surveyed_section(case.path, case.start_position, surveyed_boxes()),
    }
    try:
        with launch(port, output):
            env = _build_monitored_env(cfg, training=True)
            try:
                audit["optical"] = optical_run(env, case, output / "optical")
                direct = lambda obs, previous: np.array([2.65, 0., 0.], dtype=np.float32)
                baseline = fly(env, case, "stationary", SEED, output / "stationary_direct", direct)
                contact = fly(env, case, "moving", SEED, output / "moving_direct", direct)
                bypass = fly(env, case, "moving", SEED, output / "moving_bypass", bypass_controller(env))
                audit.update(stationary_direct=baseline, moving_direct=contact, moving_bypass=bypass)
                audit["training_smoke"] = {"moving": training_route_smoke(env)}
            finally:
                env.close()
            stationary_cfg = deepcopy(cfg)
            stationary_cfg["stage3_v2"]["crossing_mode"] = "stationary"
            stationary_env = _build_monitored_env(stationary_cfg, training=True)
            try:
                audit["training_smoke"]["stationary"] = training_route_smoke(stationary_env)
                client = stationary_env.unwrapped.client
                audit["transient_count_before_close"] = len(client.simListSceneObjects("Ch4Stage3V2_.*"))
            finally:
                stationary_env.close()
            audit["passed"] = (
                audit["optical"]["passed"] and baseline["outcome"] == "success"
                and contact["outcome"] == "collision" and contact["contact"] == "crossing_added_box"
                and bypass["outcome"] == "success" and bypass["minimum_center_to_box_surface_m"] > .55
                and all(modes["original_roofed"]["crossing_reset"]["mode"] == mode
                        and CROSSING_KEY in modes["original_roofed"]["box_keys"]
                        for mode, modes in audit["training_smoke"].items())
            )
    except BaseException as exc:
        audit["error"] = str(exc)
        raise
    finally:
        write_json(output / "verification.json", audit)
    if not audit["passed"]:
        raise RuntimeError("Packaged TrainEnv did not meet the optical, contact and bypass gate")
    print("[physical] moving depth, named collision, bypass and both ordinary training modes passed", flush=True)
