"""Collect offline AirSim demonstrations and train one gated Stage 3 slice."""

import argparse
import ast
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from datetime import datetime
from itertools import accumulate, islice
import math
from pathlib import Path
from threading import Event
import time
from zipfile import ZipFile

import numpy as np
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage
import torch

from gym_env.replay.sequence_buffer import EpisodeStorage
from tools.runtime_support import _build_monitored_env, build_parallel_runtime_env, load_model
from tools.trainenv_instances import DEFAULT_EXE, HOST, launch_trainenv
from tools.training_logging import TrainingProgressCallback
from tools.ue_benchmark.ch4_stage3_training import Stage3CourseMetrics
from tools.ue_benchmark.ch4_stage3_v2_course import CORRIDOR_FAMILIES, SPAWN_NED, TIGHT_CORRIDOR_FAMILIES, TUNNEL_Z
from tools.ue_benchmark.ch4_stage3_v2_training import ROOFED_FAMILIES, geometry_demo_action, terminal_landing_action
from tools.ue_benchmark.ch4_trainenv_3d_course import ROOT, read_json, sha256, write_json
from tools.ue_benchmark.ch4_trainenv_3d_runtime import assert_observation_contract, json_line, read_ini


REPORT = ROOT / "reports/ch4_stage3_v2_20261125"
SMOKE = REPORT / "smoke_corridor/verified/smoke_summary.json"
TEMPLATE = ROOT / "configs/ch4_stage3_trainenv_airsim.json"
OVERRIDES = ROOT / "configs/ch4_stage3_v2_overrides.ini"
SOURCE = ROOT / ("Logs/TrainData/Runs/trainenv/stage2_alt_ceil_20261125_seed0_cs2_4env_1000000_restart_022838/"
                 "run/models/best_model.zip")
SOURCE_SHA256 = "ca88e0e6e3dbe419b3592ad9239e4c9c27228eff5e5aeb34f0367001e9e09b8d"
SOURCE_CONFIG_SHA256 = "e6b0b59f04ccfea48b383a526def8239ce3cd1239f4138f510c9fc01a078fdd0"
SOURCE_STEP = 495000
PORTS = (42072, 42073)


def load_config(overrides=OVERRIDES):
    if not read_json(SMOKE)["passed"] or read_json(REPORT / "process_cleanup_smoke_corridor_final.json")["remaining"]:
        raise RuntimeError("The packaged collision and depth gate must pass before Stage 3")
    source_config = SOURCE.parent.parent / "config/config.ini"
    if sha256(SOURCE) != SOURCE_SHA256 or sha256(source_config) != SOURCE_CONFIG_SHA256:
        raise RuntimeError("Stage 2 best@495k or its input configuration changed")
    cfg = read_ini(source_config)
    cfg.read(overrides, encoding="utf-8")
    assert_observation_contract(cfg)
    if cfg.getint("mamba", "seq_len") != 16:
        raise RuntimeError("Stage 3 must retain the 16-step Mamba training sequence")
    if (tuple(ast.literal_eval(cfg.get("environment", "work_space_z"))) != TUNNEL_Z
            or cfg.getfloat("reward", "altitude_deviation_weight") != .35):
        raise RuntimeError("Stage 3 altered the Stage 2 roofed-flight envelope")
    if cfg.getboolean("airsim", "synchronous_steps", fallback=False):
        raise RuntimeError("The high-speed package uses asynchronous control")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the Stage 3 experiment")
    return cfg


def runway(output):
    return launch_trainenv(executable=DEFAULT_EXE, template=TEMPLATE, ports=PORTS, start_position=SPAWN_NED,
                           clock_speed=2.0, run_dir=output, timeout=120)


def policy_observation(obs):
    return {"image": VecTransposeImage.transpose_image(obs["image"]).copy(), "state": obs["state"].copy()}


def demonstration(env, family, seed, frozen, *, tight_roofed_landing=False):
    raw_env = env.unwrapped
    course = raw_env.stage3_course
    course.forced_family = family
    obs, _ = env.reset(seed=seed)
    if frozen is not None:
        frozen.policy.actor.reset_history()
    episode = EpisodeStorage()
    trajectory = [list(raw_env.dynamic_model.get_position())]
    episode_return = 0.0
    gx, gy, _ = course.scenario["goal_position"]
    with torch.inference_mode():
        terminated = truncated = False
        while not (terminated or truncated):
            current = policy_observation(obs)
            pos = trajectory[-1]
            yaw = raw_env.dynamic_model.get_attitude()[2]
            if frozen is not None:
                if tight_roofed_landing and math.hypot(gx - pos[0], gy - pos[1]) < 5.5:
                    action = terminal_landing_action(course.scenario["goal_position"], pos, yaw)
                else:
                    action, _ = frozen.predict(current, deterministic=True, use_history=True)
            else:
                action = geometry_demo_action(course.scenario, pos, yaw)
            obs, reward, terminated, truncated, info = env.step(action)
            action = info["actual_action"]
            episode.add(current, action.copy(), float(reward), bool(terminated))
            episode_return += float(reward)
            trajectory.append(info["position"])
    metrics = info["stage3_episode"]
    accepted = (metrics["success"] and not (metrics["collision"] or metrics["out_of_bounds"] or truncated)
                and metrics["route_complete"])
    return episode, trajectory, {"seed": seed, "accepted": accepted, "return": episode_return, **metrics}


def collect_group(cfg, port, output, seed, *, roofed, stop, supplement=False, init_checkpoint=SOURCE):
    tight_goal = cfg.getboolean("stage3_v2", "tight_goal", fallback=False)
    group = "roofed" if roofed else "corridor"
    corridor_fams = tuple(TIGHT_CORRIDOR_FAMILIES) if tight_goal else tuple(CORRIDOR_FAMILIES)
    families = ROOFED_FAMILIES if roofed else corridor_fams
    recovery_targets = {"over": 120, "under": 435, "over_under": 120, "under_over": 120}
    log = output / f"{group}_episodes.jsonl"
    accepted_dir = output / group
    accepted_dir.mkdir()
    cfg = deepcopy(cfg)
    cfg["airsim"]["rpc_port"] = str(port)
    env = _build_monitored_env(cfg, training=True)
    raw_env = env.unwrapped
    raw_env.stage3_course.recovery_demo = supplement and not roofed
    model_env = None
    episodes, steps = [], Counter()

    def enough():
        if supplement:
            if roofed:
                return sum(steps.values()) >= 875 and all(steps[name] > 0 for name in families)
            return all(steps[name] >= recovery_targets[name] for name in families)
        if roofed:
            return sum(steps.values()) >= 4600 and all(steps[name] >= 1500 for name in families)
        per_family = 900 if tight_goal else 1100
        return all(steps[name] >= per_family for name in families)

    try:
        frozen = None
        if roofed:
            # SB3 supplies the policy's channel-first spaces. Collection steps the
            # raw Gym env so the collector, not an autoreset, chooses the next family.
            model_env = VecTransposeImage(DummyVecEnv([lambda: env]))
            frozen = load_model("SAC", str(init_checkpoint), model_env, cfg=cfg)
            if frozen.policy.actor.inference_seq_len != 8:
                raise RuntimeError("Wrong frozen demonstration policy history length")
            frozen.policy.set_training_mode(False)
            frozen.policy.requires_grad_(False)
        max_attempts = 52 if (tight_goal and not roofed) else (48 if roofed else 36)
        with log.open("w", encoding="utf-8") as file:
            for i in range(max_attempts):
                if enough() or stop.is_set():
                    break
                family = min(families, key=lambda key: steps[key] / recovery_targets[key] if supplement and not roofed
                             else steps[key])
                episode, trajectory, metrics = demonstration(
                    env, family, seed + i, frozen, tight_roofed_landing=tight_goal and roofed,
                )
                json_line(file, metrics)
                file.flush()
                if metrics["accepted"]:
                    episodes.append(episode.get_state())
                    steps[family] += episode.length
                    stem = accepted_dir / f"{i:02d}_{family}"
                    np.save(stem.with_suffix(".npy"), np.asarray(trajectory, dtype=np.float32))
                    torch.save(episode.get_state(), stem.with_suffix(".pt"))
                print(f"[demo {group}] {family} success={metrics['success']} steps={episode.length} "
                      f"gates={metrics['gates_passed']}/{metrics['gates_total']} accepted_total={sum(steps.values())}",
                      flush=True)
        if not enough() and not stop.is_set():
            raise RuntimeError(f"Could not collect the required successful {group} episodes: {dict(steps)}")
        return episodes, dict(steps)
    finally:
        try:
            raw_env.stage3_course.close()
        finally:
            (model_env or env).close()
            raw_env.client.client.close()


def collect(cfg, output, seed, base_warmup=None, init_checkpoint=SOURCE):
    init_checkpoint = Path(init_checkpoint).resolve(strict=True)
    init_sha = sha256(init_checkpoint)
    output.mkdir(parents=True, exist_ok=False)
    stop = Event()
    t0 = time.monotonic()
    with runway(output) as job, ThreadPoolExecutor(max_workers=2) as pool:
        futures = {
            pool.submit(collect_group, cfg, port, output, seed + 10000 * i, roofed=(i == 0), stop=stop,
                        supplement=base_warmup is not None, init_checkpoint=init_checkpoint): i
            for i, port in enumerate(PORTS)
        }
        gathered = {}
        try:
            for future in as_completed(futures):
                index = futures[future]
                gathered[index] = future.result()
                job.stop_simulator(PORTS[index])
        finally:
            stop.set()
    roofed, roofed_steps = gathered[0]
    corridors, corridor_steps = gathered[1]
    additions = {"roofed": sum(roofed_steps.values()), "corridor": sum(corridor_steps.values())}
    base_metadata = {}
    if base_warmup is not None:
        base, base_state = read_warmup(base_warmup, expected_source_sha256=init_sha)
        roof_limit = demonstration_boundary(base_state["episodes"], base["roofed_transitions"])
        roofed = base_state["episodes"][:roof_limit] + roofed
        corridors = base_state["episodes"][roof_limit:] + corridors
        for family, count in base["steps_by_family"].items():
            counts = roofed_steps if family in ROOFED_FAMILIES else corridor_steps
            counts[family] = counts.get(family, 0) + count
        base_metadata = {"base_replay_sha256": base["replay_sha256"], "added_transitions": additions}
    a, b = sum(roofed_steps.values()), sum(corridor_steps.values())
    if not 8000 <= a + b <= 14000 or not .38 <= a / (a + b) <= .62:
        raise RuntimeError(f"Need 8,000–14,000 balanced demonstration transitions, collected roofed={a} corridor={b}")
    replay = output / "warmup_replay.pt"
    torch.save({"episodes": roofed + corridors, "total_transitions": a + b}, replay)
    write_json(output / "warmup_summary.json", {
        "passed": True, "source_checkpoint": str(init_checkpoint), "source_sha256": init_sha,
        "gradient_updates": 0, "replay_sha256": sha256(replay),
        "transitions": a + b, "steps_by_family": {**roofed_steps, **corridor_steps},
        "roofed_transitions": a, "corridor_transitions": b,
        "complete_successful_episodes": len(roofed) + len(corridors), "wall_s": time.monotonic() - t0,
        **base_metadata,
    })
    print(f"[demo] accepted {a + b} transitions, {a} roofed and {b} corridor", flush=True)


def read_warmup(directory, expected_source_sha256=None):
    summary = read_json(directory / "warmup_summary.json")
    replay = directory / "warmup_replay.pt"
    if (not summary["passed"]
            or (expected_source_sha256 is not None and summary["source_sha256"] != expected_source_sha256)
            or sha256(replay) != summary["replay_sha256"]):
        raise RuntimeError("Offline replay does not match its completed warmup audit")
    state = torch.load(replay, map_location="cpu", weights_only=False)
    if summary["transitions"] != state["total_transitions"] or sum(e["length"] for e in state["episodes"]) != summary["transitions"]:
        raise RuntimeError("Offline replay episode lengths disagree with its audit")
    if not all(ep["terminateds"][-1] for ep in state["episodes"]):
        raise RuntimeError("Offline replay contains an incomplete demonstration")
    return summary, state


def demonstration_boundary(episodes, roofed_steps):
    prefix = list(accumulate(episode["length"] for episode in episodes))
    return prefix.index(roofed_steps) + 1


def replace_demonstrations(model, previous, current):
    old = read_json(previous / "warmup_summary.json")
    summary, bank = read_warmup(current)
    if summary.get("base_replay_sha256") != old["replay_sha256"]:
        raise RuntimeError("The replacement warmup is not derived from this run's original demonstrations")
    replay = model.replay_buffer
    online = replay.get_state()["episodes"]
    base_episodes = old["complete_successful_episodes"]
    if replay._min_valid_ep_idx != 0 or sum(ep["length"] for ep in online[:base_episodes]) != old["transitions"]:
        raise RuntimeError("The original demonstration prefix cannot be identified in the resumed replay")
    episodes = bank["episodes"] + online[base_episodes:]
    total = sum(episode["length"] for episode in episodes)
    if total > replay.buffer_size:
        raise RuntimeError("Supplementary demonstrations exceed the available replay capacity")
    replay.set_state({"episodes": episodes, "total_transitions": total})


def install_warmup(model, directory, alpha):
    _, state = read_warmup(directory)
    model.replay_buffer.set_state(state)
    if model.log_ent_coef is None or model.ent_coef_optimizer is None:
        raise RuntimeError("Stage 3 needs adaptive SAC entropy")
    with torch.no_grad():
        model.log_ent_coef.fill_(math.log(alpha))
    model.ent_coef_optimizer.state.clear()


def configure_action_anchor(model, directory, weights):
    if not any(weights):
        return
    summary = read_json(directory / "warmup_summary.json")
    total = summary["complete_successful_episodes"]
    replay = model.replay_buffer
    prefix_steps = list(accumulate(ep.length for ep in islice(replay.episodes, total)))
    if (replay._min_valid_ep_idx != 0 or len(prefix_steps) != total
            or prefix_steps[-1] != summary["transitions"]):
        raise RuntimeError("The original successful episodes must still lead the replay buffer")
    roofed_end = prefix_steps.index(summary["roofed_transitions"]) + 1
    model.demo_action_anchor = (roofed_end, total, *weights)


def save_checkpoint(model, path, *, full):
    staged = path.with_name(path.stem + "_pending")
    model.save(str(staged), include_buffer=full)
    archive = staged.with_suffix(".zip")
    with ZipFile(archive) as zipped:
        bad_file = zipped.testzip()
    if bad_file is not None:
        raise RuntimeError(f"Checkpoint CRC check failed for {bad_file}")
    archive.replace(path)


def train(cfg, run, warmup, resume, added_steps, seed, anchor_weights, terminal_weight, demo_fraction, *,
          init_checkpoint=SOURCE, precision_weight=0.0):
    run.mkdir(parents=True, exist_ok=False)
    models = run / "models"
    snapshots = run / "config"
    models.mkdir()
    snapshots.mkdir()
    init_checkpoint = Path(init_checkpoint).resolve(strict=True)
    source = init_checkpoint if resume is None else resume.resolve(strict=True)
    with (snapshots / "config.ini").open("w", encoding="utf-8") as file:
        cfg.write(file)
    t0 = time.monotonic()
    with runway(run) as job:
        env = build_parallel_runtime_env(cfg, list(PORTS), base_seed=seed, training=True)
        try:
            for worker in env.venv.processes:
                job.add(worker)
            model = load_model("SAC", str(source), env, cfg=cfg, tensorboard_log=str(run / "tb_logs"),
                               load_replay=resume is not None, seed=seed)
            inherited_alpha = model.ent_coef
            if resume is None:
                install_warmup(model, warmup, cfg.getfloat("stage3_v2", "initial_alpha"))
            elif model.replay_buffer.size() < 8000:
                raise RuntimeError("Continuation needs the complete replay state from the previous slice")
            if resume is not None:
                previous_warmup = Path(read_json(source.parent.parent / "run.json")["warmup"])
                if previous_warmup != warmup:
                    replace_demonstrations(model, previous_warmup, warmup)
            configure_action_anchor(model, warmup, anchor_weights)
            model.demo_terminal_altitude_weight = terminal_weight
            model.demo_terminal_precision_weight = precision_weight
            model.demo_replay_fraction = demo_fraction
            first = model.num_timesteps
            initial = {"step": first, "updates": model._n_updates, "alpha": model.ent_coef,
                       "replay_transitions": model.replay_buffer.size()}
            write_json(run / "run.json", {
                "started_at": datetime.now().isoformat(timespec="seconds"), "slice_steps": added_steps, "seed": seed,
                "source": str(source), "source_sha256": sha256(source), "initial": initial,
                "inherited_alpha": inherited_alpha, "warmup": str(warmup), "ports": PORTS,
                "demo_action_anchor": model.demo_action_anchor,
                "demo_terminal_altitude_weight": terminal_weight,
                "demo_terminal_precision_weight": precision_weight,
                "demo_replay_fraction": demo_fraction,
            })
            print(f"[Stage 3 v2] step={first}, replay={initial['replay_transitions']}, "
                  f"alpha={initial['alpha']:.5f} on {model.device}", flush=True)
            model.learn(added_steps, reset_num_timesteps=False, tb_log_name="stage3_v2", log_interval=1,
                        callback=[Stage3CourseMetrics(run / "course_episodes.jsonl"), TrainingProgressCallback(1000)])
            if model.num_timesteps != first + added_steps or model._n_updates <= initial["updates"]:
                raise RuntimeError("Training slice did not complete its full transition and update budget")
            candidate = models / f"checkpoint_{model.num_timesteps}_steps.zip"
            save_checkpoint(model, candidate, full=False)
            complete = models / "model_sb3.zip"
            save_checkpoint(model, complete, full=True)
            write_json(run / "training_summary.json", {
                "finished_at": datetime.now().isoformat(timespec="seconds"), "initial": initial,
                "final": {"step": model.num_timesteps, "updates": model._n_updates, "alpha": model.ent_coef,
                          "replay_transitions": model.replay_buffer.size()},
                "added_steps": added_steps, "candidate": str(candidate), "candidate_sha256": sha256(candidate),
                "full_checkpoint": str(complete), "wall_s": time.monotonic() - t0,
            })
            print(f"[Stage 3 v2] saved {candidate}", flush=True)
        finally:
            env.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--overrides", type=Path, default=OVERRIDES)
    parser.add_argument("--init-checkpoint", type=Path, default=SOURCE)
    sub = parser.add_subparsers(dest="command", required=True)
    collect_parser = sub.add_parser("collect")
    collect_parser.add_argument("--output", type=Path, default=REPORT / "warmup")
    collect_parser.add_argument("--seed", type=int, default=7126)
    collect_parser.add_argument("--supplement", type=Path, help="Prior warmup to extend with real recovery flights")
    train_parser = sub.add_parser("train")
    train_parser.add_argument("--run-dir", type=Path, required=True)
    train_parser.add_argument("--warmup", type=Path, default=REPORT / "warmup")
    train_parser.add_argument("--resume", type=Path)
    train_parser.add_argument("--timesteps", type=int, default=25000)
    train_parser.add_argument("--seed", type=int, default=0)
    train_parser.add_argument("--anchor-roofed", type=float, default=0.0)
    train_parser.add_argument("--anchor-corridor", type=float, default=0.0)
    train_parser.add_argument("--anchor-terminal-altitude", type=float, default=0.0)
    train_parser.add_argument("--anchor-terminal-precision", type=float, default=0.0)
    train_parser.add_argument("--demo-replay-fraction", type=float, default=0.0)
    train_parser.add_argument("--learning-rate", type=float)
    args = parser.parse_args(argv)
    torch.set_num_threads(1)
    cfg = load_config(args.overrides.resolve())
    if args.command == "collect":
        collect(cfg, args.output.resolve(), args.seed,
                args.supplement.resolve() if args.supplement else None,
                init_checkpoint=args.init_checkpoint.resolve())
    else:
        if args.timesteps not in (20000, 25000, 30000):
            parser.error("Training slices must contain 20,000, 25,000, or 30,000 additional transitions")
        weights = (args.anchor_roofed, args.anchor_corridor)
        if any(not math.isfinite(w) or w < 0 for w in (*weights, args.anchor_terminal_altitude, args.anchor_terminal_precision)):
            parser.error("Demonstration action weights must be finite and nonnegative")
        if args.anchor_terminal_altitude and not args.anchor_corridor:
            parser.error("Terminal altitude imitation requires corridor demonstrations")
        if args.anchor_terminal_precision and not any(weights):
            parser.error("Terminal precision imitation requires demonstration action anchors")
        if not 0 <= args.demo_replay_fraction <= 1 or (args.demo_replay_fraction and not any(weights)):
            parser.error("A demo replay fraction requires action anchors and must be between zero and one")
        if args.learning_rate is not None:
            if not 3e-5 <= args.learning_rate <= 1e-4:
                parser.error("Keep the fine-tuning learning rate between 3e-5 and 1e-4")
            cfg["DRL"]["learning_rate"] = str(args.learning_rate)
        train(cfg, args.run_dir.resolve(), args.warmup.resolve(), args.resume, args.timesteps, args.seed,
              weights, args.anchor_terminal_altitude, args.demo_replay_fraction,
              init_checkpoint=args.init_checkpoint.resolve(),
              precision_weight=args.anchor_terminal_precision)


if __name__ == "__main__":
    main()
