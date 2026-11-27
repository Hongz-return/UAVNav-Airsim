"""Launch local UE 5.4 TrainEnv instances and train one depth MA-SAC policy."""
import argparse
import ast
from configparser import ConfigParser
from datetime import datetime
import math
from pathlib import Path

from scripts.training.train_single import parse_args as parse_training_args, run_training
from tools.experiment_config import assert_observation_contract, read_ini
from tools.experiment_paths import TRAIN_RUNS_ROOT
from tools.trainenv_instances import (
    DEFAULT_3D_EXE, DEFAULT_EDITOR, DEFAULT_EXE, DEFAULT_TEMPLATE, HOST,
    TRAINENV_3D_MAP, TRAINENV_MAP, launch_trainenv,
)
from tools.ue_benchmark.ch4_stage3_crossing import CROSSING_MODES, crossing_mode, enable_crossing_clock
from tools.ue_benchmark.ch4_stage3_v2_course import SPAWN_NED


TRAINENV_DYNAMIC_MAP = "/Game/UAVTraining/Maps/TrainEnv_Dynamic25"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-c", "--config", type=Path, default=Path("configs/ch4_train_trainenv.ini"))
    parser.add_argument("--ue-exe", type=Path, help="Override the executable for the selected TrainEnv map")
    parser.add_argument("--ue-project", type=Path, help="Launch the selected map from a standalone source project")
    parser.add_argument("--settings-template", type=Path, help="Override the selected map's AirSim template")
    parser.add_argument("--instances", type=int, help="Default 4 for legacy runs and 2 for moving obstacles")
    parser.add_argument("--first-port", type=int, default=41771)
    parser.add_argument("--clock-speed", type=float, help="Default 2 for legacy runs and 1 for moving obstacles")
    parser.add_argument("--crossing-mode", choices=CROSSING_MODES,
                        help="Opt into a stationary or moving box on the Stage 3 roofed routes")
    parser.add_argument("--startup-timeout", type=float, default=90.0)
    parser.add_argument("--timesteps", type=int, help="Override the total number of environment transitions")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--run-dir", type=Path, help="New directory for this run")
    parser.add_argument("--resume", type=Path, help="Load a checkpoint into this new run directory")
    parser.add_argument("--restore-replay", action="store_true",
                        help="Restore the replay buffer from a full checkpoint when resuming")
    parser.add_argument("--checkpoint-every", type=int, default=10000)
    parser.add_argument("--progress-every", type=int, default=400)
    parser.add_argument("--fast-history", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--overlap-updates", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--cache-step-state", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args(argv)

    cfg = ConfigParser()
    if not cfg.read(args.config, encoding="utf-8"):
        parser.error(f"cannot read {args.config}")
    map_path = cfg.get("trainenv", "map_path", fallback=TRAINENV_MAP)
    if args.crossing_mode is not None:
        if args.crossing_mode != "off" and not cfg.has_section("stage3_v2"):
            parser.error("Cross traffic requires the Stage 3 v2 TrainEnv configuration")
        if cfg.has_section("stage3_v2"):
            cfg.set("stage3_v2", "crossing_mode", args.crossing_mode)
    try:
        crossing = crossing_mode(cfg) != "off"
        enable_crossing_clock(cfg)
    except ValueError as exc:
        parser.error(str(exc))
    moving_obstacles = crossing or map_path == TRAINENV_DYNAMIC_MAP
    instances = args.instances if args.instances is not None else (2 if moving_obstacles else 4)
    if instances < 1 or not (1 <= args.first_port <= 65536 - instances):
        parser.error("instances and first port must define a valid non-empty port range")
    clock_speed = args.clock_speed if args.clock_speed is not None else (1.0 if moving_obstacles else 2.0)
    if not (math.isfinite(clock_speed) and clock_speed > 0
            and math.isfinite(args.startup_timeout) and args.startup_timeout > 0):
        parser.error("clock speed and startup timeout must be finite and positive")
    if (cfg.get("options", "perception") != "depth_state"
            or cfg.get("options", "algo") != "SAC"
            or not cfg.getboolean("options", "use_mamba")):
        parser.error("TrainEnv requires depth_state MA-SAC")
    if not crossing and cfg.getboolean("airsim", "synchronous_steps", fallback=False):
        parser.error("TrainEnv launch uses real-time AirSim steps. Use train_single for an existing paused simulator.")
    total_timesteps = args.timesteps if args.timesteps is not None else cfg.getint("options", "total_timesteps")
    if total_timesteps < 1 or total_timesteps % instances:
        parser.error("transition budget must be positive and divisible by the number of instances")
    cfg.read_dict({
        "airsim": {
            "ip": HOST,
            "depth_only": True,
            "cache_step_state": args.cache_step_state,
            "timeout_value": cfg.getfloat("airsim", "timeout_value", fallback=15.0),
        },
        "DRL": {
            "fast_history": args.fast_history,
            "overlap_updates": args.overlap_updates and instances > 1,
        },
    })
    ports = list(range(args.first_port, args.first_port + instances))
    start = ast.literal_eval(cfg.get("environment", "start_position"))
    if map_path not in (TRAINENV_MAP, TRAINENV_3D_MAP, TRAINENV_DYNAMIC_MAP):
        parser.error(f"unknown TrainEnv map: {map_path}")
    if crossing and map_path != TRAINENV_MAP:
        parser.error("Cross traffic requires the original roofed TrainEnv map")
    package = DEFAULT_3D_EXE if map_path == TRAINENV_3D_MAP else DEFAULT_EXE
    project = args.ue_project
    if project is None and args.ue_exe is None:
        configured_project = cfg.get("trainenv", "editor_project", fallback=None)
        if configured_project:
            project = args.config.parent / configured_project
    if map_path == TRAINENV_DYNAMIC_MAP and project is None and args.ue_exe is None:
        parser.error("TrainEnv_Dynamic25 requires its source project or a packaged executable via --ue-exe")
    executable = args.ue_exe or (DEFAULT_EDITOR if project else package)
    resume = args.resume
    if resume is not None:
        if resume.suffix != ".zip":
            resume = Path(str(resume) + ".zip")
        resume = resume.resolve(strict=True)
    if cfg.has_section("stage3"):
        if map_path != TRAINENV_MAP or resume is None:
            parser.error("Stage 3 requires the multi-obstacle TrainEnv map and --resume")
        assert_observation_contract(cfg)
        source = read_ini(resume.parent.parent / "config/config.ini")
        assert_observation_contract(source)
        for key in ("dt", "action_repeat", "v_xy_min", "v_xy_max", "v_z_max", "yaw_rate_max_deg"):
            if cfg.getfloat("multirotor", key) != source.getfloat("multirotor", key):
                parser.error(f"Stage 3 changes the checkpoint action contract at multirotor.{key}")
        course_path = (args.config.parent / cfg["stage3"]["course_file"]).resolve(strict=True)
        from tools.ue_benchmark.ch4_stage3_training import load_multiobstacle_course

        load_multiobstacle_course(course_path)
        cfg["stage3"]["course_file"] = str(course_path)
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("Stage 3 GPU training needs an available CUDA device")
    template_path = cfg.get("trainenv", "settings_template", fallback=None)
    template = args.settings_template or (args.config.parent / template_path if template_path else DEFAULT_TEMPLATE)
    # The new course's local NED origin stays at the PlayerStart, independent of the training route.
    spawn_ned = ast.literal_eval(cfg.get("trainenv", "spawn_position_ned", fallback=repr(start)))
    if crossing and tuple(spawn_ned) != SPAWN_NED:
        parser.error("Cross traffic requires TrainEnv's surveyed [0, 0, 1] NED pawn origin")
    training_args = parse_training_args([
        "--config", str(args.config.resolve()),
        "--timesteps", str(total_timesteps), "--seed", str(args.seed),
        "--checkpoint-every", str(args.checkpoint_every), "--progress-every", str(args.progress_every),
        "--airsim-ports", *map(str, ports),
        *(["--resume", str(resume)] if resume is not None else []),
        *(["--restore-replay"] if args.restore_replay else []),
    ])
    stamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%f")
    run_dir = (args.run_dir or TRAIN_RUNS_ROOT / "trainenv" / f"{stamp}_seed{args.seed}").resolve()
    run_dir.mkdir(parents=True, exist_ok=False)
    print(f"[TrainEnv] Run directory: {run_dir}", flush=True)
    with launch_trainenv(
        executable=executable.resolve(), template=template.resolve(),
        ports=ports, start_position=spawn_ned, clock_speed=clock_speed,
        run_dir=run_dir, timeout=args.startup_timeout, map_path=map_path,
        editor_project=project.resolve() if project else None,
        clock_type="SteppableClock" if crossing else None,
    ) as job:
        run_training(training_args, cfg, run_dir=run_dir, simulator_job=job)
    print("[TrainEnv] Owned UE processes exited", flush=True)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        raise SystemExit(130)
