"""Train MA-SAC through the three UE5.9 NavRL forest densities."""

import argparse
from configparser import ConfigParser
from datetime import datetime
from functools import partial
import json
import math
from pathlib import Path
import runpy

from stable_baselines3.common.callbacks import BaseCallback

from scripts.training.train_single import parse_args as parse_training_args, run_training
from tools.airsim_instances import load_settings
from tools.experiment_paths import TRAIN_RUNS_ROOT
from tools.navrl_forest_evaluation import evaluate_checkpoint
from tools.navrl_forest_runtime import CAMERA_FORWARD_M, VEHICLE_NAME, forest_config
from tools.trainenv_instances import launch_trainenv


DEFAULT_PROJECT = Path("D:/UEProjects/Ch4Tunnel_UE59/Ch4Tunnel.uproject")
DEFAULT_EDITOR = Path("D:/software/Epic Games/UE/UE_5.9/Engine/Binaries/Win64/UnrealEditor-Cmd.exe")
DEFAULT_CONFIG = Path("configs/ch4_train_trainenv.ini")
UNREAL_ARGS = (
    "-dx11", "-RenderOffscreen", "-nop4", "-NoTrace",
    "-ini:EditorPerProjectUserSettings:[/Script/ModelContextProtocol.ModelContextProtocolSettings]:bAutoStartServer=False",
)


def write_json(path, data):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def load_protocol(project, scene_version=3):
    source = project.parent / "Tools" / "NavRL" / "navrl_curriculum.py"
    rules = runpy.run_path(str(source))
    if scene_version != 1:
        stage_key = f"V{scene_version}_STAGES"
        if stage_key not in rules:
            raise ValueError(f"The Unreal project does not define the V{scene_version} forest maps in {source}.")
        rules["STAGES"] = rules[stage_key]
        rules["evaluation_scenarios"] = partial(rules["evaluation_scenarios"], scene_version=scene_version)
    for stage in rules["STAGES"]:
        relative = stage.map_path.removeprefix("/Game/")
        if relative == stage.map_path:
            raise ValueError(f"Expected a project map, got {stage.map_path}.")
        asset = project.parent / "Content" / (relative + ".umap")
        if not asset.is_file():
            raise FileNotFoundError(asset)
    return rules


def launch_stage(args, stage, ports, directory, settings):
    return launch_trainenv(
        executable=args.ue_exe, editor_project=args.ue_project, template=settings,
        ports=ports, start_position=[0.0, 0.0, 0.0], clock_speed=args.clock_speed,
        clock_type="SteppableClock", run_dir=directory, timeout=args.startup_timeout,
        map_path=stage.map_path, unreal_args=UNREAL_ARGS,
    )


class ForestPromotion(BaseCallback):
    """Evaluate the current weights independently and stop learning on a pass."""

    def __init__(self, stage, rules, stage_dir, budget, interval, evaluate, *, continuous_training=False):
        super().__init__()
        self.stage = stage
        self.rules = rules
        self.stage_dir = stage_dir
        self.budget = budget
        self.interval = interval
        self.evaluate = evaluate
        self.continuous_training = continuous_training
        self.rounds = []
        self.decision = {"decision": "stay"}
        self.promotion_checkpoint = None

    def _on_training_start(self):
        self.initial_steps = self.model.num_timesteps
        self.next_evaluation = min(self.interval, self.budget)

    def _on_step(self):
        elapsed = self.model.num_timesteps - self.initial_steps
        if elapsed < self.next_evaluation:
            return True
        round_id = len(self.rounds) + 1
        round_dir = self.stage_dir / "evaluations" / f"round_{round_id:03d}"
        round_dir.mkdir(parents=True)
        checkpoint = round_dir / "checkpoint.zip"
        pending = round_dir / "checkpoint_pending.zip"
        self.model.save(str(pending.with_suffix("")))
        pending.replace(checkpoint)
        cases = self.rules["evaluation_scenarios"](self.stage.number, round_id)
        write_json(round_dir / "scenario_manifest.json", cases)
        print(f"[NavRL] Stage {self.stage.number} evaluation {round_id} at +{elapsed} transitions", flush=True)
        if self.continuous_training:
            self.training_env.env_method("pause_for_evaluation", True)
        try:
            outcomes = self.evaluate(checkpoint, round_dir, cases["scenarios"])
        finally:
            if self.continuous_training:
                self.training_env.env_method("pause_for_evaluation", False)
        expected_ids = [case["scenario_id"] for case in cases["scenarios"]]
        if ([row["scenario_id"] for row in outcomes] != expected_ids
                or any(type(row["collision_free_success"]) is not bool for row in outcomes)):
            raise RuntimeError("The stage evaluation did not finish the prescribed scenario sequence.")
        record = {
            "stage": self.stage.number,
            "round_id": round_id,
            "completed_episodes": len(outcomes),
            "collision_free_successes": sum(row["collision_free_success"] for row in outcomes),
            "num_timesteps": self.model.num_timesteps,
            "checkpoint": str(checkpoint),
        }
        if "crossing_axis" in cases["scenarios"][0]:
            if any(type(row.get("crossed_core")) is not bool for row in outcomes):
                raise RuntimeError("Forest evaluation outcomes must include actual core crossings.")
            crossings = sum(row["crossed_core"] for row in outcomes)
            record.update(
                core_crossing_episodes=crossings,
                core_crossing_rate=crossings / len(outcomes),
                mean_core_path_fraction=math.fsum(row["core_path_fraction"] for row in outcomes) / len(outcomes),
            )
        decision = self.rules["decide"](self.stage.number, [*self.rounds, record])
        write_json(round_dir / "result.json", {**record, **decision, "outcomes": outcomes})
        self.rounds.append(record)
        self.decision = decision
        print(f"[NavRL] Stage {self.stage.number}: {record['collision_free_successes']}/"
              f"{record['completed_episodes']} collision-free arrivals, {decision['decision']}", flush=True)
        if "core_crossing_episodes" in record:
            print(f"[NavRL] Stage {self.stage.number}: {record['core_crossing_episodes']}/"
                  f"{record['completed_episodes']} crossed the core, "
                  f"mean core path fraction {record['mean_core_path_fraction']:.1%}", flush=True)
        if decision["decision"] in ("advance", "completed"):
            self.promotion_checkpoint = checkpoint
            return False
        self.next_evaluation = min(elapsed + self.interval, self.budget)
        return True


def run_course(args):
    base = ConfigParser()
    with args.config.open(encoding="utf-8-sig") as handle:
        base.read_file(handle)
    rules = load_protocol(args.ue_project, args.scene_version)
    stages = rules["STAGES"]
    route_portals = rules["V2_ROUTE_PORTALS"] if args.scene_version >= 2 else None
    stepped_training = args.step_mode == "stepped"
    training_ports = list(range(args.first_port, args.first_port + args.instances))
    evaluation_port = training_ports[-1] + 1
    template = load_settings(args.ue_project.parent / "Config" / "AirSim.settings.json")
    vehicles = template.get("Vehicles", {})
    if list(vehicles) != [VEHICLE_NAME]:
        raise ValueError("The UE5.9 AirSim template must define one SimpleFlight vehicle.")
    camera = vehicles[VEHICLE_NAME].setdefault("Cameras", {}).setdefault("0", {})
    camera["X"] = CAMERA_FORWARD_M
    template["MoveWorldOrigin"] = False
    # Validate the learning contract before creating a run or starting simulators.
    forest_config(base, stages[0], route_portals=route_portals)
    root = args.run_dir
    root.mkdir(parents=True, exist_ok=False)
    settings = root / "airsim_template.json"
    write_json(settings, template)
    write_json(root / "curriculum.json", {
        "stages": [stage._asdict() for stage in stages],
        "scene_version": args.scene_version,
        "clock_speed": args.clock_speed,
        "training_step_mode": args.step_mode,
        "evaluation_step_mode": "stepped",
        "episodes_per_round": rules["EPISODES_PER_ROUND"],
        "success_rate_strictly_above": rules["SUCCESS_RATE_THRESHOLD"],
        "consecutive_passes": rules["CONSECUTIVE_PASSES"],
        "seed": args.seed, "stage_steps": args.stage_steps, "eval_every": args.eval_every,
    })
    resume = args.resume
    print(f"[NavRL] Curriculum run: {root}", flush=True)
    for stage in stages:
        stage_dir = root / f"stage_{stage.number}"
        stage_dir.mkdir()
        cfg = forest_config(base, stage, route_portals=route_portals, synchronous_steps=stepped_training)
        cfg.read_dict({"DRL": {
            "fast_history": args.fast_history,
            "overlap_updates": args.overlap_updates and args.instances > 1,
        }})
        cfg.set("navrl_forest", "goal_dir", str(stage_dir).replace("%", "%%"))
        evaluation_cfg = forest_config(base, stage, route_portals=route_portals)
        evaluation_cfg.remove_option("navrl_forest", "goal_dir")
        evaluation_cfg.set("airsim", "rpc_port", str(evaluation_port))

        def evaluate(checkpoint, directory, cases):
            with launch_stage(args, stage, [evaluation_port], directory, settings):
                return evaluate_checkpoint(checkpoint, evaluation_cfg, cases, seed=args.seed)

        callback = ForestPromotion(stage, rules, stage_dir, args.stage_steps, args.eval_every, evaluate,
                                   continuous_training=not stepped_training)
        training_args = parse_training_args([
            "--config", str(args.config), "--timesteps", str(args.stage_steps),
            "--seed", str(args.seed), "--checkpoint-every", str(args.checkpoint_every),
            "--progress-every", str(args.progress_every),
            "--airsim-ports", *map(str, training_ports),
            *(["--resume", str(resume)] if resume is not None else []),
        ])
        print(f"[NavRL] Stage {stage.number} {stage.static_count}+{stage.dynamic_count}: {stage.map_path}", flush=True)
        print(f"[NavRL] {args.instances} training instance(s), {args.clock_speed:g}x, {args.step_mode}, "
              f"fast_history={args.fast_history}, overlap_updates={cfg.getboolean('DRL', 'overlap_updates')}; "
              "independent evaluation uses stepped control", flush=True)
        with launch_stage(args, stage, training_ports, stage_dir, settings) as job:
            run_training(training_args, cfg, run_dir=stage_dir, simulator_job=job, extra_callbacks=[callback])

        state = {
            "stage": stage.number,
            **callback.decision,
            "checkpoint": str(callback.promotion_checkpoint or stage_dir / "models" / "model_sb3.zip"),
        }
        write_json(stage_dir / "decision.json", state)
        write_json(root / "status.json", state)
        if callback.decision["decision"] != "advance":
            status = ("completed" if callback.decision["decision"] == "completed"
                      else "stage transition budget exhausted")
            print(f"[NavRL] {status}. Checkpoint: {state['checkpoint']}", flush=True)
            return state
        resume = callback.promotion_checkpoint


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-c", "--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--ue-project", type=Path, default=DEFAULT_PROJECT)
    parser.add_argument("--ue-exe", type=Path, default=DEFAULT_EDITOR)
    parser.add_argument("--scene-version", type=int, choices=(1, 2, 3), default=3,
                        help="Forest map version (default: 3 with obstacles throughout the arena)")
    parser.add_argument("--clock-speed", type=float, choices=(1.0, 2.0), default=2.0,
                        help="AirSim clock multiplier for training and evaluation (default: 2)")
    parser.add_argument("--step-mode", choices=("realtime", "stepped"), default="realtime",
                        help="Run the training worlds continuously or pause after every action")
    parser.add_argument("--fast-history", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--overlap-updates", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--resume", type=Path, help="Optional initial MA-SAC checkpoint for stage 1")
    parser.add_argument("--instances", type=int, default=4, help="Training instances, plus one separate evaluator")
    parser.add_argument("--first-port", type=int, default=41811)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--stage-steps", type=int, default=300_000, help="Maximum new transitions per stage")
    parser.add_argument("--eval-every", type=int, default=50_000, help="New transitions between evaluation rounds")
    parser.add_argument("--checkpoint-every", type=int, default=10_000)
    parser.add_argument("--progress-every", type=int, default=400)
    parser.add_argument("--startup-timeout", type=float, default=120.0)
    args = parser.parse_args(argv)
    if not 1 <= args.instances < 65_535 or not 1 <= args.first_port <= 65_535 - args.instances:
        parser.error("Training ports plus one evaluation port must be valid.")
    if (args.eval_every < args.instances or args.stage_steps < 2 * args.eval_every
            or args.stage_steps % args.instances or args.eval_every % args.instances):
        parser.error("The stage needs at least two evaluation intervals, and both budgets must divide by instances.")
    if (args.checkpoint_every < 1 or args.progress_every < 0
            or not math.isfinite(args.startup_timeout) or args.startup_timeout <= 0):
        parser.error("Checkpoint, progress and timeout settings are invalid.")
    for name in ("config", "ue_project", "ue_exe", "resume"):
        path = getattr(args, name)
        if path is not None:
            if not path.is_file():
                parser.error(f"Missing {name.replace('_', ' ')} file: {path}")
            setattr(args, name, path.resolve())
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    args.run_dir = (args.run_dir or TRAIN_RUNS_ROOT / "navrl_curriculum" / f"{stamp}_seed{args.seed}").resolve()
    return args


if __name__ == "__main__":
    run_course(parse_args())
