"""只读几何加载：障碍物与训练目标点。

仅使用只读接口 ``simListSceneObjects / simGetObjectPose / simGetObjectScale``。

障碍物来源优先级：

1. **实时场景对象**：匹配 ``Ch4Stage3V2_<gen>_<key>``（Stage3 V2 顶棚/侧墙/端墙/横梁，``simSpawnObject("Cube")``，
   scale 即盒子米制尺寸）与 ``Ch4_3D_*``；增量扫描（只对新出现的对象读取 pose / scale，已消失的对象移除）。
2. **运行目录尾部回合记录**：``course_episodes.jsonl`` 末行的 ``temporary_obstacles``（上一完成回合的盒子，近似）。

NavRL 目标来自当前训练端口和关卡的 ``live_goal_<port>.json``。旧隧道继续使用规划或回合记录。

注意坐标：对象位姿使用 PlayerStart 原点，车辆 / 相机位姿使用初始车辆原点（其下方 1 m），转换见 ``coords.object_ned_to_local_enu``。
"""

from __future__ import annotations

import json
import logging
import math
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from .coords import (
    object_ned_to_local_enu,
    ned_to_enu_point,
    ned_to_enu_quat,
    quat_wxyz,
    vec3,
)

LOG = logging.getLogger("uav_rviz.geometry")

DEFAULT_SPAWN_NED_Z = 1.0
DEFAULT_FLOOR_Z = -2.6  # = ch4_stage3_v2_course.FLOOR_Z（车辆局部 z-up）
DEFAULT_NAME_REGEX = r"^(Ch4Stage3V2_.*|Ch4_3D_.*|StaticMeshActor_.*)$"
GOAL_RADII_M = (2.0, 1.6)
GOAL_ALTITUDE_TOLERANCE_M = 0.5
NAVRL_GOAL_RADIUS_M = 0.5  # NavRL forest_config 的 accept_radius。

_TAIL_BYTES = 256 * 1024
_STAGE_DIR_NAME = re.compile(r"stage_([1-9]\d*)\Z")
_NAVRL_MAP_PREFIX = "/Game/Training/Maps/L_Train_NavRL_Forest"


@dataclass(frozen=True)
class SceneBox:
    key: str
    name: str
    center: tuple[float, float, float]  # 车辆局部 ENU / Z-up
    size: tuple[float, float, float]  # 世界轴向尺寸 (sx, sy, sz)
    quat: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)  # ENU (w, x, y, z)
    source: str = "live"

    @property
    def kind(self) -> str:
        return classify_box(self.key, self.size)

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "name": self.name,
            "kind": self.kind,
            "c": [round(v, 4) for v in self.center],
            "s": [round(v, 4) for v in self.size],
            "q": [round(v, 6) for v in self.quat],
            "source": self.source,
        }


def classify_box(key: str, size: Optional[tuple[float, float, float]] = None) -> str:
    lowered = key.lower()
    if "roof" in lowered or "ceiling" in lowered:
        return "roof"
    if "floor" in lowered or "ground" in lowered:
        return "floor"
    if "wall" in lowered or "beam" in lowered or "pillar" in lowered:
        return "wall"
    if size is not None and max(size[0], size[1]) >= 50.0:
        return "wall"
    return "block"


def is_arena_floor_slab(name: str, size: tuple[float, float, float]) -> bool:
    """过滤地图自带的整片大地面板（如 180×120×0.3m），避免遮挡工程网格。"""
    if name.startswith("StaticMeshActor_"):
        sx, sy, sz = (abs(v) for v in size)
        return max(sx, sy) >= 90.0 and min(sx, sy) >= 90.0 and sz <= 1.0
    return False


def box_key_from_name(name: str) -> str:
    """``Ch4Stage3V2_3_corridor_roof`` → ``corridor_roof``；``Ch4_3D_Over_FloorBlock`` → ``Over_FloorBlock``。"""
    if name.startswith("Ch4Stage3V2_"):
        parts = name.split("_", 2)
        return parts[2] if len(parts) == 3 else name
    if name.startswith("Ch4_3D_"):
        return name[len("Ch4_3D_"):]
    return name


@dataclass
class GoalMarker:
    position: tuple[float, float, float]  # 车辆局部 ENU z-up
    radii: tuple[float, float] = GOAL_RADII_M
    altitude_tolerance: float = GOAL_ALTITUDE_TOLERANCE_M
    source: str = "course_episodes"
    family: Optional[str] = None
    episode_hint: Optional[int] = None
    stage: Optional[int] = None

    def to_dict(self) -> dict[str, Any]:
        result = {
            "p": [round(v, 4) for v in self.position],
            "radii": list(self.radii),
            "alt_tol": self.altitude_tolerance,
            "source": self.source,
            "family": self.family,
        }
        if self.stage is not None:
            result.update(stage=self.stage, episode=self.episode_hint)
        return result


def resolve_run_dir(spec: Optional[str], root: Optional[Path] = None) -> Optional[Path]:
    """``auto`` → 取 ``Logs/TrainData/Runs`` 下最近更新的 course_episodes.jsonl 所在目录。"""
    if not spec:
        return None
    if spec != "auto":
        return Path(spec)
    base = (root or Path(__file__).resolve().parents[2]) / "Logs" / "TrainData" / "Runs"
    candidates = list(base.rglob("course_episodes.jsonl")) if base.exists() else []
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime).parent


def read_last_json_line(path: Path, tail_bytes: int = _TAIL_BYTES) -> Optional[dict[str, Any]]:
    """读取 jsonl 末尾最后一条完整 JSON 记录（只读、只取尾部，避免大文件全量读取）。"""
    try:
        size = path.stat().st_size
        with path.open("rb") as handle:
            handle.seek(max(0, size - tail_bytes))
            chunk = handle.read()
    except OSError:
        return None
    for raw in reversed(chunk.splitlines()):
        raw = raw.strip()
        if not raw:
            continue
        try:
            record = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            continue  # 可能是被截断的首行或正在写入的末行
        if isinstance(record, dict):
            return record
    return None


def _read_json_object(path: Path) -> Optional[dict[str, Any]]:
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return record if isinstance(record, dict) else None


def _stage_number(path: Path) -> Optional[int]:
    match = _STAGE_DIR_NAME.fullmatch(path.name)
    return int(match[1]) if match else None


def _goal_from_navrl_snapshot(path: Path, stage: int, rpc_port: int) -> Optional[GoalMarker]:
    record = _read_json_object(path)
    if record is None:
        return None
    if (type(record.get("stage")) is not int or record["stage"] != stage
            or type(record.get("rpc_port")) is not int or record["rpc_port"] != rpc_port):
        return None
    episode = record.get("episode")
    published_at = record.get("published_at")
    goal_ned = record.get("goal_position")
    if type(episode) is not int or episode < 1:
        return None
    if type(published_at) not in (int, float) or not 0 < published_at < math.inf:
        return None
    if not isinstance(goal_ned, list) or len(goal_ned) != 3 or any(type(v) not in (int, float) for v in goal_ned):
        return None
    try:
        position = ned_to_enu_point(goal_ned)
    except OverflowError:
        return None
    if not all(map(math.isfinite, position)):
        return None
    return GoalMarker(
        position=position,
        radii=(NAVRL_GOAL_RADIUS_M, NAVRL_GOAL_RADIUS_M),
        altitude_tolerance=NAVRL_GOAL_RADIUS_M,
        source="navrl_live",
        episode_hint=episode,
        stage=stage,
    )


def _evaluation_goal(launch_dir: Path, stage: int, port: Optional[int] = None) -> Optional[GoalMarker]:
    round_dir = launch_dir.parent if launch_dir.name.startswith("attempt_") else launch_dir
    if (round_dir / "result.json").exists():
        return None
    manifest = _read_json_object(round_dir / "scenario_manifest.json") or {}
    cases = manifest.get("scenarios")
    if not isinstance(cases, list) or not cases:
        return None
    if any(not isinstance(case, dict) or not isinstance(case.get("scenario_id"), str) for case in cases):
        return None
    active_path = round_dir / "active_scenario.json"
    port_path = round_dir / f"active_scenario_{port}.json"
    if port_path.exists():
        active_path = port_path
    radius = NAVRL_GOAL_RADIUS_M
    if active_path.exists():
        active = _read_json_object(active_path) or {}
        current = active.get("scenario_id")
        if not isinstance(current, str):
            return None
        configured = active.get("accept_radius_m")
        if type(configured) in (int, float) and math.isfinite(configured) and configured > 0:
            radius = configured
        index = next((n for n, case in enumerate(cases) if case.get("scenario_id") == current), len(cases))
    else:
        # The running continuation evaluates the ordered manifest and commits each
        # completed case before resetting to the next, including across RPC retries.
        try:
            done = json.loads((round_dir / "completed_cases.json").read_bytes())
        except FileNotFoundError:
            if launch_dir.name != "attempt_01":
                return None
            done = []
        except (ValueError, OSError):
            return None
        if not isinstance(done, list) or len(done) > len(cases):
            return None
        if any(not isinstance(row, dict) or row.get("scenario_id") != case.get("scenario_id")
               for row, case in zip(done, cases)):
            return None
        index = len(done)
    if index >= len(cases):
        return None
    goal = cases[index].get("goal_position")
    if not isinstance(goal, list) or len(goal) != 3 or not all(type(v) in (float, int) and math.isfinite(v) for v in goal):
        return None
    return GoalMarker(ned_to_enu_point(goal), (radius,) * 2, radius,
                      source="navrl_evaluation", episode_hint=index + 1, stage=stage)


def boxes_from_episode_record(record: dict[str, Any], floor_z: float = DEFAULT_FLOOR_Z) -> list[SceneBox]:
    """``temporary_obstacles``（NED y、局部 z-up 区间）→ ENU 盒子。"""
    boxes: list[SceneBox] = []
    for i, item in enumerate(record.get("temporary_obstacles") or []):
        try:
            (x0, x1), (y0, y1), (z0, z1) = item["x"], item["y"], item["z"]
        except (KeyError, TypeError, ValueError):
            continue
        key = str(item.get("key", f"box_{i}"))
        boxes.append(
            SceneBox(
                key=key,
                name=f"episode:{key}",
                center=((x0 + x1) / 2.0, -(y0 + y1) / 2.0, (z0 + z1) / 2.0),
                size=(abs(x1 - x0), abs(y1 - y0), abs(z1 - z0)),
                source="course_episodes",
            )
        )
    return boxes


def goal_from_episode_record(record: dict[str, Any], floor_z: float = DEFAULT_FLOOR_Z) -> Optional[GoalMarker]:
    goal_agl = record.get("goal_xyz_agl_m")
    goal_ned = record.get("goal_position")
    try:
        if goal_agl is not None:
            gx, gy, height = map(float, goal_agl)
            position = (gx, -gy, height + floor_z)
        elif goal_ned is not None:
            position = ned_to_enu_point(map(float, goal_ned))
        else:
            return None
    except (TypeError, ValueError):
        return None
    if not all(map(math.isfinite, position)):
        return None
    return GoalMarker(
        position=position,
        family=record.get("family"),
        episode_hint=record.get("training_updates"),
    )


class GeometryLoader:
    """只读几何加载器。``client`` 必须是 ``ReadOnlyClient``（或任何只暴露只读方法的对象）。"""

    def __init__(
        self,
        client: Any = None,
        *,
        spawn_ned_z: float = DEFAULT_SPAWN_NED_Z,
        floor_z: float = DEFAULT_FLOOR_Z,
        name_regex: str = DEFAULT_NAME_REGEX,
        run_dir: Optional[Path] = None,
        navrl_runs_root: Optional[Path] = None,
        infer_goals: bool = True,
    ) -> None:
        self.client = client
        self.spawn_ned_z = spawn_ned_z
        self.floor_z = floor_z
        self.name_regex = name_regex
        self.run_dir = Path(run_dir) if run_dir else None
        self.navrl_runs_root = Path(navrl_runs_root) if navrl_runs_root else None
        self.infer_goals = infer_goals
        self.is_navrl_run = self.navrl_runs_root is not None
        self.navrl_launch_id: Optional[str] = None
        self._navrl_stages: dict[tuple[Path, int], int] = {}
        self._followed_navrl_runs: dict[int, tuple[int, Path]] = {}
        self._live: dict[str, SceneBox] = {}
        self._ignored: set[str] = set()
        self._episode_stat: Optional[tuple[int, int]] = None
        self._episode_boxes: list[SceneBox] = []
        self.goal: Optional[GoalMarker] = None
        self.live_goal: Optional[GoalMarker] = None
        self.manual_goal: Optional[GoalMarker] = None
        self.last_scan_error: Optional[str] = None
        self.last_scan_time = 0.0
        self.scan_count = 0

    def attach(self, client: Any) -> None:
        self.client = client
        self._live.clear()
        self._ignored.clear()
        self.live_goal = None
        if self.is_navrl_run:
            self.goal = None

    # -- 实时场景对象 ---------------------------------------------------
    def scan_live(self) -> bool:
        """增量扫描实时对象；返回场景是否发生变化。失败时保留旧缓存并记录 ``last_scan_error``。"""
        if self.client is None:
            return False
        changed = False
        try:
            names = [str(n) for n in self.client.simListSceneObjects(self.name_regex)]
            wanted = set(names)
            for gone in [n for n in self._live if n not in wanted]:
                del self._live[gone]
                changed = True
            self._ignored.intersection_update(wanted)
            for name in names:
                if name in self._live or name in self._ignored:
                    continue
                pose = self.client.simGetObjectPose(name)
                scale = self.client.simGetObjectScale(name)
                pos = vec3(pose.position)
                raw_size = vec3(scale)
                if not all(map(_finite, (*pos, *raw_size))):
                    continue  # 对象刚创建尚未就绪（NaN pose）
                size = (abs(raw_size[0]), abs(raw_size[1]), abs(raw_size[2]))
                if is_arena_floor_slab(name, size):
                    self._ignored.add(name)
                    continue
                quat = ned_to_enu_quat(quat_wxyz(pose.orientation))
                self._live[name] = SceneBox(
                    key=box_key_from_name(name),
                    name=name,
                    center=object_ned_to_local_enu(pos, self.spawn_ned_z),
                    size=size,
                    quat=quat,
                    source="live",
                )
                changed = True
            self.last_scan_error = None
        except Exception as exc:  # 只读探测：任何失败都不影响主采集循环
            self.last_scan_error = f"{type(exc).__name__}: {exc}"
            LOG.warning("场景扫描失败: %s", self.last_scan_error)
        self.last_scan_time = time.perf_counter()
        self.scan_count += 1
        return changed

    def force_rescan(self) -> bool:
        self._live.clear()
        self._ignored.clear()
        self.scan_live()
        return True

    # -- 训练运行目录 --------------------------------------------------
    def _find_navrl_stage(self, run_dir: Path, rpc_port: Optional[int]) -> Optional[tuple[int, Path, Optional[str]]]:
        explicit_stage = _stage_number(run_dir)
        root = run_dir
        if explicit_stage is not None and not (root / "curriculum.json").is_file():
            root = run_dir.parent
        if (root / "curriculum.json").is_file():
            self.is_navrl_run = True
            candidates = [(stage, path) for path in root.glob("stage_*") if (stage := _stage_number(path)) is not None]
        elif explicit_stage is not None:
            candidates = [(explicit_stage, run_dir)]
        else:
            return None

        # Keep a port's stage monotonic even if a newer manifest is briefly unreadable.
        minimum_stage = self._navrl_stages.get((root, rpc_port), 0) if rpc_port is not None else 0
        for stage, path in sorted(candidates, reverse=True):
            if stage < minimum_stage:
                break
            manifest = _read_json_object(path / "launch.json") or {}
            map_path = manifest.get("map")
            if isinstance(map_path, str) and map_path.startswith(_NAVRL_MAP_PREFIX):
                self.is_navrl_run = True
            if rpc_port is None or not self.is_navrl_run:
                continue
            def matching_launches():
                yield path, manifest
                for launch in sorted((path / "evaluations").glob("round_*/**/launch.json"), reverse=True):
                    yield launch.parent, _read_json_object(launch) or {}

            for launch_dir, launch_record in matching_launches():
                instances = launch_record.get("instances")
                if not isinstance(instances, list):
                    continue
                instance = next((i for i in instances if isinstance(i, dict) and i.get("port") == rpc_port), None)
                if instance is not None:
                    settings = instance.get("settings")
                    launch_id = settings.get("TrainEnvLaunchId") if isinstance(settings, dict) else None
                    self._navrl_stages[root, rpc_port] = stage
                    return stage, launch_dir, launch_id if isinstance(launch_id, str) and launch_id else None
        return None

    def _find_latest_navrl_stage(self, runs_root: Path, rpc_port: Optional[int]) -> Optional[tuple[int, Path, Optional[str]]]:
        if rpc_port is None:
            return None
        candidates = []
        for marker in runs_root.glob("*/curriculum.json"):
            try:
                candidates.append((marker.stat().st_mtime_ns, marker.parent))
            except OSError:
                continue
        floor = self._followed_navrl_runs.get(rpc_port)
        for candidate in sorted(candidates, reverse=True):
            if floor is not None and candidate < floor:
                break
            stage = self._find_navrl_stage(candidate[1], rpc_port)
            if stage is not None:
                self._followed_navrl_runs[rpc_port] = candidate
                return stage
        return None

    def poll_run_dir(self, *, rpc_port: Optional[int] = None, launch_id: Optional[str] = None) -> bool:
        """检查 NavRL 训练目标、旧隧道的完成回合记录或计划文件。"""
        was_navrl = self.is_navrl_run
        if self.navrl_runs_root is not None:
            active_stage = self._find_latest_navrl_stage(self.navrl_runs_root, rpc_port)
        elif self.run_dir is not None:
            active_stage = self._find_navrl_stage(self.run_dir, rpc_port)
        else:
            return False
        if self.is_navrl_run:
            new_goal = None
            self.navrl_launch_id = active_stage[2] if active_stage is not None else None
            if active_stage is not None and rpc_port is not None and launch_id and launch_id == self.navrl_launch_id:
                stage, path, _ = active_stage
                if path.name.startswith(("round_", "attempt_")):
                    new_goal = _evaluation_goal(path, stage, rpc_port)
                elif not (path / "decision.json").is_file():
                    new_goal = _goal_from_navrl_snapshot(path / f"live_goal_{rpc_port}.json", stage, rpc_port)
            changed = not was_navrl or new_goal != self.goal
            self.goal = new_goal
            self.live_goal = None
            self._episode_boxes = []
            return changed
        if self.run_dir is None:
            return False
        episodes = self.run_dir / "course_episodes.jsonl"
        path = episodes if episodes.is_file() else self.run_dir / "plan.json"
        try:
            st = path.stat()
        except OSError:
            return False
        stat_key = (st.st_size, int(st.st_mtime_ns))
        if stat_key == self._episode_stat:
            return False
        if path == episodes:
            record = read_last_json_line(path)
        else:
            plan = _read_json_object(path)
            record = {"goal_position": plan.get("scene_goal_position")} if plan is not None else None
        if record is None:
            return False
        self._episode_stat = stat_key
        new_goal = goal_from_episode_record(record, self.floor_z)
        if new_goal and path != episodes:
            new_goal.source = "plan"
        new_boxes = boxes_from_episode_record(record, self.floor_z)
        changed = (new_goal != self.goal) or (new_boxes != self._episode_boxes)
        self.goal, self._episode_boxes = new_goal, new_boxes
        return changed

    # -- 输出 -----------------------------------------------------------
    @property
    def boxes(self) -> list[SceneBox]:
        if self._live:
            return list(self._live.values())
        return list(self._episode_boxes)

    def set_manual_goal(self, position: tuple[float, float, float]) -> None:
        self.manual_goal = GoalMarker(position=position, source="manual")

    def update_live_pose(
        self,
        pos_enu: tuple[float, float, float],
        vel_enu: tuple[float, float, float] = (0.0, 0.0, 0.0),
    ) -> bool:
        """根据当前端口实时走廊几何或飞行航向推断当前回合终点，避免被多进程日志文件串扰。"""
        inferred = None
        if self.infer_goals and not self.is_navrl_run:
            inferred = infer_live_goal(pos_enu, vel_enu, self._live, self.floor_z)
        if inferred == self.live_goal:
            return False
        self.live_goal = inferred
        return True

    def effective_goal(self) -> Optional[GoalMarker]:
        if self.manual_goal is not None:
            return self.manual_goal
        if self.is_navrl_run or (self.goal is not None and self.goal.source == "plan"):
            return self.goal
        return self.live_goal or self.goal

    def scene_message(self) -> dict[str, Any]:
        boxes = self.boxes
        roofs = [b for b in boxes if b.kind == "roof"]
        roof_bottom = min((b.center[2] - b.size[2] / 2.0 for b in roofs), default=None)
        goal = self.effective_goal()
        message = {
            "type": "scene",
            "t": time.time(),
            "floor_z": self.floor_z,
            "roof_z": roof_bottom,
            "boxes": [b.to_dict() for b in boxes],
            "box_source": "live" if self._live else ("course_episodes" if self._episode_boxes else "none"),
            "goal": goal.to_dict() if goal else None,
            "scan_error": self.last_scan_error,
        }
        if self.is_navrl_run:
            message["navrl"] = True
        return message


def _finite(v: float) -> bool:
    return v == v and v not in (float("inf"), float("-inf"))


def infer_live_goal(
    pos_enu: tuple[float, float, float],
    vel_enu: tuple[float, float, float],
    live_boxes: dict[str, SceneBox],
    floor_z: float = DEFAULT_FLOOR_Z,
) -> Optional[GoalMarker]:
    """根据当前连接端口上的实时障碍物与车辆位姿推断本回合终点（局部 ENU z-up）。"""
    x, y, _ = pos_enu
    if not live_boxes:
        return None

    # 1. Stage3 V2 动态封闭走廊（Ch4Stage3V2_*）：从 goal_end_wall + 左右墙直接锁定精确终点
    ch4_v2 = [b for b in live_boxes.values() if b.name.startswith("Ch4Stage3V2_")]
    end_walls = [b for b in ch4_v2 if "goal_end_wall" in b.key]
    side_walls = [b for b in ch4_v2 if b.key in ("wall_min_y", "wall_max_y")]
    if end_walls and len(side_walls) >= 2:
        ew = end_walls[0]
        w0, w1 = side_walls[0], side_walls[1]
        center_y = 0.5 * (w0.center[1] + w1.center[1])
        x_min = ew.center[0] - ew.size[0] / 2.0
        x_max = ew.center[0] + ew.size[0] / 2.0
        w_min = min(w.center[0] - w.size[0] / 2.0 for w in side_walls)
        w_max = max(w.center[0] + w.size[0] / 2.0 for w in side_walls)
        if abs(x_max - w_max) <= abs(x_min - w_min):
            goal_x = x_min - 7.9  # 走廊尽头在 xmax，终点在其前方 ~7.9 m
        else:
            goal_x = x_max + 7.9  # 走廊尽头在 xmin，终点在其前方 ~7.9 m
        gates = [b for b in ch4_v2 if any(b.key.startswith(p) for p in ("over_", "under_", "side_"))]
        gates.sort(key=lambda g: abs(g.center[0] - goal_x))
        family = None
        goal_agl = 1.9
        if gates:
            last_key = gates[0].key
            if last_key.startswith("under_"):
                goal_agl = 4.1
                family = "over_under" if len(gates) >= 2 else "under"
            elif last_key.startswith("over_"):
                goal_agl = 1.9
                family = "under_over" if len(gates) >= 2 else "over"
            elif last_key.startswith("side_"):
                goal_agl = max(1.6, min(4.3, pos_enu[2] - floor_z))
                family = "side"
        return GoalMarker(
            position=(goal_x, center_y, floor_z + goal_agl),
            radii=(2.0, 1.6),
            altitude_tolerance=0.5,
            source="live_corridor",
            family=family,
        )

    # 2. 原场 / 带顶棚隧道保底回合：TrainEnv 9 条路线起点/终点在 x=0 ↔ x=162, y∈{-15, 0, +15}, z=-1.0 (1.6m AGL)
    has_tunnel_roof = any("tunnel_roof" in b.key for b in ch4_v2)
    has_trainenv_blocks = any(b.name.startswith("StaticMeshActor_") for b in live_boxes.values())
    if has_tunnel_roof or has_trainenv_blocks:
        target_x = 162.0 if x < 81.0 else 0.0
        vx, vy, _ = vel_enu
        if abs(vx) > 0.25 and (target_x - x) * vx > 0:
            proj_y = y + (target_x - x) * (vy / vx)
        else:
            proj_y = y
        lane_y = min((-15.0, 0.0, 15.0), key=lambda c: abs(c - proj_y))
        return GoalMarker(
            position=(target_x, lane_y, -1.0),
            radii=(2.0,),
            altitude_tolerance=0.5,
            source="live_route",
            family="original_roofed" if has_tunnel_roof else "trainenv",
        )

    return None
