"""Select a quarter of the existing TrainEnv obstacles for horizontal motion."""

from collections import Counter, deque
import math

from tools.ue_benchmark.ch4_trainenv_3d_course import ROOT, read_json
from tools.ue_benchmark.ch4_stage3_v2_course import CLEAN_STRIPS


SURVEY = ROOT / "reports/ch4_stage3_3d_20261125/original_trainenv_geometry.json"
MAP = "/Game/UAVTraining/Maps/TrainEnv_Dynamic25"
REPORT = ROOT / "reports/ch4_trainenv_dynamic_20261127"
SHAPES = ("BlueBlock_", "BrownPanel_", "GreenPillar_")
END_Y = (-15, 0, 15)
PROTECTED_ENDS = [(x, y) for x in (0, 162) for y in END_Y]
COURSE_BOXES = [((a, b), (y - 6.3, y + 6.3), (-3.6, 6.0)) for a, b, y in CLEAN_STRIPS]


def overlaps(a, b, horizontal_clearance=0.0):
    margins = (horizontal_clearance, horizontal_clearance, -1e-6)
    return all(lo_a < hi_b + gap and lo_b < hi_a + gap
               for (lo_a, hi_a), (lo_b, hi_b), gap in zip(a, b, margins, strict=True))


def sweep_options(name, bounds, solids):
    for amplitude in (1.75, 1.50, 1.25, 1.00):
        for axis in (1, 0):
            swept = list(bounds)
            swept[axis] = (bounds[axis][0] - amplitude, bounds[axis][1] + amplitude)
            (xmin, xmax), (ymin, ymax), _ = swept
            if not (0 <= xmin < xmax <= 162 and -44 <= ymin < ymax <= 44):
                continue
            if any(math.hypot(max(xmin - x, 0, x - xmax), max(ymin - y, 0, y - ymax)) < 4
                   for x, y in PROTECTED_ENDS):
                continue
            if any(overlaps(swept, course, .3) for course in COURSE_BOXES):
                continue
            if any(other != name and overlaps(swept, box, .22) for other, box in solids.items()):
                continue
            yield axis, amplitude, swept


def connected_routes(envelopes):
    """Find 1 m grid routes at nominal height outside all expanded swept volumes."""
    flight_relevant = [box for box in envelopes if box[2][0] - .85 <= -1 <= box[2][1] + .85]
    blocked = set()
    for (xmin, xmax), (ymin, ymax), _ in flight_relevant:
        blocked.update((x, y) for x in range(math.ceil(xmin - .85), math.floor(xmax + .85) + 1)
                       for y in range(math.ceil(ymin - .85), math.floor(ymax + .85) + 1))
    cells = {(x, y) for x in range(-7, 170) for y in range(-58, 59)} - blocked
    routes = []
    for y in END_Y:
        start = (0, y)
        if start not in cells:
            raise ValueError(f"Blocked nominal departure {start}")
        previous = {start: None}
        frontier = deque([start])
        while frontier:
            x, zy = frontier.popleft()
            for point in ((x + 1, zy), (x, zy - 1), (x, zy + 1), (x - 1, zy)):
                if point in cells and point not in previous:
                    previous[point] = (x, zy)
                    frontier.append(point)
        for destination_y in END_Y:
            goal = (162, destination_y)
            if goal not in previous:
                raise ValueError(f"Disconnected nominal route {start} to {goal}")
            path = []
            point = goal
            while point is not None:
                path.append(point)
                point = previous[point]
            routes.append({"start_xy": start, "goal_xy": goal, "route_xy": path[::-1]})
    return routes


def layout():
    survey = read_json(SURVEY)
    solids = {item["name"]: item["axes_local_z_up_m"] for item in survey["solids"]}
    counted = [name for name in solids if name.startswith((*SHAPES, "NarrowGate_"))]
    if len(solids) != 173 or len(counted) != 160:
        raise ValueError("TrainEnv obstacle inventory differs from the surveyed baseline")
    candidates = {}
    for name in counted:
        if name.startswith(SHAPES):
            choices = list(sweep_options(name, solids[name], solids))
            if choices:
                x, y = ((lo + hi) / 2 for lo, hi in solids[name][:2])
                candidates[name] = (x, y, choices)

    selected = {}

    def accept(name):
        for axis, amplitude, box in candidates[name][2]:
            if all(not overlaps(box, sweep, .25) for _, _, sweep in selected.values()):
                selected[name] = (axis, amplitude, box)
                return True
        return False

    for family in SHAPES[:2]:
        names = sorted((n for n in candidates if n.startswith(family)),
                       key=lambda n: (abs(candidates[n][1]), -len(candidates[n][2]), n))
        accepted = 0
        for name in names:
            accepted += accept(name)
            if accepted == 4:
                break
    bands = ((0, 32), (32, 64), (64, 96), (96, 128), (128, 162))
    for a, b in bands:
        names = [n for n, (x, _, _) in candidates.items() if a <= x < b and n not in selected]
        names.sort(key=lambda n: (abs(candidates[n][1]) * .015 - len(candidates[n][2]), candidates[n][0], n))
        for name in names:
            if sum(a <= candidates[n][0] < b for n in selected) >= 8:
                break
            accept(name)
    if len(selected) * 4 != len(counted):
        raise ValueError(f"Expected 40 disjoint mover envelopes, found {len(selected)}")

    motions = []
    for i, (name, (axis, amplitude, swept)) in enumerate(sorted(selected.items())):
        excursion = [0.0, 0.0, 0.0]
        excursion[axis] = 100 * amplitude
        motions.append({"label": name, "excursion_cm": excursion, "period_s": 9.0 + 1.7 * (i % 5),
                        "phase_rad": (i * 2.399963229728653) % math.tau, "swept_local_bounds_m": swept})
    return {
        "source_map": survey["map"], "map": MAP, "survey": str(SURVEY), "solid_count": len(solids),
        "obstacle_components": len(counted), "moving_obstacles": len(motions),
        "stationary_obstacles": len(counted) - len(motions), "motions": motions,
        "preserved_obstacle_labels": sorted(set(counted) - set(selected)),
        "moving_families": dict(Counter(m["label"].split("_")[0] for m in motions)),
        "routes_outside_all_swept_volumes": connected_routes(
            [selected[n][2] if n in selected else b for n, b in solids.items()]
        ),
    }
