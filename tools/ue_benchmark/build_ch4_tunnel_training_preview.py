"""Build the first two Chapter 4 tunnel courses in the UE 5.4 editor."""

import sys
from pathlib import Path

import unreal

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_ch4_trainenv_3d import MATERIALS, spawn, static_box
from ch4_trainenv_3d_course import ROOT, sha256, write_json


MAP = "/Game/UAVTraining/Maps/TrainEnv_TunnelPreview"
FENCE = "01 Fence bypass"
CROSSING = "02 Dynamic crossing"
CAMERAS = "03 Preview cameras"


def xyz_m(x, y, z):
    return [100 * x, 100 * y, 100 * z]


def add_corridor(add, folder, center, length, width, height):
    left, right = center - width / 2, center + width / 2
    add(folder, "Floor", [-4, length + 4], [left - .24, right + .24], [-.25, 0], "Floor")
    add(folder, "Left wall", [0, length], [left - .24, left], [0, height], "Boundary")
    add(folder, "Right wall", [0, length], [right, right + .24], [0, height], "Boundary")
    add(folder, "Roof", [0, length], [left - .24, right + .24], [height, height + .24], "Boundary")
    for x, material in ((2, "Start"), (length - 2, "Goal")):
        add(folder, material, [x - .12, x + .12], [left + .2, right - .2], [.001, .011],
            material, collision=False)


def add_point_light(actors, folder, x, y, height):
    light = spawn(actors, unreal.PointLight, xyz_m(x, y, height), f"Inside light {x:g}m", folder)
    component = light.get_component_by_class(unreal.PointLightComponent)
    component.set_editor_property("mobility", unreal.ComponentMobility.MOVABLE)
    component.set_attenuation_radius(1050)
    component.set_intensity(1900)
    component.set_cast_shadows(False)


def crosses_inflated_box(start, end, axes, clearance):
    """Conservatively test a swept aircraft against the measured UE bounds."""
    entry, exit_time = 0.0, 1.0
    for first, last, (low, high) in zip(start, end, axes, strict=True):
        low, high = low - clearance, high + clearance
        movement = last - first
        if abs(movement) < 1e-9:
            if not low <= first <= high:
                return False
            continue
        near, far = sorted(((low - first) / movement, (high - first) / movement))
        entry, exit_time = max(entry, near), min(exit_time, far)
        if entry > exit_time:
            return False
    return True


def add_mover(actors, floor, cart):
    motion_class = unreal.load_class(None, "/Script/TrainEnvAirSim.TrainEnvObstacleMotion")
    if motion_class is None:
        raise RuntimeError("Cannot load the existing TrainEnv motion controller")
    rig = spawn(actors, motion_class, xyz_m(0, 0, 0), "Crossing obstacle motion", CROSSING)
    rig.set_editor_property("floor", floor)
    track = unreal.TrainEnvMotionTrack()
    track.set_editor_property("mesh_actor", cart)
    track.set_editor_property("excursion", unreal.Vector(0, 165, 0))
    track.set_editor_property("period", 8.0)
    track.set_editor_property("phase", 0.0)
    rig.set_editor_property("tracks", [track])


def main():
    levels = unreal.get_editor_subsystem(unreal.LevelEditorSubsystem)
    editor = unreal.get_editor_subsystem(unreal.UnrealEditorSubsystem)
    actors = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
    project = Path(unreal.Paths.convert_relative_path_to_full(unreal.Paths.get_project_file_path()))
    if project.stem != "TrainEnvAirSim":
        raise RuntimeError(f"Unexpected Unreal project: {project}")
    path = project.parent / "Content" / (MAP.removeprefix("/Game/") + ".umap")
    if path.exists() or unreal.EditorAssetLibrary.does_asset_exist(MAP):
        raise FileExistsError(f"Preview level already exists: {path}")

    mesh = unreal.load_asset("/Engine/BasicShapes/Cube")
    materials = {key: unreal.load_asset(MATERIALS + key) for key in
                 ("Floor", "Boundary", "BrownPanel", "BlueBlock", "GreenPillar", "Start", "Goal")}
    if mesh is None or any(material is None for material in materials.values()):
        raise RuntimeError("A required TrainEnv material or cube mesh is missing")
    if not levels.new_level(MAP):
        raise RuntimeError(f"Cannot create {MAP}")
    world = editor.get_editor_world()
    if world.get_path_name().split(".")[0] != MAP:
        raise RuntimeError("The editor did not load the preview level")
    world.get_world_settings().set_editor_property("force_no_precomputed_lighting", True)

    solids = {}
    def add(folder, label, x, y, z, material, collision=True):
        row = static_box(actors, mesh, f"{folder[:2]} {label}", [x, y, z],
                         materials[material], folder, collision=collision)
        if collision:
            solids[(folder, label)] = row
        return row

    add_corridor(add, FENCE, -8, 44, 7.2, 3.2)
    add_corridor(add, CROSSING, 8, 48, 6.8, 2.95)
    add(FENCE, "Fence from left", [13, 13.45], [-11.6, -7.40], [0, 2.92], "BrownPanel")
    add(FENCE, "Fence from right", [28, 28.45], [-8.60, -4.40], [0, 2.92], "BrownPanel")
    add(CROSSING, "Offset entrance screen", [10, 10.48], [4.6, 8.75], [0, 2.73], "BrownPanel")
    add(CROSSING, "Moving cross-aisle cart", [25.90, 27.60], [7.45, 8.55], [0, 1.55], "BlueBlock")
    add(CROSSING, "Overhead beam", [35, 36], [4.6, 11.4], [2.25, 2.95], "Boundary")

    by_name = {actor.get_name(): actor for actor in actors.get_all_level_actors()}
    cart = by_name[solids[(CROSSING, "Moving cross-aisle cart")]["ue_object_name"]]
    floor = by_name[solids[(CROSSING, "Floor")]["ue_object_name"]]
    cart.get_component_by_class(unreal.StaticMeshComponent).set_editor_property(
        "mobility", unreal.ComponentMobility.MOVABLE,
    )
    add_mover(actors, floor, cart)

    path_points = {
        FENCE: [[2,-8,1.55], [10,-6,1.55], [17,-6,1.55],
                [24,-10,1.55], [32,-10,1.55], [42,-8,1.55]],
        CROSSING: [[2,8,1.55], [7,10.1,1.55], [14,10.1,1.55],
                   [22,10.8,1.55], [30,10.8,1.55], [38,9.8,1.45], [46,8,1.55]],
    }
    for folder, points in path_points.items():
        for (owner, label), row in solids.items():
            if owner != folder:
                continue
            axes = [(low / 100, high / 100) for low, high in zip(row["lower_ue_cm"], row["upper_ue_cm"], strict=True)]
            if label == "Moving cross-aisle cart":
                axes[1] = (axes[1][0] - 1.65, axes[1][1] + 1.65)
            if any(crosses_inflated_box(a, b, axes, .42) for a, b in zip(points, points[1:])):
                raise RuntimeError(f"Reference bypass lacks clearance at {label}")

    spawn(actors, unreal.PlayerStart, xyz_m(-1.5, -8, 2.6), "Aircraft origin", "Starts")
    for folder, center, length, height in ((FENCE, -8, 44, 3.2), (CROSSING, 8, 48, 2.95)):
        for x in range(3, length, 6):
            add_point_light(actors, folder, x, center, height - .24)
    sun = spawn(actors, unreal.DirectionalLight, xyz_m(-13, -10, 18), "Exterior light",
                "Lighting", unreal.Rotator(pitch=-52, yaw=24))
    sun_component = sun.get_component_by_class(unreal.DirectionalLightComponent)
    sun_component.set_editor_property("mobility", unreal.ComponentMobility.MOVABLE)
    sun_component.set_intensity(3.3)

    for label, position, pitch in (
        ("Both entrances", [-18, 0, 6], -10),
        ("Fence bypass", [2, -8, 1.65], 0),
        ("Dynamic crossing", [2, 8, 1.60], 0),
        ("Crossing cart", [20, 9.7, 1.8], -2),
    ):
        spawn(actors, unreal.CameraActor, xyz_m(*position), label, CAMERAS, unreal.Rotator(pitch=pitch))

    actors.clear_actor_selection_set()
    if not levels.save_current_level():
        raise RuntimeError(f"Cannot save {MAP}")
    report = ROOT.parent / ".codex-sidechat-output/TrainEnv_TunnelPreview"
    report.mkdir(parents=True, exist_ok=True)
    write_json(report / "scene.json", {
        "map": MAP, "map_file": str(path), "sha256": sha256(path),
        "course_dimensions_m": {FENCE: [44, 7.2, 3.2], CROSSING: [48, 6.8, 2.95]},
        "aircraft_radius_for_clearance_m": .42,
        "reference_paths_ue_world_m": path_points,
        "moving_cart_amplitude_m": 1.65, "moving_cart_period_s": 8.0,
        "solids": list(solids.values()),
    })
    unreal.log(f"Saved tunnel training preview: {path}")


if __name__ == "__main__":
    main()
