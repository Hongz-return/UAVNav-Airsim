"""Build the isolated TrainEnv_3D level in the TrainEnvAirSim UE5.4 editor.

Run in the editor with py <absolute path to this file>, or with
-run=pythonscript -script=<absolute path to this file> on UnrealEditor-Cmd.
"""

from pathlib import Path
import sys

import unreal

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ch4_trainenv_3d_course import COURSE_FILE, ROOT, read_json, sha256, structural_boxes, write_json


TAG = "Ch4TrainEnv3D"
MATERIALS = "/Game/UAVTraining/Materials/M_TrainEnv_"
REPORT = ROOT / "reports/ch4_trainenv_3d_20261125"


def xyz(vector):
    return [float(vector.x), float(vector.y), float(vector.z)]


def spawn(actors, actor_class, position, label, folder, rotation=None):
    actor = actors.spawn_actor_from_class(
        actor_class, unreal.Vector(*position), rotation or unreal.Rotator()
    )
    if actor is None:
        raise RuntimeError(f"Could not spawn {label}")
    actor.set_actor_label(f"Ch4_3D_{label}")
    actor.set_folder_path(folder)
    actor.set_editor_property("tags", [TAG])
    return actor


def static_box(actors, mesh, name, axes, material, folder, collision=True):
    location = [(a + b) * 50 for a, b in axes]
    scale = [b - a for a, b in axes]   # Engine BasicShapes Cube is 100 cm on every axis.
    actor = spawn(actors, unreal.StaticMeshActor, location, name, folder)
    actor.set_actor_scale3d(unreal.Vector(*scale))
    component = actor.get_component_by_class(unreal.StaticMeshComponent)
    component.set_static_mesh(mesh)
    component.set_material(0, material)
    component.set_editor_property("mobility", unreal.ComponentMobility.STATIC)
    if collision:
        component.set_collision_profile_name("BlockAll")
        component.set_collision_enabled(unreal.CollisionEnabled.QUERY_AND_PHYSICS)
    else:
        component.set_collision_enabled(unreal.CollisionEnabled.NO_COLLISION)
    origin, extent = map(xyz, actor.get_actor_bounds(collision))
    lower = [p - r for p, r in zip(origin, extent)]
    upper = [p + r for p, r in zip(origin, extent)]
    for bounds, lo, hi in zip(axes, lower, upper):
        if abs(lo - bounds[0] * 100) > 0.1 or abs(hi - bounds[1] * 100) > 0.1:
            raise RuntimeError(f"{name}: invalid bounds {lower} to {upper} for {axes}")
    return {
        "name": name,
        "editor_label": actor.get_actor_label(),
        "ue_object_name": actor.get_name(),
        "lower_ue_cm": lower,
        "upper_ue_cm": upper,
        "collision_profile": str(component.get_collision_profile_name()),
        "collision_enabled": str(component.get_collision_enabled()),
    }


def measure_pawn(actors):
    pawn_class = unreal.load_class(None, "/AirSim/Blueprints/BP_FlyingPawn.BP_FlyingPawn_C")
    if pawn_class is None:
        raise RuntimeError("Cannot load the AirSim default quadrotor blueprint")
    pivot = [0, 0, 330]
    pawn = actors.spawn_actor_from_class(pawn_class, unreal.Vector(*pivot), transient=True)
    if pawn is None:
        raise RuntimeError("Cannot instantiate the AirSim blueprint for a bounds inspection")
    try:
        root = pawn.get_editor_property("root_component")
        components = []
        for component in pawn.get_components_by_class(unreal.PrimitiveComponent):
            center, box, sphere = unreal.SystemLibrary.get_component_bounds(component)
            components.append({
                "name": component.get_name(),
                "class": component.get_class().get_name(),
                "is_root": component == root,
                "center_from_pivot_cm": [v - p for v, p in zip(xyz(center), pivot)],
                "half_extent_cm": xyz(box),
                "sphere_radius_cm": float(sphere),
                "collision_profile": str(component.get_collision_profile_name()),
                "collision_enabled": str(component.get_collision_enabled()),
            })
        return {"blueprint": pawn_class.get_path_name(), "components": components}
    finally:
        actors.destroy_actor(pawn)


def main():
    course = read_json(COURSE_FILE)
    project_path = Path(unreal.Paths.convert_relative_path_to_full(unreal.Paths.get_project_file_path()))
    if project_path.stem != "TrainEnvAirSim":
        raise RuntimeError(f"Open the TrainEnvAirSim project, found {project_path}")
    package = course["map"]
    if package != "/Game/UAVTraining/Maps/TrainEnv_3D":
        raise RuntimeError(f"Unexpected destination package {package}")
    levels = unreal.get_editor_subsystem(unreal.LevelEditorSubsystem)
    actors = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
    editor = unreal.get_editor_subsystem(unreal.UnrealEditorSubsystem)
    if unreal.EditorAssetLibrary.does_asset_exist(package):
        if editor.get_editor_world().get_path_name().split(".")[0] != package:
            if not levels.load_level(package):
                raise RuntimeError(f"Cannot open {package}")
    elif not levels.new_level(package):
        raise RuntimeError(f"Cannot create {package}")
    world = editor.get_editor_world()
    if world.get_path_name().split(".")[0] != package:
        raise RuntimeError(f"Unexpected world: {world.get_path_name()}")
    for actor in actors.get_all_level_actors():
        if TAG in [str(tag) for tag in actor.get_editor_property("tags")]:
            if not actors.destroy_actor(actor):
                raise RuntimeError(f"Cannot replace {actor.get_actor_label()}")
        elif isinstance(actor, unreal.PlayerStart):
            raise RuntimeError("Generated map already has a separate PlayerStart")

    world.get_world_settings().set_editor_property("force_no_precomputed_lighting", True)
    mesh = unreal.load_asset("/Engine/BasicShapes/Cube")
    materials = {
        name: unreal.load_asset(MATERIALS + name)
        for name in ("Floor", "Boundary", "BlueBlock", "BrownPanel", "GreenPillar", "Start", "Goal")
    }
    if not mesh or not all(materials.values()):
        raise RuntimeError("A TrainEnv surface material or the Engine cube is missing")

    boxes = [
        static_box(actors, mesh, name, [x, y, z], materials[material], folder)
        for name, x, y, z, material, folder in structural_boxes(course)
    ]
    spawn(actors, unreal.PlayerStart, course["player_start_ue_cm"], "PlayerStart", "Reference")
    for family, lane in course["families"].items():
        y = lane["lane_y_m"]
        for x, mat in ((3, "Start"), (63, "Goal")):
            static_box(
                actors, mesh, f"{family}_{mat}Strip",
                [[x - 0.2, x + 0.2], [y - 5.5, y + 5.5], [0.005, 0.015]],
                materials[mat], "Reference", collision=False,
            )
        for x in (3, 15, 29, 41, 55, 67):
            light = spawn(actors, unreal.PointLight, [100 * x, 100 * y, 596], f"{family}_{x}m", "Lighting")
            component = light.get_component_by_class(unreal.PointLightComponent)
            component.set_editor_property("mobility", unreal.ComponentMobility.MOVABLE)
            component.set_attenuation_radius(1200)
            component.set_intensity(560)
            component.set_cast_shadows(False)

    light = spawn(
        actors, unreal.DirectionalLight, [0, 0, 1500], "ExteriorSun", "Lighting",
        unreal.Rotator(pitch=-55, yaw=-33),
    )
    component = light.get_component_by_class(unreal.DirectionalLightComponent)
    component.set_editor_property("mobility", unreal.ComponentMobility.MOVABLE)
    component.set_intensity(3.3)
    pawn = measure_pawn(actors)
    actors.clear_actor_selection_set()
    if not levels.save_current_level():
        raise RuntimeError(f"Cannot save {package}")
    map_file = project_path.parent / "Content" / (package.removeprefix("/Game/") + ".umap")
    REPORT.mkdir(parents=True, exist_ok=True)
    write_json(REPORT / "map_manifest.json", {
        "map": package,
        "map_file": str(map_file),
        "map_sha256": sha256(map_file),
        "geometry_sha256": sha256(COURSE_FILE),
        "engine_version": unreal.SystemLibrary.get_engine_version(),
        "player_start_ue_cm": course["player_start_ue_cm"],
        "solids": boxes,
        "physical_pawn": pawn,
    })
    unreal.log(f"[Ch4 TrainEnv 3D] Saved {map_file}")


if __name__ == "__main__":
    main()
