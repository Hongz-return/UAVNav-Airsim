"""Create TrainEnv_Dynamic25 from the audited baseline in UE 5.4."""

from pathlib import Path
import shutil
import sys

import unreal

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from tools.ue_benchmark.ch4_trainenv_3d_course import read_json, sha256, write_json


REPORT = REPO / "reports/ch4_trainenv_dynamic_20261127"


def axes_of(actor):
    center, radius = actor.get_actor_bounds(True)
    return [(getattr(center, axis) - getattr(radius, axis), getattr(center, axis) + getattr(radius, axis))
            for axis in "xyz"]


def main():
    plan_file = REPORT / "layout.json"
    plan = read_json(plan_file)
    survey = read_json(Path(plan["survey"]))
    project = Path(unreal.Paths.convert_relative_path_to_full(unreal.Paths.project_dir())).resolve()
    content = project / "Content"
    source = content / (plan["source_map"].removeprefix("/Game/") + ".umap")
    target = content / (plan["map"].removeprefix("/Game/") + ".umap")
    source_hash = sha256(source)
    if source_hash != plan["source_map_sha256"]:
        raise ValueError("The source map bytes changed since the survey was checked")
    with source.open("rb") as existing, target.open("xb") as replica:
        shutil.copyfileobj(existing, replica)
    unreal.AssetRegistryHelpers.get_asset_registry().scan_files_synchronous([str(target)], True)

    levels = unreal.get_editor_subsystem(unreal.LevelEditorSubsystem)
    worlds = unreal.get_editor_subsystem(unreal.UnrealEditorSubsystem)
    actors = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
    if not levels.load_level(plan["map"]) or worlds.get_editor_world().get_path_name().split(".")[0] != plan["map"]:
        raise RuntimeError("Failed to load the isolated dynamic map")
    by_label = {actor.get_actor_label(): actor for actor in actors.get_all_level_actors()
                if isinstance(actor, unreal.StaticMeshActor)}
    for original in survey["solids"]:
        label = original["name"]
        measured = axes_of(by_label[label])
        if any(abs(a - b) > .25 for m, n in zip(measured, original["axes_world_ue_cm"], strict=True)
               for a, b in zip(m, n, strict=True)):
            raise RuntimeError(f"Replica bounds differ from the TrainEnv survey: {label}")

    cls = unreal.load_class(None, "/Script/TrainEnvAirSim.TrainEnvObstacleMotion")
    if cls is None:
        raise RuntimeError("Build TrainEnvAirSimEditor before creating the map")
    rig = actors.spawn_actor_from_class(cls, unreal.Vector(), unreal.Rotator())
    rig.set_actor_label("TrainEnv 40 Moving Obstacles")
    rig.set_editor_property("floor", by_label["TrainEnv_Floor"])
    tracks = []
    for motion in plan["motions"]:
        actor = by_label[motion["label"]]
        mesh = actor.get_component_by_class(unreal.StaticMeshComponent)
        mesh.set_editor_property("mobility", unreal.ComponentMobility.MOVABLE)
        if mesh.get_collision_enabled() != unreal.CollisionEnabled.QUERY_AND_PHYSICS:
            raise RuntimeError(f"Expected solid mesh: {motion['label']}")
        actor.set_folder_path("Moving Obstacles")
        track = unreal.TrainEnvMotionTrack()
        track.set_editor_property("mesh_actor", actor)
        track.set_editor_property("excursion", unreal.Vector(*motion["excursion_cm"]))
        track.set_editor_property("period", motion["period_s"])
        track.set_editor_property("phase", motion["phase_rad"])
        tracks.append(track)
        motion["actor"] = actor.get_name()
    rig.set_editor_property("tracks", tracks)
    actors.clear_actor_selection_set()
    if not levels.save_current_level():
        raise RuntimeError("Saving TrainEnv_Dynamic25 failed")
    if sha256(source) != source_hash:
        raise RuntimeError("The source map was unexpectedly written")
    write_json(REPORT / "map_manifest.json", {
        **plan, "source_map_file": str(source), "map_file": str(target),
        "map_sha256": sha256(target), "layout_sha256": sha256(plan_file),
        "engine_version": unreal.SystemLibrary.get_engine_version(),
    })
    unreal.log(f"Saved {plan['map']} with {len(tracks)} original moving obstacles")


if __name__ == "__main__":
    main()
