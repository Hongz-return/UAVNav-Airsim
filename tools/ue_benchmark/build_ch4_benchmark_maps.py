"""Build the Chapter 4 maps inside the Tunnel project's UE 5.4 editor.

Run with 'py <absolute path to this script>' in the UE console. To rebuild one
map with Funplay's Python tool, import this file and call build(("t1",)), etc.
All generated packages live under Maps/04_Ch4_Benchmark.
"""

import bisect
import hashlib
import json
import math
import shutil
from pathlib import Path

import unreal


MAPS = "/Game/RailwayTunnelPrototype/Maps"
BENCH = f"{MAPS}/04_Ch4_Benchmark"
NAMES = {
    "t1": "L_Ch4_T1_DoubleStatic_Slalom",
    "t2": "L_Ch4_T2_DoubleDynamic_Workzone",
    "t3": "L_Ch4_T3_SingleSCurve_Cluttered",
}
SOURCES = {
    "t1": f"{MAPS}/02_Conditions/Crossover/L_DoubleTrack_Crossover_Reference_StaticMaintenance",
    "t2": f"{MAPS}/03_Experiments/L_Tunnel_MotionInteraction",
    "t3": f"{MAPS}/02_Conditions/SingleTrack/L_SingleTrack_SCurve_DynamicMaintenance",
}
REPORT_DIR = Path(__file__).resolve().parents[2] / "reports/ch4_benchmark_rebuild_20261125"
RAIL = "/Game/Railway_System/Meshes"
TUNNEL = "/Game/Tunnel_Col_Vol1/Meshes"
PROPS = "/Game/Asian_Village/meshes/props"
ASSETS = {
    "fence": f"{RAIL}/Platform/SM_Fence",
    "beam": f"{RAIL}/Bridge/SM_Bridge_Top",
    "support": f"{RAIL}/Bridge/SM_Bridge_Mast_Support",
    "sign": f"{RAIL}/Signs/SM_Pole_Keep_Tracks_Clear",
    "cart": f"{PROPS}/SM_RailwayMaintenanceCartBody",
    "box": f"{PROPS}/SM_box_04",
    "pylon": f"{TUNNEL}/SM_Pylon_1",
    "bollard": f"{TUNNEL}/SM_Bollard_1",
    "lamp": f"{TUNNEL}/SM_TunCol1_LED_Wall_Emergency",
    "wheel": f"{RAIL}/Intercity_Train/SM_Wheel",
}
BENCHMARK_TAG = "Ch4Benchmark"
V = unreal.Vector
UP = V(0, 0, 1)


def xyz(point, ndigits=4):
    return [round(point.x, ndigits), round(point.y, ndigits), round(point.z, ndigits)]


def yaw(degrees):
    return unreal.Rotator(roll=0.0, pitch=0.0, yaw=degrees)


def heading(vector):
    return math.degrees(math.atan2(vector.y, vector.x))


def absolute_project_dir():
    return Path(unreal.Paths.convert_relative_path_to_full(unreal.Paths.project_dir())).resolve()


def map_file(package):
    return absolute_project_dir() / "Content" / (package.removeprefix("/Game/") + ".umap")


def open_benchmark(scene, from_source):
    editor = unreal.get_editor_subsystem(unreal.LevelEditorSubsystem)
    subsystem = unreal.get_editor_subsystem(unreal.UnrealEditorSubsystem)
    target = f"{BENCH}/{NAMES[scene]}"
    source = SOURCES[scene]
    source_file, dest = map_file(source), map_file(target)
    if not source_file.is_file():
        raise FileNotFoundError(f"Missing locked base map: {source_file}")
    if from_source or not dest.is_file():
        # Unload the copy before replacing its serialized World. The original
        # package is only read, and building always saves the destination.
        if subsystem.get_editor_world().get_path_name().split(".")[0] == target:
            if not editor.load_level(source):
                raise RuntimeError(f"Could not unload {target}")
        dest.parent.mkdir(parents=True, exist_ok=True)
        staged = dest.with_suffix(".copy")
        shutil.copyfile(source_file, staged)
        staged.replace(dest)
        unreal.AssetRegistryHelpers.get_asset_registry().scan_files_synchronous([str(dest)], True)
    if subsystem.get_editor_world().get_path_name().split(".")[0] != target:
        if not editor.load_level(target):
            raise RuntimeError(f"Could not open {target}")
    world = subsystem.get_editor_world()
    if world.get_path_name().split(".")[0] != target:
        raise RuntimeError(f"Unexpected loaded world {world.get_path_name()}")
    return world, source, hashlib.sha256(source_file.read_bytes()).hexdigest()


class Centerline:
    """Polyline resampled from the actual world-space track's main spline."""

    def __init__(self, world_points):
        self.points = world_points
        self.stations = [0.0]
        for a, b in zip(world_points, world_points[1:]):
            self.stations.append(self.stations[-1] + (b - a).length())
        self.length = self.stations[-1]
        self.start = 0.0

    @classmethod
    def from_spline(cls, component):
        length = component.get_spline_length()
        segments = math.ceil(length / 25)
        return cls([
            component.get_location_at_distance_along_spline(
                length * i / segments, unreal.SplineCoordinateSpace.WORLD
            )
            for i in range(segments + 1)
        ])

    def frame(self, station_m):
        station = min(max(self.start + station_m * 100, 0), self.length)
        right = min(max(1, bisect.bisect_right(self.stations, station)), len(self.points) - 1)
        left = right - 1
        fraction = (station - self.stations[left]) / (self.stations[right] - self.stations[left])
        a, b = self.points[left], self.points[right]
        angle = heading(b - a)
        rad = math.radians(angle)
        return a + (b - a) * fraction, V(-math.sin(rad), math.cos(rad), 0), angle

    def point(self, station, offset=0, height=0):
        p, right, _ = self.frame(station)
        return p + right * (offset * 100) + UP * (height * 100)


class Build:
    def __init__(self, scene, from_source=False):
        self.scene = scene
        self.world, self.source, self.source_sha = open_benchmark(scene, from_source)
        self.actors = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
        self.assets = unreal.get_editor_subsystem(unreal.EditorAssetSubsystem)
        self.editor = unreal.get_editor_subsystem(unreal.LevelEditorSubsystem)
        self.obstacles = []
        self.restored_train = []
        self.cross_sections = []
        self.dynamic = []
        self.dynamic_corridors = []
        self.lights = []
        self.moving = []
        self.crossbeams = {}
        self.base_details = {}
        self.mesh_cache = {}
        for actor in self.actors.get_all_level_actors():
            if actor.actor_has_tag(BENCHMARK_TAG):
                self.actors.destroy_actor(actor)

    def spawn(self, actor_class, label, point, angle, folder):
        actor = self.actors.spawn_actor_from_class(actor_class, point, yaw(angle))
        if actor is None:
            raise RuntimeError(f"Failed to place {label}")
        actor.set_actor_label(label)
        actor.set_folder_path(f"Chapter 4 Benchmark/{folder}")
        actor.tags = [BENCHMARK_TAG, label]
        return actor

    def trace(self, start, end, complex_collision=False, ignore=None):
        hit = unreal.SystemLibrary.line_trace_single(
            self.world, start, end, unreal.TraceTypeQuery.TRACE_TYPE_QUERY1,
            complex_collision, ignore or [], unreal.DrawDebugTrace.NONE,
        )
        if hit is None:
            return None
        data = hit.to_tuple()
        return {"distance_m": round(data[3] / 100, 4), "point_cm": xyz(data[5]),
                "label": data[9].get_actor_label() if data[9] else None}

    def ground(self, station, offset):
        point = self.route.point(station, offset)
        hit = self.trace(point + UP * 118, point - UP * 220, complex_collision=True)
        if not hit:
            raise RuntimeError(f"No floor at s={station:.2f} m, offset={offset:.2f} m")
        return hit["point_cm"][2]

    def mesh(self, label, key, center, angle, dimensions=None, bottom=None, moving=False, collision=True):
        asset = self.mesh_cache.get(key)
        if asset is None:
            asset = self.assets.load_asset(ASSETS[key])
            if not isinstance(asset, unreal.StaticMesh):
                raise RuntimeError(f"Missing mesh {ASSETS[key]}")
            self.mesh_cache[key] = asset
        bounds = asset.get_bounding_box()
        size = bounds.max - bounds.min
        scale = V(1, 1, 1) if dimensions is None else V(
            dimensions[0] / size.x, dimensions[1] / size.y, dimensions[2] / size.z
        )
        middle = (bounds.min + bounds.max) * 0.5
        angle_rad = math.radians(angle)
        x, y = middle.x * scale.x, middle.y * scale.y
        pivot = center - V(
            x * math.cos(angle_rad) - y * math.sin(angle_rad),
            x * math.sin(angle_rad) + y * math.cos(angle_rad), middle.z * scale.z,
        )
        if bottom is not None:
            pivot.z = bottom - bounds.min.z * scale.z
        actor = self.spawn(unreal.StaticMeshActor, label, pivot, angle, "Equipment")
        component = actor.static_mesh_component
        if moving:
            component.set_mobility(unreal.ComponentMobility.MOVABLE)
        component.set_static_mesh(asset)
        component.set_collision_profile_name("BlockAllDynamic" if moving else "BlockAll")
        if not collision:
            component.set_collision_enabled(unreal.CollisionEnabled.NO_COLLISION)
        actor.set_actor_scale3d(scale)
        self.obstacles.append({"label": label, "actor_name": actor.get_name(), "asset": ASSETS[key],
                               "location_cm": xyz(pivot), "scale": xyz(scale),
                               "collision_enabled": collision})
        return actor

    def prop(self, label, key, s, offset, dimensions=None, relative_yaw=0, moving=False, collision=True):
        _, _, angle = self.route.frame(s)
        return self.mesh(label, key, self.route.point(s, offset), angle + relative_yaw,
                         dimensions, self.ground(s, offset), moving, collision)

    def fence(self, label, s, left, right, height):
        return self.prop(label, "fence", s, (left + right) * 0.5,
                         ((right - left) * 100, 9, height * 100), 90)

    def gate(self, label, station, gap, wings, height=2.85, flight_height=2.2):
        self.fence(f"{label}_LeftFence", station, wings[0], gap[0], height)
        self.fence(f"{label}_RightFence", station, gap[1], wings[1], height)
        self.measure(label, station, (gap[0] + gap[1]) * 0.5, flight_height)

    def measure(self, label, station, offset, height, ignore=None):
        _, right, _ = self.route.frame(station)
        center = self.route.point(station, offset, height)
        left_hit = self.trace(center, center - right * 1500, ignore=ignore)
        right_hit = self.trace(center, center + right * 1500, ignore=ignore)
        floor = self.trace(center, center - UP * 500, ignore=ignore)
        roof = self.trace(center, center + UP * 1000, ignore=ignore)
        beam = self.crossbeams.get(label)
        if beam:
            origin, extent = beam.get_actor_bounds(True)
            underside = origin.z - extent.z
            distance = (underside - center.z) / 100
            if (abs(center.x - origin.x) <= extent.x and abs(center.y - origin.y) <= extent.y
                    and distance >= 0 and (roof is None or distance < roof["distance_m"])):
                # Bounds cover a newly placed beam even before the editor's
                # collision scene has registered it for the upward trace.
                roof = {"distance_m": round(distance, 4), "point_cm": xyz(V(center.x, center.y, underside)),
                        "label": beam.get_actor_label()}
        self.cross_sections.append({
            "label": label, "station_m": station, "probe_cm": xyz(center),
            "effective_width_m": round(left_hit["distance_m"] + right_hit["distance_m"], 3)
            if left_hit and right_hit else None,
            "vertical_opening_m": round(floor["distance_m"] + roof["distance_m"], 3)
            if floor and roof else None,
            "left": left_hit, "right": right_hit, "floor": floor, "ceiling": roof,
        })

    def equipment_bay(self, label, station, side, clearance, height):
        # Place solid opaque cabinets ONLY on the blocking side (extending past
        # centerline) so straight flight always collides while leaving a wide
        # lateral lane without an inner-wall pinch.
        wing, left, right = ("Left", -2.48, 0.68) if side > 0 else ("Right", -0.68, 2.48)
        count = math.ceil((right - left) / 1.34)
        width = (right - left) / count
        target_top_z = self.route.point(station, 0, height).z
        for index in range(count):
            offset = left + (index + .5) * width
            floor_z = self.ground(station, offset)
            box_h_cm = max(100.0, target_top_z - floor_z)
            self.prop(f"{label}_{wing}_{index + 1}", "box", station, offset,
                      (86 + 18 * (index % 2), width * 100, box_h_cm))
        opening = side * 1.92
        self.measure(label, station, opening, 1.85)
        return opening

    def gantry(self, label, s, center_offset, width, bottom_m):
        floor = self.ground(s, center_offset)
        _, _, angle = self.route.frame(s)
        for side in (-1, 1):
            offset = center_offset + side * (width * .5 + .07)
            self.prop(f"{label}_Upright_{side}", "support", s, offset,
                      (36, 33, bottom_m * 100 + 29))
        self.crossbeams[label] = self.mesh(
            f"{label}_Crossbeam", "beam", self.route.point(s, center_offset),
            angle + 90, (width * 100 + 35, 96, 133), floor + bottom_m * 100,
        )

    def light(self, label, station, position, intensity_cd, radius_cm, color, cast_shadows=True):
        actor = self.spawn(unreal.PointLight, label, position, 0, "Practical lighting")
        component = actor.get_component_by_class(unreal.PointLightComponent)
        component.set_mobility(unreal.ComponentMobility.MOVABLE)
        component.set_intensity_units(unreal.LightUnits.CANDELAS)
        component.set_intensity(intensity_cd)
        component.set_attenuation_radius(radius_cm)
        component.set_source_radius(16)
        component.set_light_color(unreal.LinearColor(*color, 1.0))
        component.set_editor_property("cast_shadows", cast_shadows)
        self.lights.append({"station_m": station, "position_cm": xyz(position),
                            "intensity_cd": intensity_cd, "attenuation_radius_cm": radius_cm})

    def t3_exposure(self):
        volume = self.spawn(unreal.PostProcessVolume, "Ch4_T3_TunnelExposure", V(), 0,
                            "Practical lighting")
        volume.set_editor_property("unbound", True)
        volume.set_actor_enable_collision(False)
        for brush in volume.get_components_by_class(unreal.BrushComponent):
            brush.set_collision_enabled(unreal.CollisionEnabled.NO_COLLISION)
        settings = volume.get_editor_property("settings")
        # Tunnel_Exposure in both double-track maps overrides only these fields.
        for name, value in (
            ("auto_exposure_bias", -1.4),
            ("bloom_intensity", .15),
            ("scene_color_tint", unreal.LinearColor(.78, .87, 1.0, 1.0)),
        ):
            settings.set_editor_property(f"override_{name}", True)
            settings.set_editor_property(name, value)
        volume.set_editor_property("settings", settings)

    def t3_exit_sky(self, sky_actor):
        # The forward views see the existing sky sphere through the open portal.
        name = "M_Ch4_T3_ExitSky"
        path = f"{BENCH}/{name}"
        if self.assets.does_asset_exist(path):
            material = self.assets.load_asset(path)
        else:
            material = unreal.AssetToolsHelpers.get_asset_tools().create_asset(
                name, BENCH, unreal.Material, unreal.MaterialFactoryNew()
            )
        material.set_editor_property("shading_model", unreal.MaterialShadingModel.MSM_UNLIT)
        material.set_editor_property("is_sky", True)
        graph = unreal.MaterialEditingLibrary
        graph.delete_all_material_expressions(material)

        camera_vector = graph.create_material_expression(material, unreal.MaterialExpressionCameraVectorWS, -960, 0)
        vertical = graph.create_material_expression(material, unreal.MaterialExpressionComponentMask, -780, 0)
        vertical.set_editor_property("r", False)
        vertical.set_editor_property("b", True)
        vertical.set_editor_property("g", False)
        slope = graph.create_material_expression(material, unreal.MaterialExpressionMultiply, -600, 0)
        slope.set_editor_property("const_b", -.045)
        horizon = graph.create_material_expression(material, unreal.MaterialExpressionConstant3Vector, -600, 160)
        horizon.set_editor_property("constant", unreal.LinearColor(.022, .018, .015, 1))
        gradient = graph.create_material_expression(material, unreal.MaterialExpressionAdd, -420, 0)
        bounded = graph.create_material_expression(material, unreal.MaterialExpressionClamp, -240, 0)
        bounded.set_editor_property("min_default", .006)
        bounded.set_editor_property("max_default", .045)
        fixed = graph.create_material_expression(material, unreal.MaterialExpressionEyeAdaptationInverse, -60, 0)
        for source, destination, pin in (
            (camera_vector, vertical, ""), (vertical, slope, "A"), (horizon, gradient, "A"),
            (slope, gradient, "B"), (gradient, bounded, ""), (bounded, fixed, ""),
        ):
            if not graph.connect_material_expressions(source, "", destination, pin):
                raise RuntimeError(f"Could not connect {source.get_name()} to {destination.get_name()}")
        if not graph.connect_material_property(fixed, "", unreal.MaterialProperty.MP_EMISSIVE_COLOR):
            raise RuntimeError(f"Could not connect the emissive input of {name}")
        graph.recompile_material(material)
        if not self.assets.save_loaded_asset(material, False):
            raise RuntimeError(f"Could not save {path}")
        sky_actor.static_mesh_component.set_material(0, material)

    def t3_dressing_mesh(self, label, mesh, metal, point, rotation, dimensions_cm):
        actor = self.spawn(unreal.StaticMeshActor, label, point, rotation.yaw, "Wall dressing")
        component = actor.static_mesh_component
        component.set_static_mesh(mesh)
        component.set_material(0, metal)
        actor.set_actor_rotation(rotation, True)
        actor.set_actor_scale3d(V(*dimensions_cm) * .01)
        component.set_collision_profile_name("NoCollision")
        component.set_collision_enabled(unreal.CollisionEnabled.NO_COLLISION)
        actor.set_actor_enable_collision(False)
        # AirSim rasterizes DepthPlanar through a scene capture, independently of collision.
        component.set_hidden_in_scene_capture(True)

    def t3_point_light(self, label, station, point, intensity, radius, source_radius=0):
        light = self.spawn(unreal.PointLight, label, point, 0, "Practical lighting")
        light.set_actor_enable_collision(False)
        component = light.get_component_by_class(unreal.PointLightComponent)
        component.set_mobility(unreal.ComponentMobility.MOVABLE)
        component.set_intensity_units(unreal.LightUnits.CANDELAS)
        component.set_intensity(intensity)
        component.set_attenuation_radius(radius)
        component.set_source_radius(source_radius)
        component.set_light_color(unreal.LinearColor(1, 1, 1, 1))
        component.set_editor_property("use_temperature", True)
        component.set_editor_property("temperature", 5200)
        component.set_editor_property("use_inverse_squared_falloff", True)
        component.set_editor_property("cast_shadows", False)
        self.lights.append({"label": label, "station_m": round(station, 3),
                            "position_cm": xyz(point), "intensity_cd": intensity,
                            "temperature_k": 5200, "attenuation_radius_cm": radius})

    def t3_wall_dressing(self, tunnel):
        ignored = [actor for actor in self.actors.get_all_level_actors() if actor != tunnel]
        cube = self.assets.load_asset("/Engine/BasicShapes/Cube")
        cylinder = self.assets.load_asset("/Engine/BasicShapes/Cylinder")
        metal = self.assets.load_asset("/Game/Railway_System/Materials/Platform/MI_Metal")

        def wall_at(station, side, height, inset_cm):
            center, right, angle = self.route.frame(station)
            center += UP * (height * 100)
            wall = self.trace(center, center + right * (side * 800), True, ignored)
            if not wall or wall["label"] != tunnel.get_actor_label():
                raise RuntimeError(f"No T3 lining at s={station:.2f} m, side={side}, z={height} m")
            return V(*wall["point_cm"]) - right * (side * inset_cm), angle

        # The wall ends before station 111. Leave room for the last housing at the portal lip.
        end = min(111.0, (self.route.length - self.route.start) / 100 - .70)
        lamp_intervals = round((end - 2) / 11)
        stations = [2 + (end - 2) * index / lamp_intervals for index in range(lamp_intervals + 1)]
        segments = math.ceil(end / 3)
        for side in (-1, 1):
            for index, station in enumerate(stations):
                housing, angle = wall_at(station, side, 3.60, 9)
                self.t3_dressing_mesh(f"Ch4_T3_LuminaireHousing_{index:02d}_{side}", cube, metal,
                                      housing, yaw(angle), (75, 22, 18))
                mount, angle = wall_at(station, side, 3.60, 2)
                self.t3_dressing_mesh(f"Ch4_T3_LuminaireBracket_{index:02d}_{side}", cube, metal,
                                      mount, yaw(angle), (84, 8, 24))
                support, angle = wall_at(station, side, 2.90, 8)
                self.t3_dressing_mesh(f"Ch4_T3_CableSupport_{index:02d}_{side}", cube, metal,
                                      support, yaw(angle), (8, 25, 65))

                point, _ = wall_at(station, side, 3.40, 154)
                self.t3_point_light(f"Ch4_T3_WallLight_{index:02d}_{side}", station, point, 420, 950)

            # Match the two pipes and the three upper cables on each side in T1/T2.
            for height, diameter in ((1.25, 18), (1.75, 10), (2.85, 6), (2.96, 6), (3.07, 6)):
                samples = [wall_at(end * i / segments, side, height, diameter / 2 + 6)[0]
                           for i in range(segments + 1)]
                for i, (a, b) in enumerate(zip(samples, samples[1:])):
                    axis = b - a
                    self.t3_dressing_mesh(f"Ch4_T3_Conduit_{side}_{height:.2f}m_{i:02d}",
                                          cylinder, metal, (a + b) * .5,
                                          unreal.MathLibrary.make_rot_from_z(axis),
                                          (diameter, diameter, axis.length() + 3))

        # Fill box and cart faces in the gaps between the wall lights' pools.
        for index, (a, b) in enumerate(zip(stations, stations[1:])):
            station = (a + b) * .5
            self.t3_point_light(f"Ch4_T3_ReflectedFill_{index:02d}", station,
                                self.route.point(station, 0, 3.75), 110, 850, 35)

    def sequence(self, actor, positions, speed, angle, traverse):
        fps = 30
        label = actor.get_actor_label()
        name = f"LS_{label}"
        sequence = self.assets.load_asset(f"{BENCH}/{name}") if self.assets.does_asset_exist(
            f"{BENCH}/{name}"
        ) else unreal.AssetToolsHelpers.get_asset_tools().create_asset(
            name, BENCH, unreal.LevelSequence, unreal.LevelSequenceFactoryNew()
        )
        for old in sequence.get_bindings():
            old.remove()
        sequence.set_display_rate(unreal.FrameRate(fps, 1))
        sequence.set_playback_start(0)
        out_and_back = positions + positions[-2::-1]
        times = [0.0]
        for a, b in zip(out_and_back, out_and_back[1:]):
            times.append(times[-1] + (b - a).length() / (speed * 100))
        frames = [round(t * fps) for t in times]
        sequence.set_playback_end(frames[-1] + 1)
        binding = sequence.add_possessable(actor)
        section = binding.add_track(unreal.MovieScene3DTransformTrack).add_section()
        section.set_range(0, frames[-1] + 1)
        channels = section.get_all_channels()
        if len(channels) != 9:
            raise RuntimeError(f"Unexpected transform track channel count: {len(channels)}")
        for frame, p in zip(frames, out_and_back):
            for channel, value in zip(channels, xyz(p) + [0.0, 0.0, angle] + xyz(actor.get_actor_scale3d())):
                if traverse and channel == channels[5]:
                    continue
                channel.add_key(unreal.FrameNumber(frame), value,
                                interpolation=unreal.MovieSceneKeyInterpolation.LINEAR)
        if traverse:
            turn_frame = frames[len(positions) - 1]
            for frame, direction in ((0, angle), (turn_frame, angle + 180), (frames[-1], angle)):
                channels[5].add_key(unreal.FrameNumber(frame), direction,
                                    interpolation=unreal.MovieSceneKeyInterpolation.CONSTANT)
        player = self.spawn(unreal.LevelSequenceActor, f"{label}_Loop", V(), 0, "Motion")
        player.set_sequence(sequence)
        player.add_binding(sequence.get_binding_id(binding), actor, False)
        settings = player.get_editor_property("playback_settings")
        settings.set_editor_property("auto_play", True)
        settings.set_editor_property("loop_count", unreal.MovieSceneSequenceLoopCount(value=-1))
        settings.set_editor_property("disable_camera_cuts", True)
        player.set_editor_property("playback_settings", settings)
        if not self.assets.save_loaded_asset(sequence, False):
            raise RuntimeError(f"Could not save {name}")
        direction = positions[-1] - positions[0]
        self.dynamic.append({"rpc_tag": label, "actor_name": actor.get_name(),
                             "sequence": sequence.get_path_name(), "speed_mps": speed,
                             "one_way_m": round(sum((b - a).length() for a, b in
                                                    zip(positions, positions[1:])) / 100, 3),
                             "period_s": (frames[-1] + 1) / fps,
                             "require_m": 2.5, "require_lateral_m": 2.0 if traverse else 0.0,
                             "lateral_axis_ue": xyz(self.route.frame(0)[1]),
                             "projected_axis_ue": xyz(direction / direction.length()),
                             "start_cm": xyz(positions[0]), "turn_cm": xyz(positions[-1]),
                             "auto_play": True, "loop_count": -1})
        self.moving.append(actor)

    def worker(self, label, start, end, speed):
        p, q = (self.route.point(*endpoint) for endpoint in (start, end))
        a = heading(q - p)
        actor_class = unreal.EditorAssetLibrary.load_blueprint_class("/Game/ThirdPerson/Blueprints/BP_Worker")
        base_z = self.ground(*start)
        p.z = base_z + 89
        q.z = self.ground(*end) + 89
        actor = self.spawn(actor_class, label, p, a, "Motion")
        actor.set_editor_property("auto_possess_ai", unreal.AutoPossessAI.DISABLED)
        movement = actor.get_component_by_class(unreal.CharacterMovementComponent)
        movement.set_editor_property("default_land_movement_mode", unreal.MovementMode.MOVE_NONE)
        movement.set_editor_property("gravity_scale", 0)
        capsule = actor.get_component_by_class(unreal.CapsuleComponent)
        capsule.set_collision_profile_name("BlockAllDynamic")
        animation = self.assets.load_asset("/Game/Characters/Mannequins/Animations/Manny/MM_Walk_InPlace")
        skeletal = actor.get_component_by_class(unreal.SkeletalMeshComponent)
        skeletal.set_animation_mode(unreal.AnimationMode.ANIMATION_SINGLE_NODE)
        data = skeletal.get_editor_property("animation_data")
        data.set_editor_property("anim_to_play", animation)
        data.set_editor_property("saved_looping", True)
        data.set_editor_property("saved_playing", True)
        skeletal.set_editor_property("animation_data", data)
        self.sequence(actor, [p, q], speed, a, True)

    def cart(self, label, s, offset, height, end_s=None, speed=None):
        going = end_s is not None
        actor = self.prop(label, "cart", s, offset, (235, 114, height * 100),
                          moving=going)
        # Give the real cart body its existing railway wheel mesh at both axles.
        keep = unreal.AttachmentRule.KEEP_WORLD
        for index, ds in enumerate((-.76, .76)):
            wheel = self.prop(f"{label}_Axle{index+1}", "wheel", s + ds, offset,
                              (32, 75, 32), moving=going, collision=False)
            if going:
                wheel.attach_to_actor(actor, "", keep, keep, keep, False)
        if going:
            pivot = actor.get_actor_location()
            advance = 1 if end_s > s else -1
            points = [s] + [float(p) for p in range(int(s) + advance, int(end_s), advance)] + [end_s]
            positions = []
            for station in points:
                p = pivot + self.route.point(station, offset) - self.route.point(s, offset)
                p.z = pivot.z + self.ground(station, offset) - self.ground(s, offset)
                positions.append(p)
            self.sequence(actor, positions, speed, self.route.frame(s)[2], False)

    def corridor(self, label, span, signed_edge, side):
        """Conservative opening between an obstacle's swept edge and the shell."""
        samples = []
        untracked = self.moving + [a for a in self.actors.get_all_level_actors()
                                   if a.get_attach_parent_actor() in self.moving]
        for i in range(math.ceil((span[1] - span[0]) * 2) + 1):
            station = min(span[0] + i * .5, span[1])
            _, right, _ = self.route.frame(station)
            point = self.route.point(station, signed_edge, 2.0)
            wall = self.trace(point, point + right * (side * 1300), ignore=untracked)
            if wall is None:
                raise RuntimeError(f"No far wall at {label}, s={station}")
            samples.append({"station_m": station, "width_m": wall["distance_m"],
                            "edge_offset_m": signed_edge, "limiting_surface": wall["label"]})
        self.dynamic_corridors.append({"tag": label, "span_m": span, "side": side,
                                       "min_width_m": min(x["width_m"] for x in samples),
                                       "samples": samples})

    def t1(self):
        self.route = Centerline([V(500, 0, 0), V(17400, 0, 0)])
        self.length = 160.0
        self.start_height = 2.30
        # Initial Stage 2 trajectories clipped the left fence tip (y=-3.03).
        # Move the opening toward the usable lane and retain a blocked center.
        self.gate("Ch4_T1_RightTrackWorksite", 27, (-3.72, -.77), (-5.69, 5.69))
        self.prop("Ch4_T1_ToolLocker", "box", 28.5, 1.0, (102, 92, 191))
        self.fence("Ch4_T1_LeftTrackWorksite_WallEnd", 55, -5.69, -3.88, 2.85)
        self.gate("Ch4_T1_LeftTrackWorksite", 55, (.15, 3.04), (-.93, 5.69))
        self.measure("Ch4_T1_LeftTrackWorksite_AlternateLane", 55, -2.405, 2.20)
        self.cart("Ch4_T1_LeftRailServiceCart", 56.85, -1.55, 1.41)
        self.prop("Ch4_T1_DistributionCabinet", "box", 56.8, -.45, (97, 90, 188))
        self.gantry("Ch4_T1_CatenaryServiceGantry", 83, 0.0, 3.23, 2.68)
        self.fence("Ch4_T1_CatenaryServiceGantry_WallEnd", 83, -5.69, -5.00, 2.85)
        self.gate("Ch4_T1_CatenaryServiceGantry", 83, (-1.41, 1.41), (-2.06, 5.69),
                  flight_height=2.04)
        self.measure("Ch4_T1_WestTrackBypass", 83, -3.53, 2.20)
        parked_x = {"ParkedTrain_Body": 11050.0, "ParkedTrain_Bogie_0": 10972.545,
                    "ParkedTrain_Bogie_1": 12171.422}
        for actor in self.actors.get_all_level_actors():
            label = actor.get_actor_label()
            if label in parked_x:
                p = actor.get_actor_location()
                actor.set_actor_location(V(parked_x[label], p.y, p.z), False, True)
                self.restored_train.append({"label": label, "actor_name": actor.get_name(),
                                            "position_cm": xyz(actor.get_actor_location())})
        # Advance left-wall service cabinet (same geometry as T2's 63 m cabinet)
        # guides west-track traffic smoothly back toward the parked-train lane.
        self.prop("Ch4_T1_ApproachCabinet", "box", 96.0, -4.15, (105, 110, 225))
        # Match the restriction at the flank of the parked carriage with solid
        # opaque cabinets across [-5.55, -2.75] (~3.40 m clear passage to train).
        for idx, (off, h_cm) in enumerate(((-5.083, 240), (-4.150, 250), (-3.217, 255)), 1):
            self.prop(f"Ch4_T1_TrainWarningCabinet_{idx}", "box", 111.0, off, (105, 93.3, h_cm))
        for d in (108.4, 110, 112.1):
            self.prop(f"Ch4_T1_TrainPylon_{d:.1f}", "pylon", d, -2.80,
                      collision=False)
        self.measure("Ch4_T1_ParkedTrainPassage", 111, -1.05, 2.20)
        # Solid opaque exit workbank spanning [-5.60, +0.05] and [+3.35, +5.55],
        # leaving a 3.30 m opening at [+0.05, +3.35] with zero thin wire bars.
        for idx in range(5):
            off = -5.60 + (idx + 0.5) * 1.13
            self.prop(f"Ch4_T1_OffsetExitWorkscreen_Left_{idx + 1}", "box", 139.0, off,
                      (100, 113.0, 240 if idx == 0 else 255))
        for idx in range(2):
            off = 3.35 + (idx + 0.5) * 1.10
            self.prop(f"Ch4_T1_OffsetExitWorkscreen_Right_{idx + 1}", "box", 139.0, off,
                      (100, 110.0, 255 if idx == 0 else 240))
        self.measure("Ch4_T1_OffsetExitWorkscreen", 139, 1.70, 2.20)
        self.prop("Ch4_T1_ExitRailSign", "sign", 141, -.17, relative_yaw=90)
        return [
            (0, 0, 2.30), (17, 0, 2.30), (24, -2.17, 2.22), (30, -2.17, 2.22),
            (46, 0, 2.22), (52, 1.66, 2.22), (58, 1.66, 2.22), (72, 0, 2.20),
            (80, 0, 2.03), (86, 0, 2.03), (97, -.68, 2.18), (114, -.72, 2.18),
            (127, -.70, 2.18), (136, 1.70, 2.18), (142, 1.70, 2.18),
            (160, 0, 2.30),
        ], [(1, 0, 2.30), (20, -1.47, 2.3), (48, .70, 2.35),
            (77, -.25, 2.03), (105, -1.0, 2.26), (132, .73, 2.31)]

    def t2(self):
        existing = {a.get_actor_label(): a for a in self.actors.get_all_level_actors()}
        lining = [a for name, a in existing.items() if name.startswith("Lining_")]
        if len(lining) != 16 or "Tunnel_Exposure" not in existing or "Crossover_LayoutGuide" not in existing:
            raise RuntimeError("T2 must contain the original MotionInteraction lining and exposure")
        self.route = Centerline([V(500, 0, 0), V(30700, 0, 0)])
        self.length = 150.0
        self.start_height = 2.30
        self.base_details = {"lining_segments": len(lining), "lining_length_m": 308.0,
                             "exposure_actor": "Tunnel_Exposure", "shifted_carts": []}
        for name, actor in existing.items():
            if name == "MotionInteractionController" or name.startswith("MotionInspectionWorker_"):
                self.actors.destroy_actor(actor)
        for index, (object_name, x) in enumerate((("StaticMeshActor_421", 7400), ("StaticMeshActor_432", 10900))):
            label = f"MotionMaintenanceCart_{index}"
            actor = existing[label]
            if actor.get_name() != object_name:
                raise RuntimeError(f"Unexpected original actor for {label}: {actor.get_name()}")
            old = actor.get_actor_location()
            delta = V(x - old.x, 225 - old.y, 0)
            for part in (actor, *(existing[f"StressCart_{index}_Wheel_{i}"] for i in range(2))):
                part.set_actor_location(part.get_actor_location() + delta, False, True)
            final = actor.get_actor_location()
            self.base_details["shifted_carts"].append({
                "actor": object_name, "label": label, "world_x_m": final.x / 100, "world_y_m": final.y / 100,
            })
        self.base_details["cart_spacing_m"] = (
            self.base_details["shifted_carts"][1]["world_x_m"] - self.base_details["shifted_carts"][0]["world_x_m"]
        )
        self.fence("Ch4_T2_EntryWorkscreen", 31, -.48, 4.68, 2.64)
        self.prop("Ch4_T2_EntryCabinet", "box", 31.0, 0.08, (105, 105, 230))
        self.prop("Ch4_T2_OffsetCabinet", "box", 63.0, -4.25, (110, 105, 225))
        self.fence("Ch4_T2_TracksideWorkscreen", 86, 3.67, 5.13, 2.33)
        self.prop("Ch4_T2_TracksideCabinet", "box", 86.0, 4.25, (105, 100, 225))
        self.worker("Ch4_T2_WorkzoneInspectionCrewA", (39, -.42), (42, 3.30), .79)
        self.cart("Ch4_T2_RailServiceShuttle", 94, -2.25, 2.02, end_s=77, speed=1.24)
        self.worker("Ch4_T2_WorkzoneInspectionCrewB", (112, .03), (116, 3.31), .75)
        for station, side in ((29, 1), (40, 1), (63, -1), (86, 1), (111, 1), (124, 1)):
            self.prop(f"Ch4_T2_WarningBollard_{station}", "bollard", station, side * 4.18, collision=False)
            self.light(f"Ch4_T2_AmberWarning_{station}", station,
                       self.route.point(station, side * 3.90, 1.44), 58, 490, (1.0, .52, .18))
        self.corridor("Ch4_T2_WorkzoneInspectionCrewA", (38.4, 42.6), -.90, -1)
        self.corridor("Ch4_T2_RailServiceShuttle", (75.5, 95.5), -1.63, 1)
        self.corridor("Ch4_T2_WorkzoneInspectionCrewB", (111.4, 116.6), -.43, -1)
        return [
            (0, 0, 2.30), (21, 0, 2.30), (28, -2.39, 2.21), (46, -2.39, 2.21),
            (58, -1.75, 2.21), (72, -1.75, 2.21), (77, 1.95, 2.21),
            (97, 1.95, 2.21), (102, -1.97, 2.21),
            (141, -1.97, 2.21), (150, 0, 2.30),
        ], [(1, 0, 2.28), (22, -1.60, 2.22), (36, -1.70, 2.20, 16),
            (71, -1.10, 2.22, 8), (106, -1.55, 2.20, 13), (149, 0, 2.27, 180)]

    def t3(self):
        existing = {a.get_actor_label(): a for a in self.actors.get_all_level_actors()}
        if "BP_Track" not in existing or "BP_Tunnel" not in existing:
            raise RuntimeError("T3 needs a clean copy of the locked single-track base. Rebuild with from_source=('t3',).")
        spline = existing["BP_Track"].get_component_by_class(unreal.SplineComponent)
        self.route = Centerline.from_spline(spline)
        self.route.start = 380
        self.length = 106.0
        self.start_height = 1.85
        self.base_details = {"track_spline_length_m": round(self.route.length / 100, 3),
                             "goal_lining_margin_m": round((self.route.length - self.route.start) / 100 - self.length, 3),
                             "retained_geometry": ["BP_Track", "BP_Tunnel", "SM_Wire", "SM_Wire2", "SM_Wire3"]}
        for label in ("LS_MaintenanceEquipment", "BP_Worker", "SM_box_01", "SM_Wheel", "SM_Wheel2",
                      "DirectionalLight", "SkyAtmosphere", "SkyLight", "VolumetricCloud", "ExponentialHeightFog"):
            if label in existing:
                self.actors.destroy_actor(existing[label])
        # Keep the level instance's construction value consistent with the material override.
        # Apply the material without reconstructing the tunnel's spline collision components.
        tunnel = existing["BP_Tunnel"]
        lining = self.assets.load_asset("/Game/Railway_System/Materials/M_Tunnel")
        emissive = unreal.MaterialEditingLibrary.get_material_default_scalar_parameter_value(lining, "Emissive")
        tunnel.set_editor_property("Lights Emissive", emissive,
                                   notify_mode=unreal.PropertyAccessChangeNotifyMode.NEVER)
        for part in tunnel.get_components_by_class(unreal.SplineMeshComponent):
            part.set_material(0, lining)
        self.t3_exposure()
        self.t3_exit_sky(existing["SM_SkySphere"])
        self.t3_wall_dressing(tunnel)
        first = self.equipment_bay("Ch4_T3_EntryEquipment", 22, 1, 2.36, 1.98)
        second = self.equipment_bay("Ch4_T3_BlindBendServiceCases", 46, -1, 2.36, 1.96)
        third = self.equipment_bay("Ch4_T3_ReverseBendCabinets", 71, 1, 2.36, 1.98)
        fourth = self.equipment_bay("Ch4_T3_ExitMaintenanceBay", 99, -1, 2.36, 1.96)
        # Light the exposed wooden faces at the entry and reverse-facing end views.
        for label, station, opening in (("Entry", 19.7, first), ("Exit", 101.3, fourth)):
            self.t3_point_light(f"Ch4_T3_{label}BayFill", station,
                                self.route.point(station, -.5 * opening, 2.15), 95, 420, 35)
        self.cart("Ch4_T3_MovingInspectionCart", 84, -.67, 1.78, end_s=93, speed=.73)
        self.corridor("Ch4_T3_MovingInspectionCart", (82.4, 94.6), -1.42, -1)
        return [
            (0, 0, 1.85), (11, 0, 1.85), (19, first, 1.80), (25, first, 1.80),
            (42, second, 1.80), (49, second, 1.80), (66, third, 1.80),
            (73.5, third, 1.80), (81, -2.45, 1.78), (95, -2.45, 1.78),
            (97, fourth, 1.80), (101.5, fourth, 1.80), (106, 0, 1.85),
        ], [(15, 1.20, 1.85), (39, -1.05, 1.85), (65, 1.20, 1.85),
            (80, -1.15, 1.85), (92.0, -1.00, 1.85), (105, .10, 1.85, 180)]

    def finish(self):
        flight_controls, views = getattr(self, self.scene)()
        start_offset = 0
        start = self.route.point(0, start_offset, self.start_height)
        player = next(a for a in self.actors.get_all_level_actors() if isinstance(a, unreal.PlayerStart))
        player.set_actor_location(start, False, True)
        player.set_actor_rotation(yaw(0), True)
        player.tags = list(player.tags) + ["Ch4BenchOrigin"] if not player.actor_has_tag(
            "Ch4BenchOrigin"
        ) else list(player.tags)
        goal_loc = self.route.point(*flight_controls[-1])
        goal = next((a for a in self.actors.get_all_level_actors() if isinstance(a, unreal.TargetPoint)), None)
        if goal:
            goal.set_actor_location(goal_loc, False, True)
        else:
            self.spawn(unreal.TargetPoint, f"Ch4_{self.scene}_Goal", goal_loc, 0, "Route")
        def ned(p):
            return xyz(V((p.x - start.x) / 100, (p.y - start.y) / 100, (start.z - p.z) / 100))

        samples = [min(self.length, i * .5) for i in range(math.ceil(self.length * 2) + 1)]
        controls = [f[0] for f in flight_controls]
        # Include each control station explicitly even if it falls off the half-meter grid.
        samples = sorted(set(samples + controls))

        def routed_pose(s):
            index = min(max(1, bisect.bisect_right(controls, s)), len(controls) - 1)
            a, b = flight_controls[index - 1:index + 1]
            t = (s - a[0]) / (b[0] - a[0])
            return a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t

        if self.scene == "t2":
            for s in samples:
                self.measure(f"T2 static corridor {s:.2f} m", s, *routed_pose(s), ignore=self.moving)
        self.actors.clear_actor_selection_set()
        if not self.editor.save_current_level():
            raise RuntimeError(f"Could not save {self.world.get_path_name()}")

        detour = [ned(self.route.point(s, *routed_pose(s))) for s in samples]
        center = [ned(self.route.point(s, 0, self.start_height)) for s in samples]
        target = f"{BENCH}/{NAMES[self.scene]}"
        x_limit = max(p[0] for p in center) + 8
        absolute_z = (.55, 3.70)
        if self.scene == "t3":
            # The physical section supports this whole altitude range below
            # the contact wires. Keep the final bound before the open portal.
            x_limit = ned(self.route.points[-1])[0] - 1.0
            absolute_z = (.35, 4.60)
        visible_actors = {
            actor.get_name(): actor.get_actor_label()
            for actor in self.actors.get_all_level_actors()
            if not actor.hidden and (
                any(c.is_visible() and c.static_mesh for c in
                    actor.get_components_by_class(unreal.StaticMeshComponent))
                or isinstance(actor, unreal.Character)
            )
        }
        return {
            "map": target, "source": self.source, "source_sha256": self.source_sha,
            "map_sha256": hashlib.sha256(map_file(target).read_bytes()).hexdigest(),
            "base_details": self.base_details,
            "route_length_m": round(self.length, 3), "player_start_ue_cm": xyz(start),
            "start_ned_m": detour[0], "goal_ned_m": detour[-1],
            "workspace_x": [round(min(p[0] for p in center) - 5, 2), round(x_limit, 2)],
            "workspace_y": [round(min(p[1] for p in center) - 7, 2),
                            round(max(p[1] for p in center) + 7, 2)],
            "workspace_z_enu": [round(z - start.z / 100, 3) for z in absolute_z],
            "detour_ned_m": detour, "centerline_ned_m": center,
            "stills": [{"station_m": s, "ned_m": ned(self.route.point(s, lateral, h)),
                        "yaw_deg": self.route.frame(s)[2] + (angles[0] if angles else 0)}
                       for s, lateral, h, *angles in views],
            "obstacles": self.obstacles, "existing_train": self.restored_train,
            "visible_scene_actors": visible_actors,
            "static_openings": self.cross_sections, "dynamic": self.dynamic,
            "dynamic_openings": self.dynamic_corridors, "light_rig": self.lights,
        }


def build(scenes=("t1", "t2", "t3"), report_dir=REPORT_DIR, from_source=()):
    """Rebuild benchmark dressing. Pass scene IDs in from_source to copy the base again."""
    if unreal.EditorLoadingAndSavingUtils.get_dirty_map_packages():
        dirty = unreal.EditorLoadingAndSavingUtils.get_dirty_map_packages()
        outside = [p.get_name() for p in dirty if not p.get_name().startswith(BENCH + "/")]
        if outside:
            raise RuntimeError(f"Original maps have unsaved changes: {outside}")
    report_dir = Path(report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    manifest = report_dir / "benchmark_manifest.json"
    data = json.loads(manifest.read_text(encoding="utf-8")) if manifest.is_file() else {}
    for scene in scenes:
        if scene not in NAMES:
            raise ValueError(scene)
        data[scene] = Build(scene, from_source=scene in from_source).finish()
        staged = manifest.with_suffix(".json.tmp")
        staged.write_text(json.dumps(data, indent=2), encoding="utf-8")
        staged.replace(manifest)
        unreal.log(f"Built {data[scene]['map']} ({data[scene]['route_length_m']} m)")
    return data


if __name__ == "__main__":
    build()
