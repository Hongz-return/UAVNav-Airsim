"""Local Windows acknowledgement for UE's synchronous game-thread sequence command.

The same file is loaded in AirSim's Python client and once per moving episode in
Unreal Python. The client's console RPC confirms delivery to the game thread;
the request number in shared memory confirms successful sequence evaluation.
"""
import mmap
import struct
import sys
import uuid
from pathlib import Path


_STATUS = struct.Struct("<q")
_SIZE = 512
_REGISTRY = "_uavnav_sequence_clocks"


def _ack(memory, serial, error=None):
    if error is not None:
        encoded = str(error).encode("utf-8", errors="replace")[:_SIZE - _STATUS.size - 1]
        memory[_STATUS.size:_STATUS.size + len(encoded) + 1] = encoded + b"\0"
        serial = -serial
    _STATUS.pack_into(memory, 0, serial)


class LocalSequenceClock:
    def __init__(self, client):
        self.client = client
        self.tag = "UAVNavScene_" + uuid.uuid4().hex
        self.memory = mmap.mmap(-1, _SIZE, tagname=self.tag)
        self.serial = 0
        self.bound = False
        self.remote = f"__import__('sys').{_REGISTRY}[{self.tag!r}]"

    def _execute(self, expression):
        if not self.client.simRunConsoleCommand("py " + expression):
            raise RuntimeError("The UE sequence command did not reach the game thread")
        status, = _STATUS.unpack_from(self.memory)
        if status == -self.serial:
            reason = self.memory[_STATUS.size:].split(b"\0", 1)[0].decode("utf-8", errors="replace")
            raise RuntimeError(f"UE sequence clock failed: {reason}")
        if status != self.serial:
            raise RuntimeError(f"UE sequence command {self.serial} was not acknowledged (received {status})")

    def bind(self, snapshot, phase_s, play_rate):
        self.serial += 1
        self.bound = False
        script = Path(__file__).resolve().as_posix()
        args = (self.tag, self.serial, snapshot["map"], snapshot["sequence"],
                tuple(snapshot["poses"]), float(phase_s), float(play_rate))
        self._execute(f"__import__('runpy').run_path({script!r})['bind_in_unreal']{args!r}")
        self.bound = True

    def advance(self, elapsed_s):
        self.serial += 1
        self._execute(f"{self.remote}.advance({self.serial},{float(elapsed_s)!r})")

    def close(self):
        if self.memory is None:
            return
        try:
            if self.bound:
                self.serial += 1
                self._execute(f"{self.remote}.release({self.serial})")
        finally:
            self.bound = False
            self.memory.close()
            self.memory = None


class _UnrealClock:
    def __init__(self, unreal, player, actors, memory, tag, map_asset, phase_s, play_rate):
        self.map_asset = map_asset
        self.player = player
        self.actors = actors
        self.is_valid = unreal.SystemLibrary.is_valid
        self.memory = memory
        self.tag = tag
        self.phase_s = phase_s
        self.play_rate = play_rate
        self.position = unreal.MovieSceneSequencePlaybackParams()
        self.position.position_type = unreal.MovieScenePositionType.TIME
        self.position.update_method = unreal.UpdatePositionMethod.JUMP

    def advance(self, serial, elapsed_s):
        try:
            if self.player.is_playing() or not all(self.is_valid(actor) for actor in self.actors):
                raise RuntimeError("Sequence resumed or a bound obstacle disappeared")
            self.position.time = ((self.phase_s + elapsed_s) * self.play_rate) % 60
            self.player.set_playback_position(self.position)
            self.player.pause()
        except Exception as exc:
            _ack(self.memory, serial, exc)
        else:
            _ack(self.memory, serial)

    def release(self, serial):
        getattr(sys, _REGISTRY).pop(self.tag)
        _ack(self.memory, serial)
        self.memory.close()


def bind_in_unreal(tag, serial, map_asset, sequence_asset, actor_names, phase_s, play_rate):
    import unreal

    memory = mmap.mmap(-1, _SIZE, tagname=tag)
    registry = getattr(sys, _REGISTRY, None)
    if registry is None:
        registry = {}
        setattr(sys, _REGISTRY, registry)
    # Preparing a new episode transfers ownership of this world's player. A
    # previous client that exited unexpectedly must not steer the new episode.
    for previous_tag, previous in list(registry.items()):
        if previous_tag == tag or previous.map_asset == map_asset:
            registry.pop(previous_tag).memory.close()
    try:
        world_path = map_asset + "." + map_asset.rsplit("/", 1)[-1]
        world = unreal.find_object(None, world_path)
        if world is None:
            raise LookupError(f"Unloaded sequence world {world_path}")
        controllers = unreal.GameplayStatics.get_all_actors_of_class(world, unreal.LevelSequenceActor)
        matches = [actor for actor in controllers
                   if actor.get_sequence() is not None and actor.get_sequence().get_path_name() == sequence_asset]
        if len(matches) != 1:
            raise LookupError(f"Expected one controller for {sequence_asset}, found {len(matches)}")
        player = matches[0].get_editor_property("sequence_player")
        actors = [unreal.find_object(None, world_path + ":PersistentLevel." + name) for name in actor_names]
        if player.is_playing() or any(actor is None for actor in actors):
            raise RuntimeError("The prepared sequence must be paused with all its obstacle actors present")
        registry[tag] = _UnrealClock(unreal, player, actors, memory, tag, map_asset, phase_s, play_rate)
    except Exception as exc:
        _ack(memory, serial, exc)
        memory.close()
    else:
        _ack(memory, serial)
