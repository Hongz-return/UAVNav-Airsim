"""Discover RPC endpoints declared by the current NavRL training stage."""

import json
import os
import subprocess
from pathlib import Path

from .airsim_tap import connect_airsim


def _local_listeners():
    output = subprocess.check_output(
        ["netstat", "-ano", "-p", "tcp"], timeout=3,
        creationflags=subprocess.CREATE_NO_WINDOW,
    )
    listeners = set()
    for line in output.splitlines():
        fields = line.split()
        if len(fields) == 5 and fields[0] == b"TCP" and fields[2].endswith(b":0"):
            listeners.add((int(fields[1].rsplit(b":", 1)[1]), int(fields[4])))
    return listeners


def _matches_remote(host, instance):
    client = None
    try:
        client = connect_airsim(host, instance["port"], 1)
        settings = json.loads(client.getSettingsString())
        expected = instance["settings"]["TrainEnvLaunchId"]
        return bool(expected) and settings.get("TrainEnvLaunchId") == expected
    except Exception:
        return False
    finally:
        if client is not None:
            client.close()


def _online(path, host, listeners):
    try:
        data = json.loads(path.read_bytes())
        if not data["map"].startswith("/Game/Training/Maps/L_Train_NavRL_Forest"):
            return []
        instances = data["instances"]
        if listeners is not None:
            return sorted({i["port"] for i in instances if (i["port"], i["pid"]) in listeners})
        return sorted({i["port"] for i in instances if _matches_remote(host, i)})
    except (OSError, ValueError, KeyError, TypeError):
        return []


def discover_training_ports(root: Path, host: str) -> tuple[list[int], list[int]]:
    manifests = []
    for stage in root.glob("*/stage_*"):
        path = stage / "launch.json"
        try:
            marker = path if path.is_file() else stage / "route_phase.json"
            manifests.append((marker.stat().st_mtime_ns, path))
        except OSError:
            continue
    local = os.name == "nt" and host in ("localhost", "127.0.0.1")
    listeners = _local_listeners() if local else None
    for _, path in sorted(manifests, reverse=True):
        ports = _online(path, host, listeners)
        rounds = path.parent / "evaluations"
        for evaluation in sorted(rounds.glob("round_*/**/launch.json"), reverse=True):
            active = _online(evaluation, host, listeners)
            if active:
                return ports, active
            if not local:
                break
        if ports:
            return ports, []
        if not local:
            break
    return [], []
