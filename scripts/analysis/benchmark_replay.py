"""Time public CPU replay sampling on synthetic depth/motion data."""
import argparse
import hashlib
import importlib.util
import statistics
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from gymnasium import spaces

from gym_env.replay.sequence_buffer import SequenceReplayBuffer


def benchmark(transitions, batch_size, repeats, risk, implementations):
    rng = np.random.default_rng(42)
    observations = spaces.Dict({
        'image': spaces.Box(0, 255, (2, 60, 90), np.uint8),
        'state': spaces.Box(-1, 1, (6,), np.float32),
    })
    actions = spaces.Box(-1, 1, (3,), np.float32)
    buffers = [(name, cls(transitions, observations, actions, seq_len=16, device='cpu'))
               for name, cls in implementations]
    # Reuse frame payloads to exercise a large index with bounded resident memory.
    frames = [{'image': rng.integers(256, size=(2, 60, 90), dtype=np.uint8),
               'state': rng.uniform(-1, 1, 6).astype(np.float32)} for _ in range(257)]
    action = np.zeros(3, np.float32)
    for i in range(transitions + transitions // 4):
        end = i % 100 == 99
        timeout = end and i // 100 % 4 == 3
        reward = (-25., -20., 35.)[i // 100 % 3] if end else 0.
        for _, buffer in buffers:
            buffer.add(frames[i % 257], frames[(i + 1) % 257], action, reward,
                       end and not timeout, {'TimeLimit.truncated': timeout})

    timings = {name: [] for name, _ in buffers}
    digests = {}
    for i in range(3 + repeats):
        # Alternate execution order and draw identical starts in each pair.
        for name, buffer in (buffers if i % 2 == 0 else reversed(buffers)):
            np.random.seed(123 + i)
            start = perf_counter()
            batch = buffer.sample(batch_size, history_len=8, failure_horizon=30 if risk else None)
            elapsed = (perf_counter() - start) * 1000
            if i >= 3:
                timings[name].append(elapsed)
            if i == 3:
                digest = hashlib.sha256()
                for key in sorted(batch):
                    value = batch[key]
                    tensors = [value[k] for k in sorted(value)] if isinstance(value, dict) else [value]
                    for tensor in tensors:
                        digest.update(tensor.numpy().tobytes())
                digests[name] = digest.hexdigest()
            del batch

    if len(set(digests.values())) != 1:
        raise ValueError(f'Implementations returned different sampled data: {digests}')
    for name, times in timings.items():
        print(f'{name} capacity={transitions} batch={batch_size} risk={risk} '
              f'median_ms={statistics.median(times):.3f} p90_ms={np.percentile(times, 90):.3f} '
              f'sha256={digests[name]}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repeats', type=int, default=30)
    parser.add_argument('--baseline-source', type=Path, help='Extracted baseline source tree')
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error('--repeats must be positive')
    implementations = []
    if args.baseline_source:
        source = args.baseline_source / 'gym_env/replay/sequence_buffer.py'
        spec = importlib.util.spec_from_file_location('baseline_replay', source)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        implementations.append(('baseline', module.SequenceReplayBuffer))
    implementations.append(('current', SequenceReplayBuffer))
    torch.set_num_threads(1)
    for capacity, batch_size in ((5000, 16), (50000, 64), (100000, 16)):
        for risk in (False, True):
            benchmark(capacity, batch_size, args.repeats, risk, implementations)
