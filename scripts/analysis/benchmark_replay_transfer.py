"""Compare full and compact sequence replay on CUDA, without a simulator or optimizer."""
import argparse
from time import perf_counter

import numpy as np
import torch
from gymnasium import spaces

from gym_env.replay.sequence_buffer import SequenceReplayBuffer


def make_replay():
    rng = np.random.default_rng(7)
    space = spaces.Dict({
        'image': spaces.Box(0, 255, (2, 60, 90), np.uint8),
        'state': spaces.Box(0, 1, (6,), np.float32),
    })
    replay = SequenceReplayBuffer(50000, space, spaces.Box(-1, 1, (3,), np.float32),
                                  seq_len=16, device='cpu')
    frames = [{'image': rng.integers(256, size=(2, 60, 90), dtype=np.uint8),
               'state': rng.random(6, dtype=np.float32)} for _ in range(257)]
    action = np.zeros(3, dtype=np.float32)
    for i in range(60000):
        terminal = i % 100 == 99
        timeout = terminal and i % 200 == 199
        replay.add(frames[i % 257], frames[(i + 1) % 257], action, -25. if terminal else 0.,
                   terminal and not timeout, {'TimeLimit.truncated': timeout})
    return replay


@torch.inference_mode()
def benchmark(repeats):
    torch.set_num_threads(1)
    if not torch.cuda.is_available():
        raise RuntimeError('This benchmark requires a CUDA device')
    device = torch.device('cuda')
    print(torch.cuda.get_device_name(device), torch.__version__, torch.version.cuda)
    replay = make_replay()
    observations = {}
    for compact in (False, True):
        np.random.seed(19)
        cpu_batch = replay.sample(64, history_len=8, compact=compact)
        observations[compact] = [values['image'] for values in cpu_batch.values() if isinstance(values, dict)]
    results = {(compact, stage): [] for compact in (False, True) for stage in ('image_h2d', 'sample_to_cuda')}
    peak_bytes = {False: [], True: []}
    replay.device = device
    for index in range(repeats + 3):
        for compact in ((False, True) if index % 2 == 0 else (True, False)):
            torch.cuda.synchronize()
            start = perf_counter()
            gpu_images = [image.to(device) for image in observations[compact]]
            torch.cuda.synchronize()
            copy_ms = (perf_counter() - start) * 1000
            del gpu_images

            np.random.seed(29 + index)
            torch.cuda.reset_peak_memory_stats(device)
            allocated = torch.cuda.memory_allocated(device)
            start = perf_counter()
            batch = replay.sample(64, history_len=8, compact=compact)
            torch.cuda.synchronize()
            sample_ms = (perf_counter() - start) * 1000
            if index == 0:
                context = batch['context_observations']['image']
                if compact:
                    assert context.dtype == torch.uint8
                    torch.testing.assert_close(context.float(), reference, rtol=0, atol=0)
                    del reference
                else:
                    reference = context
                del context
            del batch
            if index >= 3:
                results[compact, 'image_h2d'].append(copy_ms)
                results[compact, 'sample_to_cuda'].append(sample_ms)
                peak_bytes[compact].append(torch.cuda.max_memory_allocated(device) - allocated)
    for (compact, stage), times in results.items():
        size = sum(t.numel() * t.element_size() for t in observations[compact]) / 2**20
        print(f'compact={compact} stage={stage} image_MiB={size:.3f} '
              f'median_ms={np.median(times):.3f} p90_ms={np.percentile(times, 90):.3f}')
    for compact, peaks in peak_bytes.items():
        print(f'compact={compact} sample_peak_allocated_MiB={max(peaks) / 2**20:.3f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repeats', type=int, default=20)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error('repeats must be positive')
    benchmark(args.repeats)
