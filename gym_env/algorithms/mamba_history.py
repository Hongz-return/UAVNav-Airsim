"""Gather episode-local replay windows for current and next SAC positions."""
from functools import cached_property

import torch


class HistoryWindows:
    def __init__(self, lengths, history, device, next_step):
        self.batch_size, self.steps = lengths.shape
        self.history = history
        self.indices = (torch.arange(self.steps, device=device)[:, None]
                        + torch.arange(history, device=device) + int(next_step))
        lengths = lengths.detach().cpu().reshape(-1)
        self.lengths = (lengths + 1).clamp(max=history) if next_step else lengths

    @cached_property
    def device_lengths(self):
        return self.lengths.to(self.indices.device)

    @cached_property
    def groups(self):
        lengths, positions = self.lengths.sort(stable=True)
        unique, counts = lengths.unique_consecutive(return_counts=True)
        return tuple(zip(unique.tolist(), positions.to(self.indices.device).split(counts.tolist())))

    def gather(self, features, states):
        if features.shape[1] - self.steps != self.history:
            raise ValueError("Replay context must match the actor inference history length")
        return (
            features[:, self.indices].reshape(self.batch_size * self.steps, self.history, features.shape[-1]),
            states[:, self.indices].reshape(self.batch_size * self.steps, self.history, *states.shape[2:]),
        )

    def gather_groups(self, features, states):
        feature_windows, state_windows = self.gather(features, states)
        for length, positions in self.groups:
            yield (positions,
                   feature_windows.index_select(0, positions)[:, -length:],
                   state_windows.index_select(0, positions)[:, -length:])
