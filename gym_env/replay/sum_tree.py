"""
SumTree data structure for O(log n) prioritized sampling.

Structure:
- Complete binary tree stored in array
- Leaf nodes store priorities
- Internal nodes store sum of children
- Total priority = root value
"""

import numpy as np


class SumTree:
    r"""
    SumTree for efficient prioritized sampling.

    Array layout for capacity=4:
        [0]           <- root (sum of all)
       /   \
     [1]   [2]        <- internal nodes
     / \   / \
    [3][4][5][6]      <- leaves (priorities)

    Leaves start at index (capacity - 1).
    """

    def __init__(self, capacity: int):
        """
        Initialize SumTree.

        Args:
            capacity: Maximum number of leaf nodes (must be power of 2 for simplicity,
                      but we handle non-power-of-2 gracefully)
        """
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1, dtype=np.float64)
        self.data_pointer = 0
        self.n_entries = 0

    def _propagate(self, idx: int, change: float):
        """Propagate priority change up to root."""
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    def _leaf_idx(self, data_idx: int) -> int:
        """Convert data index to tree leaf index."""
        return data_idx + self.capacity - 1

    def update(self, data_idx: int, priority: float):
        """
        Update priority for a data index.

        Args:
            data_idx: Index in data array (0 to capacity-1)
            priority: New priority value
        """
        tree_idx = self._leaf_idx(data_idx)
        change = priority - self.tree[tree_idx]
        self.tree[tree_idx] = priority
        self._propagate(tree_idx, change)

    def add(self, priority: float) -> int:
        """
        Add new priority, return data index.

        Args:
            priority: Priority for new sample

        Returns:
            Data index where sample should be stored
        """
        data_idx = self.data_pointer
        self.update(data_idx, priority)

        self.data_pointer = (self.data_pointer + 1) % self.capacity
        self.n_entries = min(self.n_entries + 1, self.capacity)

        return data_idx

    def get(self, s: float) -> tuple:
        """
        Sample a leaf by cumulative sum.

        Args:
            s: Random value in [0, total_priority)

        Returns:
            (data_idx, priority)
        """
        idx = 0  # Start at root

        while True:
            left = 2 * idx + 1
            right = left + 1

            if left >= len(self.tree):  # Reached leaf
                break

            if s <= self.tree[left]:
                idx = left
            else:
                s -= self.tree[left]
                idx = right

        data_idx = idx - (self.capacity - 1)
        return data_idx, self.tree[idx]

    @property
    def total(self) -> float:
        """Total priority (root value)."""
        return self.tree[0]

    @property
    def max(self) -> float:
        """Maximum priority among leaves."""
        leaf_start = self.capacity - 1
        leaf_end = leaf_start + self.n_entries
        if self.n_entries == 0:
            return 1.0
        return self.tree[leaf_start:leaf_end].max()

    @property
    def min(self) -> float:
        """Minimum non-zero priority among leaves."""
        leaf_start = self.capacity - 1
        leaf_end = leaf_start + self.n_entries
        if self.n_entries == 0:
            return 1.0
        leaves = self.tree[leaf_start:leaf_end]
        non_zero = leaves[leaves > 0]
        return non_zero.min() if len(non_zero) > 0 else 1.0
