"""Episode diagnostics for navigation loops."""

from __future__ import annotations

from typing import List, Sequence, Tuple

from banana_bots_rl.utils import manhattan


def count_two_cell_oscillations(positions: Sequence[Tuple[int, int]]) -> int:
    """
    Count A -> B -> A transitions where A and B are adjacent cells.
    positions should include the start position, then one entry after each step.
    """
    if len(positions) < 3:
        return 0
    count = 0
    for i in range(2, len(positions)):
        a, b, c = positions[i - 2], positions[i - 1], positions[i]
        if a == c and b != a and manhattan(a, b) == 1:
            count += 1
    return count


def count_repeated_positions(positions: Sequence[Tuple[int, int]]) -> int:
    """Number of times a position repeats (revisits), excluding the first visit."""
    seen = set()
    repeats = 0
    for p in positions:
        if p in seen:
            repeats += 1
        seen.add(p)
    return repeats


class PositionTracker:
    """Track agent positions within an episode for loop diagnostics."""

    def __init__(self) -> None:
        self.positions: List[Tuple[int, int]] = []

    def reset(self, start: Tuple[int, int]) -> None:
        self.positions = [start]

    def record(self, pos: Tuple[int, int]) -> None:
        self.positions.append(pos)

    @property
    def two_cell_oscillations(self) -> int:
        return count_two_cell_oscillations(self.positions)

    @property
    def repeated_positions(self) -> int:
        return count_repeated_positions(self.positions)
