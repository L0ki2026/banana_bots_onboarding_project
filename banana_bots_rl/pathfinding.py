"""Grid shortest-path utilities (BFS)."""

from __future__ import annotations

from collections import deque
from typing import Optional, Tuple

import numpy as np

from banana_bots_rl import config


def walkable(grid: np.ndarray, r: int, c: int) -> bool:
    h, w = grid.shape
    if r < 0 or r >= h or c < 0 or c >= w:
        return False
    return int(grid[r, c]) != config.OBSTACLE


def shortest_path_distance(
    grid: np.ndarray,
    start: Tuple[int, int],
    goal: Tuple[int, int],
) -> Optional[int]:
    """BFS shortest path length on walkable cells (empty, grass, goal)."""
    if start == goal:
        return 0
    if not walkable(grid, start[0], start[1]):
        return None
    q = deque([(start, 0)])
    seen = {start}
    while q:
        (r, c), d = q.popleft()
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            if (nr, nc) in seen:
                continue
            if (nr, nc) == goal:
                return d + 1
            if walkable(grid, nr, nc):
                seen.add((nr, nc))
                q.append(((nr, nc), d + 1))
    return None


def distance_field_to_goal(grid: np.ndarray, goal: Tuple[int, int]) -> np.ndarray:
    """
    For each walkable cell, shortest path steps to goal.
    Obstacles and unreachable walkable cells get a large sentinel distance.
    """
    h, w = grid.shape
    unreachable = h * w + 1
    dist = np.full((h, w), unreachable, dtype=np.int32)
    gr, gc = goal
    if not walkable(grid, gr, gc) and grid[gr, gc] != config.GOAL:
        return dist

    dist[gr, gc] = 0
    q = deque([(gr, gc)])
    while q:
        r, c = q.popleft()
        d = int(dist[r, c])
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            if not walkable(grid, nr, nc) and not (nr, nc) == (gr, gc):
                continue
            if dist[nr, nc] <= d + 1:
                continue
            dist[nr, nc] = d + 1
            q.append((nr, nc))
    return dist


def potential_phi(dist: int, unreachable: int) -> float:
    """phi(state) = -shortest_path_distance; unreachable uses worst-case distance."""
    return float(-dist)
