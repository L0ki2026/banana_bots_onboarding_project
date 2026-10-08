"""Procedural grid maps with guaranteed path from start to goal."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from banana_bots_rl import config


@dataclass
class MapLayout:
    grid: np.ndarray  # shape (H, W), cell type constants
    start: Tuple[int, int]
    goal: Tuple[int, int]


def _walkable(grid: np.ndarray, r: int, c: int) -> bool:
    h, w = grid.shape
    if r < 0 or r >= h or c < 0 or c >= w:
        return False
    return grid[r, c] != config.OBSTACLE


def path_exists(grid: np.ndarray, start: Tuple[int, int], goal: Tuple[int, int]) -> bool:
    """BFS on walkable cells (empty, grass, goal)."""
    if start == goal:
        return True
    q = deque([start])
    seen = {start}
    while q:
        r, c = q.popleft()
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            if (nr, nc) in seen:
                continue
            if not _walkable(grid, nr, nc) and (nr, nc) != goal:
                continue
            if (nr, nc) == goal:
                return True
            if _walkable(grid, nr, nc):
                seen.add((nr, nc))
                q.append((nr, nc))
    return False


def _random_start_goal(rng: np.random.Generator, grid: np.ndarray) -> Optional[Tuple[Tuple[int, int], Tuple[int, int]]]:
    h, w = grid.shape
    candidates = [
        (r, c)
        for r in range(h)
        for c in range(w)
        if grid[r, c] in (config.EMPTY, config.GRASS)
    ]
    if len(candidates) < 2:
        return None
    for _ in range(200):
        i, j = rng.choice(len(candidates), size=2, replace=False)
        start, goal = candidates[i], candidates[j]
        if manhattan_dist(start, goal) >= config.MIN_PATH_LENGTH:
            return start, goal
    return None


def manhattan_dist(a: Tuple[int, int], b: Tuple[int, int]) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def generate_map(
    size: int = config.GRID_SIZE,
    obstacle_density: float = config.OBSTACLE_DENSITY,
    grass_density: float = config.GRASS_DENSITY,
    rng: Optional[np.random.Generator] = None,
    max_attempts: int = 100,
) -> MapLayout:
    """
    Build a random grid with obstacles and grass; ensure start/goal are connected.
    """
    if rng is None:
        rng = np.random.default_rng()

    for _ in range(max_attempts):
        grid = np.full((size, size), config.EMPTY, dtype=np.int32)

        n_cells = size * size
        n_obstacles = int(n_cells * obstacle_density)
        n_grass = int(n_cells * grass_density)

        indices = rng.permutation(n_cells)
        for idx in indices[:n_obstacles]:
            r, c = divmod(int(idx), size)
            grid[r, c] = config.OBSTACLE

        for idx in indices[n_obstacles : n_obstacles + n_grass]:
            r, c = divmod(int(idx), size)
            if grid[r, c] == config.EMPTY:
                grid[r, c] = config.GRASS

        pair = _random_start_goal(rng, grid)
        if pair is None:
            continue
        start, goal = pair
        grid[goal[0], goal[1]] = config.GOAL

        if path_exists(grid, start, goal):
            return MapLayout(grid=grid.copy(), start=start, goal=goal)

    raise RuntimeError("Failed to generate a solvable map; try lower obstacle density.")


def generate_eval_maps(
    n: int = config.NUM_EVAL_MAPS,
    seed: int = config.EVAL_SEED,
    **kwargs,
) -> list[MapLayout]:
    """Fixed held-out maps for generalization evaluation."""
    rng = np.random.default_rng(seed)
    return [generate_map(rng=rng, **kwargs) for _ in range(n)]
