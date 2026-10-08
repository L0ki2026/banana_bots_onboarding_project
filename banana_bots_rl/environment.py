"""Custom Gymnasium environment for assistive navigation."""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from banana_bots_rl import config
from banana_bots_rl.map_generator import MapLayout, generate_map


# Facing: 0=North (-row), 1=East (+col), 2=South (+row), 3=West (-col)
DIR_DELTA = [(-1, 0), (0, 1), (1, 0), (0, -1)]
# Strafe left/right relative to facing (forward is index 0)
STRAFE_LEFT = [(0, -1), (-1, 0), (0, 1), (1, 0)]
STRAFE_RIGHT = [(0, 1), (1, 0), (0, -1), (-1, 0)]


class AssistiveNavigationEnv(gym.Env):
    """
    2D grid navigation with partial observations and orientation-based actions.

    Actions (discrete):
        0 forward, 1 backward, 2 strafe left, 3 strafe right (all relative to facing).
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": config.FPS}

    def __init__(
        self,
        grid_size: int = config.GRID_SIZE,
        view_distance: int = config.VIEW_DISTANCE,
        max_steps: int = config.MAX_STEPS,
        render_mode: Optional[str] = None,
        fixed_layout: Optional[MapLayout] = None,
        procedural: bool = True,
        rng: Optional[np.random.Generator] = None,
    ):
        super().__init__()
        self.grid_size = grid_size
        self.view_distance = view_distance
        self.max_steps = max_steps
        self.render_mode = render_mode
        self.fixed_layout = fixed_layout
        self.procedural = procedural and fixed_layout is None
        self.rng = rng if rng is not None else np.random.default_rng()

        self.action_space = spaces.Discrete(4)

        patch_side = 2 * view_distance + 1
        # Local cell types (unknown = -1 encoded as 4 in one-hot) + goal deltas + facing one-hot
        self._n_cell_types = 5  # empty, grass, obstacle, goal, unknown
        patch_size = patch_side * patch_side * self._n_cell_types
        obs_low = np.zeros(patch_size + 2 + 4, dtype=np.float32)
        obs_high = np.ones(patch_size + 2 + 4, dtype=np.float32)
        # Normalized goal direction can be negative
        obs_low[patch_size : patch_size + 2] = -1.0
        obs_high[patch_size : patch_size + 2] = 1.0
        self.observation_space = spaces.Box(
            low=obs_low,
            high=obs_high,
            dtype=np.float32,
        )

        self.grid: np.ndarray = np.zeros((grid_size, grid_size), dtype=np.int32)
        self.agent_pos: Tuple[int, int] = (0, 0)
        self.goal_pos: Tuple[int, int] = (0, 0)
        self.facing: int = 0
        self.steps: int = 0
        self.collision_count: int = 0

    def _get_obs(self) -> np.ndarray:
        return self._build_observation()

    def _build_observation(self) -> np.ndarray:
        """Local patch in agent-centric frame + normalized goal direction + facing."""
        side = 2 * self.view_distance + 1
        patch = np.zeros((side, side, self._n_cell_types), dtype=np.float32)

        for pr in range(side):
            for pc in range(side):
                # Agent-centric: forward = up in patch (pr decreases)
                dr = self.view_distance - pr
                dc = pc - self.view_distance
                # Rotate (dr, dc) from agent frame to world frame
                world_dr, world_dc = self._agent_to_world(dr, dc)
                wr, wc = self.agent_pos[0] + world_dr, self.agent_pos[1] + world_dc

                if not self._in_bounds(wr, wc):
                    patch[pr, pc, 4] = 1.0  # unknown / out of map
                    continue

                if not self._is_visible(wr, wc):
                    patch[pr, pc, 4] = 1.0
                    continue

                cell = self.grid[wr, wc]
                if cell == config.EMPTY:
                    patch[pr, pc, 0] = 1.0
                elif cell == config.GRASS:
                    patch[pr, pc, 1] = 1.0
                elif cell == config.OBSTACLE:
                    patch[pr, pc, 2] = 1.0
                elif cell == config.GOAL:
                    patch[pr, pc, 3] = 1.0
                else:
                    patch[pr, pc, 0] = 1.0

        gx = self.goal_pos[0] - self.agent_pos[0]
        gy = self.goal_pos[1] - self.agent_pos[1]
        norm = max(1.0, np.hypot(gx, gy))
        goal_vec = np.array([gx / norm, gy / norm], dtype=np.float32)

        facing_oh = np.zeros(4, dtype=np.float32)
        facing_oh[self.facing] = 1.0

        return np.concatenate([patch.flatten(), goal_vec, facing_oh]).astype(np.float32)

    def _agent_to_world(self, dr: int, dc: int) -> Tuple[int, int]:
        """Map agent-relative (forward=+dr up) offsets to world grid deltas."""
        # facing 0=N: world (-dr, dc) with forward = decrease row
        if self.facing == 0:
            return -dr, dc
        if self.facing == 1:
            return dc, dr
        if self.facing == 2:
            return dr, -dc
        return -dc, -dr

    def _is_visible(self, r: int, c: int) -> bool:
        """Line-of-sight within view_distance (Bresenham from agent)."""
        ar, ac = self.agent_pos
        if r == ar and c == ac:
            return True
        dist = abs(r - ar) + abs(c - ac)
        if dist > self.view_distance:
            return False
        return self._line_clear(ar, ac, r, c)

    def _line_clear(self, r0: int, c0: int, r1: int, c1: int) -> bool:
        """All cells along line except target may block sight; obstacles block."""
        cells = self._bresenham(r0, c0, r1, c1)
        for r, c in cells[:-1]:
            if self.grid[r, c] == config.OBSTACLE:
                return False
        return True

    @staticmethod
    def _bresenham(r0: int, c0: int, r1: int, c1: int) -> list[Tuple[int, int]]:
        points = []
        dr = abs(r1 - r0)
        dc = abs(c1 - c0)
        sr = 1 if r0 < r1 else -1
        sc = 1 if c0 < c1 else -1
        err = dr - dc
        r, c = r0, c0
        while True:
            points.append((r, c))
            if r == r1 and c == c1:
                break
            e2 = 2 * err
            if e2 > -dc:
                err -= dc
                r += sr
            if e2 < dr:
                err += dr
                c += sc
        return points

    def _in_bounds(self, r: int, c: int) -> bool:
        return 0 <= r < self.grid_size and 0 <= c < self.grid_size

    def _apply_layout(self, layout: MapLayout) -> None:
        self.grid = layout.grid.copy()
        self.agent_pos = layout.start
        self.goal_pos = layout.goal
        self.facing = int(self.rng.integers(0, 4))

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        if options and "layout" in options:
            self._apply_layout(options["layout"])
        elif self.fixed_layout is not None:
            self._apply_layout(self.fixed_layout)
        elif self.procedural:
            layout = generate_map(size=self.grid_size, rng=self.rng)
            self._apply_layout(layout)
        else:
            raise ValueError("No layout source: set procedural, fixed_layout, or pass options['layout'].")

        self.steps = 0
        self.collision_count = 0
        obs = self._get_obs()
        return obs, {}

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        assert self.action_space.contains(action)
        self.steps += 1
        reward = config.STEP_PENALTY
        collision = False

        dr, dc = 0, 0
        if action == 0:
            dr, dc = DIR_DELTA[self.facing]
        elif action == 1:
            dr, dc = -DIR_DELTA[self.facing][0], -DIR_DELTA[self.facing][1]
        elif action == 2:
            dr, dc = STRAFE_LEFT[self.facing]
        elif action == 3:
            dr, dc = STRAFE_RIGHT[self.facing]

        nr, nc = self.agent_pos[0] + dr, self.agent_pos[1] + dc
        if not self._in_bounds(nr, nc) or self.grid[nr, nc] == config.OBSTACLE:
            collision = True
            self.collision_count += 1
            reward += config.COLLISION_PENALTY
        else:
            self.agent_pos = (nr, nc)
            cell = self.grid[nr, nc]
            if cell == config.GRASS:
                reward += config.GRASS_PENALTY

        terminated = self.agent_pos == self.goal_pos
        if terminated:
            reward += config.GOAL_REWARD

        truncated = self.steps >= self.max_steps
        obs = self._get_obs()
        info = {
            "collision": collision,
            "collisions_total": self.collision_count,
            "steps": self.steps,
            "success": terminated,
        }
        return obs, float(reward), terminated, truncated, info

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()
        return None

    def _render_frame(self) -> np.ndarray:
        """RGB frame for pygame / Colab (lazy import in visualize module)."""
        from banana_bots_rl.visualize import render_grid_to_array

        return render_grid_to_array(
            self.grid,
            self.agent_pos,
            self.goal_pos,
            self.facing,
        )
