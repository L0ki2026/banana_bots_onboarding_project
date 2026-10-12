"""Potential-based reward shaping wrapper."""

from __future__ import annotations

from typing import Any, Dict, Optional, SupportsFloat, Tuple

import gymnasium as gym
import numpy as np

from banana_bots_rl import config
from banana_bots_rl.pathfinding import distance_field_to_goal, potential_phi


class PotentialBasedRewardShaping(gym.Wrapper):
    """
    r' = r + beta * (gamma * phi(s') - phi(s)), phi = -shortest_path_distance to goal.

    Shaping is skipped when the agent position does not change (collisions).
    Terminal goal states use phi=0 at the goal. Truncation uses the same formula.
    """

    def __init__(
        self,
        env: gym.Env,
        *,
        beta: float = config.REWARD_SHAPING_BETA,
        gamma: float = config.PPO_GAMMA,
        enabled: bool = True,
    ):
        super().__init__(env)
        self.beta = beta
        self.gamma = gamma
        self.enabled = enabled
        self._dist_field: Optional[np.ndarray] = None
        self._unreachable: int = 1

    def _phi_at(self, pos: Tuple[int, int]) -> float:
        assert self._dist_field is not None
        r, c = pos
        return potential_phi(int(self._dist_field[r, c]), self._unreachable)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        base = self.env.unwrapped
        self._unreachable = base.grid_size * base.grid_size + 1
        self._dist_field = distance_field_to_goal(base.grid, base.goal_pos)
        info = dict(info)
        info["reward_shaping"] = self.enabled
        return obs, info

    def step(self, action):
        base = self.env.unwrapped
        pos_before = base.agent_pos
        phi_before = self._phi_at(pos_before) if self.enabled else 0.0

        obs, reward, terminated, truncated, info = self.env.step(action)
        info = dict(info)
        original = float(reward)

        if self.enabled and not info.get("collision", False):
            pos_after = base.agent_pos
            if terminated and pos_after == base.goal_pos:
                phi_after = 0.0
            else:
                phi_after = self._phi_at(pos_after)
            shaping = self.beta * (self.gamma * phi_after - phi_before)
            reward = original + shaping
            info["reward_shaping_bonus"] = shaping
        else:
            info["reward_shaping_bonus"] = 0.0

        info["reward_original"] = original
        return obs, reward, terminated, truncated, info
