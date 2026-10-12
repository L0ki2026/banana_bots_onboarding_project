"""Training curriculum: easy -> medium -> hard maps."""

from __future__ import annotations

from collections import deque
from typing import Deque

from stable_baselines3.common.callbacks import BaseCallback

from banana_bots_rl import config
from banana_bots_rl.environment import AssistiveNavigationEnv
from banana_bots_rl.map_generator import Difficulty


def unwrap_env(env):
    while hasattr(env, "env"):
        env = env.env
    return env


class CurriculumCallback(BaseCallback):
    """
    Advance map difficulty when rolling goal success exceeds a threshold.
    Operates on AssistiveNavigationEnv.difficulty (easy -> medium -> hard).
    """

    def __init__(
        self,
        env,
        success_threshold: float = config.CURRICULUM_SUCCESS_THRESHOLD,
        window_episodes: int = config.CURRICULUM_WINDOW_EPISODES,
        verbose: int = 1,
    ):
        super().__init__(verbose)
        self._env = env
        self.success_threshold = success_threshold
        self.window_episodes = window_episodes
        self._recent_success: Deque[float] = deque(maxlen=window_episodes)

    def _base_env(self) -> AssistiveNavigationEnv:
        return unwrap_env(self._env)

    def _on_step(self) -> bool:
        dones = self.locals.get("dones", [])
        infos = self.locals.get("infos", [])
        for done, info in zip(dones, infos):
            if not done:
                continue
            if "success" in info:
                self._recent_success.append(float(info["success"]))
        if len(self._recent_success) < self.window_episodes:
            return True

        rate = sum(self._recent_success) / len(self._recent_success)
        base = self._base_env()
        current: Difficulty = base.difficulty
        if rate < self.success_threshold:
            return True

        next_level: Difficulty | None = None
        if current == "easy":
            next_level = "medium"
        elif current == "medium":
            next_level = "hard"

        if next_level is None:
            return True

        base.difficulty = next_level
        self._recent_success.clear()
        if self.verbose:
            print(
                f"Curriculum: success {rate:.1%} over last {self.window_episodes} "
                f"episodes -> advancing to '{next_level}'"
            )
        return True
