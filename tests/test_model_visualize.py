"""Checkpoint loading and visualization smoke tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from banana_bots_rl import config
from banana_bots_rl.environment import AssistiveNavigationEnv
from banana_bots_rl.map_generator import MapLayout
from banana_bots_rl.model_utils import load_ppo_for_env, validate_observation_space
from banana_bots_rl.train import build_model, make_train_env
from banana_bots_rl.visualize import play_episode_pygame


def test_checkpoint_save_and_load(tmp_path: Path) -> None:
    env = make_train_env(seed=0, reward_shaping=False, difficulty="easy")
    model = build_model(env, seed=0)
    ckpt = tmp_path / "test_policy.zip"
    model.save(str(ckpt))

    probe = AssistiveNavigationEnv(procedural=False)
    loaded = load_ppo_for_env(ckpt, probe)
    validate_observation_space(loaded, probe)


def test_play_episode_headless_short() -> None:
    grid = np.zeros((8, 8), dtype=np.int32)
    grid[7, 7] = config.GOAL
    layout = MapLayout(grid=grid, start=(3, 3), goal=(7, 7))
    env = AssistiveNavigationEnv(procedural=False, max_steps=10)
    result = play_episode_pygame(
        env,
        model=None,
        seed=123,
        layout=layout,
        max_steps=5,
        use_display=False,
    )
    assert len(result.frames) >= 2
    assert len(result.positions) >= 2
    assert result.steps <= 5
